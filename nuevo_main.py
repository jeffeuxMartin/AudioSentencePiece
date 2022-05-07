#!/usr/bin/env python3

# region         === importations ===         NOIGER #
import logging
from typing import Dict, Optional

from transformers import training_args
FORMAT = '\033[01;31m%(asctime)s\033[0m %(message)s'
logging.basicConfig(format=FORMAT)
logging.warning('== START ==')

import pathlib
from pathlib import Path
MAXUNITLEN = 1024
MAXTEXTLEN = 512
DATADIR_PREFIX = pathlib.Path("data/fairseq_data/data")
PRETRAINED_PREFIX = pathlib.Path("pret")
CKPT_PREFIX = pathlib.Path("ckpts")
EXP_PREFIX = pathlib.Path("exp")
LIBRISPEECH_UNIT_PATH = "data/LibriSpeechUnits"

pathlib.Path(EXP_PREFIX / "hf_ckpts/basic_trial1"
    ).mkdir(0o755, parents=True, exist_ok=True)    
pathlib.Path(PRETRAINED_PREFIX / "hf_pretrains"
    ).mkdir(0o755, parents=True, exist_ok=True)    
pathlib.Path(PRETRAINED_PREFIX / "hf_toks"
    ).mkdir(0o755, parents=True, exist_ok=True)    

import os
os.environ['WANDB_PROJECT'] = "HuggingFaceSentASR_May08"
# os.environ['WANDB_PROJECT'] = "HFDebug"

import sys
from dataclasses import dataclass
from itertools import groupby
from pprint import pprint
from datetime import datetime
strftime, now = datetime.strftime, datetime.now

import numpy as np
import pandas as pd
from tqdm import tqdm   
from tqdm.contrib.concurrent import process_map

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
# from transformers import BartForConditionalGeneration
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import BartTokenizer
from transformers import BartConfig
from transformers import AutoConfig
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_sagemaker_mp_enabled
from transformers.utils import is_apex_available
from transformers.trainer_callback import TrainerState
from datasets import load_metric
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp  # noqa
    from transformers.trainer_pt_utils import smp_forward_backward
if is_apex_available():
    from apex import amp  # noqa
from transformers.trainer_pt_utils import nested_detach


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.strategies.ddp import DDPStrategy
from torchmetrics import WordErrorRate

# --- self written --- #
from src.newmodels import SentBartForConditionalGeneration
from src.newmodels import pure_advanced_load_pretrained
from src.newmodels import advanced_load_pretrained

from src.newutils import get_args
from src.build_tok import build_tokenizer

# endregion      === importations ===      NOIGERDNE #

# region       === classes ===        NOIGER #
class MyUnitDataset(Dataset):
    def __init__(self, units, texts=None, wordlen=None, lower=False): 
        self.units = units
        if texts is not None:
            assert len(texts) == len(self.units)
        self.texts = texts
        if lower:
            self.texts = [s.lower() for s in self.texts]
        if wordlen is not None:
            assert len(wordlen) == len(self.units)
        self.wordlen = wordlen
    def __len__(self): 
        return len(self.units)
    def __getitem__(self, idx): 
        return (
            self.units[idx], 
            self.texts[idx] 
                if self.texts is not None else 
                None,
            self.wordlen[idx] 
                if self.wordlen is not None else 
                None,
        )

@dataclass
class AugTrainerState(TrainerState):
    masked_lm_loss: Optional[torch.FloatTensor] = None
    real_length_loss: Optional[torch.FloatTensor] = None

class LogCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        return super().on_train_begin(
            args, AugTrainerState(**(vars(state))), control, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        # print(dict(
        #     masked_lm_loss=state.masked_lm_loss,
        #     real_length_loss=state.real_length_loss,
        # ))
        pass

    def on_log(self, args, state, control, **kwargs):
        pass


class AugTrainer(Trainer):
    def log(self, logs):
        # return super().log(logs)
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if getattr(self.state, "masked_lm_loss", None) is not None:
            logs["masked_lm_loss"] = round(self.state.masked_lm_loss, 2)
        if getattr(self.state, "real_length_loss", None) is not None:
            logs["real_length_loss"] = round(self.state.real_length_loss, 2)
            
        output = {
            **logs, 
            **{"step": self.state.global_step},
        }
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs)

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss, other_outputs = self.compute_loss(model, inputs, return_outputs=True)
            if 'masked_lm_loss' in other_outputs:
                self.state.masked_lm_loss = other_outputs.get('masked_lm_loss').detach().item()
            if 'real_length_loss' in other_outputs:
                self.state.real_length_loss = other_outputs.get('real_length_loss').detach().item()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
      ):
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():  # nottodo~~
                raw_outputs = smp_forward_only(model, inputs)  # noqa
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)  # noqa
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)  # noqa
            else:
                if has_labels:
                    with self.autocast_smart_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()
                    # @@
                    self.state.masked_lm_loss = outputs.masked_lm_loss.detach().item()
                    self.state.real_length_loss = outputs.real_length_loss.detach().item()
                    # $$

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**inputs)
                    # @@
                    self.state.masked_lm_loss = outputs.masked_lm_loss.detach().item()
                    self.state.real_length_loss = outputs.real_length_loss.detach().item()
                    # $$
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

class AugSeq2SeqTrainer(Seq2SeqTrainer, AugTrainer):
    def prediction_step(self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
      ):
        if not self.args.predict_with_generate or prediction_loss_only:
            return super(Seq2SeqTrainer, self).prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)
        if "word_length_tensor" in inputs:
            gen_kwargs["word_length_tensor"] = inputs.get("word_length_tensor", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)



def DataSetCollector(infix, collapsed=True, lower=False):
    suffix = "_coll" if collapsed else ""

    logging.warning('== ....      ==')
    with open(DATADIR_PREFIX /
       f'train-clean-100{suffix}/{infix}.en') as f:
        texts = f.read().strip().split('\n')

    with open(DATADIR_PREFIX /
       f'train-clean-100{suffix}/{infix}.unit') as f:
        original_units = f.read().strip().split('\n')

    with open(DATADIR_PREFIX /
       f'train-clean-100{suffix}/{infix}.len') as f:
        wordlens = f.read().strip().split('\n')
    
    assert len(texts) == len(original_units)
    assert len(wordlens) == len(original_units)

    mydataset = MyUnitDataset(original_units, texts, wordlens, lower=lower)

    return mydataset
    
def DataSetCollectorBetter(filepath, lower=False):

    logging.warning('== ....      ==')
    with open(f'{filepath}.en') as f:
        texts = f.read().strip().split('\n')

    with open(f'{filepath}.unit') as f:
        original_units = f.read().strip().split('\n')

    with open(f'{filepath}.len') as f:
        wordlens = f.read().strip().split('\n')

    assert len(texts) == len(original_units)
    assert len(wordlens) == len(original_units)

    mydataset = MyUnitDataset(original_units, texts, wordlens, lower=lower)

    return mydataset


# TODO: 獨立出去 (可以較晚 XXX)
def DataSetCollectorUnlength(infix, collapsed=True, lower=False):
    suffix = "_coll" if collapsed else ""

    logging.warning('== ....      ==')
    with open(DATADIR_PREFIX /
       f'train-clean-100{suffix}/{infix}.en') as f:
        texts = f.read().strip().split('\n')

    with open(DATADIR_PREFIX /
       f'train-clean-100{suffix}/{infix}.unit') as f:
        original_units = f.read().strip().split('\n')

    mydataset = MyUnitDataset(original_units, texts)

    return mydataset

        
def DataSetCollectorGeneral(
    prefix_path, split, dtype2subdir_ext=None, lower=False,
):
    dtype2subdir_ext = ({} 
        if dtype2subdir_ext is None else 
        dtype2subdir_ext)
    dtype2subdir_ext_default = {
        'texts': dict(
            subdir='texts',
            ext='txt',
        ),
        'original_units': dict(
            subdir='collunits',
            ext='collunit',
        ),
        'wordlens': dict(
            subdir='lengths',
            ext='len',
        ),
    }
    
    dtype2subdir_ext_default.update(dtype2subdir_ext)
    dtype2subdir_ext = dtype2subdir_ext_default

    logging.warning('== ....      ==')
    with open(Path(prefix_path) / '{subdir}/{split}.{ext}'.format(
        split=split, 
        subdir=dtype2subdir_ext['texts']['subdir'],
        ext=dtype2subdir_ext['texts']['ext'],
    )) as f:
        texts = f.read().strip().split('\n')

    if 'texts' in dtype2subdir_ext:
        with open(Path(prefix_path) / '{subdir}/{split}.{ext}'.format(
            split=split, 
            subdir=dtype2subdir_ext['original_units']['subdir'],
            ext=dtype2subdir_ext['original_units']['ext'],
        )) as f:
            original_units = f.read().strip().split('\n')
        assert len(texts) == len(original_units)
    else:
        # print("NO "
        #       "\033[01;31m"
        #       "`{texts}`!"
        #       "\033[0m")
        texts = None

    if 'wordlens' in dtype2subdir_ext:
        with open(Path(prefix_path) / '{subdir}/{split}.{ext}'.format(
            split=split, 
            subdir=dtype2subdir_ext.get('wordlens', {}).get('subdir'),
            ext=dtype2subdir_ext.get('wordlens', {}).get('ext'),
        )) as f:
            wordlens = f.read().strip().split('\n')
        assert len(wordlens) == len(original_units)
    else:
        # print("NO "
        #       "\033[01;31m"
        #       "`{wordlens}`!"
        #       "\033[0m")
        wordlens = None

    mydataset = MyUnitDataset(original_units, texts, wordlens, lower=lower)

    return mydataset


def Data_collate_fn(unit_tokenizer, text_tokenizer):
    # done: combine & 應該要都可以處理，沒 label 或 length
    def prepend_append(tok):
        def f(s): return f"{tok.bos_token} {s} {tok.eos_token}"
        return f 
    unit_tokenizer.prepend_append = (prepend_append(unit_tokenizer) 
        if unit_tokenizer(['']).get("input_ids", None) == [[]] else 
        lambda x: x)
    text_tokenizer.prepend_append = (prepend_append(text_tokenizer) 
        if text_tokenizer(['']).get("input_ids", None) == [[]] else 
        lambda x: x)

    def collate_fn(batch):
        input_ids, labels, wordlens = list(zip(*batch))
        output_dict = dict(
            **unit_tokenizer(
                list(map(unit_tokenizer.prepend_append, input_ids)), 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=1024,
            ))
        if labels[0] is not None:
            output_dict["labels"] = text_tokenizer(
                list(map(text_tokenizer.prepend_append, labels)), 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=1024,
            )['input_ids']
        if wordlens[0] is not None:
            output_dict["word_length_tensor"] = torch.tensor(
                np.array(wordlens, dtype=int))
        return output_dict
    return collate_fn
   
def load_cached(cls, obj_name, saved_path, msg="Loading ..."):
    logging.warning(msg)
    if os.path.isdir(saved_path):
        if list(pathlib.Path(saved_path).glob('*')) == []:
            pathlib.Path(saved_path).rmdir()
    if os.path.isdir(saved_path):
        logging.warning('    (Using local cache...)')
        obj = cls.from_pretrained(saved_path)
        # obj = advanced_load_pretrained(obj_name, cls, type(config), **config)
    else:
        logging.warning('    (Loading pretrained...)')
        obj = cls.from_pretrained(obj_name)
        # obj = advanced_load_pretrained(obj_name, cls, type(config), **config)
        obj.save_pretrained(saved_path)
    return obj

def load_cached_config(cls, obj_name, saved_path, config=None, msg="Loading ..."):
    """ 
    If enter this, USE pretrained (local or remot) is confirmed! 
    We try to match maximized!
    Otherwise, 
        config = AutoConfig(collapse_n=-1)
        model = cls(config)
    """
    
    config = {} if config is None else config
    logging.warning(msg)
    if os.path.isdir(saved_path):
        if list(pathlib.Path(saved_path).glob('*')) == []:
            pathlib.Path(saved_path).rmdir()
    if os.path.isdir(saved_path):
        # print(saved_path)
        logging.warning('    (Using local cache...)')
        # obj = cls.from_pretrained(saved_path)
        obj = advanced_load_pretrained(saved_path, cls, AutoConfig, **config)
    else:
        pathlib.Path(saved_path
            ).mkdir(0o755, parents=True, exist_ok=True)    
        logging.warning('    (Loading pretrained...)')
        # pretrained_config = config
        # obj = cls.from_pretrained(obj_name, config)
        obj = advanced_load_pretrained(obj_name, cls, AutoConfig, **config)
        obj.save_pretrained(saved_path)
    return obj

def load_cached_tokenizer(cls, obj_name, saved_path, msg="Loading ..."):
    if os.path.isdir(saved_path):
        if list(pathlib.Path(saved_path).glob('*')) == []:
            pathlib.Path(saved_path).rmdir()
    tokenizer = load_cached(cls, obj_name, saved_path, msg)
    speech_units = ['uni_{:04d}'.format(d) for d in range(500)]  # TODOLATER: modify format
    if speech_units[0] not in tokenizer.get_vocab():  
        tokenizer.add_tokens(speech_units)
        tokenizer.save_pretrained(saved_path)
    return tokenizer

def compute_metrics_WER_logits(tokenizer):  # For ASR, FIXME
    def fn(eval_preds):  # For ASR, FIXME
        metric = load_metric("wer")
        predictions = eval_preds.predictions
        predicted_texts = predictions.argmax(-1)
        label_texts = eval_preds.label_ids

        attention_masks = (label_texts != -100) & (label_texts != tokenizer.pad_token_id)
        sent_lengths = attention_masks.sum(1)
        overlapped = (predicted_texts == label_texts) * attention_masks
        accuracy = (overlapped.sum(1) / sent_lengths).mean(0).item()
        label_texts = [s[m] for s, m in zip(label_texts, attention_masks)]
        predicted_texts = [s[m] for s, m in zip(predicted_texts, attention_masks)]
        REAL = tokenizer.batch_decode(label_texts, skip_special_tokens=True)  # TODO: 直接傳進來！不用
        PRED = tokenizer.batch_decode(predicted_texts, skip_special_tokens=True)
        
        return {"acc": accuracy, "wer": metric.compute(predictions=PRED, references=REAL)}
    return fn

def compute_metrics_WER(tokenizer, metric_batch=10, verbose_batch=20):  # For ASR, FIXME
    # 1. logits --> id (because of "generate")
    # 2. acc removed
    (pathlib.Path('./.cache/preds') / strftime(now(), r'%Y%m%d_%H%M%S')).mkdir(parents=True, exist_ok=True)
    def fn(eval_preds):  # For ASR, FIXME
        metric = load_metric("wer", cache_dir=(pathlib.Path('./.cache/preds') / strftime(now(), r'%Y%m%d_%H%M%S')))
        predicted_texts = eval_preds.predictions
        label_texts = eval_preds.label_ids

        def batch(iterable, n=1):
            iterable = list(iterable)
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        for ni, bat in enumerate(batch(zip(label_texts, predicted_texts), metric_batch)):
            bat_label_texts, bat_predicted_texts = zip(*bat)
            bat_label_texts = np.array(bat_label_texts)
            bat_label_texts[bat_label_texts == -100] = tokenizer.pad_token_id
            bat_predicted_texts = np.array(bat_predicted_texts)
            bat_REAL = tokenizer.batch_decode(bat_label_texts, skip_special_tokens=True)  # TODO: 直接傳進來！不用
            bat_PRED = tokenizer.batch_decode(bat_predicted_texts, skip_special_tokens=True)
            metric.add_batch(
                predictions=bat_PRED,
                references=bat_REAL,
            )
            if verbose_batch > 0:
                if ni % verbose_batch == verbose_batch - 1:
                    print(f"\n"f"Pred: \033[01;31m{bat_PRED[0]}\033[0m\n"f"Refe: \033[01;32m{bat_REAL[0]}\033[0m\n""")
        
        return {"wer": metric.compute()}
    return fn

logging.warning('== import DONE ==')
# endregion    === classes ===     NOIGERDNE #

if __name__ == "__main__":
    args = get_args()

    tokenizer = load_cached_tokenizer(
        cls=AutoTokenizer,
        # obj_name='facebook/bart-base',
        obj_name='facebook/s2t-wav2vec2-large-en-de',
        saved_path=PRETRAINED_PREFIX / "hf_toks" / "endeunit",
        msg="Loading ...")

    collate_fn = Data_collate_fn(
        unit_tokenizer=tokenizer,
        text_tokenizer=tokenizer,
    )

    train_dataset        = DataSetCollectorGeneral(LIBRISPEECH_UNIT_PATH, split='train-clean-100', lower=args.lower,
        dtype2subdir_ext={
            'original_units': dict(
                subdir='collunits' if args.coll else 'symbolunits',
                # subdir='symbolunits',
                ext='collunit' if args.coll else 'symbolunit',
                # ext='symbolunit',
            ),
            'texts': dict(
                subdir='texts',
                # subdir='collunits',
                # subdir='collunits' if args.coll else 'symbolunits',
                ext='txt',
                # ext='collunit',
                # ext='collunit' if args.coll else 'symbolunit',
            ),
        }
    )
    dev_dataset          = DataSetCollectorGeneral(LIBRISPEECH_UNIT_PATH, split='dev-clean', lower=args.lower,
        dtype2subdir_ext={
            'original_units': dict(
                subdir='collunits' if args.coll else 'symbolunits',
                # subdir='symbolunits',
                ext='collunit' if args.coll else 'symbolunit',
                # ext='symbolunit',
            ),
            'texts': dict(
                subdir='texts',
                # subdir='collunits',
                # subdir='collunits' if args.coll else 'symbolunits',
                ext='txt',
                # ext='collunit',
                # ext='collunit' if args.coll else 'symbolunit',
            ),
        }
    )
    # test_dataset       = DataSetCollectorGeneral(LIBRISPEECH_UNIT_PATH, split='test-clean', lower=args.lower)
    # dummy_dataset      = DataSetCollectorGeneral(LIBRISPEECH_UNIT_PATH, split='dummy', lower=args.lower)
    dummy_train_dataset = DataSetCollectorGeneral(LIBRISPEECH_UNIT_PATH, split='dummy', lower=args.lower,
        dtype2subdir_ext={
            'original_units': dict(
                subdir='collunits' if args.coll else 'symbolunits',
                ext='collunit' if args.coll else 'symbolunit',
            ),
            'texts': dict(
                subdir='texts',
                # subdir='collunits',
                # subdir='collunits' if args.coll else 'symbolunits',
                ext='txt',
                # ext='collunit',
                # ext='collunit' if args.coll else 'symbolunit',
            ),
        }
    )
    dummy_dev_dataset   = DataSetCollectorGeneral(LIBRISPEECH_UNIT_PATH, split='dummy', lower=args.lower,
        dtype2subdir_ext={
            'original_units': dict(
                subdir='collunits' if args.coll else 'symbolunits',
                # subdir='symbolunits',
                ext='collunit' if args.coll else 'symbolunit',
                # ext='symbolunit',
            ),
            'texts': dict(
                subdir='texts',
                # subdir='collunits',
                # subdir='collunits' if args.coll else 'symbolunits',
                ext='txt',
                # ext='collunit',
                # ext='collunit' if args.coll else 'symbolunit',
            ),
        }
    )

    exp_config = dict(
        collapse_n=-1 if args.original else 0,
    )
    if args.weight_len is not None:
        exp_config['weight_len'] = args.weight_len
    model = load_cached_config(
        SentBartForConditionalGeneration,
        # "voidful/asr_hubert_cluster_bart_base",
        "facebook/bart-base",
        saved_path=PRETRAINED_PREFIX / "hf_pretrains" / "default",
        config=exp_config,
    )
    assert all([getattr(model.config, i) == exp_config[i] for i in exp_config])

    # TODOLATER: unshared embeddings
    if model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    # CHECK XXX: resize embedding or config 正確的 embedding 數從頭？

    if args.fix_encoder:
        model.fix_encoder_()
        
    training_args_dict = dict(
        save_total_limit=3,
        run_name=args.run_name,
        output_dir=EXP_PREFIX / "hf_ckpts/basic_trial1"
            / pathlib.Path(strftime(now(), r'%Y%m%d_%H%M%S')),
        
        do_train=True,
        logging_steps=5,
        per_device_train_batch_size=args.batch_size,
        
        do_eval=True,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        eval_accumulation_steps=25,
        per_device_eval_batch_size=args.batch_size * 2,
        
        learning_rate=args.lr,
        warmup_ratio=0.1,
        
        report_to='wandb' if args.wandb else 'none',
        
        num_train_epochs=args.epochs,
        save_steps=500,
    )
    training_args_dict["predict_with_generate"] = True
    training_args_dict["generation_max_length"] = 1024
    
    
    autoencoder_option = args.autoencoder

    trainer_cls = (
        AugTrainer 
        if autoencoder_option else 
        AugSeq2SeqTrainer)
    trainer_args_cls = (
        TrainingArguments 
        if autoencoder_option else 
        Seq2SeqTrainingArguments)

    compute_metrics_fn = (
        compute_metrics_WER_logits(tokenizer)
        if autoencoder_option else 
        compute_metrics_WER(
            tokenizer,
            metric_batch=20,
            verbose_batch=50,
        ))

    trainer_args = dict(
        model=model,
        args=trainer_args_cls(**training_args_dict),
        
        # optimizers=optimizers,
        train_dataset=(
            train_dataset
            # dummy_train_dataset
        ),
        eval_dataset=(
            dev_dataset
            # dummy_dev_dataset
        ),
        
        data_collator=collate_fn,
        
        compute_metrics=compute_metrics_fn,
        callbacks=[LogCallback],
    )
    trainer = trainer_cls(**trainer_args)

    trainer.train(
        # 也可以直接寫進 config!
        ignore_keys_for_eval=[
            'encoder_last_hidden_state', 
            'encoder_last_hidden_out_attention_mask',
            'encoder_length_loss',
            'encoder_pred_word_lengths',
            'encoder_hidden_states',
            'encoder_attentions',
            'masked_lm_loss',  # store in other formats!
            'real_length_loss',  # store in other formats!
        ] + getattr(model.config, "keys_to_ignore_at_inference", []),
    )
    # breakpoint()
    # from IPython import embed as e; e()

# TODO: compute_metrics or PL!
# TODO: from scratch --> yaml config (一個 config 一個 setting)
# TODO: AE pretraining
# TODO: Speech translation
# TODO: return Self WORDLEN pred output! (penalized? 叫他自己用 sum of alphas 當成 wordlen)
# FIXME: ddp repeat 3 times?
# NOT TODO: other losses? every batch
