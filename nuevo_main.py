#!/usr/bin/env python3

# region         === importations ===         NOIGER #
import logging
FORMAT = '\033[01;31m%(asctime)s\033[0m %(message)s'
logging.basicConfig(format=FORMAT)
logging.warning('== START ==')

import pathlib
LOG_WANDB = True
# LOG_WANDB = False
MAXUNITLEN = 1024
MAXTEXTLEN = 512
DATADIR_PREFIX = pathlib.Path("data/fairseq_data/data")
PRETRAINED_PREFIX = pathlib.Path("pret")
CKPT_PREFIX = pathlib.Path("ckpts")
EXP_PREFIX = pathlib.Path("exp")

pathlib.Path(EXP_PREFIX / "hf_ckpts/basic_trial1"
    ).mkdir(0o755, parents=True, exist_ok=True)    
pathlib.Path(PRETRAINED_PREFIX / "hf_pretrains"
    ).mkdir(0o755, parents=True, exist_ok=True)    
pathlib.Path(PRETRAINED_PREFIX / "hf_toks"
    ).mkdir(0o755, parents=True, exist_ok=True)    

import os
os.environ['WANDB_PROJECT'] = "HuggingFaceSentASR_May05"

import sys
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
from transformers import BartForConditionalGeneration
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import BartTokenizer
from transformers import BartConfig
from transformers.deepspeed import is_deepspeed_zero3_enabled
from datasets import load_metric

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.strategies.ddp import DDPStrategy
from torchmetrics import WordErrorRate

# --- self written --- #
from src.newmodels import SentBartForConditionalGeneration as BartForConditionalGeneration
from src.newmodels import pure_advanced_load_pretrained
from src.newmodels import advanced_load_pretrained

from src.newutils import get_args
from src.build_tok import build_tokenizer
# endregion      === importations ===      NOIGERDNE #

# region       === classes ===        NOIGER #
class MyUnitDataset(Dataset):
    def __init__(self, units, texts=None, wordlen=None): 
        self.units = units
        if texts is not None:
            assert len(texts) == len(self.units)
        self.texts = texts
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

class AugSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
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



def DataSetCollector(infix, collapsed=True):
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

    mydataset = MyUnitDataset(original_units, texts, wordlens)

    return mydataset

# TODO: 獨立出去 (可以較晚 XXX)
def DataSetCollectorUnlength(infix, collapsed=True):
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

def Data_collate_fn(unit_tokenizer, text_tokenizer):
    # done: combine & 應該要都可以處理，沒 label 或 length
    def collate_fn(batch):
        input_ids, labels, wordlens = list(zip(*batch))
        output_dict = dict(
            **unit_tokenizer(
                list(input_ids), 
                return_tensors='pt', 
                padding=True, 
                truncation=True))
        if labels[0] is not None:
            output_dict["labels"] = text_tokenizer(
                list(labels), 
                return_tensors='pt', 
                padding=True, 
                truncation=True)['input_ids']
        if wordlens[0] is not None:
            output_dict["word_length_tensor"] = torch.tensor(
                np.array(wordlens, dtype=int))
        return output_dict
    return collate_fn
   
def load_cached(cls, obj_name, saved_path, msg="Loading ..."):
    logging.warning(msg)
    if list(pathlib.Path(saved_path).glob('*')) == []:
        pathlib.Path(saved_path).rmdir()
    if os.path.isdir(saved_path):
        logging.warning('    (Using local cache...)')
        obj = cls.from_pretrained(saved_path)
    else:
        logging.warning('    (Loading pretrained...)')
        obj = cls.from_pretrained(obj_name)
        obj.save_pretrained(saved_path)
    return obj

def load_cached_tokenizer(cls, obj_name, saved_path, msg="Loading ..."):
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

        attention_masks = label_texts != -100
        sent_lengths = attention_masks.sum(1)
        overlapped = (predicted_texts == label_texts) * attention_masks
        accuracy = (overlapped.sum(1) / sent_lengths).mean(0).item()
        label_texts = [s[m] for s, m in zip(label_texts, attention_masks)]
        predicted_texts = [s[m] for s, m in zip(predicted_texts, attention_masks)]
        REAL = tokenizer.batch_decode(label_texts, skip_special_tokens=True)  # TODO: 直接傳進來！不用
        PRED = tokenizer.batch_decode(predicted_texts, skip_special_tokens=True)
        
        return {"acc": accuracy, "wer": metric.compute(predictions=PRED, references=REAL)}
    return fn

def compute_metrics_WER(tokenizer):  # For ASR, FIXME
    # 1. logits --> id (because of "generate")
    # 2. acc removed
    import pathlib
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

        for ni, bat in enumerate(batch(zip(label_texts, predicted_texts), 10)):
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
            if ni % 20 == 20 - 1:
                print(f"\n"f"Pred: \033[01;31m{bat_PRED[0]}\033[0m\n"f"Refe: \033[01;32m{bat_REAL[0]}\033[0m\n""")
        
        return {"wer": metric.compute()}
    return fn

logging.warning('== import DONE ==')
# endregion    === classes ===     NOIGERDNE #

if __name__ == "__main__":
    args = get_args()

    tokenizer = load_cached_tokenizer(
        cls=BartTokenizer,
        obj_name='facebook/bart-base',
        saved_path=PRETRAINED_PREFIX / "hf_toks",
        msg="Loading ...")

    collate_fn = Data_collate_fn(
        unit_tokenizer=tokenizer,
        text_tokenizer=tokenizer,
    )

    train_dataset = DataSetCollector('train')
    dev_dataset = DataSetCollector('dev')
    test_dataset = DataSetCollector('test')
    dummy_dataset = DataSetCollector('dummy')

    model = load_cached(
        BartForConditionalGeneration,
        "voidful/asr_hubert_cluster_bart_base",
        PRETRAINED_PREFIX / "hf_pretrains",
    )

    # TODOLATER: unshared embeddings
    if model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    # CHECK XXX: resize embedding or config 正確的 embedding 數從頭？

    trainer = AugSeq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            run_name=args.run_name,
            output_dir=EXP_PREFIX / "hf_ckpts/basic_trial1"
                / pathlib.Path(strftime(now(), r'%Y%m%d_%H%M%S')),
            
            do_train=True,
            logging_steps=5,
            per_device_train_batch_size=args.batch_size,
            
            do_eval=True,
            eval_steps=500,
            evaluation_strategy="steps",
            eval_accumulation_steps=25,
            per_device_eval_batch_size=args.batch_size,
            predict_with_generate=True,
            generation_max_length=512,
            
            learning_rate=args.lr,
            warmup_ratio=0.1,
            
            report_to='wandb' if LOG_WANDB else 'none',
            
            num_train_epochs=args.epochs,
            save_steps=500,
        ),
        
        # optimizers=optimizers,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        # train_dataset=dummy_dataset,
        # eval_dataset=dummy_dataset,
        
        data_collator=collate_fn,
        
        compute_metrics=compute_metrics_WER(tokenizer),
    )

    trainer.train(
        # 也可以直接寫進 config!
        ignore_keys_for_eval=[
            'encoder_last_hidden_state', 
            'encoder_last_hidden_out_attention_mask',
        ] + getattr(model.config, "keys_to_ignore_at_inference", []),
    )
    # breakpoint()
    # from IPython import embed as e; e()

# TODO: compute_metrics or PL!
# TODO: from scratch --> yaml config (一個 config 一個 setting)
# TODO: AE pretraining
# TODO: Speech translation
# TODO: return Self WORDLEN pred output! (penalized? 叫他自己用 sum of alphas 當成 wordlen)
