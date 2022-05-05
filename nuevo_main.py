#!/usr/bin/env python3

# region         === importations ===         NOIGER #
import logging
FORMAT = '\033[01;31m%(asctime)s\033[0m %(message)s'
logging.basicConfig(format=FORMAT)
logging.warning('== START ==')

import pathlib
LOG_WANDB = True
LOG_WANDB = False
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
from transformers import BartTokenizer
from transformers import BartConfig
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

def compute_metrics_WER(tokenizer):  # For ASR, FIXME
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

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            run_name=args.run_name,
            output_dir=EXP_PREFIX / "hf_ckpts/basic_trial1"
                / pathlib.Path(strftime(now(), r'%Y%m%d_%H%M%S')),
            
            do_train=True,
            logging_steps=1,
            per_device_train_batch_size=args.batch_size,
            
            do_eval=True,
            eval_steps=50,
            evaluation_strategy="steps",
            eval_accumulation_steps=15,
            per_device_eval_batch_size=args.batch_size,
            
            learning_rate=args.lr,
            warmup_steps=100,
            
            report_to='wandb' if LOG_WANDB else 'none',
            
            num_train_epochs=args.epochs,
            save_steps=500,
        ),
        
        # optimizers=optimizers,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        
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
