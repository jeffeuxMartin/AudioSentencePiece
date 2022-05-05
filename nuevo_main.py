#!/usr/bin/env python3

import logging
FORMAT = '\033[01;31m%(asctime)s\033[0m %(message)s'
logging.basicConfig(format=FORMAT)
logging.warning('== START ==')

LOG_WANDB = True
MAXUNITLEN = 1024
MAXTEXTLEN = 512

import os
os.environ['WANDB_PROJECT'] = "HuggingFaceSentASR"

import sys
from itertools import groupby
from pprint import pprint

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

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torchmetrics import WordErrorRate

from src.newmodels import SentBartForConditionalGeneration
from src.newmodels import pure_advanced_load_pretrained
from src.newmodels import advanced_load_pretrained

from src.build_tok import build_tokenizer


class MyUnitDataset(Dataset):
    def __init__(self, units, texts): 
        self.units = units
        self.texts = texts
    def __len__(self): 
        return len(self.units)
    def __getitem__(self, idx): 
        return self.units[idx], self.texts[idx]


def DataSetCollector(infix, collapsed=True):
    suffix = "_coll" if collapsed else ""

    logging.warning('== ....      ==')
    with open('/storage/LabJob/Projects'
        '/FairseqCollapse'
       f'/data/train-clean-100{suffix}/{infix}.en') as f:
        texts = f.read().split('\n')

    with open('/storage/LabJob/Projects'
        '/FairseqCollapse'
       f'/data/train-clean-100{suffix}/{infix}.unit') as f:
        original_units = f.read().split('\n')

    mydataset = MyUnitDataset(original_units, texts)

    return mydataset

def Data_collate_fn(unit_tokenizer, text_tokenizer):
    def collate_fn(batch):
        input_ids, labels = list(zip(*batch))
        return dict(
            **unit_tokenizer(
                list(input_ids), 
                return_tensors='pt', 
                padding=True, 
                truncation=True),
            labels=text_tokenizer(
                list(labels), 
                return_tensors='pt', 
                padding=True, 
                truncation=True)['input_ids'],
        )
    return collate_fn

def load_cached(cls, obj_name, saved_path, msg="Loading ..."):
    logging.warning(msg)
    if os.path.isdir(saved_path):
        logging.warning('    (Using local cache...)')
        obj = cls.from_pretrained(saved_path)
    else:
        logging.warning('    (Loading pretrained...)')
        obj = cls.from_pretrained(obj_name)
        obj.save_pretrained(saved_path)
    return obj

def load_cached_tokenizer(cls, obj_name, saved_path, msg="Loading ..."):
    tokenizer = load_cached(cls, obj_name, saved_path, msg)
    speech_units = ['uni_{:04d}'.format(d) for d in range(500)]  # TODOLATER: modify format
    if speech_units[0] not in tokenizer.get_vocab():  
        tokenizer.add_tokens(speech_units)
        tokenizer.save_pretrained(saved_path)
    return tokenizer

logging.warning('== import DONE ==')

tokenizer = load_cached_tokenizer(
    cls=BartTokenizer,
    obj_name='facebook/bart-base',
    saved_path="/tmp/hf_toks",
    msg="Loading ...")

collate_fn = Data_collate_fn(
    unit_tokenizer=tokenizer,
    text_tokenizer=tokenizer,
)

train_dataset = DataSetCollector('train')
dev_dataset = DataSetCollector('dev')
test_dataset = DataSetCollector('test')

model = load_cached(
    BartForConditionalGeneration,
    "voidful/asr_hubert_cluster_bart_base",
    "/tmp/hf_pretrains",
)
    
# TODOLATER: unshared embeddings
if model.config.vocab_size != len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="/tmp/hf_ckpts/basic_trial1",
        do_train=True,
        logging_steps=1,
        per_device_train_batch_size=6,
        
        do_eval=True,
        eval_steps=50,
        evaluation_strategy="steps",
        per_device_eval_batch_size=6,
        
        learning_rate=2e-4,
        warmup_steps=100,
        
        report_to='none',
        
        num_train_epochs=1,
        save_steps=500,
    ),
    
    # optimizers=optimizers,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=collate_fn,
)

trainer.train()
# from IPython import embed as e; e()
