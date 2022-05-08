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
from src.models import SentBartForConditionalGeneration

from src.utils import get_args
from src.build_tok import build_tokenizer

# endregion      === importations ===      NOIGERDNE #

# region       === classes ===        NOIGER #

from src.utils import load_cached_tokenizer
from src.utils import load_cached_config

from src.datasets import DataSetCollectorGeneral
from src.datasets import Data_collate_fn

from src.trainers import LogCallback
from src.trainers import AugSeq2SeqTrainer
from src.trainers import AugTrainer
from src.metrics import compute_metrics_WER
from src.metrics import compute_metrics_WER_logits

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
