#!/usr/bin/env python3

# region         === importations ===         NOIGER #
from dataclasses import dataclass
from functools import partial
import logging
from typing import Callable

from src.metrics import postprocess_text

FORMAT = '\033[01;31m%(asctime)s\033[0m %(message)s'
logging.basicConfig(format=FORMAT)
logging.warning('== START ==')

from pathlib import Path
PRETRAINED_PREFIX = Path("pret")
CKPT_PREFIX = Path("ckpts")
EXP_PREFIX = Path("exp")

Path(PRETRAINED_PREFIX / "hf_pretrains").mkdir(0o755, parents=True, exist_ok=True)    
Path(PRETRAINED_PREFIX / "hf_toks").mkdir(0o755, parents=True, exist_ok=True)    

import os
import sys

from datetime import datetime
strftime, now = datetime.strftime, datetime.now

import numpy as np

from torch.optim import AdamW
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer
from transformers.models.auto.configuration_auto import AutoConfig
from transformers import get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from torchmetrics import WordErrorRate, SacreBLEUScore

# --- self written --- #
from src.models import SentBartForConditionalGeneration
from src.utils import get_args
from src.utils import load_cached_tokenizer
from src.utils import load_cached_config
from src.datasets import DataSetCollectorGeneral
from src.datasets import Data_collate_fn
from src.trainers import LogCallback, TrainerCallback
from src.config_registers import TASK_CONFIG_DICT
from src.plmodel import PLModel
# endregion      === importations ===      NOIGERDNE #
logging.warning('== import DONE ==')



def main():
    args = get_args()
    os.environ['WANDB_PROJECT'] = args.proj_name
    task_config = TASK_CONFIG_DICT(args.coll)[args.task]

    # === collect dataset for setup === #
    def LIBRISPEECH_UNIT_ASR_SPLIT(split, lower=args.lower):
        return DataSetCollectorGeneral(
            args.datapath, 
            split=split, 
            lower=lower,
            dtype2subdir_ext=task_config.data_structure_def,
        )

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

    train_dataset = (LIBRISPEECH_UNIT_ASR_SPLIT(args.train_split)
        if args.train_split != 'none' else None)

    dev_dataset = (LIBRISPEECH_UNIT_ASR_SPLIT(args.dev_split)
        if args.dev_split != 'none' else None)
    
    test_dataset = (LIBRISPEECH_UNIT_ASR_SPLIT(args.test_split)
        if args.test_split is not None else None)

    # === experiment setup === #
    exp_config = dict(
        collapse_n=(-1 if args.original else args.collapse_n),  # default = 0
        **(dict(weight_len=args.weight_len) if args.weight_len is not None else {}),
        **(dict(use_self=True) if args.use_self else dict(use_self=False)),
        **(dict(minimize_len=True) if args.minimize_len else dict(minimize_len=False)),
    )

    # === build the model === #
    if args.scratch:
        config = AutoConfig.from_pretrained(args.pretrained_name, **exp_config)
        config.update(exp_config)
        model = SentBartForConditionalGeneration(config)
    else:
        model = load_cached_config(
            SentBartForConditionalGeneration,
            # "voidful/asr_hubert_cluster_bart_base",
            args.pretrained_name,
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
        
    return args, task_config, model, tokenizer, train_dataset, dev_dataset, test_dataset, collate_fn
        
       
        


if __name__ == "__main__": 
    args, task_config, model, tokenizer, train_dataset, dev_dataset, test_dataset, collate_fn = main()
    


    # main2()

    plmodel_kwargs = dict(
        model=model,
        datasets=(train_dataset, dev_dataset, test_dataset),
        tokenizer=tokenizer,
        hparams=dict(
            lr=args.lr,
            batch_size=args.batch_size,  # train val different? FIXME
            warmup_ratio=args.warmup_ratio,
            eval_batch_size=args.eval_batch_size,
            generation_max_length=args.generation_max_length,
            verbose_batch=args.verbose_batch,
            metric_batch=args.metric_batch,
            eval_in_train=args.eval_in_train,
            num_beams=args.num_beams,
            weight_decay=args.weight_decay,
        ),
        collate_fn=collate_fn,
        taskconfig=task_config,
    )
    if args.resume_from_checkpoint is None:
        plmodel = PLModel(**plmodel_kwargs)
    else:
        plmodel = PLModel.load_from_checkpoint(
            args.resume_from_checkpoint,
            **plmodel_kwargs)

    



    if "check data":
        print()
        if args.train_split != 'none':
            print()
            print('Checking Trainloader...')
            for i in plmodel.train_dataloader():
                print(i)
                break
        if args.dev_split != 'none':
            print()
            print('Checking Evalloader...')
            for i in plmodel.val_dataloader():
                print(i)
                break
        if args.test_split is not None:
            print()
            print('Checking Testloader...')
            for i in plmodel.test_dataloader():
                print(i)
                break









    checkpoint_callback = ModelCheckpoint(
        # ref.: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html
        # ref.: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
        monitor=f"valid_{task_config.metric_pl.metricname}",
        mode=task_config.metric_pl.metric_mode,
        save_top_k=args.save_total_limit,
        every_n_train_steps=args.save_steps,
        save_on_train_epoch_end=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args.mode != 'train': 
        args.wandb = True
    trainer = pl.Trainer(
        gpus=-1,
        logger=WandbLogger(args.run_name, project=args.proj_name) if args.wandb else True,
        log_every_n_steps=args.logging_steps,
        val_check_interval=min(args.eval_steps / len(plmodel.train_dataloader()), 1.0),
        default_root_dir=args.output_dir,
        max_epochs=args.epochs,
        strategy=DDPStrategy(find_unused_parameters=True) if any('--local_rank' in i for i in sys.argv) else None,
        enable_progress_bar=True,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        callbacks=[checkpoint_callback, lr_monitor],
    )
    if args.mode == "train":
        trainer.fit(
            plmodel,
            ckpt_path=args.resume_from_checkpoint,
        )  # dataloader here? <-- FIXME
    else:
        pass
    if args.mode == 'saveHf':
        assert args.hf_tosave is not None
        plmodel.model.save_pretrained(args.hf_tosave)

# TODO: from scratch --> yaml config (一個 config 一個 setting)
# TODO: AE pretraining
# TODO: Speech translation
# TODO: return Self WORDLEN pred output! (penalized? 叫他自己用 sum of alphas 當成 wordlen)
# FIXME: ddp repeat 3 times?
# NOT TODO: other losses? every batch
