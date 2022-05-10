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

    train_dataset = LIBRISPEECH_UNIT_ASR_SPLIT(args.train_split)
    
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
        
        
# model, datasets, tokenizer, metric, hparams, collate_fn,
class PLModel(pl.LightningModule):
    def __init__(self,
        model, datasets, tokenizer, hparams, collate_fn, taskconfig,
    ):
        super().__init__()
        self.hparams.update(hparams)
        
        self.model = model
        self.tokenizer = tokenizer
        
        self.trainset, self.valset = datasets
        
        self.taskconfig = taskconfig
        self.metric = dict(
            train=self.taskconfig.metric_pl.metric(),
            valid=self.taskconfig.metric_pl.metric(),
        )
        self.collate_fn = collate_fn
        
    def forward(self, inputs):
        return self.model(**inputs)
        
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
        
    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.batch_size * max(1, self.trainer.num_devices)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size
        print(f"{tb_size = }")
        print(f"{ab_size = }")
        print(f"{self.total_steps = }")
        print(f"{len(train_loader.dataset) = }")

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.get("weight_decay", 0.1),
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr=self.hparams["lr"],
            eps=self.hparams.get('adam_epsilon', 1e-8))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.get(
                "warmup_steps", 
                self.hparams.get("warmup_ratio", 0.1) * self.total_steps,
            ),
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        assert self.training
        self.log(f"train_loss", outputs.loss, batch_size=self.hparams.batch_size, prog_bar=True) 
        
        # predicted_ids = outputs.logits.argmax(-1)
        # predicted_texts = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        # groundtruth_texts = batch["texts"]
        # if batch_idx % 10 == 0:
        predicted = False
        gen_len = None
        if self.hparams.eval_in_train and self.hparams.metric_batch > 0:
            if batch_idx % self.hparams.metric_batch == 0:
                ar_preds = self.generate(
                    **{
                        k: batch[k]
                        for k in batch
                        if k != 'labels'
                    }, 
                    num_beams=self.hparams.num_beams,
                    max_length=self.hparams.generation_max_length,
                )
                
                ar_texts = self.tokenizer.batch_decode(ar_preds, skip_special_tokens=True)
                ar_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                ar_texts, ar_labels = self.taskconfig.metric_pl.postprocess_fn(ar_texts, ar_labels)
                self.metric['train'] = self.metric['train'].to(ar_preds.device)
                self.metric['train'].update(preds=ar_texts, target=ar_labels)
                
                gen_len = np.mean([np.count_nonzero(pred.detach().cpu().numpy() != self.tokenizer.pad_token_id) for pred in ar_preds])
                predicted = True
                
    
        # ~~~ BUILD: demo dataframe ~~~ #
        return {**dict(
            loss=outputs.loss,
            predicted=predicted,
            # target=groundtruth_texts,
        ), **(dict(gen_len=gen_len) if gen_len is not None else {})}
        
    def training_step_end(self, outputs):
        if 'gen_len' in outputs:
            self.log("gen_len", round(outputs['gen_len'], 4),
                batch_size=self.hparams.batch_size,
            )
        if getattr(outputs, "predicted", False):
            self.log(
                f'train_{self.taskconfig.metric_pl.metricname}',
                self.metric['train'].compute(),
                on_step=True,
                on_epoch=False,
                
                batch_size=self.hparams.batch_size,
                prog_bar=True,
            )

    def training_epoch_end(self, outputs):
        if getattr(outputs, "predicted", False):
            self.metric['train'].reset()

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        assert not self.training

        self.log(f"valid_loss", outputs.loss, batch_size=self.hparams.eval_batch_size, prog_bar=True) 
        loss = outputs.loss.item()
        del outputs
        
        ar_preds = self.generate(
            **{
                k: batch[k]
                for k in batch
                if k != 'labels'
            }, 
            num_beams=self.hparams.num_beams,
            max_length=self.hparams.generation_max_length,
        )
        
        ar_texts = self.tokenizer.batch_decode(ar_preds, skip_special_tokens=True)
        ar_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        ar_texts, ar_labels = self.taskconfig.metric_pl.postprocess_fn(ar_texts, ar_labels)
        self.metric['valid'] = self.metric['valid'].to(ar_preds.device)
        self.metric['valid'].update(preds=ar_texts, target=ar_labels)
        
        if self.hparams.verbose_batch > 0:
            if batch_idx % self.hparams.verbose_batch == self.hparams.verbose_batch - 1:
                print('\n'
                   'Pred: \033[01;35m' + ar_texts[0] + '\n\033[0m'
                 + 'GrTr: \033[01;32m' +
                    (ar_labels[0][0] if isinstance(ar_labels[0], list) else ar_labels[0])
                    + '\n\033[0m')
        gen_len = np.mean([np.count_nonzero(
            pred.detach().cpu().numpy() != self.tokenizer.pad_token_id) for pred in ar_preds])
        
        return dict(
            loss=loss,
            gen_len=gen_len,
        )

    def validation_step_end(self, outputs):
        """
        0. check metric! --> FIXME: metric, compute_on_step?
            # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
        FIXME: eval_accum? now using metric_batch!
        """

        self.log("gen_len", round(outputs['gen_len'], 4),
            batch_size=self.hparams.eval_batch_size,
        )
        self.log(
            f'valid_{self.taskconfig.metric_pl.metricname}', 
            self.metric['valid'].compute(),
            on_step=True,
            on_epoch=True,
            
            batch_size=self.hparams.eval_batch_size,
            prog_bar=True,
        )

    def validation_epoch_end(self, outputs):
        self.metric['valid'].reset()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.trainset.num_workers,
            collate_fn=self.collate_fn,
        )
        
    def val_dataloader(self):
        if self.valset is not None:
            self.hparams.update(dict(eval_batch_size=getattr(self.hparams, 
                            "eval_batch_size", None
                        ) or self.hparams.batch_size))
            return DataLoader(
                dataset=self.valset,
                batch_size=self.hparams.eval_batch_size,
                shuffle=False,
                num_workers=self.valset.num_workers,
                collate_fn=self.collate_fn,
            )
        
          
        
        


if __name__ == "__main__": 
    args, task_config, model, tokenizer, train_dataset, dev_dataset, test_dataset, collate_fn = main()
    


    # main2()

    plmodel = PLModel(
        model=model,
        datasets=(train_dataset, dev_dataset),
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
        ),
        collate_fn=collate_fn,
        taskconfig=task_config,
    )
    



    if "check data":
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
    trainer.fit(
        plmodel,
        ckpt_path=args.resume_from_checkpoint,
    )  # dataloader here? <-- FIXME
    

# TODO: from scratch --> yaml config (一個 config 一個 setting)
# TODO: AE pretraining
# TODO: Speech translation
# TODO: return Self WORDLEN pred output! (penalized? 叫他自己用 sum of alphas 當成 wordlen)
# FIXME: ddp repeat 3 times?
# NOT TODO: other losses? every batch
