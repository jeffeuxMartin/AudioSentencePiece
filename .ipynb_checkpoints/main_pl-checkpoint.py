#!/usr/bin/env python3

# region         === importations ===         NOIGER #
import logging

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
from torchmetrics import WordErrorRate

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
    # CHECK XXX: resize embedding or config ????????? embedding ????????????

    if args.fix_encoder:
        model.fix_encoder_()
        
    return args, task_config, model, tokenizer, train_dataset, dev_dataset, test_dataset, collate_fn
        
        
# model, datasets, tokenizer, metric, hparams, collate_fn,
class PLModel(pl.LightningModule):
    def __init__(self,
        model, datasets, tokenizer, metric, hparams, collate_fn,
    ):
        super().__init__()
        self.hparams.update(hparams)
        
        self.model = model
        self.tokenizer = tokenizer
        
        self.trainset, self.valset = datasets
        
        self.metric = dict(
            train=metric(),
            valid=metric(),
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
                # 500,
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
                
                ar_texts = self.tokenizer.batch_decode(
                    ar_preds, skip_special_tokens=True)
                ar_labels = self.tokenizer.batch_decode(
                    batch['labels'], skip_special_tokens=True)
                self.metric['train'] = self.metric['train'].to(ar_preds.device)
                self.metric['train'].update(ar_texts, ar_labels)
                predicted = True
    
        # ~~~ BUILD: demo dataframe ~~~ #
        
        return dict(
            loss=outputs.loss,
            predicted=predicted,
            # target=groundtruth_texts,
        )

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        assert not self.training

        self.log(f"valid_loss", outputs.loss, batch_size=self.hparams.batch_size, prog_bar=True) 

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
        ar_labels = self.tokenizer.batch_decode(
            batch['labels'], skip_special_tokens=True)
        self.metric['valid'] = self.metric['valid'].to(ar_preds.device)
        self.metric['valid'].update(ar_texts, ar_labels)
        
        if self.hparams.verbose_batch > 0:
            if batch_idx % self.hparams.verbose_batch == self.hparams.verbose_batch - 1:
                print()
                print('Pred: \033[01;35m', end='')
                print(ar_texts[0])
                print('\033[0m', end='')
                print('GrTr: \033[01;32m', end='')
                print(ar_labels[0])
                print('\033[0m', end='')
                print()
        
        
        
        return dict(
            loss=outputs.loss,
        )
        
    def training_step_end(self, outputs):
        if getattr(outputs, "predicted", False):
            self.log(
                'train_wer', 
                self.metric['train'].compute(),
                on_step=True,
                on_epoch=False,
                
                batch_size=self.hparams.batch_size,
                prog_bar=True,
            )
    def training_epoch_end(self, outputs):
        if getattr(outputs, "predicted", False):
            self.metric['train'].reset()
        pass

    def validation_step_end(self, outputs):
        self.log(
            'valid_wer', 
            self.metric['valid'].compute(),
            on_step=True,
            on_epoch=True,
            
            batch_size=self.hparams.batch_size,
            prog_bar=True,
        )
    def validation_epoch_end(self, outputs):
        self.metric['valid'].reset()
        pass
        
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.trainset.num_workers,
            collate_fn=self.collate_fn,
        )
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valset,
            batch_size=(getattr(self.hparams, 
                        "eval_batch_size", None
                       ) or self.hparams.batch_size),
            shuffle=False,
            num_workers=self.valset.num_workers,
            collate_fn=self.collate_fn,
        )
        
          
        
        
def main2(args, task_config, model, tokenizer, train_dataset, dev_dataset, test_dataset, collate_fn):
    # ~~~ training args ~~~ #
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    training_args_dict = dict(
        save_total_limit=args.save_total_limit,
        run_name=args.run_name,
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        do_train=True,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        
        do_eval=True,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        eval_accumulation_steps=args.eval_accumulation_steps,
        per_device_eval_batch_size=args.eval_batch_size,
        
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        
        report_to='wandb' if args.wandb else 'none',
        
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
    )
    
    if task_config.seq2seq:
        training_args_dict["predict_with_generate"] = True
        training_args_dict["generation_max_length"] = args.generation_max_length
    
    # ~~~ trainer ~~~ #
    compute_metrics_fn = task_config.metric_func(
        tokenizer,
        **(dict(metric_batch=args.metric_batch, 
                verbose_batch=args.verbose_batch)
            if task_config.seq2seq else {}))

    trainer_args = dict(
        model=model,
        args=task_config.training_arg_class(**training_args_dict),
        
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
        callbacks=[LogCallback] if args.callback else [],
    )

    trainer = task_config.trainer_class(**trainer_args)
    
    if "check data":
        print()
        print('Checking Trainloader...')
        for i in trainer.get_train_dataloader():
            print(i)
            break
        if args.dev_split != 'none':
            print()
            print('Checking Evalloader...')
            for i in trainer.get_eval_dataloader():
                print(i)
                break

    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
        ignore_keys_for_eval=[  # ????????????????????? config!
            'encoder_last_hidden_state', 
            'encoder_last_hidden_out_attention_mask',
            'encoder_length_loss',
            'encoder_pred_word_lengths',
            'encoder_hidden_states',
            'encoder_attentions',
            'masked_lm_loss',    # store in other formats!
            'real_length_loss',  # store in other formats!
        ] + getattr(model.config, 
            "keys_to_ignore_at_inference", []),
    )


if __name__ == "__main__": 
    args, task_config, model, tokenizer, train_dataset, dev_dataset, test_dataset, collate_fn = main()
    


    # main2()
    plmodel = PLModel(
        model=model,
        datasets=(train_dataset, dev_dataset),
        tokenizer=tokenizer,
        metric=WordErrorRate,
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
        monitor="val_loss",
        save_top_k=args.save_total_limit,
        every_n_train_steps=args.save_steps,
        save_on_train_epoch_end=True,
        mode="min",  # wer
    )

    trainer = pl.Trainer(
        gpus=-1,
        logger=WandbLogger(args.run_name, project=args.proj_name) if args.wandb else True,
        log_every_n_steps=args.logging_steps,
        val_check_interval=0.1,
        # check_val_every_n_epoch=5,
        default_root_dir=args.output_dir,
        max_epochs=args.epochs,
        strategy=DDPStrategy(find_unused_parameters=True) if any('--local_rank' in i for i in sys.argv) else None,
        
        # weights_save_path=
        enable_progress_bar=True,
        # enable_checkpointing=
        
        accumulate_grad_batches=args.gradient_accumulation_steps,
        callbacks=[checkpoint_callback],

    )
    trainer.fit(
        plmodel,
        # ckpt=
    )  # dataloader here? <-- FIXME
    

# TODO: from scratch --> yaml config (?????? config ?????? setting)
# TODO: AE pretraining
# TODO: Speech translation
# TODO: return Self WORDLEN pred output! (penalized? ??????????????? sum of alphas ?????? wordlen)
# FIXME: ddp repeat 3 times?
# NOT TODO: other losses? every batch
"""
0. check metric! --> FIXME: metric, compute_on_step?
    # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
1. eval_accumulation_steps=args.eval_accumulation_steps,
1. callbacks=[LogCallback] if args.callback else [],
1. gen_len
"""