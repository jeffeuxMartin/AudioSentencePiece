#!/usr/bin/env python3  # ~~~ VERIFIED ~~~ #

# region         === importations ===         NOIGER #
import math
import random


import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup
from transformers import logging

import pytorch_lightning as pl

try:
    from .torch_cif import cif_function
    from .utils import mask_generator
    from .datasets import DataSetCollectorGeneral
except:
    from torch_cif import cif_function
    from utils import mask_generator
    from datasets import DataSetCollectorGeneral
# endregion      === importations ===      NOIGERDNE #


logger = logging.get_logger("transformers.models.bart.modeling_bart")

   
# model, datasets, tokenizer, metric, hparams, collate_fn,
class PLModel(pl.LightningModule):
    def __init__(self,
        model, datasets, tokenizer, hparams, collate_fn, taskconfig,
    ):
        super().__init__()
        self.hparams.update(hparams)
        
        self.model = model
        self.tokenizer = tokenizer
        
        self.trainset, self.valset, self.testset = datasets
        
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
        ab_size = self.trainer.accumulate_grad_batches / float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.get("weight_decay", 0.01),
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
                self.hparams.get("warmup_ratio", 0.1) * self.total_steps // self.trainer.max_epochs,
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
                if self.taskconfig.seq2seq:
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
                else:
                    predicted_ids = outputs.logits.argmax(-1)

                    predicted_texts = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                    ar_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                    self.metric['train'] = self.metric['train'].to(predicted_ids.device)
                    self.metric['train'].update(preds=predicted_texts, target=ar_labels)
                    
                    gen_len = np.mean([np.count_nonzero(pred.detach().cpu().numpy() != self.tokenizer.pad_token_id) for pred in predicted_ids])
                    predicted = True
                    
                    pass
                
    
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
        if self.taskconfig.seq2seq:
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
        else:
            predicted_ids = outputs.logits.argmax(-1)
            
            decoded_ids = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            ar_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            decoded_ids, ar_labels = self.taskconfig.metric_pl.postprocess_fn(decoded_ids, ar_labels)
            self.metric['valid'] = self.metric['valid'].to(predicted_ids.device)
            self.metric['valid'].update(preds=decoded_ids, target=ar_labels)
            
            if self.hparams.verbose_batch > 0:
                if batch_idx % self.hparams.verbose_batch == self.hparams.verbose_batch - 1:
                    print('\n'
                    'Pred: \033[01;35m' + decoded_ids[0] + '\n\033[0m'
                    + 'GrTr: \033[01;32m' +
                        (ar_labels[0][0] if isinstance(ar_labels[0], list) else ar_labels[0])
                        + '\n\033[0m')
            gen_len = np.mean([np.count_nonzero(
                pred.detach().cpu().numpy() != self.tokenizer.pad_token_id) for pred in predicted_ids])
        
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
        
    def test_dataloader(self):
        if self.testset is not None:
            self.hparams.update(dict(eval_batch_size=getattr(self.hparams, 
                            "eval_batch_size", None
                        ) or self.hparams.batch_size))
            return DataLoader(
                dataset=self.testset,
                batch_size=self.hparams.eval_batch_size,
                shuffle=False,
                num_workers=self.testset.num_workers,
                collate_fn=self.collate_fn,
            )
        
          
      