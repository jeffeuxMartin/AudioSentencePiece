#!/usr/bin/env python3

LOG_WANDB = False
LOG_WANDB = True
import os; os.environ['WANDB_PROJECT'] = ("BartUnitWordSemanticsASR")
if "real":
    MAXUNITLEN = 1024
    MAXTEXTLEN = 512
else:
    MAXUNITLEN = 200
    MAXTEXTLEN = 28
PAD_ID = 1


if "loggings":
    import logging, logging.config; from src.logging import MYCONFIG; 
    logging.basicConfig(format='\033[0;36m''%(message)s''\033[0m'); 
    logging.config.dictConfig(MYCONFIG); mylogger = logging.getLogger('main')
    PRINTINFO = mylogger.info; PRINTDEBUG = mylogger.debug
PRINTINFO('~~~ !!! START !!! ~~~')

import os, sys

import pandas as pd

from transformers import BartForCausalLM
from transformers import BartTokenizer
from transformers import BartConfig

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torchmetrics import WordErrorRate

from src.newmodels import SentBart
from src.newmodels import SentBartForConditionalGeneration
from src.newmodels import pure_advanced_load_pretrained
from src.newmodels import advanced_load_pretrained

from src.datasets import UnitDataset
from src.newutils import get_args
from src.newutils import load_cached; load_cached = load_cached(PRINTDEBUG)

def on_device(inputs, device, tolabel=False):
    input_ids = inputs["input_ids"]
    to_return = dict(
        input_ids=input_ids.to(device),
        word_length_tensor=inputs.get("word_length_tensor", None).to(device),
        attention_mask=inputs.get("attention_mask", None).to(device),
    )
    if tolabel:
        labels = input_ids.masked_fill(input_ids==PAD_ID, -100).to(device)
        return {"labels": labels, **to_return}
    else:
        return to_return

if __name__ == '__main__':
    PRINTINFO('~~~ !!! __main__ START !!! ~~~')    
    args = get_args()
    
    bart_tokenizer = load_cached(BartTokenizer, "facebook/bart-base", saved_path="./exp/tokenizer/", msg='Load tokenizer...')
    if """PRINTDEBUG('Load dataset...')""":
        PRINTDEBUG('Load dataset...')
        fname = (".""/data/corpus_train-clean-100.tsv"".sorted.tsv")
        if 'sorted' not in fname or not os.path.isfile(fname):
            PRINTDEBUG('    (Sorting data...        )')
            fname = fname.replace('.sorted.tsv', '')
            fname = UnitDataset.dataframe_sorter(fname)
        else:
            PRINTDEBUG('    (Using Presorted data...)')
        
        df = pd.read_csv(fname, sep='\t')
        split_num = 200
        train_dataset = UnitDataset(df[df.index % split_num != split_num - 1], bart_tokenizer)
        # train_dataset = UnitDataset(df[87 + 0 : 87 + 1 * 60], bart_tokenizer)
        dev_dataset = UnitDataset(df[df.index % split_num == split_num - 1], bart_tokenizer)
        # dev_dataset = UnitDataset(df[87 + 60 : 87 + 2 * 60], bart_tokenizer)
        deviced_dataset = UnitDataset(df[:20], bart_tokenizer)

    PRINTDEBUG('data loading done!!')

    PRINTDEBUG('loading self defined AE!')
    checkpoint_name = 'facebook/bart-base'
    if not "AE":
        if "trained":
            if 1:
                AEModel = SentBartForConditionalGeneration.from_pretrained(
                    "jjjjj/checkpoint-80")
            if 0:
                AEModel = advanced_load_pretrained(
                    checkpoint_name="jjjjj/checkpoint-80",
                    model_class=SentBartForConditionalGeneration, 
                    config_class=BartConfig, 
                    # new_config_options...
                    max_position_embeddings=MAXUNITLEN, 
                    vocab_size=500 + 4,
                    num_labels=3, is_encoder_decoder=True, pad_token_id=1, bos_token_id=0, eos_token_id=2, decoder_start_token_id=2,
                    # unk=3,
                )

        else:
            AEModel = advanced_load_pretrained(
                checkpoint_name=checkpoint_name,
                model_class=SentBartForConditionalGeneration, 
                config_class=BartConfig, 
                # new_config_options...
                max_position_embeddings=MAXUNITLEN, 
                vocab_size=500 + 4,
                num_labels=3, is_encoder_decoder=True, pad_token_id=1, bos_token_id=0, eos_token_id=2, decoder_start_token_id=2,
                # unk=3,
            )
        assert AEModel.get_encoder().config.hidden_size > 0  # 768
    else:
        BEModel = advanced_load_pretrained(
            checkpoint_name=checkpoint_name,
            # checkpoint_name="jjjjj/checkpoint-80",
            # checkpoint_name="./jRRMM/checkpoint-4700",
            
            model_class=SentBartForConditionalGeneration, 
            config_class=BartConfig, 
            # new_config_options...
            max_position_embeddings=MAXUNITLEN, 
            vocab_size=500 + 4,
            num_labels=3, is_encoder_decoder=True, pad_token_id=1, bos_token_id=0, eos_token_id=2, decoder_start_token_id=2,
            tgt_vocab_size=bart_tokenizer.vocab_size,  # FIXME???
            # unk=3,
        )
        # TODO or FIXME? PURE???
        # BEModel.fix_encoder_()  # FIXME: Yes later!
    
    PRINTDEBUG('definition done!!')
    
    # breakpoint()
    from transformers import Trainer, TrainingArguments
    

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer.
            """
            input_ids = inputs["input_ids"]
            labels = input_ids.masked_fill(input_ids==PAD_ID, -100)
            outputs = model(
                input_ids=input_ids,
                word_length_tensor=inputs.get("word_length_tensor", None),
                attention_mask=inputs.get("attention_mask", None),
                labels=labels,
            )
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            return (loss, outputs) if return_outputs else loss

    class CustomTrainer2(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer.
            """
            input_ids = inputs["input_ids"]
            # labels = input_ids.masked_fill(input_ids==PAD_ID, -100)
            labels = inputs.get("text_tokens", {}).get("input_ids", None)
            outputs = model(
                input_ids=input_ids,
                # decoder_input_ids
                word_length_tensor=inputs.get("word_length_tensor", None),
                attention_mask=inputs.get("attention_mask", None),
                decoder_attention_mask=inputs.get("text_tokens", {}).get("attention_mask", None),
                
                labels=labels,
            )
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            return (loss, outputs) if return_outputs else loss

        def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=None,
        ):
            from transformers.trainer_pt_utils import (nested_detach)
            from transformers.utils import (is_sagemaker_mp_enabled)
            if is_sagemaker_mp_enabled():
                from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat


            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            if has_labels:
                # labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                # if len(labels) == 1:
                #     labels = labels[0]
                pass  # FIXME!!!
            else:
                labels = None

            import torch
            with torch.no_grad():
                if is_sagemaker_mp_enabled():
                    raw_outputs = smp_forward_only(model, inputs)
                    if has_labels:
                        if isinstance(raw_outputs, dict):
                            loss_mb = raw_outputs["loss"]
                            logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                        else:
                            loss_mb = raw_outputs[0]
                            logits_mb = raw_outputs[1:]

                        loss = loss_mb.reduce_mean().detach().cpu()
                        logits = smp_nested_concat(logits_mb)
                    else:
                        loss = None
                        if isinstance(raw_outputs, dict):
                            logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                        else:
                            logits_mb = raw_outputs
                        logits = smp_nested_concat(logits_mb)
                else:
                    if has_labels:
                        with self.autocast_smart_context_manager():
                            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                        loss = loss.mean().detach()

                        if isinstance(outputs, dict):
                            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                        else:
                            logits = outputs[1:]
                    else:
                        loss = None
                        with self.autocast_smart_context_manager():
                            outputs = model(**inputs)
                            ## outputs = model(
                            ##     input_ids=input_ids,
                            ##     # decoder_input_ids
                            ##     word_length_tensor=inputs.get("word_length_tensor", None),
                            ##     attention_mask=inputs.get("attention_mask", None),
                            ##     decoder_attention_mask=inputs.get("text_tokens", {}).get("attention_mask", None),
                            ##     
                            ##     labels=labels,
                            ## )
                            
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

from torch.utils.data import DataLoader
import torch
class PLNew(pl.LightningModule):
    def __init__(self, 
        model,
        tokenizer,
        wandb_logger,
        datasets,
        metric_cls,
        hparams,
    ):
        super().__init__()
        # self.hparams = hparams
        self.hparams.update(hparams)
        self.model = model
        self.tokenizer = tokenizer
        self.wandb_logger = wandb_logger
        self.trainset, self.valset = datasets
        
        self.metrics = {
            "train": metric_cls(),
            "valid": metric_cls(),
            "validREAL": metric_cls(),
        }
    pass
    # def forward(self, *a, **k):
    def forward(self, inputs, return_outputs=False):
        pass
        # return self.model(*a, **k)
        
        input_ids = inputs["input_ids"]
        # labels = input_ids.masked_fill(input_ids==PAD_ID, -100)
        labels = inputs.get("text_tokens", {}).get("input_ids", None)
        outputs = self.model(
            input_ids=input_ids,
            # decoder_input_ids
            word_length_tensor=inputs.get("word_length_tensor", None),
            attention_mask=inputs.get("attention_mask", None),
            decoder_attention_mask=inputs.get("text_tokens", {}).get("attention_mask", None),
            
            labels=labels,
        )
        ### # Save past state if it exists
        ### # TODO: this needs to be fixed and made cleaner later.
        ### if self.args.past_index >= 0:
        ###     self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        return (loss, outputs) if return_outputs else loss

        
    def configure_optimizers__old(self):
        # Ref.: https://github.com/PyTorchLightning/pytorch-lightning/issues/328
        optimizer = torch.optim.AdamW(
            params=self.parameters(), 
            # lr=self.my_hparams["lr"],
            lr=self.hparams["lr"],
            # weight_decay=self.my_hparams["weight_decay"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=1,
            verbose=True,
        )
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
           'monitor': 'valid_loss',
       }
       
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
        from torch.optim import AdamW
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams["weight_decay"],
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

        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.get("warmup_steps", 250),
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    
       
    def training_step(self, batch, batch_idx):
        loss, outputs = self(batch, return_outputs=True)
        assert self.training
        self.log(f"train_loss", outputs.loss, batch_size=self.hparams.batch_size, prog_bar=True) 
        
        predicted_ids = outputs.logits.argmax(-1)
        predicted_texts = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        groundtruth_texts = batch["texts"]
        # groundtruth_texts = [i.lower() for i in groundtruth_texts]  # FIXME: not lower!
    
        # ~~~ BUILD: demo dataframe ~~~ #
        
        return dict(
            loss=outputs.loss,
            preds=predicted_texts,
            target=groundtruth_texts,
        )

    def validation_step(self, batch, batch_idx):
        loss, outputs = self(batch, return_outputs=True)
        # print(outputs.logits.argmax(-1)[0])
        
        assert not self.training
        UUpon = self.model.generate(
            
                inputs=batch.get('input_ids', None),
                attention_mask=batch.get("attention_mask", None),
                word_length_tensor=batch.get("word_length_tensor", None),
                
        )

        self.log(f"valid_loss", outputs.loss, batch_size=self.hparams.batch_size, prog_bar=True) 
        return (loss, outputs, 
            batch['texts'], 
            # [i.lower() for i in batch['texts']],  # FIXME: not lower!
            UUpon)


    def training_step_end(self, outputs):
        # return   # FIXME!!!
        mode = "train"
        metrics = self.metrics[mode]
        self.task_name = "ASR"  # FIXME!!!! $$$$$$$$$$$$$$$$$$$$$$
        if 'preds' in outputs:
            if self.task_name == "ASR":
                eval_result = metrics(
                    preds=outputs['preds'], 
                    target=list(outputs['target']))
            elif self.task_name == "ST":
                eval_result = metrics(
                    preds=outputs['preds'], 
                    target=[[i] for i in outputs['target']])
            metric_name = {"ASR": "WER", "ST": "BLEU"}[self.task_name]
            self.log(f"{mode}_{metric_name}", eval_result, batch_size=self.hparams.batch_size, prog_bar=True)

    def validation_step_end(self, outputs):
        # return   # FIXME!!!
        loss, myout, texts, realpreds = outputs
        mode = "valid"
        metrics = self.metrics[mode]
        metrics_REAL = self.metrics["validREAL"]
        if not "old":
            if 'preds' in outputs:
                if self.task_name == "ASR":
                    eval_result = metrics(
                        preds=outputs['preds'], 
                        target=list(outputs['target']))
                    eval_resultREAL = metrics_REAL(
                        preds=outputs['reals'], 
                        target=list(outputs['reals_target']))
                elif self.task_name == "ST":
                    eval_result = metrics(
                        preds=outputs['preds'], 
                        target=[[i] for i in outputs['target']])
                    eval_resultREAL = metrics_REAL(
                        preds=outputs['reals'], 
                        target=[[i] for i in outputs['reals_target']])
        else:
            LLLLL = myout.logits.argmax(-1)
            ZZZ = bart_tokenizer.batch_decode(LLLLL, skip_special_tokens=True)
            TTT = texts
            eval_result = metrics(preds=ZZZ, target=list(TTT))
            realpreds = bart_tokenizer.batch_decode(realpreds, skip_special_tokens=True)
            # breakpoint()
            eval_resultREAL = metrics(preds=realpreds, target=list(TTT))
        if 1:
            if 1:
                self.task_name = "ASR"  # FIXME
                metric_name = {"ASR": "WER", "ST": "BLEU"}[self.task_name]
                self.log(f"{mode}_{metric_name}", eval_result, batch_size=self.hparams.batch_size, prog_bar=True)
                if 1:
                    self.log(f"validREAL_{metric_name}", eval_resultREAL, batch_size=self.hparams.batch_size, prog_bar=True)
        for pred, gt, realss in zip(
            ZZZ,
            TTT,
            realpreds):
            print()
            print('\033[01;32m' + 'GT: '
                + 
                gt
                + 
                '\033[0m')
            print('\033[01;34m' + 'PR: '
                + 
                pred
                + 
                '\033[0m')
            print('\033[01;33m' + 'AR: '
                + 
                realss
                + 
                '\033[0m')
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset, 
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.trainset.num_workers,
            collate_fn=UnitDataset.tokenized_collate_fn(
                self.tokenizer, 
                padding_value=PAD_ID, 
                max_unit_length=MAXUNITLEN, 
                max_text_length=MAXTEXTLEN,
                unit_shift_ids=4,
            ),
        )
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valset, 
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.valset.num_workers,
            collate_fn=UnitDataset.tokenized_collate_fn(
                self.tokenizer, 
                padding_value=PAD_ID, 
                max_unit_length=MAXUNITLEN, 
                max_text_length=MAXTEXTLEN,
                unit_shift_ids=4,
            ),
        )


if __name__ == "__main__":
    if LOG_WANDB:
        wandb_logger = WandbLogger("new_run__NEWNUEVO", save_dir=".""/exp/wandb_savior222")

    if not "HuggingFace":
        hf_args = TrainingArguments(
            output_dir='./jRRMM/',
            do_train=True,
            # logging_steps=1,
            logging_steps=10,
            # per_device_train_batch_size=2,
            per_device_train_batch_size=8,
            
            
            do_eval=True,
            eval_steps=100,
            # eval_steps=1,
            evaluation_strategy="steps",
            # # eval_accumulation_steps=10,
            per_device_eval_batch_size=10,
            # label_names=[
            #     "input_ids",
            #     "attention_mask",
            #     # "texts",
            # ],
            label_names=["text_tokens"],
            
            
            
            num_train_epochs=10,
            # learning_rate=args.lr,
            learning_rate=8e-4,
            # learning_rate=1e-5,
        
            # warmup_steps=300,
            warmup_ratio=0.2,
            # warmup_ratio=0.1 / 2,
            # report_to='none',
            report_to='wandb' if LOG_WANDB else 'none',
            save_steps=100,
            save_strategy="steps",
        )
        mytrainer = CustomTrainer2(
            model=BEModel,
            args=hf_args, 
            # train_dataset=deviced_dataset, 
            train_dataset=train_dataset, 
            # train_dataset=dev_dataset, 
            # eval_dataset=deviced_dataset, 
            # eval_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=UnitDataset.tokenized_collate_fn(
                bart_tokenizer,
                padding_value=PAD_ID, 
                max_unit_length=MAXUNITLEN, 
                max_text_length=MAXTEXTLEN,
                unit_shift_ids=4,
            ), 
            # compute_metrics=compute_metrics_ACC,
            tokenizer=bart_tokenizer,
            
            
            
            
            
        )

        mytrainer.train(
            ignore_keys_for_eval=getattr(
                BEModel.config, 
                "keys_to_ignore_at_inference", []
            ) + ["texts"],
        )
    else:
        semantic_model = PLNew(BEModel, tokenizer=bart_tokenizer, wandb_logger=wandb_logger if LOG_WANDB else None,
        datasets=(train_dataset, dev_dataset), metric_cls=WordErrorRate, hparams={
            "lr": 3e-5,
            # "lr": 6e-4,
            "batch_size": 12,
            "weight_decay": 0.01,
        }
        )
        PRINTDEBUG('Initialize Trainer...')
        lightning_trainer = pl.Trainer(
            gpus=-1,
            logger=wandb_logger if LOG_WANDB else True,
            log_every_n_steps=10,
            val_check_interval=0.05,
            # check_val_every_n_epoch=1,
            default_root_dir="exp/rewritten22/checkpoints_debugged222",
            max_epochs=20 + 50,
            strategy=DDPStrategy(find_unused_parameters=True) if any('--local_rank' in i for i in sys.argv) else None,
            # resume_from_checkpoint='./exp/wandb_savior222/3guoe4x2'
            # ckpt_path='./exp/wandb_savior222/BartUnitWordSemanticsASR/3guoe4x2'
            # /home/jeffeuxmartin/AudioWords/./exp/wandb_savior222/BartUnitWordSemanticsASR/3guoe4x2/checkpoints/epoch=19-step=28400.ckpt
            
        )

        PRINTINFO('=== START TRAINING! ===')
        lightning_trainer.fit(
            semantic_model,
            # ckpt_path='./exp/wandb_savior222/BartUnitWordSemanticsASR/3guoe4x2/checkpoints/epoch=19-step=28400.ckpt',
            ckpt_path='/home/jeffeuxmartin/AudioWords/exp/wandb_savior222/BartUnitWordSemanticsASR/h9im5sjh/checkpoints/epoch=0-step=455.ckpt',
        )

        pass
    if 0:
        input("DONOOOOOOEEEEEEEEEEEE!!!!!1")
            
        
        
        
        # B = next(iter(mytrainer.get_train_dataloader()))
        
        
        
        
        ####################33
        B0 = next(iter(mytrainer.get_train_dataloader()))
        B = on_device(B0, "cuda")
        import torch
        with torch.no_grad():
            # L = AEModel(**B)
            L = BEModel(**B)
            TFOutput = L.logits.argmax(-1)
            # OOO = AEModel.generate(**B, num_beams=2)
            OOO = BEModel.generate(**B, num_beams=2)
        
        
        
        
        # breakpoint()
        from IPython import embed; embed()
    exit()
    configuration = BartConfig.from_pretrained('facebook/bart-base')
    # configuration = (
    #     vocab_size=504,
    #     
    #     # d_model=18,
    #     
    #     # encoder_layers=2,
    #     # decoder_layers=1,
    #     # encoder_attention_heads=2,
    #     # decoder_attention_heads=1,
    #     # encoder_ffn_dim=32,
    #     # decoder_ffn_dim=12,
    #     # # scale_embedding=True,
    #     # dropout=0.1,
    #     # attention_dropout=0.05,
    #     # activation_dropout=0.05,
    #     # classifier_dropout=0.05,
    #     max_position_embeddings=MAXUNITLEN,
    #     num_labels=504,
    # )
    configuration.vocab_size = 504
    configuration.max_position_embeddings = MAXUNITLEN
    configuration.num_labels = 504

    AEModel = WordLevelBartAutoEncoder(
        configuration
    )

    PRINTDEBUG('definition done!!')


    from src.trainers import HuggingFaceTrainer, compute_metrics_ACC
    Trainer = HuggingFaceTrainer
    from transformers import TrainingArguments

    a = TrainingArguments(
    output_dir='./bbbbbc/',
        do_train=True,
        logging_steps=10,
        per_device_train_batch_size=3,
        
        
        do_eval=True,
        eval_steps=100,
        evaluation_strategy="steps",
        # eval_accumulation_steps=10,
        per_device_eval_batch_size=3,
        label_names=[
            "input_ids",
            "attention_mask",
            # "texts",
        ],
        
        
        
        num_train_epochs=3,
        # learning_rate=args.lr,
        learning_rate=4e-4,
    
        warmup_steps=300,
        # warmup_ratio=0.1,
        report_to='none',
        save_steps=200,
        save_strategy="steps",
    )
    mytrainer = HuggingFaceTrainer(
        model=AEModel,args=a, 
        train_dataset=train_dataset, 
        # train_dataset=dev_dataset, 
        # eval_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=UnitDataset.tokenized_collate_fn(
            bart_tokenizer,
                padding_value=503, 
                max_unit_length=MAXUNITLEN, 
                max_text_length=512,
            ), 
        compute_metrics=compute_metrics_ACC,
        tokenizer=bart_tokenizer)

    mytrainer.train()

    if 0:
        # ################3

        """
        training_args = TrainingArguments(
        #     output_dir="./GoBattleship_frompretrained_continue_from_60",
            output_dir="./ASR_leoyang",

            
            do_train=True,
            logging_steps=10,
        #     # logging_strategy="epoch",
            per_device_train_batch_size=2,
            
            num_train_epochs=20,
            learning_rate=6e-5,
            warmup_ratio=0.3,
            optim="adamw_torch",  # FIXME!
            
            do_eval=True,
            evaluation_strategy="steps",
        #     label_names=["input_ids", "attention_mask"],
            label_names=[
                "text_tokens__input_ids",
                "text_tokens__attention_mask",
                # "texts",
            ],

            # eval_steps=1,
            # eval_steps=5,
            eval_steps=200,
        #     # evaluation_strategy="epoch",
            per_device_eval_batch_size=4,
            eval_accumulation_steps=20,
            
            save_strategy="steps",
            save_steps=500,
            
            # report_to="none",
            report_to="wandb",
            
            ddp_find_unused_parameters=False,
        )
        trainer = Trainer(
            # model=model,
            model=fmdl,
            
            args=training_args,
            
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=train_dataset.collate_fn2,
            
            tokenizer=tokenizer_bart,
            compute_metrics=partialme(tokenizer_bart),
            
        #     # optimizers=(torch.optim.AdamW, None),
        )
        trainer.train()


        def train_dataloader(self):
                return DataLoader(
                    dataset=self.trainset, 
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.trainset.num_workers,
                    collate_fn=UnitDataset.tokenized_collate_fn(
                        self.tokenizer, 
                        padding_value=503, 
                        max_unit_length=MAXUNITLEN, 
                        max_text_length=512,
                    ),
                )
        """

# TODO: 我現在就不要 AE 了直接 train，但請注意 embed 不 share 所以 decoder 那邊會出問題！
