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

from datetime import datetime
strftime, now = datetime.strftime, datetime.now

import transformers
from transformers import AutoTokenizer
from transformers.models.auto.configuration_auto import AutoConfig

# import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.strategies.ddp import DDPStrategy
# from torchmetrics import WordErrorRate

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
    task_config = TASK_CONFIG_DICT[args.task]

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
    if args.dev_split != 'none':
        dev_dataset = LIBRISPEECH_UNIT_ASR_SPLIT(args.dev_split)
    if args.test_split is not None:
        test_dataset = LIBRISPEECH_UNIT_ASR_SPLIT(args.test_split)

    # === experiment setup === #
    exp_config = dict(
        collapse_n=(-1 if args.original else args.collapse_n),  # default = 0
        **(dict(weight_len=args.weight_len) if args.weight_len is not None else {}),
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
        
    # ~~~ training args ~~~ #
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    training_args_dict = dict(
        save_total_limit=args.save_total_limit,
        run_name=args.run_name,
        output_dir=args.output_dir,
        
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
        ignore_keys_for_eval=[  # 也可以直接寫進 config!
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

if __name__ == "__main__": main()

# TODO: compute_metrics or PL!
# TODO: from scratch --> yaml config (一個 config 一個 setting)
# TODO: AE pretraining
# TODO: Speech translation
# TODO: return Self WORDLEN pred output! (penalized? 叫他自己用 sum of alphas 當成 wordlen)
# FIXME: ddp repeat 3 times?
# NOT TODO: other losses? every batch
