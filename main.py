#!/usr/bin/env python3

LOG_WANDB = True

import os
import logging
logging.basicConfig(format=''
    '(\033[0;32m%(asctime)s\033[0m) \033[0;33m%(message)s\033[0m',
    datefmt=r'%Y/%m/%d %p %I:%M:%S')
logger_handler = logging.StreamHandler()

logging.warning('[INFO] ~~~ !!! START !!! ~~~')
os.environ['WANDB_PROJECT'] = ("UnitWordSemantics")

import pandas as pd


from transformers import AutoTokenizer
from transformers import BartForCausalLM
from transformers import BartTokenizer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import WordErrorRate

from src.trainers import HuggingFaceTrainer
from src.trainers import compute_metrics_ACC
from src.trainers import compute_metrics_WER

from src.models import PLSpeechToSemantics
from src.models import WordLevelBartAutoEncoder

from src.datasets import UnitDataset


if __name__ == '__main__':
    logging.warning('[INFO] ~~~ !!! __main__ START !!! ~~~')
    
    logger_handler.setFormatter(logging.Formatter('(\033[0;33m%(asctime)s\033[0m) %(message)s'))

    logging.warning('[NOTE] Load tokenizer...')
    if os.path.isdir("./models/tokenizer/"):
        logging.warning('[NOTE]     (Using local cache...           )')
        bart_tokenizer = BartTokenizer.from_pretrained("./models/tokenizer/")
    else:
        logging.warning('[NOTE]     (Loading pretrained tokenizer...)')
        bart_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-base")
        bart_tokenizer.save_pretrained("./models/tokenizer/")

    logging.warning('[NOTE] Load dataset...')
    fname = ("/home/jeffeuxmartin"
             "/TestBed"
             "/data/corpus_train-clean-100.tsv"
             ".sorted.tsv"
            )
    if 'sorted' not in fname or not os.path.isfile(fname):
        logging.warning('[NOTE]     (Sorting data...                )')
        fname = fname.replace('.sorted.tsv', '')
        fname = UnitDataset.dataframe_sorter(fname)
    else:
        logging.warning('[NOTE]     (Using Presorted data...        )')
        
    df = pd.read_csv(fname, sep='\t')
    split_idx = int(round(len(df) * 0.8))
    train_dataset = UnitDataset(df[:split_idx], bart_tokenizer)
    dev_dataset = UnitDataset(df[split_idx:], bart_tokenizer)

    logging.warning('[NOTE] Initialize Word Model...')
    if os.path.isdir("./models/wordlevel/"):
        logging.warning('[NOTE]     (Using local pretrained...      )')
        AEModel = WordLevelBartAutoEncoder.from_pretrained(
            "facebook/bart-base")  # TODO: Check OK?!
    else:
        logging.warning('[NOTE]     (Loading pretrained tokenizer...)')
        AEModel = WordLevelBartAutoEncoder.from_pretrained(
            "facebook/bart-base")  # TODO: Check OK?!
        AEModel.save_pretrained("./models/wordlevel/")

    logging.warning('[NOTE] Initialize Unit-to-text Model...')
    if os.path.isdir("./models/worddecode/"):
        logging.warning('[NOTE]     (Using local pretrained...      )')
        WordReprModel = BartForCausalLM.from_pretrained(
            "facebook/bart-base")  # TODO: Check OK?!
    else:
        logging.warning('[NOTE]     (Loading pretrained tokenizer...)')
        WordReprModel = BartForCausalLM.from_pretrained(
            "facebook/bart-base")  # TODO: Check OK?!
        WordReprModel.save_pretrained("./models/worddecode/")

    if LOG_WANDB:
        wandb_logger = WandbLogger(
            "test_run", 
            save_dir="."
                    "/AudioSentencePiece/savior")

    logging.warning('[NOTE] Initialize Full Model...')
    semantic_model = PLSpeechToSemantics(
        datasets=(train_dataset, dev_dataset),
        metric_cls=WordErrorRate,
        word_level_encoder=AEModel,
        representation_decoder=WordReprModel,
        tokenizer=bart_tokenizer,
        task_name="ASR",
        **(dict(wandb_logger=wandb_logger) 
           if LOG_WANDB else {}),
    )

    logging.warning('[NOTE] Initialize Trainer...')
    lightning_trainer = pl.Trainer(
        # accelerator="cpu", devices=1,
        gpus=-1,
        # precisioin=16,
        # limit_train_batches=0.5,
        logger=wandb_logger if LOG_WANDB else True,
        log_every_n_steps=20,
        val_check_interval=0.1,
        # auto_scale_batch_size="binsearch",
        strategy='ddp',
        default_root_dir="rewritten/checkpoints",
        max_epochs=-1,
    )

    logging.warning('[NOTE] === START TRAINING! ===')
    lightning_trainer.fit(
        semantic_model,
        # ckpt_path=None,
    )


# TODO: evaluation (check reset or by batch/epoch?)
# TODO: log text
# TODO: reg. e.g. dropout weight_decay
# TODO: fix feat extractor
# TODO: inv sqrt scheduler
# TODO: BOS id problem
# TODO: SHIFT!!!
