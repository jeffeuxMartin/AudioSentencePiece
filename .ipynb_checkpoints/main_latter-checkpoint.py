#!/usr/bin/env python3

LOG_WANDB = True

if "loggings":
    import logging, logging.config; from src.logging import MYCONFIG; 
    logging.basicConfig(format='\033[0;36m''%(message)s''\033[0m'); 
    logging.config.dictConfig(MYCONFIG); mylogger = logging.getLogger('main')
    PRINTINFO = mylogger.info; PRINTDEBUG = mylogger.debug
PRINTINFO('~~~ !!! START !!! ~~~')

import os, sys

import pandas as pd

from transformers import BartForCausalLM, BartModel
from transformers import BartTokenizer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torchmetrics import WordErrorRate

from src.models import PLSpeechToSemantics
from src.models import WordLevelBartAutoEncoder
from src.models import NewNewBartForCausalLM

from src.datasets import UnitDataset
from src.newutils import get_args
from src.newutils import load_cached; load_cached = load_cached(PRINTDEBUG)


if __name__ == '__main__':
    PRINTINFO('~~~ !!! __main__ START !!! ~~~')    
    args = get_args()
    
    os.environ['WANDB_PROJECT'] = ("UnitWordSemanticsASR")
    
    bart_tokenizer = load_cached(BartTokenizer, "facebook/bart-base", saved_path="./exp/tokenizer/", msg='Load tokenizer...')
    if """PRINTDEBUG('Load dataset...')""":
        PRINTDEBUG('Load dataset...')
        fname = (".""/data/corpus_train-clean-100.tsv"".sorted.tsv")
        if 'sorted' not in fname or not os.path.isfile(fname):
            PRINTDEBUG('    (Sorting data...                )')
            fname = fname.replace('.sorted.tsv', '')
            fname = UnitDataset.dataframe_sorter(fname)
        else:
            PRINTDEBUG('    (Using Presorted data...        )')
        
        df = pd.read_csv(fname, sep='\t')
        split_num = 200
        train_dataset = UnitDataset(df[df.index % split_num != split_num - 1], bart_tokenizer)
        dev_dataset = UnitDataset(df[df.index % split_num == split_num - 1], bart_tokenizer)

    AEModel = load_cached(WordLevelBartAutoEncoder, "facebook/bart-base", saved_path="./models/wordlevel/", msg='Initialize Word Model...')
    WordReprModel = load_cached(NewNewBartForCausalLM, "facebook/bart-base", saved_path="./models/worddecode/", msg='Initialize Unit-to-text Model...')
    if LOG_WANDB:
        wandb_logger = WandbLogger("new_run__NEW", save_dir=".""/exp/wandb_savior")

    PRINTDEBUG('Initialize Full Model...')
    semantic_model = PLSpeechToSemantics(
        datasets=(train_dataset, dev_dataset),
        metric_cls=WordErrorRate,
        word_level_encoder=AEModel,
        representation_decoder=WordReprModel,
        tokenizer=bart_tokenizer,
        task_name="ASR",
        batch_size=12,
        # batch_size=args.batch_size,
        fixed_encoder=True,
        wandb_logger=wandb_logger if LOG_WANDB else None,
        lr=args.lr,
    )

    PRINTDEBUG('Initialize Trainer...')
    lightning_trainer = pl.Trainer(
        gpus=-1,
        logger=wandb_logger if LOG_WANDB else True,
        log_every_n_steps=10,
        val_check_interval=0.05,
        # check_val_every_n_epoch=1,
        default_root_dir="exp/rewritten/checkpoints_debugged",
        max_epochs=20,
        strategy=DDPStrategy(find_unused_parameters=False) if any('--local_rank' in i for i in sys.argv) else None,
    )

    PRINTINFO('=== START TRAINING! ===')
    lightning_trainer.fit(
        semantic_model,
    )

# TODO: beam decode!