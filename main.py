#!/usr/bin/env python3

LOG_WANDB = True

if "loggings":
    import logging, logging.config; from src.logging import MYCONFIG; 
    logging.basicConfig(format='\033[0;36m''%(message)s''\033[0m'); 
    logging.config.dictConfig(MYCONFIG); mylogger = logging.getLogger('main')
    PRINTINFO = mylogger.info; PRINTDEBUG = mylogger.debug
PRINTINFO('~~~ !!! START !!! ~~~')

if "imports":
    import os, argparse

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
    from src.newutils import load_cached; load_cached = load_cached(PRINTDEBUG)

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-b", "--batch_size",
        type=int, default=9,
    )
    args = parser.parse_args()
    return args  # , config, backup_files


if __name__ == '__main__':
    PRINTINFO('~~~ !!! __main__ START !!! ~~~')    
    args = get_args()
    
    os.environ['WANDB_PROJECT'] = ("UnitWordSemantics")
    
    bart_tokenizer = load_cached(
        BartTokenizer, "facebook/bart-base", 
        saved_path="./exp/tokenizer/", 
        msg='Load tokenizer...')
    if """PRINTDEBUG('Load dataset...')""":
        PRINTDEBUG('Load dataset...')
        fname = ("."
                "/data/corpus_train-clean-100.tsv"
                ".sorted.tsv"
                )
        if 'sorted' not in fname or not os.path.isfile(fname):
            PRINTDEBUG('    (Sorting data...                )')
            fname = fname.replace('.sorted.tsv', '')
            fname = UnitDataset.dataframe_sorter(fname)
        else:
            PRINTDEBUG('    (Using Presorted data...        )')
        
        df = pd.read_csv(fname, sep='\t')
        # split_idx = int(round(len(df) * 0.8))
        split_num = 10
        train_dataset = UnitDataset(
            # df[:split_idx], 
            df[df.index % split_num != split_num - 1],
            bart_tokenizer)
        dev_dataset = UnitDataset(
            # df[split_idx:], 
            df[df.index % split_num == split_num - 1],
            bart_tokenizer)
    AEModel = load_cached(
        WordLevelBartAutoEncoder, "facebook/bart-base",
        saved_path="./models/wordlevel/",  # TODO: Check OK?!
        msg='Initialize Word Model...')
    WordReprModel = load_cached(
        BartForCausalLM, "facebook/bart-base", 
        saved_path="./models/worddecode/",
        msg='Initialize Unit-to-text Model...')
    if LOG_WANDB:
        wandb_logger = WandbLogger(
            "new_run", 
            save_dir="."
                    "/exp/wandb_savior")

    PRINTDEBUG('Initialize Full Model...')
    semantic_model = PLSpeechToSemantics(
        datasets=(train_dataset, dev_dataset),
        metric_cls=WordErrorRate,
        word_level_encoder=AEModel,
        representation_decoder=WordReprModel,
        tokenizer=bart_tokenizer,
        task_name="ASR",
        batch_size=args.batch_size,
        fixed_encoder=True,
        **(dict(wandb_logger=wandb_logger) 
           if LOG_WANDB else {}),
    )

    PRINTDEBUG('Initialize Trainer...')
    lightning_trainer = pl.Trainer(
        # accelerator="cpu", devices=1,
        gpus=-1,
        # precisioin=16,
        # limit_train_batches=0.5,
        logger=wandb_logger if LOG_WANDB else True,
        log_every_n_steps=20,
        val_check_interval=0.005,
        # auto_scale_batch_size="binsearch",
        # strategy='ddp',
        default_root_dir="exp/rewritten/checkpoints",
        max_epochs=-1,
    )

    PRINTINFO('=== START TRAINING! ===')
    lightning_trainer.fit(
        semantic_model,
        # ckpt_path=None,
    )


# TODO:  evaluation (check reset or by batch/epoch?)
# TODO:  log text
# TODO:  reg. e.g. dropout weight_decay
# TODO:  fix feat extractor
# TODO:  inv sqrt scheduler
# TODO:  BOS id problem
# TODO:  SHIFT!!!

# zIXME: shift problem
# zIXME: input 長度 problem  XXX 😨😨😨😨😨😨 確認模型有沒有輸入輸出錯
# zIXME: wandb logger problem
# zIXME: train val split 因為排序不隨機了！
# FIXME: pretrain problem  XXX 🙇‍♂️🙇‍♂️🙇‍♂️🙇‍♂️🙇‍♂️🙇‍♂️🙇‍♂️🙇‍♂️🙇‍♂️🙇‍♂️ # <~~ 用 pl TRAIN!!!
# TODO:  load MOST pretrained weight AND SCRATCH
# TODO:  add ddp (check!)
# TODO:  更多的 ｕｎｉｔｓ＆ａｎａｌｙｓｉｓ required!!!  # <--- # <~~
# 🙇‍♂️🙇‍♂️🙇‍♂️🙇‍♂️🙇‍♂️ BOS problem
