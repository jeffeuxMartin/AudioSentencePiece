#!/usr/bin/env python3

LOG_WANDB = False
MAXUNITLEN = 200

if "loggings":
    import logging, logging.config; from src.logging import MYCONFIG; 
    logging.basicConfig(format='\033[0;36m''%(message)s''\033[0m'); 
    logging.config.dictConfig(MYCONFIG); mylogger = logging.getLogger('main')
    PRINTINFO = mylogger.info; PRINTDEBUG = mylogger.debug
PRINTINFO('~~~ !!! START !!! ~~~')

import os, sys

import pandas as pd

# from transformers import BartForCausalLM
from transformers import BartTokenizer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torchmetrics import WordErrorRate

from src.models import PLSpeechToSemantics
from src.models import WordLevelBartAutoEncoder

from src.datasets import UnitDataset
from src.newutils import get_args
from src.newutils import load_cached; load_cached = load_cached(PRINTDEBUG)


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

    PRINTDEBUG('data loading done!!')
    # AEModel = load_cached(WordLevelBartAutoEncoder, "facebook/bart-base", saved_path="./models/wordlevel/", msg='Initialize Word Model...')
    from transformers import BartConfig
    # vocab_size (`int`, *optional*, defaults to 50265):
    # d_model (`int`, *optional*, defaults to 1024):
    # encoder_layers (`int`, *optional*, defaults to 12):
    # decoder_layers (`int`, *optional*, defaults to 12):
    # encoder_attention_heads (`int`, *optional*, defaults to 16):
    # decoder_attention_heads (`int`, *optional*, defaults to 16):
    # decoder_ffn_dim (`int`, *optional*, defaults to 4096):
    # encoder_ffn_dim (`int`, *optional*, defaults to 4096):
    # activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
    # dropout (`float`, *optional*, defaults to 0.1):
    # attention_dropout (`float`, *optional*, defaults to 0.0):
    # activation_dropout (`float`, *optional*, defaults to 0.0):
    # classifier_dropout (`float`, *optional*, defaults to 0.0):
    # max_position_embeddings (`int`, *optional*, defaults to 1024):
    # init_std (`float`, *optional*, defaults to 0.02):
    # encoder_layerdrop: (`float`, *optional*, defaults to 0.0):
    # decoder_layerdrop: (`float`, *optional*, defaults to 0.0):
    # scale_embedding (`bool`, *optional*, defaults to `False`):
    # use_cache (`bool`, *optional*, defaults to `True`):
    # num_labels: (`int`, *optional*, defaults to 3):
    # forced_eos_token_id (`int`, *optional*, defaults to 2):


    PRINTDEBUG('loading self defined AE!')
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
