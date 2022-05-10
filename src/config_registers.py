#!/usr/bin/env python3  # ~~~ VERIFIED ~~~ #
from typing import Callable, Dict, Type, Optional
from dataclasses import dataclass
from functools import partial

from transformers import TrainingArguments
from transformers import Trainer
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

from .trainers import AugSeq2SeqTrainer
from .trainers import AugTrainer
from .trainers import AugTrainerState
from .metrics import compute_metrics_WER
from .metrics import compute_metrics_WER_logits
from .metrics import compute_metrics_WER_HF
from .metrics import compute_metrics_BLEU
from .metrics import compute_metrics_BLEU_HF
from .metrics import postprocess_text

from torchmetrics import WordErrorRate, SacreBLEUScore


@dataclass
class TrainingDataType:
    subdir: str = ""
    ext: str = ""

@dataclass
class TaskConfig:
    trainer_class: Type = None
    training_arg_class: Type = None
    metric_func: Callable = None
    data_structure_def: Dict = None
    seq2seq: bool = True
    metric_pl: object = None

@dataclass
class MetricClass:
    metric: object = None
    metricname: str = ""
    postprocess_fn: Callable = None
    metric_mode: str = ''
    
asrmetric = MetricClass(WordErrorRate, 'wer', partial(postprocess_text, translation=False), 'min')
stmetric = MetricClass(SacreBLEUScore, 'bleu', partial(postprocess_text, translation=True), 'max')



CollUnits = TrainingDataType("collunits", "collunit")
FullUnits = TrainingDataType("symbolunits", "symbolunit")
IntUnits = TrainingDataType("units", "unit")

TextData = TrainingDataType("texts", "txt")
SubwordData = TrainingDataType("subwords", "subword")
EnDeSubwordData = TrainingDataType("endesubwords", "endesubword")
DeSubwordData = TrainingDataType("desubwords", "desubword")
TranslationData = TrainingDataType("translation", "de")

EnLengthData = TrainingDataType("lengths", "len")
DeLengthData = TrainingDataType("delengths", "delength")
EnDeLengthData = TrainingDataType("endelengths", "endelength")
WordLengthData = TrainingDataType("wordlengths", "wordlen")
PathData = TrainingDataType("paths", "path")


autoencoder_config = lambda coll: TaskConfig(
    # trainer_class=AugTrainer,
    trainer_class=Trainer,
    training_arg_class=TrainingArguments,
    metric_func=compute_metrics_WER_logits,
    data_structure_def=dict(
        src=(CollUnits if coll else FullUnits),
        tgt=(CollUnits if coll else FullUnits),
        hint=WordLengthData,
    ),
    seq2seq=False,
)

asr_config = lambda coll: TaskConfig(
    # trainer_class=AugSeq2SeqTrainer,
    trainer_class=Seq2SeqTrainer,
    training_arg_class=Seq2SeqTrainingArguments,
    # metric_func=compute_metrics_WER,
    metric_func=compute_metrics_WER_HF,
    data_structure_def=dict(
        src=(CollUnits if coll else FullUnits),
        tgt=TextData,
        hint=WordLengthData,
    ),
    seq2seq=True,
    metric_pl=asrmetric,
)

st_config = lambda coll: TaskConfig(
    # trainer_class=AugSeq2SeqTrainer,
    trainer_class=Seq2SeqTrainer,
    training_arg_class=Seq2SeqTrainingArguments,
    # metric_func=compute_metrics_BLEU,
    metric_func=compute_metrics_BLEU_HF,
    data_structure_def=dict(
        src=(CollUnits if coll else FullUnits),
        tgt=TranslationData,
        hint=WordLengthData,
    ),
    seq2seq=True,
    metric_pl=stmetric,
)

TASK_CONFIG_DICT = lambda coll: {
    "AE": autoencoder_config(coll),
    "ASR": asr_config(coll),
    "ST": st_config(coll),
}
