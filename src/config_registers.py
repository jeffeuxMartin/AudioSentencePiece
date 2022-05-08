#!/usr/bin/env python3  # ~~~ VERIFIED ~~~ #
from typing import Callable, Dict, Type, Optional
from dataclasses import dataclass

from transformers import TrainingArguments
from transformers import Seq2SeqTrainingArguments

from .trainers import AugSeq2SeqTrainer
from .trainers import AugTrainer
from .trainers import AugTrainerState
from .metrics import compute_metrics_WER
from .metrics import compute_metrics_WER_logits
from .metrics import compute_metrics_BLEU


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


CollUnits = TrainingDataType("collunits", "collunit")
FullUnits = TrainingDataType("symbolunits", "symbolunit")
IntUnits = TrainingDataType("units", "unit")

TextData = TrainingDataType("texts", "txt")
SubwordData = TrainingDataType("subwords", "subword")
TranslationData = TrainingDataType("translation", "de")

LengthData = TrainingDataType("lengths", "len")
PathData = TrainingDataType("paths", "path")


autoencoder_config = TaskConfig(
    trainer_class=AugTrainer,
    training_arg_class=TrainingArguments,
    metric_func=compute_metrics_WER_logits,
    data_structure_def=dict(
        src=CollUnits,
        tgt=CollUnits,
        hint=LengthData,
    ),
    seq2seq=False,
)

asr_config = TaskConfig(
    trainer_class=AugSeq2SeqTrainer,
    training_arg_class=Seq2SeqTrainingArguments,
    metric_func=compute_metrics_WER,
    data_structure_def=dict(
        src=CollUnits,
        tgt=TextData,
        hint=LengthData,
    ),
    seq2seq=True,
)

st_config = TaskConfig(
    trainer_class=AugSeq2SeqTrainer,
    training_arg_class=Seq2SeqTrainingArguments,
    metric_func=compute_metrics_BLEU,
    data_structure_def=dict(
        src=CollUnits,
        tgt=TranslationData,
        hint=LengthData,
    ),
    seq2seq=True,
)

TASK_CONFIG_DICT = {
    "AE": autoencoder_config,
    "ASR": asr_config,
    "ST": st_config,
}
