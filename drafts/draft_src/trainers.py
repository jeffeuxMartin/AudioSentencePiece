#!/usr/bin/env python3

import numpy as np

from torch import nn

from transformers import Trainer
from datasets import load_metric


class HuggingFaceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        
        original_units = inputs.get("input_ids")
        reconstructed_logits = outputs.get("last_hidden_state")
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=501)  # FIXME: check -100 or 501 or ...more?
        loss = loss_fct(
            reconstructed_logits.transpose(1, -1),
            original_units,
        )
        return (loss, outputs) if return_outputs else loss

def compute_metrics_ACC(eval_preds):  # For ASR, FIXME
    reconstructed_logits, shortend_features = eval_preds.predictions
    reconstructed_units = reconstructed_logits.argmax(-1)
    original_units, attention_masks = eval_preds.label_ids
    # Ref.: /home/jeffeux/anaconda3/envs/unit_extraction/lib/python3.8/site-packages/transformers/trainer.py:2468
    assert np.array_equal((attention_masks == -100), (original_units == -100))
    
    overlapped = (reconstructed_units == original_units) * attention_masks # TODO: mask == -100 or 501?
    accuracy = overlapped.mean(1).mean(0).item()

    return {"acc": accuracy}

def compute_metrics_WER(eval_preds, tokenizer):  # For ASR, FIXME
    metric = load_metric("wer")
    predictions = eval_preds.predictions
    predicted_texts = predictions.argmax(-1)
    label_texts, attention_masks = eval_preds.label_ids

    overlapped = (predicted_texts == label_texts) * attention_masks # TODO: mask?
    accuracy = overlapped.mean(1).mean(0).item()
    new_mask = (label_texts != -100)  # TODO: Why different???
    label_texts = [s[m] for s, m in zip(label_texts, new_mask)]
    predicted_texts = [s[m] for s, m in zip(predicted_texts, new_mask)]
    REAL = tokenizer.batch_decode(label_texts, skip_special_tokens=True)  # TODO: 直接傳進來！不用
    PRED = tokenizer.batch_decode(predicted_texts, skip_special_tokens=True)
    
    return {"acc": accuracy, "wer": metric.compute(predictions=PRED, references=REAL)}
