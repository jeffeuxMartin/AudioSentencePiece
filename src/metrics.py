#!/usr/bin/env python3  # ~~~ VERIFIED ~~~ #
import pathlib
from datetime import datetime

import numpy as np

from datasets import load_metric
strftime, now = datetime.strftime, datetime.now

def compute_metrics_WER_logits(tokenizer): 
    metric = load_metric("wer")
    def fn(eval_preds): 
        predicted_texts = eval_preds.predictions.argmax(-1)
        label_texts = eval_preds.label_ids

        attention_masks = (label_texts != -100) & (label_texts != tokenizer.pad_token_id)
        sent_lengths = attention_masks.sum(1)
        overlapped = (predicted_texts == label_texts) * attention_masks
        accuracy = (overlapped.sum(1) / sent_lengths).mean(0).item()

        label_texts = [s[m] for s, m in zip(label_texts, attention_masks)]
        predicted_texts = [s[m] for s, m in zip(predicted_texts, attention_masks)]
        REAL = tokenizer.batch_decode(label_texts, skip_special_tokens=True)  # TODO: 直接傳進來！不用
        PRED = tokenizer.batch_decode(predicted_texts, skip_special_tokens=True)
        
        return {"acc": accuracy, "wer": metric.compute(predictions=PRED, references=REAL)}
    return fn

def compute_metrics_WER(tokenizer, metric_batch=10, verbose_batch=20): 
    # 1. logits --> id (because of "generate")
    # 2. acc removed
    
    # XXX: 因為儲存空間問題
    cache_dir = pathlib.Path('./.cache/preds') / strftime(now(), r'%Y%m%d_%H%M%S')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    metric = load_metric("wer", cache_dir=cache_dir)
    def fn(eval_preds): 
        
        predicted_texts = eval_preds.predictions
        label_texts = eval_preds.label_ids

        def batchify(iterable, n=1):
            iterable = list(iterable)
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx : min(ndx + n, l)]

        for batch_idx, batch in enumerate(batchify(zip(label_texts, predicted_texts), metric_batch)):
            # expand
            bat_label_texts, bat_predicted_texts = zip(*batch)
            # numpify
            bat_label_texts, bat_predicted_texts = np.array(bat_label_texts), np.array(bat_predicted_texts)
            # replace -100 to pad_id
            bat_label_texts[bat_label_texts == -100] = tokenizer.pad_token_id

            bat_REAL = tokenizer.batch_decode(bat_label_texts, skip_special_tokens=True)  # TODO: 直接傳進來！不用
            bat_PRED = tokenizer.batch_decode(bat_predicted_texts, skip_special_tokens=True)

            metric.add_batch(predictions=bat_PRED, references=bat_REAL)

            if verbose_batch > 0:
                if batch_idx % verbose_batch == verbose_batch - 1:
                    print(f"\nPred: \033[01;31m{bat_PRED[0]}\033[0m\nRefe: \033[01;32m{bat_REAL[0]}\033[0m\n")
        
        return {"wer": metric.compute()}

    return fn

def postprocess_text(preds, labels, translation=False):
    preds = [pred.strip() for pred in preds]
    if translation:
        labels = [[label.strip()] for label in labels]
    else:
        labels = [label.strip() for label in labels]

    return preds, labels

# Ref.: https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py
def compute_metrics_WER_HF(tokenizer, **kwargs):
    cache_dir = pathlib.Path('./.cache/preds') / strftime(now(), r'%Y%m%d_%H%M%S')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    metric = load_metric("wer", cache_dir=cache_dir)

    def fn(eval_preds): 
        predicted_texts, label_texts = eval_preds
        if isinstance(predicted_texts, tuple): [predicted_texts] = predicted_texts            
        label_texts = np.where(label_texts != -100, label_texts, tokenizer.pad_token_id)  # replace -100 to pad_id

        bat_REAL = tokenizer.batch_decode(label_texts, skip_special_tokens=True)  # TODO: 直接傳進來！不用
        bat_PRED = tokenizer.batch_decode(predicted_texts, skip_special_tokens=True)
        bat_PRED, bat_REAL = postprocess_text(preds=bat_PRED, labels=bat_REAL)
        
        result = {"wer": metric.compute(predictions=bat_PRED, references=bat_REAL)}
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predicted_texts]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        
        return result

    return fn

def compute_metrics_BLEU(tokenizer, metric_batch=10, verbose_batch=20): 
    # XXX: 因為儲存空間問題
    cache_dir = pathlib.Path('./.cache/preds') / strftime(now(), r'%Y%m%d_%H%M%S')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    metric = load_metric("sacrebleu", cache_dir=cache_dir)
    def fn(eval_preds): 
        
        predicted_texts = eval_preds.predictions
        label_texts = eval_preds.label_ids

        def batchify(iterable, n=1):
            iterable = list(iterable)
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx : min(ndx + n, l)]

        for batch_idx, batch in enumerate(batchify(zip(label_texts, predicted_texts), metric_batch)):
            # expand
            bat_label_texts, bat_predicted_texts = zip(*batch)
            # numpify
            bat_label_texts, bat_predicted_texts = np.array(bat_label_texts), np.array(bat_predicted_texts)
            # replace -100 to pad_id
            bat_label_texts[bat_label_texts == -100] = tokenizer.pad_token_id

            bat_REAL = tokenizer.batch_decode(bat_label_texts, skip_special_tokens=True)  # TODO: 直接傳進來！不用
            bat_REAL = [[s] for s in bat_REAL]
            bat_PRED = tokenizer.batch_decode(bat_predicted_texts, skip_special_tokens=True)

            metric.add_batch(predictions=bat_PRED, references=bat_REAL)

            if verbose_batch > 0:
                if batch_idx % verbose_batch == verbose_batch - 1:
                    print(f"\nPred: \033[01;31m{bat_PRED[0]}\033[0m\nRefe: \033[01;32m{bat_REAL[0]}\033[0m\n")
        
        return {"sacreBLEU": metric.compute()}

    return fn

def compute_metrics_BLEU_HF(tokenizer, **kwargs):
    cache_dir = pathlib.Path('./.cache/preds') / strftime(now(), r'%Y%m%d_%H%M%S')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    metric = load_metric("sacrebleu", cache_dir=cache_dir)

    def fn(eval_preds): 
        predicted_texts, label_texts = eval_preds
        if isinstance(predicted_texts, tuple): [predicted_texts] = predicted_texts            
        label_texts = np.where(label_texts != -100, label_texts, tokenizer.pad_token_id)  # replace -100 to pad_id

        bat_REAL = tokenizer.batch_decode(label_texts, skip_special_tokens=True)  # TODO: 直接傳進來！不用
        bat_PRED = tokenizer.batch_decode(predicted_texts, skip_special_tokens=True)
        bat_PRED, bat_REAL = postprocess_text(preds=bat_PRED, labels=bat_REAL, translation=True)
        
        result_dict = metric.compute(predictions=bat_PRED, references=bat_REAL)
        result = {"bleu": result_dict["score"]}
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predicted_texts]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        
        return result

    return fn
