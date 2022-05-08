import pathlib
from datetime import datetime

import numpy as np

from datasets import load_metric
strftime, now = datetime.strftime, datetime.now

def compute_metrics_WER_logits(tokenizer):  # For ASR, FIXME
    def fn(eval_preds):  # For ASR, FIXME
        metric = load_metric("wer")
        predictions = eval_preds.predictions
        predicted_texts = predictions.argmax(-1)
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

def compute_metrics_WER(tokenizer, metric_batch=10, verbose_batch=20):  # For ASR, FIXME
    # 1. logits --> id (because of "generate")
    # 2. acc removed
    (pathlib.Path('./.cache/preds') / strftime(now(), r'%Y%m%d_%H%M%S')).mkdir(parents=True, exist_ok=True)
    def fn(eval_preds):  # For ASR, FIXME
        metric = load_metric("wer", cache_dir=(pathlib.Path('./.cache/preds') / strftime(now(), r'%Y%m%d_%H%M%S')))
        predicted_texts = eval_preds.predictions
        label_texts = eval_preds.label_ids

        def batch(iterable, n=1):
            iterable = list(iterable)
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        for ni, bat in enumerate(batch(zip(label_texts, predicted_texts), metric_batch)):
            bat_label_texts, bat_predicted_texts = zip(*bat)
            bat_label_texts = np.array(bat_label_texts)
            bat_label_texts[bat_label_texts == -100] = tokenizer.pad_token_id
            bat_predicted_texts = np.array(bat_predicted_texts)
            bat_REAL = tokenizer.batch_decode(bat_label_texts, skip_special_tokens=True)  # TODO: 直接傳進來！不用
            bat_PRED = tokenizer.batch_decode(bat_predicted_texts, skip_special_tokens=True)
            metric.add_batch(
                predictions=bat_PRED,
                references=bat_REAL,
            )
            if verbose_batch > 0:
                if ni % verbose_batch == verbose_batch - 1:
                    print(f"\n"f"Pred: \033[01;31m{bat_PRED[0]}\033[0m\n"f"Refe: \033[01;32m{bat_REAL[0]}\033[0m\n""")
        
        return {"wer": metric.compute()}
    return fn
