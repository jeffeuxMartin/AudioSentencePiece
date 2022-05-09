#!/usr/bin/env python3  # ~~~ VERIFIED ~~~ #
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class WrappedDataclass:
    data: object
    gendata: object
    labels: object = None


class MyUnitDataset(Dataset):
    def __init__(self, units, texts=None, wordlen=None, lower=False): 
        self.units = units
        if texts is not None:
            assert len(texts) == len(self.units)
        self.texts = texts
        if lower:
            self.texts = [s.lower() for s in self.texts]
        if wordlen is not None:
            assert len(wordlen) == len(self.units)
        self.wordlen = wordlen
    
    def __len__(self): 
        return len(self.units)
    
    def __getitem__(self, idx): 
        return (
            self.units[idx], 
            self.texts[idx] 
                if self.texts is not None else 
                None,
            self.wordlen[idx] 
                if self.wordlen is not None else 
                None,
        )


def DataSetCollectorGeneral(
    prefix_path: Path, 
    split: str, 
    dtype2subdir_ext: dict, 
    lower: bool = False,
  ):
    logging.warning('== Loading data... ==')
    with open(Path(prefix_path) / '{subdir}/{split}.{ext}'.format(
        split=split, **(vars(dtype2subdir_ext["src"])))) as f:
        src_data = f.read().strip().split('\n')

    if 'tgt' in dtype2subdir_ext:
        with open(Path(prefix_path) / '{subdir}/{split}.{ext}'.format(
            split=split, **(vars(dtype2subdir_ext["tgt"])))) as f:
            tgt_data = f.read().strip().split('\n')
        assert len(tgt_data) == len(src_data)
    else:
        tgt_data = None

    if 'hint' in dtype2subdir_ext:
        with open(Path(prefix_path) / '{subdir}/{split}.{ext}'.format(
            split=split, **(vars(dtype2subdir_ext["hint"])))) as f:
            hint_data = f.read().strip().split('\n')
        assert len(hint_data) == len(src_data)
    else:
        hint_data = None

    mydataset = MyUnitDataset(src_data, tgt_data, hint_data, lower=lower)

    return mydataset

def Data_collate_fn(unit_tokenizer, text_tokenizer):
    # done: combine & 應該要都可以處理，沒 label 或 length

    def prepend_append(tok):
        """ Solve the problem of <s> ... </s> """
        def f(s): return f"{tok.bos_token} {s} {tok.eos_token}"
        return f 
    unit_tokenizer.prepend_append = (prepend_append(unit_tokenizer) 
        if unit_tokenizer(['']).get("input_ids", None) == [[]] else 
        lambda x: x)
    text_tokenizer.prepend_append = (prepend_append(text_tokenizer) 
        if text_tokenizer(['']).get("input_ids", None) == [[]] else 
        lambda x: x)

    def collate_fn(batch):
        input_ids, labels, wordlens = list(zip(*batch))
        output_dict = dict(
            **unit_tokenizer(
                list(map(unit_tokenizer.prepend_append, input_ids)), 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=1024,
            ))

        if labels[0] is not None:
            output_dict["labels"] = text_tokenizer(
                list(map(text_tokenizer.prepend_append, labels)), 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=1024,
            )['input_ids']

        if wordlens[0] is not None:
            output_dict["word_length_tensor"] = torch.tensor(
                np.array(wordlens, dtype=int))
        
        genoutput_dict = {
            k: output_dict[k]
            for k in output_dict
            if k != 'labels'
        }

        batch_data = WrappedDataclass(
            output_dict, 
            genoutput_dict,
            labels)
        return batch_data
        
    return collate_fn

