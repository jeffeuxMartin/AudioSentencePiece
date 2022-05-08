import logging
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

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
    prefix_path, split, dtype2subdir_ext=None, lower=False,
):
    dtype2subdir_ext = ({} 
        if dtype2subdir_ext is None else 
        dtype2subdir_ext)
    dtype2subdir_ext_default = {
        'texts': dict(
            subdir='texts',
            ext='txt',
        ),
        'original_units': dict(
            subdir='collunits',
            ext='collunit',
        ),
        'wordlens': dict(
            subdir='lengths',
            ext='len',
        ),
    }
    
    dtype2subdir_ext_default.update(dtype2subdir_ext)
    dtype2subdir_ext = dtype2subdir_ext_default

    logging.warning('== ....      ==')
    with open(Path(prefix_path) / '{subdir}/{split}.{ext}'.format(
        split=split, 
        subdir=dtype2subdir_ext['texts']['subdir'],
        ext=dtype2subdir_ext['texts']['ext'],
    )) as f:
        texts = f.read().strip().split('\n')

    if 'texts' in dtype2subdir_ext:
        with open(Path(prefix_path) / '{subdir}/{split}.{ext}'.format(
            split=split, 
            subdir=dtype2subdir_ext['original_units']['subdir'],
            ext=dtype2subdir_ext['original_units']['ext'],
        )) as f:
            original_units = f.read().strip().split('\n')
        assert len(texts) == len(original_units)
    else:
        # print("NO "
        #       "\033[01;31m"
        #       "`{texts}`!"
        #       "\033[0m")
        texts = None

    if 'wordlens' in dtype2subdir_ext:
        with open(Path(prefix_path) / '{subdir}/{split}.{ext}'.format(
            split=split, 
            subdir=dtype2subdir_ext.get('wordlens', {}).get('subdir'),
            ext=dtype2subdir_ext.get('wordlens', {}).get('ext'),
        )) as f:
            wordlens = f.read().strip().split('\n')
        assert len(wordlens) == len(original_units)
    else:
        # print("NO "
        #       "\033[01;31m"
        #       "`{wordlens}`!"
        #       "\033[0m")
        wordlens = None

    mydataset = MyUnitDataset(original_units, texts, wordlens, lower=lower)

    return mydataset


def Data_collate_fn(unit_tokenizer, text_tokenizer):
    # done: combine & 應該要都可以處理，沒 label 或 length
    def prepend_append(tok):
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
        return output_dict
    return collate_fn
   