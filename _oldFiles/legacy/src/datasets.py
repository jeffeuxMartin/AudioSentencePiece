#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .newutils import mask_generator

def _string_parser(s):
    return np.fromstring(s, dtype=int, sep=' ')

class UnitDataset(Dataset):
    def dataframe_reader(self):
        units = self.df["units"].map(_string_parser).map(torch.tensor)
        unit_lengths = units.map(len)
        self.df = pd.DataFrame.from_dict({
            **self.df.to_dict(),
            "units": units,
            "unit_lengths": unit_lengths,
        })
        
    @classmethod
    def dataframe_sorter(cls, fname):
        df = pd.read_csv(fname, sep='\t')
        unit_lengths = df["units"].map(_string_parser).map(torch.tensor).map(len)
        df = pd.DataFrame.from_dict({**df.to_dict(),"unit_lengths": unit_lengths})
        df = df.sort_values("unit_lengths", ascending=False)
        df[["data_path", "src_txt", "units", "pseudotexts", "length_src", "spwords"]].to_csv(fname + '.sorted.tsv', sep='\t')
        return fname + '.sorted.tsv'
        
    # TODO: How about ST?
    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        
        # ref.: https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
        self.num_workers = len(os.sched_getaffinity(0))
        
        self.df = dataframe
        self.dataframe_reader()

        self.units = self.df["units"].values
        self.lengths = torch.LongTensor(self.df["length_src"].values)
        self.texts = self.df["src_txt"].values
        self.unit_lengths = self.df["unit_lengths"].values
        
    def __getitem__(self, idx):
        return (
            self.units[idx],
            self.lengths[idx],
            self.texts[idx],
            self.unit_lengths[idx],
        )
        
    def __len__(self):
        return len(self.df)

    @classmethod
    def tokenized_collate_fn(cls, 
        tokenizer,
        padding_value=-100, 
        max_unit_length=1024, 
        max_text_length=512,
        unit_shift_ids=4,
    ):

        def collate_fn(batch):
            units, lengths, texts, unit_lengths = list(zip(*batch))

            if not "unshifted":
                units = pad_sequence(units, 
                    batch_first=True, 
                    padding_value=padding_value)
                units = units[..., :max_unit_length]
            else:
                units = [torch.concat([
                    torch.tensor([0 - unit_shift_ids]),  # bos_token_id
                    unit_seq,
                    torch.tensor([2 - unit_shift_ids]),  # eos_token_id
                ]) for unit_seq in units]
                units = pad_sequence(units, 
                    batch_first=True, 
                    padding_value=padding_value - unit_shift_ids)
                units = units + unit_shift_ids
                units = units[..., :max_unit_length]

            lengths = torch.tensor(lengths
            ).clip(max=min(max_unit_length, max_text_length))
            
            text_tokens = tokenizer(list(texts), 
                return_tensors='pt', padding=True, truncation=True, max_length=max_text_length)
            
            if not "unshifted":
                unit_lengths = torch.tensor(unit_lengths).clip(max=max_unit_length)
            else:
                unit_lengths = (torch.tensor(unit_lengths) + 2).clip(max=max_unit_length)
            unit_attention_mask = mask_generator(unit_lengths)
            
            return dict(
                input_ids=units,
                word_length_tensor=lengths,
                texts=texts,
                unit_lengths=unit_lengths,
                attention_mask=unit_attention_mask,
                text_tokens=text_tokens,
            )

        return collate_fn

