#!python
import pandas as pd

split = "dummy"

for split in 'dummy train test dev'.split():
    data = pd.read_csv(
        f'../golden_corpus/{split}_result_adv.tsv', 
        sep='\t')
    for folder, ext, colname in [
        ("lengths", "len", "length_src"),
        ("paths", "path", "data_id"),
        ("subwords", "subword", "spwords"),
        ("texts", "txt", "src_txt"),
        ("units", "unit", "units"),
        # ("collunits", "collunit", ""),
        # ("symbolunits", "symbolunit", ""),
        # ("wordlengths", "wordlen", ""),
        # ("endelengths", "endelength", ""),

        ("translation", "de", "tgt_txt"),
        # ("desubwords", "desubword", ""),
    ]:
    
        with open(f'{folder}/{split}.{ext}', 'w') as f:
            for i in data[colname]:
                print(i, file=f)	

r"""
collunits	collunit
endelengths	endelength
lengths	len
paths	path
subwords	subword
symbolunits	symbolunit
texts	txt
units	unit
wordlengths	wordlen

desubwords	desubword
translation	de
"""
