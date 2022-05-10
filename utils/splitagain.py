#!python
splits = 'dummy test dev train'.split()
import os
import pathlib
from tqdm import tqdm
for split in splits:
    fname = f'data/golden_corpus/{split}_result_adv.tsv'
    rawfile = open(fname)
    with os.popen(f'wc -l {fname}') as f:
        total = int(f.read().split()[0])

    fileptrs = {}
    for dirname, ext in [
        ("collunits", "collunit"),
        ("paths", "path"),
        ("endelengths", "endelength"),
        ("units", "unit"),
        ("subwords", "subword"),
        ("delengths", "delength"),
        ("translation", "de"),
        ("wordlengths", "wordlen"),
        ("symbolunits", "symbolunit"),
        ("desubwords", "desubword"),
        ("lengths", "len"),
        ("endesubwords", "endesubword"),
        ("texts", "txt"),
    ]:
        pathlib.Path(f'data/CoVoSTUnitsNew/{dirname}').mkdir(exist_ok=True, parents=True)
        fileptrs[dirname] = open(f'data/CoVoSTUnitsNew/{dirname}/{split}.{ext}', 'w')
        



    print(rawfile.readline())

    for line in tqdm(rawfile, total=total):
        linesplit = line.strip().split('\t')
        for (datacol, ext), content in zip([
        ("paths", "path"),
        ("texts", "txt"),
        ("translation", "de"),
        ("units", "unit"),
        ("lengths", "len"),
        ("subwords", "subword"),
        
    ], linesplit):
            print(content, file=fileptrs[datacol])
            

[
    ("symbolunits", "symbolunit"),  # for i in units/*; do cat $i | python3 '/storage/LabJob/Projects/AudioWords/AudioSentencePiece/utils/unitfy.py' > symbolunits/$(basename $i .unit).symbolunit; done                           
    ("collunits", "collunit"),  # mkdir collunits; for i in symbolunits/*; do cat $i | python3 '/storage/LabJob/Projects/AudioWords/AudioSentencePiece/utils/collapse.py' > collunits/$(basename $i .symbolunit).collunit; done

    ("wordlengths", "wordlen"),  # zsh '/storage/LabJob/Projects/AudioWords/AudioSentencePiece/utils/word_len.sh'

    # zsh '/storage/LabJob/Projects/AudioWords/AudioSentencePiece/utils/split_len.sh'
    ("delengths", "delength"),
    ("desubwords", "desubword"),
    ("endelengths", "endelength"),
    ("endesubwords", "endesubword"),
    
]
