import pandas as pd
import glob 
D = glob.glob('../../golden_corpus/*_result_adv.tsv')

C = [
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
    ]

from tqdm import tqdm

# os.mkdirs("lengths")
# os.mkdirs("paths")
# os.mkdirs("subwords")
# os.mkdirs("texts")
# os.mkdirs("units")
# os.mkdirs("translation")
splits = [d[20:-15] for d in D]
for split in splits:
    print(split)
    import os
    with os.popen(f'wc -l ../../golden_corpus/{split}_result_adv.tsv') as ff:
        total = int(ff.read().strip().split()[0])

    with open(f"lengths/{split}.len", 'w') as flengths:
        with open(f"paths/{split}.path", 'w') as fpaths:
            with open(f"subwords/{split}.subword", 'w') as fsubwords:
                with open(f"texts/{split}.txt", 'w') as ftexts:
                    with open(f"units/{split}.unit", 'w') as funits:
                        with open(f"translation/{split}.de", 'w') as ftranslation:
                            with open(f'../../golden_corpus/{split}_result_adv.tsv') as FFF:
                                FFF.readline()
                                for LL in tqdm(FFF, total=total):
                                    liner = LL.strip().split('\t')
                                    assert len(liner) == 6
                                    row = {}
                                    row['length_src'], row['data_id'], row['spwords'], row['src_txt'], row['units'], row['tgt_txt'] = liner
                                    print(row['length_src'], file=flengths)
                                    print(row['data_id'], file=fpaths)
                                    print(row['spwords'], file=fsubwords)
                                    print(row['src_txt'], file=ftexts)
                                    print(row['units'], file=funits)
                                    print(row['tgt_txt'], file=ftranslation)
