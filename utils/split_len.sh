#!sh

mkdir -p endelengths
mkdir -p endesubwords

# for SPLIT in dummy \
#     dev-clean dev-other \
#     test-clean test-other \
#     train-clean-100 train-clean-360 train-other-500; do
for SPLIT in dummy train dev test; do
echo $SPLIT
python -c '

import os
from transformers import AutoTokenizer
from tqdm import tqdm
tok = AutoTokenizer.from_pretrained("./_tok")

split = "'$SPLIT'"
with os.popen("wc -l "f"texts/{split}.txt") as f:
    total = int(f.read().split()[0].strip())
with open(f"texts/{split}.txt") as f:
    with open(f"endelengths/{split}.endelength", "w") as fout:
        with open(f"endesubwords/{split}.endesubword", "w") as foutw:
            for i in tqdm(f, total=total):
                s = i.lower().strip()
                w = (tok.tokenize(s))
                ss = len(w)
                print(ss, file=fout)
                print(" ".join(w), file=foutw)
'
# read
done

# ---------------------------
mkdir -p delengths
mkdir -p desubwords

# for SPLIT in dummy \
#     dev-clean dev-other \
#     test-clean test-other \
#     train-clean-100 train-clean-360 train-other-500; do
for SPLIT in dummy train dev test; do
echo $SPLIT
python -c '

import os
from transformers import AutoTokenizer
from tqdm import tqdm
tok = AutoTokenizer.from_pretrained("./_tok")

split = "'$SPLIT'"
with os.popen("wc -l "f"translation/{split}.de") as f:
    total = int(f.read().split()[0].strip())
with open(f"translation/{split}.de") as f:
    with open(f"delengths/{split}.delength", "w") as fout:
        with open(f"desubwords/{split}.desubword", "w") as foutw:
            for i in tqdm(f, total=total):
                s = i.lower().strip()
                w = (tok.tokenize(s))
                ss = len(w)
                print(ss, file=fout)
                print(" ".join(w), file=foutw)
'
# read
done
