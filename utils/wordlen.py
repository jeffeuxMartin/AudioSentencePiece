from transformers import AutoTokenizer
PATHname = 'test-clean.en'
subwordName = PATHname.split('.')[0] + '.subword'
lenName = PATHname.split('.')[0] + '.len'
tok = AutoTokenizer.from_pretrained('facebook/bart-base')
with open(subwordName, 'w') as fa:
    with open(lenName, 'w') as fb:
        with open(PATHname) as fr:
            for l in fr:
                s = (' '.join(tok.tokenize(l)))
                print(s, file=fa)
                print(len(s), file=fb)
