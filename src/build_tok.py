#!/usr/bin/env python3  # ~~~ VERIFIED ~~~ #
from pathlib import Path
from transformers import BartTokenizerFast, BartTokenizer
from tokenizers.processors import RobertaProcessing
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer

def build_tokenizer(
    tokenizer_path='./unit_tokenizerfast',
    train=False,
    save=False,
    program_testing=False,
  ):
    if Path(tokenizer_path).exists():
        return BartTokenizerFast.from_pretrained(tokenizer_path)
        

    tok = BartTokenizer.from_pretrained('facebook/bart-base')
    # unk_token = '<unk>'
    unk_token = tok.unk_token

    Ftokenizer = Tokenizer(WordLevel(unk_token=unk_token))
    trainer = WordLevelTrainer(
        special_tokens=[
            # tok.all_special_tokens_extended
            tok.special_tokens_map_extended['cls_token'],
            # tok.special_tokens_map_extended['bos_token'],
            tok.special_tokens_map_extended['pad_token'],
            tok.special_tokens_map_extended['sep_token'],
            # tok.special_tokens_map_extended['eos_token'],
            tok.special_tokens_map_extended['unk_token'],
            tok.special_tokens_map_extended['mask_token'],
        ],
        min_frequency=1,
        vocab_size=500 + len(tok.all_special_tokens_extended),
    )

    Ftokenizer.add_tokens([
        tok.special_tokens_map_extended['cls_token'],
        # tok.special_tokens_map_extended['bos_token'],
        tok.special_tokens_map_extended['pad_token'],
        tok.special_tokens_map_extended['sep_token'],
        # tok.special_tokens_map_extended['eos_token'],
        tok.special_tokens_map_extended['unk_token'],
        tok.special_tokens_map_extended['mask_token'],
    ])
    Ftokenizer.add_tokens(['uni_{:04d}'.format(d) for d in range(500)])
    Ftokenizer.add_tokens([
        tok.special_tokens_map_extended['mask_token']])

    Ftokenizer.pre_tokenizer = Whitespace()
    Ftokenizer.post_processor = RobertaProcessing(
        sep=(tok.sep_token, tok.sep_token_id),
        cls=(tok.cls_token, tok.cls_token_id),
    )
    PREFIX = Path(
        '/storage/LabJob/Projects/FairseqCollapse'
        '/data/train-clean-100'
    )
    corpus = [
        str(PREFIX / 'train.unit'),
        # str(PREFIX / 'dev.unit'),
        # str(PREFIX / 'test.unit'),
    ]

    if train:
        Ftokenizer.train(corpus, trainer)

    tokenizer = BartTokenizerFast(
        tokenizer_object=Ftokenizer,
        model_max_length=tok.model_max_length,
        **tok.special_tokens_map_extended,
    )

    if save:
        tokenizer.save_pretrained(tokenizer_path)

    if program_testing:
        from pprint import pprint
        pprint({k: v for v, k in sorted([(v, k) for k, v in tokenizer.get_vocab().items()])})

    for i in range(50):
        assert tokenizer.get_vocab()['uni_{:04d}'.format(i)] == i + 5, (
            str(tokenizer.get_vocab()['uni_{:04d}'.format(i)]))

    return tokenizer

def test(tokenizer):
    tokenizer([
      'uni_0032 uni_0033',
      'uni_0012 uni_0023 uni_0043 uni_0036 uni_0026',
      'uni_0032 uni_0044 uni_0033',
    ], return_tensors='pt', 
    padding=True, truncation=True)
