LOG_WANDB = True
MAXUNITLEN = 1024
MAXTEXTLEN = 512
DATADIR_PREFIX = pathlib.Path("data/fairseq_data/data")
PRETRAINED_PREFIX = pathlib.Path("pret")
CKPT_PREFIX = pathlib.Path("ckpts")
EXP_PREFIX = pathlib.Path("exp")

from src.newmodels import SentBartForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import logging
import os
import pathlib
from transformers import BartTokenizer
import torch, numpy as np
from tqdm import tqdm
import jiwer

# region       === classes ===        NOIGER #
class MyUnitDataset(Dataset):
    def __init__(self, units, texts=None, wordlen=None):
        self.units = units
        if texts is not None:
            assert len(texts) == len(self.units)
        self.texts = texts
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
def DataSetCollector(infix, collapsed=True):
    suffix = "_coll" if collapsed else ""

    logging.warning('== ....      ==')
    with open(DATADIR_PREFIX /
       f'train-clean-100{suffix}/{infix}.en') as f:
        texts = f.read().strip().split('\n')

    with open(DATADIR_PREFIX /
       f'train-clean-100{suffix}/{infix}.unit') as f:
        original_units = f.read().strip().split('\n')

    with open(DATADIR_PREFIX /
       f'train-clean-100{suffix}/{infix}.len') as f:
        wordlens = f.read().strip().split('\n')

    assert len(texts) == len(original_units)
    assert len(wordlens) == len(original_units)

    mydataset = MyUnitDataset(original_units, texts, wordlens)

    return mydataset
def Data_collate_fn(unit_tokenizer, text_tokenizer):
    # done: combine & 應該要都可以處理，沒 label 或 length
    def collate_fn(batch):
        input_ids, labels, wordlens = list(zip(*batch))
        output_dict = dict(
            **unit_tokenizer(
                list(input_ids),
                return_tensors='pt',
                padding=True,
                truncation=True))
        if labels[0] is not None:
            output_dict["labels"] = text_tokenizer(
                list(labels),
                return_tensors='pt',
                padding=True,
                truncation=True)['input_ids']
        if wordlens[0] is not None:
            output_dict["word_length_tensor"] = torch.tensor(
                np.array(wordlens, dtype=int))
        return output_dict
    return collate_fn
def load_cached(cls, obj_name, saved_path, msg="Loading ..."):
    logging.warning(msg)
    if list(pathlib.Path(saved_path).glob('*')) == []:
        pathlib.Path(saved_path).rmdir()
    if os.path.isdir(saved_path):
        logging.warning('    (Using local cache...)')
        obj = cls.from_pretrained(saved_path)
    else:
        logging.warning('    (Loading pretrained...)')
        obj = cls.from_pretrained(obj_name)
        obj.save_pretrained(saved_path)
    return obj
def load_cached_tokenizer(cls, obj_name, saved_path, msg="Loading ..."):
    if list(pathlib.Path(saved_path).glob('*')) == []:
        pathlib.Path(saved_path).rmdir()
    tokenizer = load_cached(cls, obj_name, saved_path, msg)
    speech_units = ['uni_{:04d}'.format(d) for d in range(500)]  # TODOLATER: modify format
    if speech_units[0] not in tokenizer.get_vocab():
        tokenizer.add_tokens(speech_units)
        tokenizer.save_pretrained(saved_path)
    return tokenizer

if __name__ == "__main__":
    tokenizer = load_cached_tokenizer(
        cls=BartTokenizer,
        obj_name='facebook/bart-base',
        saved_path=PRETRAINED_PREFIX / "hf_toks",
        msg="Loading ...")

    collate_fn = Data_collate_fn(
        unit_tokenizer=tokenizer,
        text_tokenizer=tokenizer,
    )
    dev_dataset = DataSetCollector('dev')
    test_dataset = DataSetCollector('test')
    model = load_cached(
        SentBartForConditionalGeneration,
        "voidful/asr_hubert_cluster_bart_base",
        'Golden/checkpoint-8000/',
    )
    model = model.cuda()


    test_dataloader = DataLoader(
        test_dataset,
        batch_size=6,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model.eval()
    outtexts_tfor_testset = []
    with torch.no_grad():
        for inputs in tqdm(test_dataloader):
            outputs = model.generate(
                input_ids=inputs['input_ids'].cuda(),
                attention_mask=inputs['attention_mask'].cuda(),
                word_length_tensor=inputs['word_length_tensor'].cuda(),
            )

            outtexts_tfor = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outtexts_tfor_testset.extend(outtexts_tfor)

    print("WER = {:6.2f} % (teacherforcing)".format(jiwer.wer(truth=test_dataset.texts, hypothesis=outtexts_tfor_testset)) * 100)

    outtexts_real_testset = []
    with torch.no_grad():
        for inputs in tqdm(test_dataloader):
            outputs_real = model.generate(
                input_ids=inputs['input_ids'].cuda(),
                attention_mask=inputs['attention_mask'].cuda(),
                word_length_tensor=inputs['word_length_tensor'].cuda(),
                max_length=512,
            )

            outtexts_real = tokenizer.batch_decode(outputs_real, skip_special_tokens=True)
            outtexts_real_testset.extend(outtexts_real)

    print("WER = {:6.2f} % (AR generation)".format(jiwer.wer(truth=test_dataset.texts, hypothesis=outtexts_real_testset)) * 100)

    outtexts_tfor_testset = []
    with torch.no_grad():
        for inputs in tqdm(test_dataloader):
            outputs = model.generate(
                input_ids=inputs['input_ids'].cuda(),
                attention_mask=inputs['attention_mask'].cuda(),
            )

            outtexts_tfor = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outtexts_tfor_testset.extend(outtexts_tfor)

    print("WER = {:6.2f} % (teacherforcing, no length)".format(jiwer.wer(truth=test_dataset.texts, hypothesis=outtexts_tfor_testset)) * 100)

    outtexts_real_testset = []
    with torch.no_grad():
        for inputs in tqdm(test_dataloader):
            outputs_real = model.generate(
                input_ids=inputs['input_ids'].cuda(),
                attention_mask=inputs['attention_mask'].cuda(),
                max_length=512,
            )

            outtexts_real = tokenizer.batch_decode(outputs_real, skip_special_tokens=True)
            outtexts_real_testset.extend(outtexts_real)

    print("WER = {:6.2f} % (AR generation, no length)".format(jiwer.wer(truth=test_dataset.texts, hypothesis=outtexts_real_testset)) * 100)
