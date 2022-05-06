LOG_WANDB = True
MAXUNITLEN = 1024
MAXTEXTLEN = 512
import pathlib
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
        batch_size=24,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model.eval()
    
    # ---------------------
    outtexts_real_testset = []
    outtexts_real_testset2 = []
    with torch.no_grad():
        # for inputs, transcription in tqdm(zip(test_dataloader, test_dataset.texts), total=len(test_dataset.texts)):
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            outputs_real = model.generate(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                word_length_tensor=batch['word_length_tensor'].cuda(),
                max_length=1024,
            )
            outputs_real2 = model.generate(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                word_length_tensor=None,
                max_length=1024,
            )

            outtexts_real = tokenizer.batch_decode(outputs_real, skip_special_tokens=True)
            outtexts_real_testset.extend(outtexts_real)
            outtexts_real2 = tokenizer.batch_decode(outputs_real2, skip_special_tokens=True)
            outtexts_real_testset2.extend(outtexts_real2)

            demo_template = (
                "PRED:         \033[01;33m{PRED}\033[0m\n"
                "PRED(no len): \033[00;33m{PRED2}\033[0m\n"
                "REAL:         \033[01;32m{REAL}\033[0m\n")
            print('\n'.join([
                demo_template.format(PRED=pred, PRED2=pred2, REAL=transcription) 
                    for pred, pred2, transcription in zip(
                        outtexts_real, 
                        outtexts_real2, 
                        test_dataset.texts[
                            test_dataloader.batch_size * batch_idx
                            :
                            test_dataloader.batch_size * (batch_idx + 1)
                        ])]))
    print()    
    print("WER = {:6.4f} % (AR generation)".format(100 * jiwer.wer(truth=test_dataset.texts, hypothesis=outtexts_real_testset)))
    print("WER = {:6.4f} % (AR generation, no length)".format(100 * jiwer.wer(truth=test_dataset.texts, hypothesis=outtexts_real_testset2)))
    exit()
    # ---------------------
    # ---------------------
    outtexts_real_testset = []
    with torch.no_grad():
        # for inputs, transcription in tqdm(zip(test_dataloader, test_dataset.texts), total=len(test_dataset.texts)):
        for batch_idx, batch in tqdm(test_dataloader):
            outputs_real = model.generate(
                input_ids=inputs['input_ids'].cuda(),
                attention_mask=inputs['attention_mask'].cuda(),
                word_length_tensor=inputs['word_length_tensor'].cuda(),
                max_length=1024,
            )

            outtexts_real = tokenizer.batch_decode(outputs_real, skip_special_tokens=True)
            outtexts_real_testset.extend(outtexts_real)

            demo_template = (
                "PRED: \033[01;33m{PRED}\033[0m\n"
                "REAL: \033[01;32m{REAL}\033[0m\n")
            print('\n'.join([
                demo_template.format(PRED=pred, REAL=transcription) 
                    for pred, transcription in zip(
                        outtexts_real, 
                        test_dataset.texts[
                            test_dataloader.batch_size * batch_idx
                            :
                            test_dataloader.batch_size * (batch_idx + 1)
                        ])]))
    print("WER = {:6.4f} % (AR generation)".format(100 * jiwer.wer(truth=test_dataset.texts, hypothesis=outtexts_real_testset)))
    
    # ---------------------