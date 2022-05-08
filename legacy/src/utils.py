from typing import List
import argparse
import numpy as np
import torch
from torch import Tensor

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-b", "--batch_size_train",
        type=int, default=8,
    )

    parser.add_argument(
        "-B", "--batch_size_eval",
        type=int, default=8,
    )

    parser.add_argument(
        "-l", "--lr", "--learning_rate",
        type=float, default=3e-4,
    )

    parser.add_argument(
        "-L", "--max_len", "--max_length",
        type=int, default=700,
    )

    parser.add_argument(
        "-v", "--validation", "--validation_portioin",
        type=int, default=10,
    )

    parser.add_argument(
        "-e", "--epochs", 
        type=int, default=6,
    )
    
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        default="./token_autoencoder",
    )

    parser.add_argument(
        "-w", "--warmup_steps", 
        type=int, 
        default=1000,
    )
    
    args = parser.parse_args()
    return args  # , config, backup_files

def compute_metrics(eval_preds, PAD_TOKEN=503):
    logits, (labels, attn_mask) = eval_preds
    predictions = np.argmax(logits, axis=-1)

    counted = (labels != -100) & (labels != PAD_TOKEN)
    # assert torch.equal(counted, attn_mask)
    if np.array_equal(counted, attn_mask):
        correct = ((predictions == labels) * counted).sum(-1)
    else:  # ??? FIXME!
        correct = ((predictions == labels) * attn_mask).sum(-1)
    return {
        "acc":
            (correct / counted.sum(-1)).mean()
    }

def range_checker(stacked_codes, PREPADDING_ID=-1):
    MAX = stacked_codes.max().item()
    MIN = stacked_codes.min().item()
    ORIGINAL_MIN = (
        stacked_codes[stacked_codes != PREPADDING_ID
        ].min().item() if MIN == PREPADDING_ID else 
        MIN)
    PREPADDING = MIN == PREPADDING_ID
    
    return ORIGINAL_MIN, MAX, PREPADDING

def mask_generator(X_len, X=None, max_len=None):
    """
    X_len:   mask 的長度們
    X:       要被 mask 的 sequences
    max_len: 最長的 seq. len
    
    X 和 max_len 有一即可
    
    return --> mask 的 tensor
    """
    # XXX: unneeded??
    # if isinstance(X_len, torch.LongTensor):
    #     X_len = X_len.clone()
    # else:  # CHECK!
    #     X_len = torch.LongTensor(X_len)
    if max_len is not None:
        X_size = max_len
    elif X is not None:
        X_size = X.size(1)
    else:
        X = torch.zeros(max(X_len), len(X_len))
        X_size = X.size(0)
    return ((
        (torch.arange(X_size)[None, :]
        ).to(X_len.device) 
        < X_len[:, None]).long()
    )

def unpad_sequence(padded_sequences, lengths, batch_first=False):
    # type: (Tensor, Tensor, bool) -> List[Tensor]
    # ~~~ Copied from PyTorch
    r"""Unpad padded Tensor into a list of variable length Tensors

    ``unpad_sequence`` unstacks padded Tensor into a list of variable length Tensors.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> sequences = [a, b, c]
        >>> padded_sequences = pad_sequence(sequences)
        >>> lengths = torch.as_tensor([v.size(0) for v in sequences])
        >>> unpadded_sequences = unpad_sequence(padded_sequences, lengths)
        >>> torch.allclose(sequences[0], unpadded_sequences[0])
        True
        >>> torch.allclose(sequences[1], unpadded_sequences[1])
        True
        >>> torch.allclose(sequences[2], unpadded_sequences[2])
        True

    Args:
        padded_sequences (Tensor): padded sequences.
        lengths (Tensor): length of original (unpadded) sequences.
        batch_first (bool, optional): whether batch dimension first or not. Default: False.

    Returns:
        a list of :class:`Tensor` objects
    """

    unpadded_sequences = []

    if not batch_first:
        padded_sequences.transpose_(0, 1)

    max_length = padded_sequences.shape[1]
    idx = torch.arange(max_length)

    for seq, length in zip(padded_sequences, lengths):
        mask = idx < length
        unpacked_seq = seq[mask]
        unpadded_sequences.append(unpacked_seq)

    return unpadded_sequences

def shift_embeds_right(
    input_embeds: torch.Tensor, 
    pad_token_embed: torch.Tensor, 
    decoder_start_token_embed: torch.Tensor,
  ):
    """
    Shift input ids one embed to the right.
    """
    shifted_input_embeds = input_embeds.new_zeros(input_embeds.shape)
    shifted_input_embeds[:, 1:] = input_embeds[:, :-1].clone()
    shifted_input_embeds[:, 0] = decoder_start_token_embed

    if pad_token_embed is None:
        raise ValueError("self.model.config.pad_token_embed has to be defined.")
    # replace possible -100 values in labels by `pad_token_embed`
    # XXX: to fix? how to "recognize" [PAD]?
    shifted_input_embeds.masked_fill_(shifted_input_embeds == -100, pad_token_embed)

    return shifted_input_embeds

