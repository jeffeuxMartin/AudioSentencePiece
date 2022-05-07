import os
import argparse
import torch
import logging
FORMAT = '\033[01;31m%(asctime)s\033[0m %(message)s'
logging.basicConfig(format=FORMAT)

def load_cached(PRINTDEBUG):
    def load_cached(cls, obj_name, saved_path, msg="Loading ..."):
        PRINTDEBUG(msg)
        if os.path.isdir(saved_path):
            PRINTDEBUG('    (Using local cache...)')
            obj = cls.from_pretrained(saved_path)
        else:
            PRINTDEBUG('    (Loading pretrained...)')
            obj = cls.from_pretrained(obj_name)
            obj.save_pretrained(saved_path)
        return obj
    return load_cached
    
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

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=6)
    parser.add_argument("-lr", "--lr", type=float, default=2e-4)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--vram", type=float, default=10)
    args = parser.parse_args()

    batch_scaled_up = max(int(args.vram / 10.), 1)
    # args.batch_size *= batch_scaled_up
    # if batch_scaled_up > 1:
    #     logging.warning(
    #         f"Batch size resized to {args.batch_size:3d}..."
    #     )

    if args.run_name is None:
        args.run_name = (
            # f"lr = {args.lr}, bsz = {args.batch_size} ({batch_scaled_up} scaled_up)"
            f"lr = {args.lr}, bsz = {args.batch_size}"
        )    
    return args
