import os
import argparse
import logging
import pathlib

import torch

from transformers import AutoConfig


FORMAT = '\033[01;31m%(asctime)s\033[0m %(message)s'
logging.basicConfig(format=FORMAT)



def pure_advanced_load_pretrained(
    pretrained_model, 
    model, 
    verbose=(lambda *a: None),
):
    pretrained_dict = dict(pretrained_model.state_dict())
    new_dict = model.state_dict()

    for key_src in pretrained_model.state_dict():
        val_src = pretrained_dict[key_src]
        val_tgt = new_dict.get(key_src, None)
        if val_tgt is not None:
            # 都有
            if val_src.shape != val_tgt.shape:
                # 但重新更新了
                verbose(f"{key_src} reshaped! {val_src.shape} | {val_tgt.shape}")
                pretrained_dict.pop(key_src)
            else:
                # OK
                # verbose('Matched!')
                pass
        else:
            # 舊的有新的沒有
            verbose(f"{key_src} missing in new model! {val_src.shape}")
            pretrained_dict.pop(key_src)
    # 舊的沒有新的有，應該會被忽略！
    model.load_state_dict(pretrained_dict)
    return model

def advanced_load_pretrained(
    checkpoint_name, 
    model_class, 
    config_class, 
    verbose=(lambda *a: None),  # verbose=print,
    **new_config_options,
):
    """
    config_class = AutoConfig
    because with a pretrained, with an AutoConfig
    """
    newconfig = config_class.from_pretrained(
        checkpoint_name, 
        **new_config_options)
    newconfig.update(new_config_options)  # FIXME: redundant???
        # No! 如果有會蓋，但新的不會加，要 update
        # Yes? 其實可以只用後面這步

    pretrained_model = model_class.from_pretrained(checkpoint_name)
    model = model_class(config=newconfig)
    
    model = pure_advanced_load_pretrained(
        pretrained_model=pretrained_model,
        model=model,
        verbose=verbose,
    )

    del pretrained_model
    return model



def load_cached_pure(PRINTDEBUG):
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
    

def load_cached(cls, obj_name, saved_path, msg="Loading ..."):
    logging.warning(msg)
    if os.path.isdir(saved_path):
        if list(pathlib.Path(saved_path).glob('*')) == []:
            pathlib.Path(saved_path).rmdir()
    if os.path.isdir(saved_path):
        logging.warning('    (Using local cache...)')
        obj = cls.from_pretrained(saved_path)
        # obj = advanced_load_pretrained(obj_name, cls, type(config), **config)
    else:
        logging.warning('    (Loading pretrained...)')
        obj = cls.from_pretrained(obj_name)
        # obj = advanced_load_pretrained(obj_name, cls, type(config), **config)
        obj.save_pretrained(saved_path)
    return obj

def load_cached_config(cls, obj_name, saved_path, config=None, msg="Loading ..."):
    """ 
    If enter this, USE pretrained (local or remot) is confirmed! 
    We try to match maximized!
    Otherwise, 
        config = AutoConfig(collapse_n=-1)
        model = cls(config)
    """
    
    config = {} if config is None else config
    logging.warning(msg)
    if os.path.isdir(saved_path):
        if list(pathlib.Path(saved_path).glob('*')) == []:
            pathlib.Path(saved_path).rmdir()
    if os.path.isdir(saved_path):
        # print(saved_path)
        logging.warning('    (Using local cache...)')
        # obj = cls.from_pretrained(saved_path)
        obj = advanced_load_pretrained(saved_path, cls, AutoConfig, **config)
    else:
        pathlib.Path(saved_path
            ).mkdir(0o755, parents=True, exist_ok=True)    
        logging.warning('    (Loading pretrained...)')
        # pretrained_config = config
        # obj = cls.from_pretrained(obj_name, config)
        obj = advanced_load_pretrained(obj_name, cls, AutoConfig, **config)
        obj.save_pretrained(saved_path)
    return obj

def load_cached_tokenizer(cls, obj_name, saved_path, msg="Loading ..."):
    if os.path.isdir(saved_path):
        if list(pathlib.Path(saved_path).glob('*')) == []:
            pathlib.Path(saved_path).rmdir()
    tokenizer = load_cached(cls, obj_name, saved_path, msg)
    speech_units = ['uni_{:04d}'.format(d) for d in range(500)]  # TODOLATER: modify format
    if speech_units[0] not in tokenizer.get_vocab():  
        tokenizer.add_tokens(speech_units)
        tokenizer.save_pretrained(saved_path)
    return tokenizer


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

    parser.add_argument("--weight_len", type=float, default=None)
    parser.add_argument("--notcoll", 
        action='store_false', dest='coll')
    parser.set_defaults(coll=True)

    parser.add_argument("--nowandb", 
        action='store_false', dest='wandb')
    parser.set_defaults(wandb=True)

    parser.add_argument("--nolower", 
        action='store_false', dest='lower')
    parser.set_defaults(lower=True)

    parser.add_argument('--fix_encoder', action='store_true')
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--autoencoder', action='store_true')

    args = parser.parse_args()

    batch_scaled_up = max(int(args.vram / 10.), 1)
    # args.batch_size *= batch_scaled_up
    # if batch_scaled_up > 1:
    #     logging.warning(
    #         f"Batch size resized to {args.batch_size:3d}..."
    #     )

    default_run_name = (
        # f"lr = {args.lr}, bsz = {args.batch_size} ({batch_scaled_up} scaled_up)"
        f"lr = {args.lr}, bsz = {args.batch_size}, {args.epochs} epochs"
        + (" (coll)" if args.coll else " (orig)")
        + (" (lower)" if args.lower else " (normalcase)")
        + (" (fix_encoder)" if args.fix_encoder else "")
        + (" (orignalTfm)" if args.original else "")
        + (" (autoencoder)" if args.autoencoder else "")
        + (f" weight_len = {args.weight_len}" if args.weight_len is not None else "")
    )
    if args.run_name is None:
        args.run_name = default_run_name
    else:
        args.run_name = args.run_name + " " + default_run_name
    return args


