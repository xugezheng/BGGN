import pandas as pd
import logging

from utils.utils import df_reduce
from data.celebA import get_loader_celebA
from data.toxic import get_loader_toxic

logger = logging.getLogger("intersectionalFair")


def f_model_loaders_load(df, mode, args):
    if args.dset == "celebA":
        return f_model_loaders_load_celebA(df, mode, args)
    elif args.dset == "toxic":
        return f_model_loaders_load_toxic(df, mode, args)
    else:
        return
    
     
def a_bias_loaders_load(df, args):
    if args.dset == "celebA":
        return a_bias_loaders_load_celebA(df, args)
    elif args.dset == "toxic":
        return a_bias_loaders_load_toxic(df, args)
    else:
        return
            
        
# ====================================================
# ====================   CelebA  =====================
# ====================================================
def f_model_loaders_load_celebA(df, mode, args):
    loaders_bias = {}
    df_train = pd.concat([df["train"], df["val"], df["test"]])
    logger.info(f"f_model train using train, val and test (all) data...")
    loaders_bias['train'] = get_loader_celebA(df=df_train, data_path=f"{args.data_root}/Img/img_align_celeba/", target_id=args.target_id, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode=mode, mode_shuffle=True)
    loaders_bias['tr'] = get_loader_celebA(df=df["train"], data_path=f"{args.data_root}/Img/img_align_celeba/", target_id=args.target_id, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode=mode,mode_shuffle=False)
    loaders_bias['val'] = get_loader_celebA(df=df["val"], data_path=f"{args.data_root}/Img/img_align_celeba/", target_id=args.target_id, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode=mode,mode_shuffle=False)
    loaders_bias['te'] = get_loader_celebA(df=df["test"], data_path=f"{args.data_root}/Img/img_align_celeba/", target_id=args.target_id, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode=mode,mode_shuffle=False)
    return loaders_bias


def a_bias_loaders_load_celebA(df, args):
    df_tr = df["train"]
    df_te = df["test"]
    df_all = df["all"]
    
    df_tr_reduced = df_reduce(df_tr, args.sens_attr_ids)
    df_te_reduced = df_reduce(df_te, args.sens_attr_ids)
    df_all_reduced = df_reduce(df_all, args.sens_attr_ids)
    
    loaders_bias = {}
    
    # for predictor and vae
    loaders_bias['train_reduced'] = get_loader_celebA(df=df_tr_reduced, data_path=None, target_id=args.target_id, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode="bias", mode_shuffle=True, args=args)
    # for bias_vae_trA_sample process
    loaders_bias['train_reduced_ft'] = get_loader_celebA(df=df_tr_reduced, data_path=None, target_id=args.target_id, batch_size=args.ft_batch_size, sens_attr_ids=args.sens_attr_ids, mode="bias", mode_shuffle=True, args=args)
    loaders_bias['all_reduced'] = get_loader_celebA(df=df_all_reduced, data_path=None, target_id=args.target_id, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode="bias", mode_shuffle=True, args=args)
    loaders_bias['tr_reduced'] = get_loader_celebA(df=df_tr_reduced, data_path=None, target_id=args.target_id, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode="bias",mode_shuffle=False, args=args)
    loaders_bias['te_reduced'] = get_loader_celebA(df=df_te_reduced, data_path=None, target_id=args.target_id, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode="bias",mode_shuffle=False, args=args)
    return loaders_bias


# ====================================================
# ======================  TOXIC  =====================
# ====================================================
def f_model_loaders_load_toxic(df, mode, args):
    
    loaders_bias = {}
    df_train = pd.concat([df["train"], df["val"], df["test"]]) # this order should be matched with all_xy.h5 generation order: train, val and finally test (see toxic_data_preparation.py - generate_multi_attrs_toxic())
    logger.info(f"f_model train using train, val and test (all) data...")
    loaders_bias['train'] = get_loader_toxic(df=df_train, xy_h5_path=f"{args.data_root}/xy_split/all_xy.h5", batch_size=64, sens_attr_ids=args.sens_attr_ids, mode=mode, mode_shuffle=True)
    loaders_bias['tr'] = get_loader_toxic(df=df["train"], xy_h5_path=f"{args.data_root}/xy_split/train_xy.h5", batch_size=64, sens_attr_ids=args.sens_attr_ids, mode=mode,mode_shuffle=False)
    loaders_bias['val'] = get_loader_toxic(df=df["val"], xy_h5_path=f"{args.data_root}/xy_split/val_xy.h5", batch_size=64, sens_attr_ids=args.sens_attr_ids, mode=mode,mode_shuffle=False)
    loaders_bias['te'] = get_loader_toxic(df=df["test"], xy_h5_path=f"{args.data_root}/xy_split/test_xy.h5", batch_size=64, sens_attr_ids=args.sens_attr_ids, mode=mode,mode_shuffle=False)
    return loaders_bias


def a_bias_loaders_load_toxic(df, args):
    
    df_tr = df["train"]
    df_te = df["test"]
    df_all = df["all"]
    
    df_tr_reduced = df_reduce(df_tr, args.sens_attr_ids)
    df_te_reduced = df_reduce(df_te, args.sens_attr_ids)
    df_all_reduced = df_reduce(df_all, args.sens_attr_ids)
    
    loaders_bias = {}
    # useless
    loaders_bias['train'] = get_loader_toxic(df=df_tr, xy_h5_path=None, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode="bias", mode_shuffle=True, args=args)
    loaders_bias['tr'] = get_loader_toxic(df=df_tr, xy_h5_path=None, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode="bias",mode_shuffle=False, args=args)
    loaders_bias['te'] = get_loader_toxic(df=df_te, xy_h5_path=None, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode="bias",mode_shuffle=False, args=args)
    
    # for predictor and vae
    loaders_bias['train_reduced'] = get_loader_toxic(df=df_tr_reduced, xy_h5_path=None, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode="bias", mode_shuffle=True, args=args)
    loaders_bias['train_reduced_ft'] = get_loader_toxic(df=df_tr_reduced, xy_h5_path=None, batch_size=args.ft_batch_size, sens_attr_ids=args.sens_attr_ids, mode="bias", mode_shuffle=True, args=args)
    loaders_bias['all_reduced'] = get_loader_toxic(df=df_all_reduced, xy_h5_path=None, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode="bias", mode_shuffle=True, args=args)
    loaders_bias['tr_reduced'] = get_loader_toxic(df=df_tr_reduced, xy_h5_path=None, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode="bias",mode_shuffle=False, args=args)
    loaders_bias['te_reduced'] = get_loader_toxic(df=df_te_reduced, xy_h5_path=None, batch_size=64, sens_attr_ids=args.sens_attr_ids, mode="bias",mode_shuffle=False, args=args)
    return loaders_bias