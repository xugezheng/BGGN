import random
import torch
import numpy as np
import pandas as pd
import os
import os.path as osp
from collections import defaultdict
import pickle
import copy
from random import sample

import logging
logger = logging.getLogger("intersectionalFair")

def seed_set(args):
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
def onehot2vec(one_hot_batch, output_dim, class_dim):
    '''
    one_hot_batch: (batch, class_num * sens_attrs_num), numpy array
    '''
    bs = one_hot_batch.shape[0]
    one_hot_batch = one_hot_batch.reshape(bs, output_dim, class_dim)
    vec = np.argmax(one_hot_batch, axis=2)
    return vec.astype(float)

##############################################
########### df operations ####################
##############################################
def df_2_dict(df_dict, target_id, sens_attr_ids):
    '''
    function: x based dataframe to a based sens attrs dict, for evaluation
    bias is group bias
    df_dict: dict, contains train and test dataframe (df_regroup)
    return: dict[str(name), dict[str(a_str), dict[str, list/float]]]
    inner element: {a: info dict (including y_true, y_score, idx, losses, sample_num, bias)}
    '''
    df = copy.deepcopy(df_dict)
    re_dict = {}
    for name in ["train", "test"]:
        df_array = df[name].to_numpy()
        df_y_scores = df_array[:, -1]
        df_biases = df_array[:, -2]
        df_biases_group = df_array[:, -3]
        df_y = df_array[:, target_id]
        df_sens_attrs = df_array[:, sens_attr_ids]
        sens_attrs_dict = defaultdict(lambda: {'y_true':[], 'y_scores':[], 'idx':[], 'losses':[], 'bias':0.0})
        
        for idx, a in enumerate(df_sens_attrs):
            a_str = ' '.join(list(map(str, a)))
            sens_attrs_dict[a_str]["y_true"].append(df_y[idx])
            sens_attrs_dict[a_str]["y_scores"].append(df_y_scores[idx])
            sens_attrs_dict[a_str]["idx"].append(idx)
            sens_attrs_dict[a_str]["losses"].append(df_biases[idx])
            sens_attrs_dict[a_str]['bias'] = df_biases_group[idx]
        
        for (key, values) in sens_attrs_dict.items():
            sens_attrs_dict[key]['sample_num'] = len(values["y_true"])
        
        re_dict[name] = sens_attrs_dict
    return re_dict


def df_2_lists(df_dict, sens_attr_ids):
    '''
    function: x based dataframe to a based sens attrs list, in order to pass to models(NN or DT)
    df_dict is x based
    bias is group bias
    df_org: dict, contains train and test dataframe
    return: dict[str, tuple[list[list[float]], list[float]]]
    inner element: a(list of float) - group bias
    '''
    df = df_dict.copy()
    re_dict = {}
    for name in ["train", "test"]:
        df_array = df[name].to_numpy()
        df_biases_group = df_array[:, -3]
        df_sens_attrs = df_array[:, sens_attr_ids]
        sens_attrs_dict = defaultdict(lambda: {'y_true':[], 'y_scores':[], 'idx':[], 'losses':[]})
        for idx, a in enumerate(df_sens_attrs):
            a_str = ' '.join(list(map(str, a)))
            sens_attrs_dict[a_str]['bias'] = df_biases_group[idx]
        a_all = []
        group_bias_all = []
        for (key, values) in sens_attrs_dict.items():
            a = list(map(float,key.split(' ')))
            a_all.append(a)
            group_bias_all.append(values['bias'])
        assert len(a_all) == len(sens_attrs_dict)
        re_dict[name] = (a_all, group_bias_all)
    return re_dict


def df_reduce(df_org, sens_attr_ids):
    """_summary_
    x based data to a based data
    per subgroup, retain one sample (the first x under a subgroup occurs in the given dataframe(e.g., train, test))
    df_org: dataframe
    """
    df = df_org.copy()    
    df_array = df.to_numpy()
    df_sens_attrs = df_array[:, sens_attr_ids]
    remain_row = []
    sens_attrs_dict = defaultdict(lambda: {'idx':[]})
    for idx, a in enumerate(df_sens_attrs):
        a_str = ' '.join(list(map(str, a)))
        sens_attrs_dict[a_str]["idx"].append(idx)
    for (key, values) in sens_attrs_dict.items():
        remain_row.append(values["idx"][0])
    df_remained = df.iloc[remain_row]
    logger.info(f"[A BIAS DATALOADER] Reducing X to A | Sample Num (Org/Reduced): {len(df)}/{len(df_remained)}")
    logger.info(f"[A BIAS DATALOADER] Reducing X to A | Sens A Num (Org/Reduced): {len(sens_attrs_dict)}/{len(df_remained)}")
    return df_remained
    
    
def bias_value_df_generation(df_bias, args):
    '''
    recalculate the group bias value for each a accros the whole dataset (including train, test and val)
    regroup the whole dataset into train (observation) and test (holdout)
    with NO OVERLAP
    returned df_regroup: "all", "train", "test"
    '''
    
    df_all = pd.concat([df_bias["train"], df_bias["val"], df_bias["test"]])
    df_regroup = regroup_a_bias_df(df_all, args)
    data_dir = osp.join(args.output_dir, "data")
    if not osp.exists(data_dir):
        os.makedirs(data_dir)
    regroup_name = f"data_frame_regroup_{args.f_model_df_regroup_norm}.pickle"
    with open(f"{data_dir}/{regroup_name}", "wb") as handle:
        pickle.dump(df_regroup, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return df_regroup


def regroup_a_bias_df(df_org, args):
    #===================================
    # df bias value regroup
    #===================================
    df = df_org.copy() # a merged dataframe
    df_array = df.to_numpy()
    df_y = df_array[:, args.target_id]
    df_sens_attrs = df_array[:, args.sens_attr_ids]
    # 3 added metrics
    df_y_scores = df_array[:, -1]
    df_biases = df_array[:, -2]
    if args.f_model_df_regroup_norm == "minmax":
        min_val = np.min(df_biases)
        max_val = np.max(df_biases)
        df_biases = (df_biases - min_val) / (max_val - min_val)
    elif args.f_model_df_regroup_norm == 'zscore':
        mean = np.mean(df_biases)
        std_dev = np.std(df_biases)
        df_biases = (df_biases - mean) / std_dev
    else:
        # "org" no scale for bias value
        pass
    df_biases_group = df_array[:, -3]
    
    # acc
    ap = np.sum(np.greater(df_y_scores, 0.5) == df_y) / float(len(df_y_scores))
    
    sens_attrs_dict = defaultdict(lambda: {'y_true':[], 'y_scores':[], 'idx':[], 'losses':[]})
    bias_group = np.zeros(len(df_biases_group))
    for idx, a in enumerate(df_sens_attrs):
        a_str = ' '.join(list(map(str, a)))
        sens_attrs_dict[a_str]["y_true"].append(df_y[idx])
        sens_attrs_dict[a_str]["y_scores"].append(df_y_scores[idx])
        sens_attrs_dict[a_str]["losses"].append(df_biases[idx])
        sens_attrs_dict[a_str]["idx"].append(idx)
    
    for (key, values) in sens_attrs_dict.items():
        cur_a_ap = np.sum(np.greater(np.array(values["y_scores"]), 0.5) == np.array(values["y_true"])) / float(len(values["y_scores"]))
        sens_attrs_dict[key]['ap'] = cur_a_ap
        sens_attrs_dict[key]['ap_dataset'] = ap
        # sample number
        sens_attrs_dict[key]['sample_num'] = len(values["y_true"])
        # loss based bias
        sens_attrs_dict[key]['loss_group'] = np.mean(sens_attrs_dict[key]['losses'])
        for i in sens_attrs_dict[key]['idx']:
            bias_group[i] = sens_attrs_dict[key]['loss_group']
           
    
    # In the original full-set (train+test+val) dataframe
    # after merging the three dataframes
    # reorganize the sensattr and recalculate the mean bias and replace the original group bias (index: -3).
    df["bias_group"] = bias_group
    df["bias"] = df_biases
    logger.info(f"[DATA PREPARATION] Regroup Dataset (train + val + test) Bias Group Important Values:\n Mean: [{np.mean(bias_group):.4f}] | Medium: [{np.median(bias_group):.4f}] | 70%: {np.percentile(bias_group, 70):.4f}")
    #===================================
    # tr te generation
    #===================================
    te_par = args.te_par
    df_regroup = df.copy()
    te_row = []
    tr_row = []
    te_dict = {}
    tr_dict = {}
    # non-repetition sampling
    te_a = sample(list(sens_attrs_dict.keys()), int(te_par * len(list(sens_attrs_dict.keys()))))
    
    # sample based dict, saving every sample, repeated a
    for (key, values) in sens_attrs_dict.items():
        if key in te_a:
            te_row += sens_attrs_dict[key]['idx']
            te_dict[key] = sens_attrs_dict[key]
        else:
            tr_row += sens_attrs_dict[key]['idx']
            tr_dict[key] = sens_attrs_dict[key]   
    df_dict = {}
    te_row = sorted(te_row)
    tr_row = sorted(tr_row)
    df_tr = df_regroup.iloc[tr_row]
    df_te = df_regroup.iloc[te_row]
    df_dict["all"] = df_regroup
    df_dict["train"] = df_tr
    df_dict["test"] = df_te
    logger.info(f"DATA PREPARATION] All   Set:\nLen: {len(df_regroup)} | Sens Attr Num: {len(tr_dict) + len(te_dict)}")
    logger.info(f"DATA PREPARATION] Train Set:\nLen: {len(tr_row)} | Sens Attr Num: {len(tr_dict)} | Sample: {tr_row[0:5]}")
    logger.info(f"DATA PREPARATION] Test  Set:\nLen: {len(te_row)} | Sens Attr Num: {len(te_dict)} | Sample: {te_row[0:5]}")
    
    return df_dict


##############################################
########### save operations ##################
##############################################
def save_dict_2_df(dicts, df_dir):
    """
    dicts: [dict_org, dict_gen_init, dict_gen_ft]
    train_org_df
    train_gen_initVae_df
    train_gen_ftVae_df
    test_org_df
    test_gen_initVae_df
    test_gen_ftVae_df
    """
    
    for dict_idx, dict_name in enumerate(["org", "gen_initVae", "gen_ftVae"]):
        for i in ["test", "train"]:
            df_name = f"{i}_{dict_name}"
            
            if dict_idx == 0:
                data_list = list(zip(*(dicts[dict_idx][i])))
                data = {
                    "A": data_list[0],
                    "GT_group_bias": data_list[1]
                }
            else:
                data_list = list(zip(*(dicts[dict_idx][f"{i}_a_ordered"])))
                data = {
                    "A": data_list[0],
                    "pred_group_bias": data_list[1],
                    "GT_group_bias": data_list[2]
                }
            df_cur = pd.DataFrame.from_dict(data)
            df_cur.to_csv(osp.join(df_dir, f"{df_name}.csv"))


def save_dset_dict_2_df(d, df_dir):
    """
    d: dict_org
    """
    for i in ["test", "train"]:
        df_name = f"{i}_dataset"
        data_list = list(zip(*(d[i])))
        data = {
            "A": data_list[0],
            "GT_group_bias": data_list[1],
            "X_num": data_list[2]
        }
        df_cur = pd.DataFrame.from_dict(data)
        df_cur.to_csv(osp.join(df_dir, f"{df_name}.csv"))


def save_train_test_dict_2_df(dicts, df_dir, name):
    """
    dicts: [dict_org, dict_gen_init, dict_gen_ft]
    train_org_df
    train_gen_initVae_df
    train_gen_ftVae_df
    test_org_df
    test_gen_initVae_df
    test_gen_ftVae_df
    """
    
    
    for i in ["test", "train", "new"]:
        df_name = f"{name}_{i}"
        # data_list = list(zip(*(dicts[f"{i}_a_ordered"])))
        data_list = dicts[f"{i}_a_ordered"]
        data = {
            "A": data_list[0],
            "pred_group_bias": data_list[1],
            "GT_group_bias": data_list[2]
        }
        df_cur = pd.DataFrame.from_dict(data)
        df_cur.to_csv(osp.join(df_dir, f"{df_name}_A.csv"))
        
        # data_list_x = list(zip(*(dicts[f"{i}_x_ordered"])))
        data_list_x = dicts[f"{i}_x_ordered"]
        data_x = {
            "A": data_list_x[0],
            "pred_group_bias": data_list_x[1],
            "GT_group_bias": data_list_x[2]
        }
        df_cur_x = pd.DataFrame.from_dict(data_x)
        df_cur_x.to_csv(osp.join(df_dir, f"{df_name}_X.csv"))
        
def save_baseline_list_2_df(baseline_list, df_dir, name):
    data_list = list(zip(*(baseline_list)))
    data = {
            "A": data_list[0],
            "pred_group_bias": data_list[1],
            "GT_group_bias": data_list[2]
        }
    df_cur = pd.DataFrame.from_dict(data)
    df_cur.to_csv(osp.join(df_dir, f"{name}_A.csv"))



if __name__ == '__main__':
    pass