import pandas as pd
import numpy as np
import collections

import torch
import time
from collections import defaultdict

from vae_bias_utils import sample_from_multi_softmax_distribution, get_bias
from utils import onehot2vec
import logging
logger = logging.getLogger("intersectionalFair")

def sampling_for_evaluation(decoder, pred, args, a_bank=None, a_fea_bank=None, gp_bias_bank=None):
    a_generated_all = []
    b_all = []
    with torch.no_grad():
        T1 = time.time()
        for _ in range(args.evaluation_batch_num):
            z_samples = torch.randn(args.batch_size, args.latent_dim).to(args.device)
            hat_x = decoder(z_samples)
            dis_x = sample_from_multi_softmax_distribution(
                hat_x, args.class_dim, normal=True, one_hot=True
            )
            a_generated = dis_x.view(-1, args.class_dim * args.output_dim)
            if a_bank is not None:
                b = get_bias(dis_x, pred, a_bank, a_fea_bank, gp_bias_bank, args)
            else:
                b, _, __ = pred(a_generated)
            a_generated_all.append(a_generated.data.cpu().numpy())
            b_all.append(b.data.cpu().numpy())
        T2 = time.time()
        sampling_time = (T2 - T1)*1000
    a_generated_all = np.concatenate(a_generated_all)
    a_generated_all = onehot2vec(a_generated_all, args.output_dim, args.class_dim)
    b_all = np.concatenate(b_all).squeeze()
    return a_generated_all, b_all, sampling_time


def filter_and_regroup_genA(a_generated_all, b_all, a_dicts):
    '''
    Function: transfer all generated a into 3 dict, train/test/new, repeated （x based) and non-repeated (a based)
    a_generated_all: ndarray, (bs*sens_attrs), float - 0.0 or 1.0
    [tuple[str: a, float: pred, float: gt]] -> [tuple[str: a], tuple[float: pred], tuple[folat: gt]]
    '''
    tr_dict = a_dicts["train"] 
    te_dict = a_dicts["test"] 
    
    # temporary vars
    temps_pred_all_dict = {
        "train": {
            "x": [],
            "a": [],
        },  # (str, pred, gt)
        "test": {
            "x": [],
            "a": [],
        },
        "new": {
            "x": [],
            "a": [],
        }
    }
    temps_a_in_tr = defaultdict(lambda: {"num": 0, "bias_pred": [], "bias_gt": []})
    temps_a_in_te = defaultdict(lambda: {"num": 0, "bias_pred": [], "bias_gt": []})
    temps_a_new = defaultdict(lambda: {"num": 0, "bias_pred": [], "bias_gt": []})
    
    # train/test/new evaluation
    for idx, a in enumerate(a_generated_all):
        a_str = " ".join(list(map(str, a.tolist())))
        if a_str in tr_dict:
            # all result recording
            temps_pred_all_dict["train"]["x"].append(
                (a_str, b_all[idx], tr_dict[a_str]["bias"])
            )
            if temps_a_in_tr[a_str]["num"] == 0:
                temps_pred_all_dict["train"]["a"].append(
                    (a_str, b_all[idx], tr_dict[a_str]["bias"])
                )
            temps_a_in_tr[a_str]["num"] += 1
            temps_a_in_tr[a_str]["bias_pred"].append(b_all[idx])
            temps_a_in_tr[a_str]["bias_gt"].append(
                tr_dict[a_str]["bias"]
            ) 
        elif a_str in te_dict:
            # all result recording
            temps_pred_all_dict["test"]["x"].append(
                (a_str, b_all[idx], te_dict[a_str]["bias"])
            )
            if temps_a_in_te[a_str]["num"] == 0:
                temps_pred_all_dict["test"]["a"].append(
                    (a_str, b_all[idx], te_dict[a_str]["bias"])
                )
            temps_a_in_te[a_str]["num"] += 1
            temps_a_in_te[a_str]["bias_pred"].append(b_all[idx])
            temps_a_in_te[a_str]["bias_gt"].append(te_dict[a_str]["bias"])
        else:
            # all result recording
            temps_pred_all_dict["new"]["x"].append((a_str, b_all[idx], 0.0))
            if temps_a_new[a_str]["num"] == 0:
                temps_pred_all_dict["new"]["a"].append((a_str, b_all[idx], 0.0))
            temps_a_new[a_str]["num"] += 1
            temps_a_new[a_str]["bias_pred"].append(b_all[idx])
            temps_a_new[a_str]["bias_gt"].append(0)
            
    temps_pred_all_dict["train_a_ordered"] = sorted(
    temps_pred_all_dict["train"]["a"], key=lambda x: (x[1], x[0], x[2]), reverse=True
    )
    temps_pred_all_dict["test_a_ordered"] = sorted(
        temps_pred_all_dict["test"]["a"], key=lambda x: (x[1], x[0], x[2]), reverse=True
    )
    temps_pred_all_dict["new_a_ordered"] = sorted(
        temps_pred_all_dict["new"]["a"], key=lambda x: (x[1], x[0], x[2]), reverse=True
    )
    temps_pred_all_dict["train_x_ordered"] = sorted(
        temps_pred_all_dict["train"]["x"], key=lambda x: (x[1], x[0], x[2]), reverse=True
    )
    temps_pred_all_dict["test_x_ordered"] = sorted(
        temps_pred_all_dict["test"]["x"], key=lambda x: (x[1], x[0], x[2]), reverse=True
    )
    temps_pred_all_dict["new_x_ordered"] = sorted(
        temps_pred_all_dict["new"]["x"], key=lambda x: (x[1], x[0], x[2]), reverse=True
    )
    # decoupling after ordering
    dict_gen_sorted_listTuple = {}
    for k, v in temps_pred_all_dict.items():
        if "ordered" in k:
            dict_gen_sorted_listTuple[k] = list(zip(*v))
    
    return dict_gen_sorted_listTuple


def regroup_dataset(a_dicts):
    tr_dict = a_dicts["train"] # training/observation - a based, but with x sample number
    te_dict = a_dicts["test"]  # test/holdout - a based, but with x sample number
    tr_group_bias_true = [(key, tr_dict[key]["bias"], tr_dict[key]["sample_num"]) for key in list(tr_dict.keys())]
    tr_group_bias_true = sorted(
        tr_group_bias_true, key=lambda x: (x[1], x[2], x[0]), reverse=True
    )
    te_group_bias_true = [(key, te_dict[key]["bias"], te_dict[key]["sample_num"]) for key in list(te_dict.keys())]
    te_group_bias_true = sorted(
        te_group_bias_true, key=lambda x: (x[1], x[2], x[0]), reverse=True
    )
    dict_dataset_AX_sorted_tupleList = {"train": tr_group_bias_true, "test": te_group_bias_true}
    return dict_dataset_AX_sorted_tupleList


def bias_num_ratio(gen_A_sorted_listTuple, gen_X_sorted_listTuple, dataset_AX_sorted_tupleList, bias_ratio):
    # datatset
    (_, dataset_a_biasList, dataset_a_numList) = list(zip(*dataset_AX_sorted_tupleList))
    dataset_a_biasList = np.array(dataset_a_biasList)
    dataset_a_highbias_idx = np.greater(dataset_a_biasList, bias_ratio).astype(int)
    dataset_a_highbias_num = np.sum(dataset_a_highbias_idx)
    dataset_x_highbias_num = np.sum(dataset_a_highbias_idx * dataset_a_numList)
    dataset_a_highbias_ratio = dataset_a_highbias_num / len(dataset_a_numList)
    dataset_x_highbias_ratio = dataset_x_highbias_num / np.sum(dataset_a_numList)
    
    # Generated A
    (_, generated_a_pred_biasList, generated_a_gt_biasList) = gen_A_sorted_listTuple
    generated_a_highbias_idx = np.greater(generated_a_gt_biasList, bias_ratio).astype(int)
    generated_a_highbias_num = np.sum(generated_a_highbias_idx)
    generated_a_highbias_ratio = generated_a_highbias_num / len(generated_a_highbias_idx)
    generated_a_pred_biasMean = np.mean(generated_a_pred_biasList)
    generated_a_gt_biasMean = np.mean(generated_a_gt_biasList)
    
    # Generated X
    (_, generated_x_pred_biasList, generated_x_gt_biasList) = gen_X_sorted_listTuple
    generated_x_highbias_idx = np.greater(generated_x_gt_biasList, bias_ratio).astype(int)
    generated_x_highbias_num = np.sum(generated_x_highbias_idx)
    generated_x_highbias_ratio = generated_x_highbias_num / len(generated_x_highbias_idx)
    generated_x_pred_biasMean = np.mean(generated_x_pred_biasList)
    generated_x_gt_biasMean = np.mean(generated_x_gt_biasList)
    
    
    re_dict = {
        'dataset':{
            'a_bias_num':dataset_a_highbias_num,
            'a_bias_ratio': dataset_a_highbias_ratio,
            'x_bias_num': dataset_x_highbias_num,
            'x_bias_ratio': dataset_x_highbias_ratio,
            "a_num": len(dataset_a_numList),
            "x_num": np.sum(dataset_a_numList),
                   },
        'generation': {
            'a_bias_num':generated_a_highbias_num,
            'a_bias_ratio': generated_a_highbias_ratio,
            'x_bias_num': generated_x_highbias_num,
            'x_bias_ratio': generated_x_highbias_ratio,
            "a_num": len(generated_a_highbias_idx),
            "x_num": len(generated_x_highbias_idx),
            "x_pred_mean": generated_x_pred_biasMean,
            "x_gt_mean": generated_x_gt_biasMean,
            "a_pred_mean": generated_a_pred_biasMean,
            "a_gt_mean": generated_a_gt_biasMean,
        }
    }
    return re_dict



def rr(list_true, list_pred):
    """_summary_

    Args:
        list_true (_type_): ordered tuple (str, b_gt), ordered by b_gt, test set
        list_pred (_type_): ordered tuple (str, b_pred, b_gt), ordered by b_pred, generated A list
    """
    flag = False
    # (str_pred, b_pred, b_gt) = list(zip(*list_pred))
    (str_pred, b_pred, b_gt) = list_pred
    top_bias_a = list_true[0][0]
    if top_bias_a in str_pred:
        flag = True
        idx = (str_pred.index(top_bias_a) + 1)
        return flag, 1.0 / idx
    else:
        return flag, 0.0

def rr_at_k(list_true, list_pred, k):
    """_summary_

    Args:
        list_true (_type_): ordered tuple (str, b_gt), ordered by b_gt, test set
        list_pred (_type_): ordered tuple (str, b_pred, b_gt), ordered by b_pred, generated A list
    """
    
    (str_pred, b_pred, b_gt) = list_pred
    str_gt = list(zip(*list_true))[0]
    top_k_bias_a = str_gt[0:k]
    score = 0.0
    indicator = 0.0
    for gt_idx, top_bias_a in enumerate(top_k_bias_a):
        if top_bias_a in str_pred:
            idx = (str_pred.index(top_bias_a) + 1)
            dis = np.abs(idx - gt_idx -1)
            score += np.exp(-dis)
            indicator += 1.0
    return score, indicator / k
        
    
def dcg_at_k(list_true, list_pred, k=10, gains="exponential"):
    """_summary_

    Args:
        dict_true (_type_): keys are subgroup str, value is dict including bias info
        list_pred (_type_): ordered tuple (str, b_pred, b_gt), ordered by b_pred, generated A list
        k (int, optional): count for the top k bias group. Defaults to 10.
        gains (str, optional): gains calculation method, exp or linear. Defaults to "exponential".

    Returns: dcg@k
    """
    (data_a, data_bias, _) = list(zip(*list_true))
    pred_top_k_strs = list_pred[0][0:k]
    pred_top_k_b_true = np.array([data_bias[data_a.index(a)] for a in pred_top_k_strs])
    
    if gains == "exponential":
        gains = 2 ** pred_top_k_b_true - 1
    elif gains == "linear":
        gains = pred_top_k_b_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(pred_top_k_b_true)) + 2)
    return np.sum(gains / discounts)


def precision_at_k(list_pred, k, vae_eval_bias_thresh):
    b_gt = list_pred[2]
    y_true = np.greater(b_gt, vae_eval_bias_thresh).astype(int)
    n_relevant = np.sum(y_true[:k] == 1)
    return float(n_relevant) / k


def recall_at_k(list_true, list_pred, k, vae_eval_bias_thresh):
    '''
    list_true: dataset ground_truth bias value, a based
    list_pred: generated pred bias value, a based
    k = args.recall_k
    args.vae_eval_bias_thresh
    '''
    b_gt = list_pred[2]
    y_true_gen = np.greater(b_gt, vae_eval_bias_thresh).astype(int)
    b_gt_true = list(zip(*list_true))[1]
    y_true_all = np.greater(b_gt_true, vae_eval_bias_thresh).astype(int)
    n_relevant = np.sum(y_true_gen[:k] == 1) if len(y_true_gen) > k else np.sum(y_true_gen == 1)
    
    return float(n_relevant) / y_true_all.sum()
    
    
def avg_evaluations(evluation_dicts):
    merged_dict = {
        "new": collections.defaultdict(list),
        "train": collections.defaultdict(list),
        "test": collections.defaultdict(list),
    }
    avg_dict = {
        "dataset": collections.defaultdict(float),
        "new": collections.defaultdict(float),
        "train": collections.defaultdict(float),
        "test": collections.defaultdict(float),
    }
    methods = ["train", "test", "new"]
    for key, single_dict in evluation_dicts.items():
        avg_dict["dataset"] = single_dict["dataset"]
        for m in methods:
            for item_key, value in single_dict[m].items():
                merged_dict[m][item_key].append(value)
    
    for m in methods:
        for item_key, value in merged_dict[m].items():
            avg_dict[m][item_key] = np.array(value).mean()
    
    log_str = print_dict(avg_dict)
    return avg_dict, log_str


def compare_evaluations(baseline_dict, cur_dict, print_flag=False):
    # use train set evaluation metric to select best training spot
    # can change to other metrics
    eval_pairs = [("train", "gen_high_bias_A_num"), ("train", "dcg"), ("train", "recall")]
    eval_flags = np.zeros(len(eval_pairs))
    for idx, p in enumerate(eval_pairs):
        m = p[0]
        key = p[1]
        if "A_num" in key:
            if cur_dict[m][key] >= 0.9 * baseline_dict[m][key] :
                eval_flags[idx] = 1
        else:
            if cur_dict[m][key] >= baseline_dict[m][key] :
                eval_flags[idx] = 1
    if int(eval_flags.prod()) != 0:
        if print_flag:
            logger.info("Current is better, Baseline is replaced.")
        return True
    else:
        if print_flag:
            logger.info("Baseline is better.")
        return False
    


def result_dict_2_list(re_dict, epo, type):
    '''
    type = single, best, or avg
    '''
    l = [epo, type, 0]
    index = ["epo", "type", "betterThanBaseline"]
    methods = ["train", "test", "new", "dataset"]
    for m in methods:
        d_m = re_dict[m]
        d_m_keys = sorted(d_m.keys())
        for k in d_m_keys:
            content = d_m[k]
            index.append(f"{m}_{k}")
            l.append(content)
    return l, index

 
def log_evaluation_dict(evaluation_dict, args):
    args.rr_k_tr = max(1, int(args.rr_k_ratio * evaluation_dict['dataset']['train_high_bias_A_num'])) # celebA 0.01; toxic 0.02)
    args.precision_k_tr = int(evaluation_dict['dataset']['train_high_bias_A_ratio'] * 1.5 * evaluation_dict['train']['gen_X_num'])
    args.recall_k_tr = evaluation_dict['dataset']['train_high_bias_A_num'] # set a num larger than maximum that can be generated, which means cover all gen A
    
    args.rr_k_te = max(1, int(args.rr_k_ratio * evaluation_dict['dataset']['test_high_bias_A_num'])) # celebA 0.01; toxic 0.02)
    args.precision_k_te = int(evaluation_dict['dataset']['test_high_bias_A_ratio'] * 1.5 * evaluation_dict['test']['gen_X_num'])
    args.recall_k_te = evaluation_dict['dataset']['test_high_bias_A_num'] # set a num larger than maximum that can be generated, which means cover all gen A
    
    # Dataset Info (FIXED)
    d_tr_a_num = evaluation_dict["dataset"]["train_A_num"]
    d_te_a_num = evaluation_dict["dataset"]["test_A_num"]
    d_tr_bias_a_num = evaluation_dict["dataset"]["train_high_bias_A_num"]
    d_te_bias_a_num = evaluation_dict["dataset"]["test_high_bias_A_num"]
    d_tr_bias_a_ratio = evaluation_dict["dataset"]["train_high_bias_A_ratio"]
    d_te_bias_a_ratio = evaluation_dict["dataset"]["test_high_bias_A_ratio"]
    
    # Generation
    g_tr_a_num = evaluation_dict["train"]["gen_A_num"]
    g_te_a_num = evaluation_dict["test"]["gen_A_num"]
    g_new_a_num = evaluation_dict["new"]["gen_A_num"]
    g_all_a_num = g_tr_a_num + g_te_a_num + g_new_a_num
    g_tr_bias_a_num = evaluation_dict["train"]["gen_high_bias_A_num"]
    g_te_bias_a_num = evaluation_dict["test"]["gen_high_bias_A_num"]
    d_tr_bias_a_mean = evaluation_dict["dataset"]["train_A_gt_bias_mean"]
    d_te_bias_a_mean = evaluation_dict["dataset"]["test_A_gt_bias_mean"]
    g_tr_bias_a_gt_mean = evaluation_dict["train"]["gen_A_gt_bias_mean"]
    g_tr_bias_a_pred_mean = evaluation_dict["train"]["gen_A_pred_bias_mean"]
    g_te_bias_a_gt_mean = evaluation_dict["test"]["gen_A_gt_bias_mean"]
    g_te_bias_a_pred_mean = evaluation_dict["test"]["gen_A_pred_bias_mean"]
    g_tr_bias_a_ratio = evaluation_dict["train"]["gen_high_bias_A_ratio"]
    g_te_bias_a_ratio = evaluation_dict["test"]["gen_high_bias_A_ratio"]
    
    # Ranking
    rr_value_tr = evaluation_dict["train"]["rr"]
    rr_value_te = evaluation_dict["test"]["rr"]
    rr_ranked_score_tr = evaluation_dict["train"]["rr_score"]
    rr_ranked_score_te = evaluation_dict["test"]["rr_score"]
    rr_k_ind_tr = evaluation_dict["train"]["rr_ind"]
    rr_k_ind_te = evaluation_dict["test"]["rr_ind"]
    precision_tr = evaluation_dict["train"]["precision_X"]
    precision_te = evaluation_dict["test"]["precision_X"]
    recall_tr = evaluation_dict["train"]["recall"]
    recall_te = evaluation_dict["test"]["recall"]
    dcg_tr = evaluation_dict["train"]["dcg"] / args.dcg_k
    dcg_te = evaluation_dict["test"]["dcg"] / args.dcg_k

    logger.info("[Generation Eval] Result Summary:")
    logger.info("[Generation Eval] Dataset Info (FIXED):")
    logger.info(f"[Generation Eval] | A | All A num (tr/te): {d_tr_a_num}/{d_te_a_num}")
    logger.info(f"[Generation Eval] | A | Bias A num (bar={args.vae_eval_bias_thresh}) (tr/te): {d_tr_bias_a_num}/{d_te_bias_a_num}")
    logger.info(f"[Generation Eval] | A | Bias A ratio (tr/te): {d_tr_bias_a_ratio:.4f}/{d_te_bias_a_ratio:.4f}")
    logger.info("[Generation Eval] Number Analysis:")
    logger.info(f"[Generation Eval] | A | Gen A num (all/tr/te/new): {g_all_a_num}/{g_tr_a_num}/{g_te_a_num}/{g_new_a_num}")
    logger.info(f"[Generation Eval] | A | Gen Bias A num (bar={args.vae_eval_bias_thresh}) (tr/te): {g_tr_bias_a_num}/{g_te_bias_a_num}")
    logger.info(f"[Generation Eval] | A | Gen Bias A ratio (tr/te): {g_tr_bias_a_ratio:.4f}/{g_te_bias_a_ratio:.4f}")
    logger.info("[Generation Eval] Overall Bias Mean Value Analysis:")
    logger.info(f"[Generation Eval] | A | Dataset Bias   (tr/te): {d_tr_bias_a_mean:.4f}/{d_te_bias_a_mean:.4f}")
    logger.info(f"[Generation Eval] | A | Gen Bias (True) (tr/te):{g_tr_bias_a_gt_mean:.4f}/{g_te_bias_a_gt_mean:.4f}")
    logger.info(f"[Generation Eval] | A | Gen Bias (Pred) (tr/te):{g_tr_bias_a_pred_mean:.4f}/{g_te_bias_a_pred_mean:.4f}")
    logger.info("[Generation Eval] Ranking Analysis:")
    logger.info(
        f"[Generation Eval] | A | RR Value (tr/te): {rr_value_tr:.4f}/{rr_value_te:.4f}"
    )
    logger.info(
        f"[Generation Eval] | A | RR@{args.rr_k_tr}/{args.rr_k_te} score - ind (tr/te): {rr_ranked_score_tr:.4f} - {rr_k_ind_tr:.4f}/{rr_ranked_score_te} - {rr_k_ind_te:.4f}"
    )
    logger.info(
        f"[Generation Eval] | X | precision@{args.precision_k_tr}/{args.precision_k_te} (tr/te): {precision_tr:.4f}/{precision_te:.4f}"
    )
    logger.info(
        f"[Generation Eval] | A | recall@{args.recall_k_tr}/{args.recall_k_te} (tr/te): {recall_tr:.4f}/{recall_te:.4f}"
    )
    logger.info(
        f"[Generation Eval] | A | DCG@{args.dcg_k}({args.dcg_gains}) (tr/te): {dcg_tr:.4f}/{dcg_te:.4f}"
    )


def print_dict(d):
    methods = ["dataset", "train", "test", "new"]
    s = "==========================================\n"
    for m in methods:
        d_m = d[m]
        s += f"{m}:\n"
        for k, content in d_m.items():
            if isinstance(content, float):
                s += f"[{k}]: {content:.4f}\n"
            else:
                s += f"[{k}]: {content}\n"
    return s