import torch
import numpy as np
from collections import defaultdict, Counter
import torch.nn as nn
import os
import os.path as osp
from sklearn.metrics import confusion_matrix

from utils import save_train_test_dict_2_df, df_2_dict
from evaluation_utils import avg_evaluations, compare_evaluations, bias_num_ratio


import logging
logger = logging.getLogger("intersectionalFair")


# ======================================================
# Evaluation
# ======================================================
def default_sens_attr_dict():
    return {"y_true": [], "y_scores": [], "losses": []}


# ======================================================
# evaluate the basic task model
# ======================================================
def evaluate_f_model(model, model_linear, loaders, print_all=False):
    """_summary_

    return {
        'tr': {
            "y_true": [],
            "y_scores": [],
            "losses": [],
            'sample_num': Int,
            "ap": float, # sens_attrs ap
            'ap_dataset': float, # overall dataset ap
            'loss_dataset': float, # overall dataset loss
        }
        'te': ...
        'val': ...
    }
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_result = {}
    all_new_record = {}
    for cur_dset in ["tr", "val", "te"]:
        dataloader = loaders[cur_dset]
        y_scores = []
        y_true = []
        losses = []
        sens_attrs = []
        sens_attrs_dict = defaultdict(default_sens_attr_dict)
        bias = []
        test_criterion = nn.BCELoss(reduction="none")

        with torch.no_grad():
            for i, (inputs, target, sens_attr) in enumerate(dataloader):
                inputs, target = inputs.to(device), target.float().to(device)

                if model_linear is None:
                    pred = model(inputs).detach()
                else:
                    feat = model(inputs)
                    pred = model_linear(feat).detach()
                loss = test_criterion(pred, target)

                losses.append(loss.data.cpu().numpy())
                y_scores.append(pred.data.cpu().numpy())
                y_true.append(target.data.cpu().numpy())
                sens_attrs.append(sens_attr.data.cpu().numpy())

        y_scores = np.concatenate(y_scores)
        y_true = np.concatenate(y_true)
        losses = np.concatenate(losses)
        sens_attrs = np.concatenate(sens_attrs)

        # fairness def: accuracy parity
        ap = np.sum(np.greater(y_scores, 0.5) == y_true) / float(len(y_scores))
        loss_avg = np.mean(losses)

        # per sens_attr acc/bias metric
        # dict {'sens_attr'}
        for idx, a in enumerate(sens_attrs):
            a_str = " ".join(list(map(str, a)))
            sens_attrs_dict[a_str]["y_true"].append(y_true[idx])
            sens_attrs_dict[a_str]["y_scores"].append(y_scores[idx])
            sens_attrs_dict[a_str]["losses"].append(losses[idx])

        for key, values in sens_attrs_dict.items():
            sens_attrs_dict[key]["ap"] = np.sum(
                np.greater(np.array(values["y_scores"]), 0.5)
                == np.array(values["y_true"])
            ) / float(len(values["y_scores"]))
            sens_attrs_dict[key]["sample_num"] = len(values["y_true"])
            sens_attrs_dict[key]["ap_dataset"] = ap
            sens_attrs_dict[key]["loss_dataset"] = loss_avg
           

        bias = losses

        all_result[cur_dset] = sens_attrs_dict
        all_new_record[cur_dset] = {
            "bias": bias,
            "y_scores": y_scores,
            "acc": ap,
            "loss_avg": loss_avg,
        }

        logger.info(f"Data Set: [{cur_dset}]")
        logger.info(f"Data Sample Num: [{idx+1}]")
        logger.info(f"Sens Attr   Num: [{len(list(sens_attrs_dict.keys()))}]")


    # dataset stats and compare
    if print_all:
        sens_attr_set_tr = set(all_result["tr"].keys())
        sens_attr_set_te = set(all_result["te"].keys())
        sens_attr_set_val = set(all_result["val"].keys())
        logger.info(
            f"[Sens Attrs]      All: {len(sens_attr_set_tr | sens_attr_set_te | sens_attr_set_val)}"
        )
        logger.info(f"[Sens Attrs]       Tr: {len(sens_attr_set_tr)}")
        logger.info(f"[Sens Attrs]       Te: {len(sens_attr_set_te)}")
        logger.info(f"[Sens Attrs]      Val: {len(sens_attr_set_val)}")
        logger.info(
            f"[Sens Attrs] Tr & Val: {len(sens_attr_set_tr & sens_attr_set_val)}"
        )
        logger.info(
            f"[Sens Attrs]  Tr & Te: {len(sens_attr_set_tr & sens_attr_set_te)}"
        )

    return all_result, all_new_record


# ======================================================
# evaluate bias predictor
# ======================================================
def evaluate_bias_predictor(pred, dataloader, bias_bar):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_scores = []
    y_true = []
    y_true_group = []
    start_test = True
    with torch.no_grad():
        for i, (_, __, a_one_hot, bias, bias_group, weight, cls_label) in enumerate(
            dataloader
        ):
            a_one_hot, bias = a_one_hot.float().to(device), bias.float().to(device)
            bias_pred, cls_pred, _ = pred(a_one_hot)
            bias_pred = torch.squeeze(bias_pred.detach())
            y_scores.append(bias_pred.data.cpu().numpy())
            y_true.append(bias.data.cpu().numpy())
            y_true_group.append(bias_group.data.cpu().numpy())

            # cls result
            if cls_pred is not None:
                if start_test:
                    all_cls_pred = cls_pred.float().cpu()
                    all_cls_label = cls_label.float()
                    start_test = False
                else:
                    all_cls_pred = torch.cat(
                        (all_cls_pred, cls_pred.float().cpu()), dim=0
                    )
                    all_cls_label = torch.cat(
                        (all_cls_label, cls_label.float().cpu()), dim=0
                    )

    # y_true : individual loss
    # y_true_group: group loss
    y_scores = np.concatenate(y_scores)
    y_true = np.concatenate(y_true)
    y_true_group = np.concatenate(y_true_group)
    mae_ind = np.abs(y_scores - y_true).mean()
    mae_group = np.abs(y_scores - y_true_group).mean()

    logger.info(
        f"[L_f(a)] evaluation | Pred Bias Max : {y_scores.max():.5f} | Real Ind Bias Max Num: {y_true.max():.5f} | Real Grp Bias Max Num: {y_true_group.max():.5f}"
    )
    logger.info(
        f"[L_f(a)] evaluation| Pred Bias Mean: {y_scores.mean():.5f} | Sample     Prediction: {y_scores[0:5]}"
    )
    logger.info(
        f"[L_f(a)] evaluation | Grp  Bias Mean: {y_true_group.mean():.5f} | Real  Grp Bias Values: {y_true_group[0:5]}"
    )

    # classification
    if cls_pred is not None:
        _, cls_predict = torch.max(all_cls_pred, 1)
        cls_acc = (
            100
            * torch.sum(cls_predict.squeeze().float() == all_cls_label).item()
            / float(all_cls_label.size()[0])
        )
        cls_c = Counter(all_cls_label.cpu().numpy())
        matrix = confusion_matrix(all_cls_label, torch.squeeze(cls_predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        logger.info(f"[L_f(a)] evaluation | CLS | Overall Acc: {cls_acc:.2f}%")
        logger.info(f"[L_f(a)] evaluation | CLS | Per Class Acc: \n{acc}")
        logger.info(f"[L_f(a)] evaluation | CLS | True Class Num: \n{cls_c}")

    # Take bias - unbias as a classification problem - loss bar
    # larger than the bar, bias subgroup; otherwise, unbiased
    y_true_01 = np.less(y_true_group, bias_bar)
    y_pred_01 = np.less(y_scores, bias_bar)
    bias_cls_acc = np.sum(y_pred_01 == y_true_01) / float(len(y_pred_01))

    return mae_ind, mae_group, bias_cls_acc


# ======================================================
# evaluate whole generation process
# ======================================================
from evaluation_utils import rr, rr_at_k, dcg_at_k, precision_at_k, recall_at_k, result_dict_2_list
from evaluation_utils import sampling_for_evaluation, regroup_dataset, filter_and_regroup_genA
from evaluation_utils import log_evaluation_dict

def evaluate_genModel(decoder, pred, df_regroup, args, a_bank=None, a_fea_bank=None, gp_bias_bank=None, for_draw=False):
    '''
    evaluation_dict = {
        "dataset": {
            "train_A_num": 
            "train_high_bias_A_num": 
            "train_high_bias_A_ratio": 
            "train_X_gt_bias_mean": 
            "test_A_num": 
            "test_high_bias_A_num": 
            "test_high_bias_A_ratio": 
            "test_X_gt_bias_mean": 
        },
        "new": {
            "gen_A_num": 
            "gen_X_pred_bias_mean": 
        },
        "train": {
            "gen_A_num": 
            "gen_X_num":
            "gen_high_bias_A_num": 
            "gen_high_bias_A_ratio": 
            "gen_X_gt_bias_mean": 
            "gen_X_pred_bias_mean": 
            "precision_X": 
            "recall": 
            "dcg": 
            "rr": 
            "rr_score": 
            "rr_ind": 
        },
        "test": {
            "gen_A_num": 
            "gen_X_num":
            "gen_high_bias_A_num": 
            "gen_high_bias_A_ratio": 
            "gen_X_gt_bias_mean": 
            "gen_X_pred_bias_mean": 
            "precision_X": 
            "recall": 
            "dcg": 
            "rr": 
            "rr_score": 
            "rr_ind": 
        },
    }
'''
    a_generated_all, b_all, sampling_time = sampling_for_evaluation(decoder, pred, args, a_bank, a_fea_bank, gp_bias_bank)
    a_dicts = df_2_dict(df_regroup, args.target_id, args.sens_attr_ids)
    
    dict_dataset_AX_sorted_tupleList = regroup_dataset(a_dicts)
    dict_gen_sorted_listTuple = filter_and_regroup_genA(a_generated_all, b_all, a_dicts)
    
    evaluation_dict = {}
    draw_dict = {}
    # dataset
    evaluation_dict["dataset"] = evaluate_dataset(dict_dataset_AX_sorted_tupleList, args.vae_eval_bias_thresh)
    # train/test
    for mode in ["train", "test"]:
        gen_A_sorted_listTuple = dict_gen_sorted_listTuple[f"{mode}_a_ordered"]
        gen_X_sorted_listTuple = dict_gen_sorted_listTuple[f"{mode}_x_ordered"]
        dataset_AX_sorted_tupleList = dict_dataset_AX_sorted_tupleList[mode]
        evaluation_re, draw_re = evaluate_seen(gen_A_sorted_listTuple, gen_X_sorted_listTuple, dataset_AX_sorted_tupleList, args, mode=mode, print_flag=for_draw)
        evaluation_dict[mode] = evaluation_re
        draw_dict[mode] = draw_re
    # new
    evaluation_dict["new"] = evaluate_unseen(dict_gen_sorted_listTuple["new_a_ordered"], dict_gen_sorted_listTuple["new_x_ordered"])
    
    if for_draw:
        return draw_dict
    else:
        log_evaluation_dict(evaluation_dict, args)
        return dict_dataset_AX_sorted_tupleList, dict_gen_sorted_listTuple, evaluation_dict


def evaluate_genModel_multi_round(
    k, decoder, pred, df, args, a_bank=None, a_fea_bank=None, gp_bias_bank=None, epo=' '
):
    all_dict = {}
    df_list = []
    df_index = None
    best_evaluation_dict = None
    
    if args.save_all_gen_result:
        df_all_dir = osp.join(args.output_dir, "df", args.key_info, "all")
        if not osp.exists(df_all_dir):
            os.makedirs(df_all_dir)
    for i in range(k):
        org_all_dict, gen_all_dict, evaluation_dict = evaluate_genModel(
                decoder, pred, df, args, a_bank, a_fea_bank, gp_bias_bank
            )
        if args.save_all_gen_result:
            # save all sampling results into csv file, to evaluate the generated/discovered a
            save_train_test_dict_2_df(gen_all_dict, df_all_dir, f"{epo}_{i}")
        
        
        all_dict[i] = evaluation_dict
        # prepare all result into line in the final csv file
        i_df_list, i_index = result_dict_2_list(evaluation_dict, f"{epo}_{i}", "single")
        
        if i == 0:
            best_evaluation_dict = evaluation_dict
            df_index = i_index
        else:
            assert df_index == i_index # to make sure each line with the same item order
            replace = compare_evaluations(best_evaluation_dict, evaluation_dict)
            if replace:
                i_df_list[2] = 1
                best_evaluation_dict = evaluation_dict
        df_list.append(i_df_list)
    merged_dict, log_str = avg_evaluations(all_dict)
    best_df_list, _ = result_dict_2_list(best_evaluation_dict, f"{epo}_best", "best")
    df_list.append(best_df_list)
    avg_df_list, _ = result_dict_2_list(merged_dict, f"{epo}_avg", "avg")
    df_list.append(avg_df_list)
    return org_all_dict, best_evaluation_dict, merged_dict, df_list, df_index


def evaluate_seen(gen_A_sorted_listTuple, gen_X_sorted_listTuple, dataset_AX_sorted_tupleList, args, mode="unknown", print_flag=False):
    
    bias_num_ratio_dict = bias_num_ratio(gen_A_sorted_listTuple, gen_X_sorted_listTuple, dataset_AX_sorted_tupleList, args.vae_eval_bias_thresh)
    ## k values for different metric
    args.rr_k = int(args.rr_k_ratio * bias_num_ratio_dict['dataset']['a_bias_num']) # celebA 0.01; toxic 0.02)
    if args.rr_k == 0:
        args.rr_k = 1
    args.precision_k = int(bias_num_ratio_dict['dataset']['a_bias_ratio'] * 1.5 * bias_num_ratio_dict['generation']['x_num'])
    args.recall_k = bias_num_ratio_dict['dataset']['a_bias_num'] # set a num larger than maximum that can be generated, which means cover all gen A

    
    ## rr
    rr_flag, rr_value = rr(dataset_AX_sorted_tupleList, gen_A_sorted_listTuple)
    rr_ranked_score, rr_k_ind = rr_at_k(dataset_AX_sorted_tupleList, gen_A_sorted_listTuple, args.rr_k)
    ## precision
    precision = precision_at_k(gen_X_sorted_listTuple, args.precision_k, args.vae_eval_bias_thresh)
    ## recall
    recall = recall_at_k(dataset_AX_sorted_tupleList, gen_A_sorted_listTuple, args.recall_k ,args.vae_eval_bias_thresh)
    # DCG
    dcg = dcg_at_k(dataset_AX_sorted_tupleList, gen_A_sorted_listTuple, k=args.dcg_k, gains=args.dcg_gains)
    
    draw_re = {
        "Bias Ratio": bias_num_ratio_dict['generation']['a_bias_ratio'],
        "Precision": precision,
        "Recall": recall,
        "Avg DCG": (dcg/args.dcg_k),
        "RR k Score": rr_ranked_score,
        "Generation Bias Number": bias_num_ratio_dict['generation']['a_bias_num'],
        "Dataset Bias Number": bias_num_ratio_dict['dataset']['a_bias_num'],
        "Bias Bar": args.vae_eval_bias_thresh,
        # "RR k Ind Score": rr_k_ind_tr,
        # "RR": rr_value_tr
    }
    
    evaluation_re = {
            "gen_A_num": len(gen_A_sorted_listTuple[0]),
            "gen_X_num": len(gen_X_sorted_listTuple[0]),
            "gen_high_bias_A_num": bias_num_ratio_dict["generation"]["a_bias_num"],
            "gen_high_bias_A_ratio":  bias_num_ratio_dict["generation"]["a_bias_ratio"],
            "gen_A_gt_bias_mean": bias_num_ratio_dict["generation"]["a_gt_mean"],
            "gen_A_pred_bias_mean": bias_num_ratio_dict["generation"]["a_pred_mean"],
            "precision_X": precision,
            "recall": recall,
            "dcg": dcg,
            "rr": rr_value,
            "rr_score": rr_ranked_score,
            "rr_ind": rr_k_ind,
        }
    
    if print_flag:
        print(f"Current Evaluation Dataset: [{mode}].")
        print(f"Bias Ratio: {args.vae_eval_bias_thresh}")
        print(bias_num_ratio_dict['generation'])
        print(f"Dataset | A | High Bias Num - Ratio: {bias_num_ratio_dict['dataset']['a_bias_num']} - {bias_num_ratio_dict['dataset']['a_bias_ratio']:.4f}")
        print(f"Dataset | X | High Bias Num - Ratio: {bias_num_ratio_dict['dataset']['x_bias_num']} - {bias_num_ratio_dict['dataset']['x_bias_ratio']:.4f}")
        print(f"Generation | A | High Bias Num - Ratio: {bias_num_ratio_dict['generation']['a_bias_num']} - {bias_num_ratio_dict['generation']['a_bias_ratio']:.4f}")
        print(f"Generation | X | High Bias Num - Ratio: {bias_num_ratio_dict['generation']['x_bias_num']} - {bias_num_ratio_dict['generation']['x_bias_ratio']:.4f}")
        print(f"Generation | A | RR Generated?-Value: {rr_flag} - {rr_value:.4f}")
        print(
            f"Generation | A | RR@{args.rr_k} score - ind: {rr_ranked_score:.4f} - {rr_k_ind:.4f}"
        )
        print(
            f"Generation | X | precision@{args.precision_k} : {precision:.4f}"
        )
        print(f"Generation | A | recall@{args.recall_k}: {recall:.4f}")
        print(f"Generation | A | Avg DCG@{args.dcg_k}=dcg/dcg_k({args.dcg_gains}): {dcg:.4f} / {args.dcg_k} = {(dcg/args.dcg_k):.4f}")
    
    return evaluation_re, draw_re


def evaluate_unseen(gen_A_sorted_listTuple, gen_X_sorted_listTuple):
    new_dict = {
        "gen_A_num": len(gen_A_sorted_listTuple[0]),
        "gen_X_pred_bias_mean": np.mean(gen_X_sorted_listTuple[1])
    }
    return new_dict


def evaluate_dataset(dict_dataset_AX_sorted_tupleList, bias_ratio):
    dataset_info_dict = {}
    for mode in ["train", "test"]:
        dataset_AX_sorted_tupleList = dict_dataset_AX_sorted_tupleList[mode]
        (_, dataset_a_biasList, dataset_a_numList) = list(zip(*dataset_AX_sorted_tupleList))
        dataset_a_biasList = np.array(dataset_a_biasList)
        dataset_a_highbias_idx = np.greater(dataset_a_biasList, bias_ratio).astype(int)
        dataset_a_highbias_num = np.sum(dataset_a_highbias_idx)
        dataset_a_highbias_ratio = dataset_a_highbias_num / len(dataset_a_numList)
        dataset_x_gt_mean = np.sum(dataset_a_biasList * np.array(dataset_a_numList)) / np.sum(dataset_a_numList) # x based lower
        dataset_a_gt_mean = np.mean(dataset_a_biasList) # a based higher
        
        dataset_info_dict[f"{mode}_A_num"] = len(dataset_a_biasList)
        dataset_info_dict[f"{mode}_high_bias_A_num"] = dataset_a_highbias_num
        dataset_info_dict[f"{mode}_high_bias_A_ratio"] = dataset_a_highbias_ratio
        dataset_info_dict[f"{mode}_X_gt_bias_mean"] = dataset_x_gt_mean
        dataset_info_dict[f"{mode}_A_gt_bias_mean"] = dataset_a_gt_mean
    '''
    "dataset": {
            "train_A_num": ,
            "train_high_bias_A_num": ,
            "train_high_bias_A_ratio": ,
            "train_X_gt_bias_mean": ,
            "test_A_num":,
            "test_high_bias_A_num": ,
            "test_high_bias_A_ratio": ,
            "test_X_gt_bias_mean": ,
        }
    '''
    return dataset_info_dict
    
