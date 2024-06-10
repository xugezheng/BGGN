import argparse
import numpy as np
import os
import os.path as osp
import pickle
import torch
import time
import copy

# baseline - decision tree
from sklearn import tree
from sklearn.tree import  DecisionTreeRegressor

from baseline_utils import (
    calculate_bias_mean_from_dict,
    get_regression_tree_paths,
    recover_path,
    replace_negatives,
    filter_by_lf,
)
from utils import df_reduce, seed_set
from hypers import SENS_FEAS_ALL, CELEBA_2_FEAS_ID, TOXIC_25_FEAS_ID
from loggers import TxtLogger, set_logger
import logging
logger = logging.getLogger("intersectionalFair")


# check path
def check_relaxedSearchResult(relaxed_recovered_a, relaxed_recovered_a_pred, x, y_gt_scores, x_te, y_te_gt_scores, mode, logger):
    final_a_bias_dict_relaxed = {}
    final_a_sets_relaxed = {}
    # evaluation for relaxation based recovered a in train and also test set
    logger.info(f"[Mode] {mode} | Relaxation recovered A num: {len(relaxed_recovered_a)}")
    relaxed_a_in_x_tr = 0
    relaxed_recovered_a_in_tr = []
    relaxed_recovered_a_in_tr_pred = []
    relaxed_recovered_a_in_tr_gt = []
            
    relaxed_a_in_x_te = 0
    relaxed_recovered_a_in_te = []
    relaxed_recovered_a_in_te_pred = []
    relaxed_recovered_a_in_te_gt = []
    
    relaxed_a_new = 0
    relaxed_recovered_a_new = []
    relaxed_recovered_a_new_pred = []
 
    for idx_a, a in enumerate(relaxed_recovered_a):
        a_str = " ".join(list(map(str, a.tolist())))
        if np.any(np.all(a == x, axis=1)):
            for i, row in enumerate(x):
                if np.array_equal(row, a):
                    idx_dset = i
                    relaxed_recovered_a_in_tr.append(a)
                    relaxed_recovered_a_in_tr_pred.append(
                        float(relaxed_recovered_a_pred[idx_a])
                    )
                    relaxed_recovered_a_in_tr_gt.append(y_gt_scores[idx_dset])
                    break
            relaxed_a_in_x_tr += 1
        elif np.any(np.all(a == x_te, axis=1)):
            for i, row in enumerate(x_te):
                if np.array_equal(row, a):
                    idx_dset = i
                    relaxed_recovered_a_in_te.append(a)
                    relaxed_recovered_a_in_te_pred.append(
                        float(relaxed_recovered_a_pred[idx_a])
                    )
                    relaxed_recovered_a_in_te_gt.append(y_te_gt_scores[idx_dset])
                    break
            relaxed_a_in_x_te += 1
        else:
            relaxed_a_new += 1
            relaxed_recovered_a_new.append(a)
            relaxed_recovered_a_new_pred.append(float(relaxed_recovered_a_pred[idx_a]))
        
    # merge for discovered training data
    relaxed_tr_pred_bias_mean = np.mean(relaxed_recovered_a_in_tr_pred)
    relaxed_tr_gt_bias_mean = np.mean(relaxed_recovered_a_in_tr_gt)
    relaxed_tr_recovered_dict = {}
    relaxed_tr_a_all = set()
    for idx, a in enumerate(relaxed_recovered_a_in_tr):
        a_str = " ".join(list(map(str, a.tolist())))
        relaxed_tr_recovered_dict[a_str] = {
            "bias_pred": relaxed_recovered_a_in_tr_pred[idx],
            "bias_gt": relaxed_recovered_a_in_tr_gt[idx],
        }
        relaxed_tr_a_all.add(a_str)
    final_a_bias_dict_relaxed['train'] = copy.deepcopy(relaxed_tr_recovered_dict)
    final_a_sets_relaxed['train'] = relaxed_tr_a_all
    logger.info(f"Relaxed recovered A num in Training Data: {relaxed_a_in_x_tr} | Merged: {len(relaxed_tr_a_all)}")
    logger.info(f"Predicted Bias (mean): {relaxed_tr_pred_bias_mean:.4f}")
    logger.info(f"Ground Truth Bias (mean): {relaxed_tr_gt_bias_mean:.4f}")
            
    # merge for discovered test data
    relaxed_te_pred_bias_mean = np.mean(relaxed_recovered_a_in_te_pred)
    relaxed_te_gt_bias_mean = np.mean(relaxed_recovered_a_in_te_gt)
    relaxed_te_recovered_dict = {}
    relaxed_te_a_all = set()
    for idx, a in enumerate(relaxed_recovered_a_in_te):
        a_str = " ".join(list(map(str, a.tolist())))
        relaxed_te_recovered_dict[a_str] = {
            "bias_pred": relaxed_recovered_a_in_te_pred[idx],
            "bias_gt": relaxed_recovered_a_in_te_gt[idx],
        }
        relaxed_te_a_all.add(a_str)
    final_a_bias_dict_relaxed['test'] = copy.deepcopy(relaxed_te_recovered_dict)
    final_a_sets_relaxed['test'] = relaxed_te_a_all
    logger.info(f"Relaxed recovered A num in Test Data: {relaxed_a_in_x_te} | Merged: {len(relaxed_te_a_all)}")
    logger.info(f"Predicted Bias (mean): {relaxed_te_pred_bias_mean:.4f}")
    logger.info(f"Ground Truth Bias (mean): {relaxed_te_gt_bias_mean:.4f}")
            
    # merge for discovered new data
    relaxed_new_pred_bias_mean = np.mean(relaxed_recovered_a_new_pred)
    relaxed_new_recovered_dict = {}
    relaxed_new_a_all = set()
    for idx, a in enumerate(relaxed_recovered_a_new):
        a_str = " ".join(list(map(str, a.tolist())))
        relaxed_new_recovered_dict[a_str] = {
            "bias_pred": relaxed_recovered_a_new_pred[idx],
            "bias_gt": 0.0,
        }
        relaxed_new_a_all.add(a_str)
    final_a_bias_dict_relaxed['new'] = copy.deepcopy(relaxed_new_recovered_dict)
    final_a_sets_relaxed['new'] = relaxed_new_a_all
    logger.info(f"Relaxed recovered A num New: {relaxed_a_new}")
    logger.info(f"Predicted Bias (mean): {relaxed_new_pred_bias_mean:.4f}")
    logger.info(f"Ground Truth Bias (mean): Unknown")
    return final_a_bias_dict_relaxed, final_a_sets_relaxed


# evaluate path old version - slow
def evaluate_paths(paths, bias_paths, unbias_paths, x, y_gt_scores, x_te, y_te_gt_scores, args, logger):
    '''
    bias_path: 
        for the tree mode bias value estimator, this is the biased paths based on the decision predictive results;
        for the lf mode bias value estimator, this is the whole path set extracted from the decision tree. Then, in the evaluate_paths function, we will use l_f to re-calculate the bias values.
    '''
    logger.info(f"Before completing path, raw paths extracted from the decision tree.")
    logger.info(f"All Path num: {len(paths)}")
    logger.info(f"Unbias Path Num: {len(unbias_paths)}")
    logger.info(f"Bias Path Length: {len(bias_paths)}")
    path_dict = {"bias": bias_paths, "un_bias": unbias_paths}
    final_a_bias_dict = {}
    final_a_sets = {}
    final_a_bias_dict_relaxed = {'bias':{}, 'un_bias':{}}
    final_a_sets_relaxed = {'bias':{}, 'un_bias':{}}

    for mode in ["bias", "un_bias"]:
        print(f"Curretn Mode: {mode}")
        cur_path = path_dict[mode]
        complete_recovered_a = []
        complete_recovered_a_pred = []
        relaxed_recovered_a = []
        relaxed_recovered_a_pred = []
        for path in cur_path:
            recovered_a, path_len, final_value = recover_path(path, args)
            if -1 not in recovered_a:
                complete_recovered_a.append(recovered_a)
                complete_recovered_a_pred.append(final_value)
            elif mode == "bias":
                relax_a_index = np.where(recovered_a == -1)
                if len(relax_a_index[0]) <= args.relaxation_num:
                    combinations = []
                    replace_negatives(recovered_a, results=combinations)
                    for c in combinations: 
                        relaxed_recovered_a.append(np.array(c))
                        relaxed_recovered_a_pred.append(final_value)
            else:
                continue
            
        if mode == 'bias' and args.path_selection == 'lf':
            if args.relaxation_num > 0:
                relaxed_recovered_a, relaxed_recovered_a_pred = filter_by_lf(relaxed_recovered_a, relaxed_recovered_a_pred, args)
            complete_recovered_a, complete_recovered_a_pred = filter_by_lf(complete_recovered_a, complete_recovered_a_pred, args)
            
        
        logger.info(f"[Mode] {mode} | Completely recovered A num: {len(complete_recovered_a)}")
        a_in_x = 0
        complete_recovered_a_in_dset = []
        complete_recovered_a_in_dset_pred = []
        complete_recovered_a_in_dset_gt = []
        for idx_a, a in enumerate(complete_recovered_a):
            if a in x:
                for i, row in enumerate(x):
                    if np.array_equal(row, a):
                        idx_dset = i
                        complete_recovered_a_in_dset.append(a)
                        complete_recovered_a_in_dset_pred.append(
                            float(complete_recovered_a_pred[idx_a])
                        )
                        complete_recovered_a_in_dset_gt.append(y_gt_scores[idx_dset])
                        break
                a_in_x += 1
        pred_bias_mean = np.mean(complete_recovered_a_in_dset_pred)
        gt_bias_mean = np.mean(complete_recovered_a_in_dset_gt)

        logger.info(f"[Mode] {mode} | Completely recovered A num in Training Data: {a_in_x}")
        logger.info(f"[Mode] {mode} | Predicted Bias (mean): {pred_bias_mean:.4f}")
        logger.info(f"[Mode] {mode} | Ground Truth Bias (mean): {gt_bias_mean:.4f}")

        complete_recovered_dict = {}
        a_all = set()
        for idx, a in enumerate(complete_recovered_a_in_dset):
            a_str = " ".join(list(map(str, a.tolist())))
            complete_recovered_dict[a_str] = {
                "bias_pred": pred_bias_mean,
                "bias_gt": gt_bias_mean,
            }
            a_all.add(a_str)
        final_a_bias_dict[mode] = copy.deepcopy(complete_recovered_dict)
        final_a_sets[mode] = a_all
        
        if mode == 'bias':
            # ================================================================================================
            final_a_bias_dict_relaxed_, final_a_sets_relaxed_ = check_relaxedSearchResult(relaxed_recovered_a, relaxed_recovered_a_pred, x, y_gt_scores, x_te, y_te_gt_scores, mode, logger)
            final_a_bias_dict_relaxed[mode] = final_a_bias_dict_relaxed_
            final_a_sets_relaxed[mode] = final_a_sets_relaxed_
        
    return final_a_bias_dict, final_a_sets, final_a_bias_dict_relaxed, final_a_sets_relaxed



def build_tree(df_org, args, logger=None):
    # =========== Data Preperation ============
    df_tr_x = copy.deepcopy(df_org["train"])
    df_tr = df_reduce(df_tr_x, args.sens_attr_ids)
    x = df_tr.to_numpy()[:, args.sens_attr_ids]  # X
    y_gt_scores = df_tr.to_numpy()[:, -3]

    
    df_te_x = copy.deepcopy(df_org["test"])
    df_te = df_reduce(df_te_x, args.sens_attr_ids)
    x_te = df_te.to_numpy()[:, args.sens_attr_ids]  # X
    y_te_gt_scores = df_te.to_numpy()[:, -3]

    # =========== Build Model and Get Path============
    model = DecisionTreeRegressor(
        random_state=args.seed, splitter="random",
    )
    model.fit(x, y_gt_scores)
    
    # evaluate
    
    y_tr_pred = model.predict(x)
    y_te_pred = model.predict(x_te)
    mae_tr = np.mean(np.abs(y_tr_pred - y_gt_scores))
    mae_te = np.mean(np.abs(y_te_pred - y_te_gt_scores))
    logger.info(f"DT MEA Train: {mae_tr:.4f}")
    logger.info(f"DT MEA Test: {mae_te:.4f}")
    logger.info(f"DT Prediction Result Train: {y_tr_pred.mean():.4f}")
    logger.info(f"DT Prediction Result Test: {y_te_pred.mean():.4f}")
    
    
    T1 = time.time()
    paths = get_regression_tree_paths(model, args.feature_names) # all paths with target value, no partition for bias and unbias yet

    bias_paths = []
    unbias_paths = []
    for p in paths:
        if p[-1][1] > args.vae_eval_bias_thresh:
            bias_paths.append(p)
        elif p[-1][1] <= args.vae_eval_bias_thresh:
            unbias_paths.append(p)
        else:
            print(f"Error path {p}")
    if args.path_selection == 'lf':
            input_bias_path = paths
    elif args.path_selection == 'tree':
        input_bias_path = bias_paths
    else:
        print('Unknown Path selection method... Use tree found bias path for input bias path')
        input_bias_path = bias_paths
        
    complete_recovered_dict, a_all, relaxed_recovered_dict, relaxed_a_all = evaluate_paths(paths, input_bias_path, unbias_paths, x, y_gt_scores, x_te, y_te_gt_scores, args, logger)
    T2 = time.time()
    logger.info('DT Search Time (Once):%s ms' % ((T2 - T1)*1000))
    
    # =============== save tree model ===============
    text_representation = tree.export_text(model)

    cur_out_dir = osp.join(args.output_dir, "baseline", args.key_info)
    if not osp.exists(cur_out_dir):
        os.makedirs(cur_out_dir)
    with open(os.path.join(cur_out_dir, "decistion_tree_reg.log"), "w") as fout:
        fout.write(text_representation)
    # ===============================================
    
    
    return complete_recovered_dict, a_all, relaxed_recovered_dict, relaxed_a_all


def train_multi_seed_bias(args):
    # ===============================================================
    # log
    # ===============================================================
    log_dir, log_file = set_logger(args.key_info, args.seed, args.output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger, fh, sh = TxtLogger(filename=osp.abspath(osp.join(log_dir, log_file)))
    logger.info(args)

    # ===============================================================
    # data preperation
    # ===============================================================
    with open(
        f"{osp.abspath(osp.join(args.output_dir, args.f_model_df_regroup_dir))}", "rb"
    ) as handle:
        df_regroup = pickle.load(handle)

   
    # complete path
    a_all_bias = set()
    complete_recovered_dict_bias = {}
    a_all_unbias = set()
    complete_recovered_dict_unbias = {}
    
    # relaxed path
    relaxed_recovered_a_all_bias_tr = set()
    relaxed_recovered_dict_bias_tr = {}
    relaxed_recovered_a_all_bias_te = set()
    relaxed_recovered_dict_bias_te = {}
    relaxed_recovered_a_all_bias_new = set()
    relaxed_recovered_dict_bias_new = {}
    for s in range(args.rd_num):
        args.seed = s
        cur_complete_recovered_dict, cur_a_all, cur_relaxed_recovered_dict, cur_relaxed_a_all = build_tree(df_regroup, args, logger)
        # post process
        complete_recovered_dict_bias.update(cur_complete_recovered_dict["bias"])
        a_all_bias = a_all_bias | set(cur_a_all["bias"])
        complete_recovered_dict_unbias.update(cur_complete_recovered_dict["un_bias"])
        a_all_unbias = a_all_unbias | set(cur_a_all["un_bias"])
        # relaxation
        # merged with full path a
        relaxed_recovered_dict_bias_tr.update(cur_complete_recovered_dict["bias"])
        relaxed_recovered_a_all_bias_tr = relaxed_recovered_a_all_bias_tr | set(cur_a_all["bias"])
        # train
        relaxed_recovered_dict_bias_tr.update(cur_relaxed_recovered_dict["bias"]["train"])
        relaxed_recovered_a_all_bias_tr = relaxed_recovered_a_all_bias_tr | set(cur_relaxed_a_all["bias"]["train"])
        # test
        relaxed_recovered_dict_bias_te.update(cur_relaxed_recovered_dict["bias"]["test"])
        relaxed_recovered_a_all_bias_te = relaxed_recovered_a_all_bias_te | set(cur_relaxed_a_all["bias"]["test"])
        # new
        relaxed_recovered_dict_bias_new.update(cur_relaxed_recovered_dict["bias"]["new"])
        relaxed_recovered_a_all_bias_new = relaxed_recovered_a_all_bias_new | set(cur_relaxed_a_all["bias"]["new"])

    # Complete Recovered Result
    logger.info(f"Final Result of Baseline on [{args.dset}], by dTree [{args.dtree_type}] ")
    # quantity
    logger.info(f"Merged | Complete | ALL A Num: {len(a_all_unbias) + len(a_all_bias)}")
    logger.info(f"Merged | Complete | bias A Num: {len(complete_recovered_dict_bias)}")
    all_recovered_a_num = (len(a_all_unbias) + len(a_all_bias))
    if all_recovered_a_num == 0:
        all_recovered_a_num = 1
    logger.info(
        f"Merged | High bias A ratio: {(len(a_all_bias) / all_recovered_a_num):.4f}"
    )

    # bias mean value
    merged_all_recovered_a_dict = {
        **complete_recovered_dict_bias,
        **complete_recovered_dict_unbias,
    }  # recover here means generation
    gt_bias_mean, pred_bias_mean = calculate_bias_mean_from_dict(
        merged_all_recovered_a_dict
    )
    logger.info(f"Merged | ALL GT bias mean: {gt_bias_mean:.4f}")
    logger.info(f"Merged | ALL Pred bias mean: {pred_bias_mean:.4f}")
    # Relaxed Recovered Result
    logger.info(f"================= Relaxation ====================")
    logger.info(f"Merged | Relaxed (ReNum={args.relaxation_num}) | bias A Num (Train): {len(relaxed_recovered_dict_bias_tr)}")
    logger.info(f"Merged | Relaxed (ReNum={args.relaxation_num}) | bias A Num (Test): {len(relaxed_recovered_dict_bias_te)}")
    logger.info(f"Merged | Relaxed (ReNum={args.relaxation_num}) | bias A Num (New): {len(relaxed_recovered_dict_bias_new)}")

    # ===============================================================
    
    logger.removeHandler(fh)  # remove current file handler to avoid log file error
    logger.removeHandler(sh)
    
    return (len(relaxed_recovered_dict_bias_tr), len(relaxed_recovered_dict_bias_te), len(relaxed_recovered_dict_bias_new))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="subGroupBias_baseline")

    parser.add_argument("--dset", type=str, choices=["celebA", "toxic"], default="toxic")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./DATASOURCE/DATASET_NAME",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/EXPS_NAME",
    )
    parser.add_argument(
        "--target_id",
        default=2,
        type=int,
        help="prediction index, 0 for toxic, 2 (attractive) or 39 (young) for celebA",
    )
    # data load
    parser.add_argument(
        "--f_model_df_regroup_dir",
        type=str,
        default="data/data_frame_regroup_org.pickle",
        help="pre-prepare the data file into ./output/EXPS_NAME/data/data_frame_regroup_org.pickle"
    )
    # basic param
    parser.add_argument(
        "--output_dim", default=20, type=int, help="sensitive attributes number, 20 for celebA and 25 for toxic"
    )
    parser.add_argument(
        "--class_dim",
        default=2,
        type=int,
        help="per sensitive attribute's class number, 2 for both toxic and celebA datasets",
    )
    parser.add_argument("--seed", type=int, default=2024, help="seed for 1) pre-setting 2) decision tree construction")
    parser.add_argument("--rd_num", type=int, help="multi seed rounds", default=5)
    
    # decision tree
    parser.add_argument("--dtree_type", type=str, default="reg")
    parser.add_argument(
        "--vae_eval_bias_thresh",
        default=0.4,
        type=float,
        help="bias bar, default 0.3",
    )
    
    # relaxation
    parser.add_argument("--relaxation_num", type=int, default=0, help="0: search tree; 1 or more: relaxed search tree")

    #######################################
    # if path_selection == lf:
    #######################################
    # bias value predictor for filtering
    parser.add_argument(
        "--pred_type",
        type=str,
        choices=["tf_reg_cls", "tf_reg", "mlp_reg_cls", "mlp_reg"],
        default="mlp_reg_cls",
    )
    parser.add_argument(
        "--pred_hidden_dim",
        default=512,
        type=int,
    )
    parser.add_argument("--pred_fea_dim", default=256, type=int)
    ## classification task
    parser.add_argument("--pred_cls_interval", type=float, default=0.05)
    parser.add_argument("--pred_cls_num", type=int, default=10)
    parser.add_argument("--pred_reg_loss_coef", type=float, default=0.1)
    parser.add_argument("--pred_ce_loss_coef", type=float, default=1)
    parser.add_argument(
        "--reweight", type=str, choices=["inverse", "sqrt_inv", "none"], default="inverse"
    )
    parser.add_argument("--ft_batch_size", type=int, default=256)
    parser.add_argument("--path_selection", type=str, default='lf', choices=['tree', 'lf'])
    
    # log related
    parser.add_argument("--key_info", type=str, default=r"toxic_lf_test_seed2024", help="for log name and output files name")

    
    args = parser.parse_args()
    

    seed_set(args)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.dset == "celebA":
        # basic
        args.sens_attr_ids = CELEBA_2_FEAS_ID
        args.data_root = osp.abspath(r"./DATASOURCE/celebA")
        args.output_dir = osp.abspath(r"./output/celebA_attractive_fast_train")
        args.target_id = 2
        args.vae_eval_bias_thresh = 0.3
        args.output_dim = len(args.sens_attr_ids)
        
    elif args.dset == "toxic":
        # basic
        args.sens_attr_ids = TOXIC_25_FEAS_ID
        args.target_id = 0
        args.vae_eval_bias_thresh = 0.3
        args.data_root = osp.abspath(r"./DATASOURCE/toxic")
        args.output_dir = osp.abspath(r"./output/toxic_fast_train")
        args.output_dim = len(args.sens_attr_ids)
        
    print(f"sens attrs [{args.output_dim}]: {args.sens_attr_ids}")
    args.input_dim = int(args.output_dim * args.class_dim)
    args.dtree_type = "reg"
    args.rd_num = 5

    args.feature_names = [SENS_FEAS_ALL[args.dset][i] for i in args.sens_attr_ids]
    args.target_names = ["un_bias", "bias"]  # True/1.0 for bias, False/0.0 for un_bias
    
    org_key_info = args.key_info
    org_seed = args.seed
    
    bias_bar_dir = osp.join(args.output_dir, "baseline", "different_threshold")
    if not osp.exists(bias_bar_dir):
        os.makedirs(bias_bar_dir)
    bias_bar_txt = osp.join(bias_bar_dir, f'{org_key_info}_0508')
    with open(bias_bar_txt, 'a+') as f:
        f.write(f"Current Dataset is {args.dset}, path selection is {args.path_selection}, target id is {args.target_id} \n")

    
    bias_bar_result_dict = {"train":[], "test":[], 'new':[]}
    bias_bar = 0.3
    args.vae_eval_bias_thresh = bias_bar
    args.key_info =  f'baseline_{bias_bar}_final'
    args.seed = org_seed
    tr_num, te_num, new_num = train_multi_seed_bias(args)
    bias_bar_result_dict["train"].append(tr_num)
    bias_bar_result_dict["test"].append(te_num)
    bias_bar_result_dict["new"].append(new_num)
    with open(bias_bar_txt, 'a+') as f:
        f.write(f"Train/Observation Dataset: {bias_bar_result_dict['train']} \n")
        f.write(f"Test/Holdout Dataset: {bias_bar_result_dict['test']} \n")
        f.write(f"New Dataset: {bias_bar_result_dict['new']} \n")
    
