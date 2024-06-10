import numpy as np
from sklearn.tree import _tree
import logging

######################################################
# Decision Tree V.S. Bias Value Predictor 
from network import pred_init
from loaders import a_bias_loaders_load
import os.path as osp
import pickle
import torch
######################################################

logger = logging.getLogger("intersectionalFair")
TARGET_TO_VALUE = {"bias": 1.0, "un_bias": 0.0}


    
def calculate_bias_mean_from_dict(d):
    bias_pred = []
    bias_gt = []
    for key, value in d.items():
        bias_pred.append(value["bias_pred"])
        bias_gt.append(value["bias_gt"])
    return np.mean(bias_gt), np.mean(bias_pred)


# for regression tree
def get_regression_tree_paths(tree, feature_names):
    """
    Extracts all paths from a decision tree regressor along with the predicted value of each leaf.
    Each path is represented as a list of tuples (feature, threshold, direction), ending with the predicted value.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # Left child
            recurse(tree_.children_left[node], path + [(name, threshold, "left")])
            # Right child
            recurse(tree_.children_right[node], path + [(name, threshold, "right")])
        else:
            # Leaf - append the predicted value
            predicted_value = np.mean(tree_.value[node])
            paths.append(path + [("target", predicted_value)])

    recurse(0, [])
    return paths


# recover path
def recover_path(path, args):
    possible_a = -1 * np.ones(len(args.sens_attr_ids))
    path_len = 0
    for step in path:
        if step[0] != "target":
            cur_a_name = step[0]
            cur_a_idx = args.feature_names.index(cur_a_name)
            cur_a_value = 0 if step[-1] == "left" else 1
            possible_a[cur_a_idx] = cur_a_value
            path_len += 1
        else:
            final_value = step[-1]
    assert path_len == (len(path) - 1)
    return possible_a, path_len, final_value # np.array, int, float


######################################################
#### Relaxed Tree
######################################################
# enumeration to transfer incomplete tree into complete 
def replace_negatives(arr, index=0, current=None, results=None):
    if results is None:
        results = []
    if current is None:
        current = arr.copy()
    
    # Base case: If index is beyond the last element, add the current combination to the results.
    if index == len(arr):
        results.append(current.copy())
        return
    
    # If the current element is -1, branch into two possibilities: replace with 0 and with 1.
    if arr[index] == -1:
        for replacement in (0, 1):
            current[index] = replacement
            replace_negatives(arr, index + 1, current, results)
    else:
        replace_negatives(arr, index + 1, current, results)
        
        

######################################################
#++++++ Decision Tree V.S. Bias Value Predictor +++++
######################################################
def convert_a_to_lfinput(relaxed_recovered_a):
    relaxed_recovered_a_lf = []
    for a in relaxed_recovered_a:
        a_one_hot = []
        for i in a:
            if int(i) == 0:
                a_one_hot.append(1)
                a_one_hot.append(0)
            elif int(i) == 1:
                a_one_hot.append(0)
                a_one_hot.append(1)
            else:
                print("unknown a")
        a_one_hot = np.array(a_one_hot)
        relaxed_recovered_a_lf.append(a_one_hot)
    relaxed_recovered_a_lf_numpy = np.stack(relaxed_recovered_a_lf)
    print(relaxed_recovered_a_lf_numpy.shape)
    return torch.tensor(relaxed_recovered_a_lf_numpy, dtype=torch.float)

def calculate_pred_mean(pred, dataloader):
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


    # y_true : individual loss
    # y_true_group: group loss
    y_scores = np.concatenate(y_scores)
    y_true = np.concatenate(y_true)
    y_true_group = np.concatenate(y_true_group)

    print(
        f"Bias Pred | Pred Bias Mean: {y_scores.mean():.5f} | Sample     Prediction: {y_scores[0:5]}"
    )
    print(
        f"Bias True | Grp  Bias Mean: {y_true_group.mean():.5f} | Real  Grp Bias Values: {y_true_group[0:5]}"
    )

def filter_by_lf(relaxed_recovered_a, relaxed_recovered_a_pred, args):
    pred = pred_init(args).to(args.device)
    pred.load_state_dict(
            torch.load(
                osp.join(
                    args.output_dir,
                    "bias_pred_model",
                    f"{args.pred_type}_bias_pred.pt",
                ),
                map_location=args.device,
            )
        )
    with open(
            f"{osp.join(args.output_dir, 'data', 'data_frame_regroup_org.pickle')}", "rb"
        ) as handle:
        df_regroup = pickle.load(handle)
    
    loaders_bias = a_bias_loaders_load(df_regroup, args)
    
    # calculate mean
    calculate_pred_mean(pred, loaders_bias["te_reduced"])
    
    # pred on all recovered a:
    relaxed_recovered_a_lf_tensor = convert_a_to_lfinput(relaxed_recovered_a)
    relaxed_recovered_a_lf_tensor.float().to(args.device)
    bias_pred, cls_pred, _ = pred(relaxed_recovered_a_lf_tensor)
    bias_pred_numpy = bias_pred.data.cpu().numpy()
    high_bias_index = np.nonzero(bias_pred_numpy > args.vae_eval_bias_thresh)[0]
    
    relaxed_recovered_a_numpy = np.stack(relaxed_recovered_a)
    
    return relaxed_recovered_a_numpy[high_bias_index], bias_pred_numpy[high_bias_index]
        
        
        
