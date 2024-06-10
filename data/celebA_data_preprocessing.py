import os
import os.path as osp
import zipfile
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import shutil
import pickle
import copy


def celebA_raw_to_dfpickle(data_dir="../DATASOURCE/celebA"):
    labels_path = f"{data_dir}/Anno/list_attr_celeba.txt"
    image_path = f"{data_dir}/Img/img_align_celeba/"
    split_path = f"{data_dir}/Eval/list_eval_partition.txt"

    labels_df = pd.read_csv(labels_path, delim_whitespace=True, skiprows=1)
    labels_df[labels_df == -1] = 0

    # generate train/val/test
    files = glob(image_path + "*.jpg")

    split_file = open(split_path, "r")
    lines = split_file.readlines()

    if not os.path.exists(f"{data_dir}/tmp/"):
        os.mkdir(f"{data_dir}/tmp/")
    for i in ["train", "val", "test"]:
        if not os.path.exists(os.path.join(f"{data_dir}/tmp/", i)):
            os.mkdir(os.path.join(f"{data_dir}/tmp/", i))

    train_file_names = []
    train_dict = {}
    valid_file_names = []
    valid_dict = {}
    test_file_names = []
    test_dict = {}
    for i in tqdm(range(len(lines))):
        file_name, sp = lines[i].split()
        sp = sp.split("\n")[0]
        if sp == "0":
            labels = np.array(labels_df.loc[file_name])
            assert len(labels) == 40
            train_dict[file_name] = labels
            train_file_names.append(file_name)
            source_path = image_path + file_name
            shutil.copy2(source_path, os.path.join(f"{data_dir}/tmp/train", file_name))
        elif sp == "1":
            labels = np.array(labels_df.loc[file_name])
            assert len(labels) == 40
            valid_dict[file_name] = labels
            valid_file_names.append(file_name)
            source_path = image_path + file_name
            shutil.copy2(source_path, os.path.join(f"{data_dir}/tmp/val", file_name))
        elif sp == "2":
            labels = np.array(labels_df.loc[file_name])
            assert len(labels) == 40
            test_dict[file_name] = labels
            test_file_names.append(file_name)
            source_path = image_path + file_name
            shutil.copy2(source_path, os.path.join(f"{data_dir}/tmp/test", file_name))
        else:
            print(f"Unknown img {file_name} with sp {sp}")

    train_df = pd.DataFrame(train_dict.values())
    train_df.index = train_file_names
    train_df.columns = labels_df.columns
    print(train_df.head)

    valid_df = pd.DataFrame(valid_dict.values())
    valid_df.index = valid_file_names
    valid_df.columns = labels_df.columns

    test_df = pd.DataFrame(test_dict.values())
    test_df.index = test_file_names
    test_df.columns = labels_df.columns

    df = {}
    df["train"] = train_df
    df["val"] = valid_df
    df["test"] = test_df
    with open(f"{data_dir}/Anno/data_frame.pickle", "wb") as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("data frame saved")


def f_model_df_merge_from_seed(data_dir="../DATASOURCE/celebA/Models/f_model_train", debug=False):
    df_list = []
    seed_list = [2021, 2022, 2023, 2024, 2025]
    for s in seed_list:
        seed_df_path = osp.join(data_dir, str(s), "data_frame_with_yscores.pickle")
        df_s = pd.read_pickle(seed_df_path)
        df_list.append(df_s)

    # only for testing equal length
    for w in ["Young", "Attractive"]:
        assert (df_list[0]["train"][w] == df_list[1]["train"][w]).all()
        assert (df_list[0]["test"][w] == df_list[1]["test"][w]).all()
        assert (df_list[0]["val"][w] == df_list[1]["val"][w]).all()

    df_merged = copy.deepcopy(df_list[0])
    if debug:
        df_debug = copy.deepcopy(df_list[0])

    for dset in ["train", "test", "val"]:
        new_bias_group = np.zeros_like(df_merged[dset].to_numpy()[:, -3])
        new_bias = np.zeros_like(df_merged[dset].to_numpy()[:, -2])
        new_y_score = np.zeros_like(df_merged[dset].to_numpy()[:, -1])
        for s_idx, s in enumerate(seed_list):
            new_bias_group += df_list[s_idx][dset].to_numpy()[:, -3]
            new_bias += df_list[s_idx][dset].to_numpy()[:, -2]
            new_y_score += df_list[s_idx][dset].to_numpy()[:, -1]
            if debug:
                df_debug[dset][f"bias_group_seed_{str(s)}"] = df_list[s_idx][dset].to_numpy()[:, -3]
                df_debug[dset][f"bias_seed_{str(s)}"] = df_list[s_idx][dset].to_numpy()[:, -2]
                df_debug[dset][f"y_score_seed_{str(s)}"] = df_list[s_idx][dset].to_numpy()[:, -1]
        df_merged[dset]["bias_group"] = new_bias_group / len(seed_list)
        df_merged[dset]["bias"] = new_bias / len(seed_list)
        df_merged[dset]["y_score"] = new_y_score / len(seed_list)
        
        if debug:
            df_debug[dset][f"bias_group_mean"] = df_merged[dset]["bias_group"]
            df_debug[dset][f"bias_seed_mean"] = df_merged[dset]["bias"]
            df_debug[dset][f"y_score_seed_mean"] = df_merged[dset]["y_score"]
            df_debug[dset].to_csv(osp.join(data_dir, f"debug_{dset}_all.csv"))

    with open(f"{data_dir}/all_data_frame_with_yscores.pickle", "wb") as handle:
        pickle.dump(df_merged, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    celebA_raw_to_dfpickle()
    f_model_df_merge_from_seed(debug=False)
