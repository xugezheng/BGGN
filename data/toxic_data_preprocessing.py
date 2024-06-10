import pandas as pd
import numpy as np
import os
import os.path as osp
import h5py
import pickle
import logging
import copy

logger = logging.getLogger("intersectionalFair")


TOXIC_SENS_FEAS = [
    "toxicity",
    "male",
    "female",
    "transgender",
    "other_gender",
    "na_gender",  # 1-5；5
    "heterosexual",
    "homosexual_gay_or_lesbian",
    "bisexual",
    "other_sexual_orientation",
    "LGBTQ",
    "na_orientation",  # 6-11；6
    "christian",
    "jewish",
    "muslim",
    "hindu",
    "buddhist",
    "atheist",
    "other_religion",
    "other_religions",
    "na_religion",  # 12-20；9
    "black",
    "white",
    "asian",
    "latino",
    "other_race_or_ethnicity",
    "asian_latino_etc",
    "identity_any",
    "na_race",  # 21-28；8
    "physical_disability",
    "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness",
    "other_disability",
    "disability_any",
    "na_disability",  # 29-34； 6
]


def generate_multi_attrs_toxic(bar=0.4):
    h5_file = osp.abspath(
        osp.join(
            osp.dirname(__file__),
            "../DATASOURCE/toxic/toxic_from_wilds_all.h5",
        )
    )
    csv_file = osp.abspath(
        osp.join(
            osp.dirname(__file__),
            "../DATASOURCE/toxic/all_data_with_identities.csv",
        )
    )
    df_split_file_path = osp.join(osp.dirname(__file__),"../DATASOURCE/toxic/df_split")
    if not osp.exists(df_split_file_path):
        os.makedirs(df_split_file_path)

    x_y_split_file_path = osp.join(osp.dirname(__file__),"../DATASOURCE/toxic/xy_split")
    if not osp.exists(x_y_split_file_path):
        os.makedirs(x_y_split_file_path)
    
    all_df = pd.read_csv(csv_file)
    all_h5 = h5py.File(h5_file)
    X = np.array(all_h5["X"])
    Y = np.array(all_h5["Y"])
    X_all = []
    Y_all = []

    print("TOXIC dataset Preprocessing ...")
    print(f"Extracting Features: {TOXIC_SENS_FEAS}")

    df = {}
    print(f"Original DataSet Length: {len(all_df)}")
    for dset in ["train", "val", "test"]:
        dset_idx = np.flatnonzero(all_df["split"] == dset)
        cur_df = all_df.loc[dset_idx]
        cur_x = X[dset_idx]
        cur_y = Y[dset_idx]
        cur_y_01 = np.where(cur_y > bar, 1, 0)
        cur_df["toxicity"] = np.where(cur_df["toxicity"] > bar, 1, 0)
        assert (cur_y_01 == cur_df["toxicity"]).all()
        for fea in TOXIC_SENS_FEAS[1:]:
            cur_df[fea] = np.where(cur_df[fea] > 0.1, 1, 0)
        df[dset] = copy.deepcopy(cur_df[TOXIC_SENS_FEAS])
        print(f"Dset: {dset}, Length: {len(cur_df)}")
        print(f"X feature dim: {cur_x[0].shape}")
        print(f"y label sample: {cur_y_01[0:5]}")
        print(df[dset].head())
        X_all.append(copy.deepcopy(cur_x))
        Y_all.append(copy.deepcopy(cur_y_01))

        
        cur_df_path = f"{df_split_file_path}/{dset}.csv"
        print(cur_df_path)
        df[dset].to_csv(cur_df_path)

        out_file_name = f"{x_y_split_file_path}/{dset}_xy.h5"
        with h5py.File(out_file_name, "w") as f:
            f.create_dataset("X", data=cur_x)
            f.create_dataset("Y", data=cur_y_01)

    with open(osp.join(osp.dirname(__file__),"../DATASOURCE/toxic/data_frame.pickle"), "wb") as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_a = np.concatenate(X_all, axis=0)
    Y_a = np.concatenate(Y_all, axis=0)
    print(X_a.shape)
    print(Y_a.shape)
    
    
    out_file_name = f"{x_y_split_file_path}/all_xy.h5"
    with h5py.File(out_file_name, "w") as f:
        f.create_dataset("X", data=X_a)
        f.create_dataset("Y", data=Y_a)
        
    return

def f_model_df_merge_from_seed(data_dir="../DATASOURCE/toxic/Models/f_model_train", debug=False):
    df_list = []
    seed_list = [2021, 2022, 2023, 2024, 2025]
    for s in seed_list:
        seed_df_path = osp.join(data_dir, str(s), "data_frame_with_yscores.pickle")
        df_s = pd.read_pickle(seed_df_path)
        df_list.append(df_s)

    # only for testing equal length
    for w in ["toxicity", "male"]:
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
    # generate_multi_attrs_toxic(bar=0.4)
    # f_model_df_merge_from_seed(debug=False)
    pass