import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score
from torch._C import dtype

from torchvision import transforms
from torch.utils.data import Dataset

# imbalanced weight
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import h5py

import logging
logger = logging.getLogger("intersectionalFair")


class TOXIC(Dataset):
    def __init__(
        self, dataframe, h5_dir, sens_attr_ids
    ):
        self.dataframe = dataframe
        all_h5 = h5py.File(h5_dir, 'r')
        self.X = np.array(all_h5["X"])
        self.Y = np.array(all_h5["Y"])
        self.file_names = dataframe.index
        self.labels = dataframe.to_numpy()
        self.sens_attr_ids = sens_attr_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        return self.X[index], self.Y[index], label[self.sens_attr_ids]


class TOXIC_bias(Dataset):
    def __init__(
        self,
        dataframe,
        sens_attr_ids,
        pred_cls_interval=0.1,
        cls_num=10,
        args=None
    ):
        self.dataframe = dataframe
        self.labels = dataframe.to_numpy()[:, :-3]
        self.biases = dataframe.to_numpy()[:, -2]
        self.bias_group = dataframe.to_numpy()[:, -3]
        self.sens_attr_ids = sens_attr_ids
        # reweight
        # self.weights = self._prepare_weights(reweight="none", lds=False)
        self.weights = self._prepare_weights(args)
        self.cls_labels = self._prepare_cls_label(pred_cls_interval, cls_num)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        bias = self.biases[index]
        bias_group = self.bias_group[index]
        weight = self.weights[index]
        cls_label = self.cls_labels[index]
        a = label[self.sens_attr_ids]
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
        return (
            np.zeros_like(bias),
            np.zeros_like(bias),
            a_one_hot,
            bias,
            bias_group,
            weight,
            cls_label,
        )

    def _prepare_weights(self, args):
        """
        https://github.com/YyzHarry/imbalanced-regression/
        """
        
        reweight = args.reweight
        lds=False
        lds_kernel="gaussian"
        lds_ks=5
        lds_sigma=2
        
        assert reweight in {"none", "inverse", "sqrt_inv"}
        assert (
            reweight != "none" if lds else True
        ), "Set reweight to 'sqrt_inv' (default) or 'inverse' when using LDS"

        # bias bins
        interval = args.pred_cls_interval
        max_target = args.pred_cls_num

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.bias_group

        for label in labels:
            value_dict[min(max_target - 1, int(label / interval))] += 1
        logger.info(f"[A BIAS DATALOADER] Using re-weighting: [{reweight.upper()}]")
        logger.info(f"[A BIAS DATALOADER] Initial Value Dict: {value_dict}")
        if reweight == "sqrt_inv":
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == "inverse":
            value_dict = {
                k: np.clip(v, 5, 5000) for k, v in value_dict.items()
            }  # clip weights for inverse re-weight
        num_per_label = [
            value_dict[min(max_target - 1, int(label / interval))] for label in labels
        ]
        if not len(num_per_label) or reweight == "none":
            return len(self.bias_group) * [1.0]
        logger.info(f"[A BIAS DATALOADER] Changed Value Dict: {value_dict}")
        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            logger.info(f"[A BIAS DATALOADER] Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})")
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]),
                weights=lds_kernel_window,
                mode="constant",
            )
            num_per_label = [
                smoothed_value[min(max_target - 1, int(label / interval))]
                for label in labels
            ]
            logger.info(f"[A BIAS DATALOADER] {num_per_label}")

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights

    def _prepare_cls_label(self, pred_cls_interval, cls_num):
        cls_labels = (self.bias_group / pred_cls_interval).astype(int)
        cls_labels = np.where(cls_labels <= (cls_num - 1), cls_labels, cls_num - 1)
        # print(cls_labels.max())
        return cls_labels



def get_loader_toxic(
    df,
    xy_h5_path,
    batch_size,
    sens_attr_ids,
    mode="org",
    mode_shuffle=True,
    args=None,
):
    # for f_model training
    if mode == "org":
        dl = TOXIC(df, xy_h5_path, sens_attr_ids)
    # for predictor and vae training
    elif mode == "bias":
        pred_cls_interval = 0.1 if args is None else args.pred_cls_interval
        cls_num = 30 if args is None else args.pred_cls_num
        dl = TOXIC_bias(df, sens_attr_ids,pred_cls_interval=pred_cls_interval,cls_num=cls_num,args=args)
    else:
        logger.info("[DATALOADER] Unknown loader mode")

    if mode_shuffle:
        dloader = torch.utils.data.DataLoader(
            dl,
            shuffle=mode_shuffle,
            batch_size=batch_size,
            num_workers=3,
            drop_last=True,
        )
    else:
        dloader = torch.utils.data.DataLoader(
            dl,
            shuffle=mode_shuffle,
            batch_size=batch_size,
            num_workers=3,
            drop_last=False,
        )

    return dloader


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ["gaussian", "triang", "laplace"]
    half_ks = (ks - 1) // 2
    if kernel == "gaussian":
        base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(
            gaussian_filter1d(base_kernel, sigma=sigma)
        )
    elif kernel == "triang":
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2.0 * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1))
        )

    return kernel_window


if __name__ == "__main__":
    pass
