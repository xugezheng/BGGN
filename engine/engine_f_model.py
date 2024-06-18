import pickle
import os
import os.path as osp
import torch
import torch.nn as nn
from torch import optim
import pandas as pd

import logging
logger = logging.getLogger("intersectionalFair")

from data.loaders import f_model_loaders_load
from layers.network import ResNet18_Encoder, LinearModel, F_Model_TOXIC
from utils.evaluation import evaluate_f_model


def f_model_train(df, args):
    if args.dset == "celebA":
        return f_model_train_celebA(df, args)
    elif args.dset == 'toxic':
        return f_model_train_toxic(df, args)
    else:
        logger.info(f"Error Dataset: {args.dset}")
        return None

# f_model TRAIN
def f_model_train_celebA(df, args):
    loaders = f_model_loaders_load(df, "org", args)
    f_model = ResNet18_Encoder(pretrained=True).to(args.device)
    f_model_linear = LinearModel().to(args.device)
    f_model_criterion = nn.BCELoss()
    f_model_optimizer = optim.Adam(f_model.parameters(), lr=1e-4)
    f_model_linear_optimizer = optim.Adam(f_model_linear.parameters(), lr=1e-4)
    schedular = optim.lr_scheduler.ExponentialLR(f_model_optimizer, gamma=0.9)
    schedular_linear = optim.lr_scheduler.ExponentialLR(f_model_linear_optimizer, gamma=0.9)

    loader_train = loaders["train"]
    train_iter = iter(loader_train)

    best_tr_ap = 0

    # INITIAL TEST
    f_model.eval()
    f_model_linear.eval()
    all_result, all_new_record = evaluate_f_model(
        f_model, f_model_linear, loaders, print_all=True
    )
    logger.info(
        f"F_MODEL | Initial | Tr Acc: {all_new_record['tr']['acc']:.5f} | Loss: {all_new_record['tr']['loss_avg']:.5f}"
    )
    logger.info(
        f"F_MODEL | Initial | Te Acc: {all_new_record['te']['acc']:.5f} | Loss: {all_new_record['te']['loss_avg']:.5f}"
    )
    logger.info(
        f"F_MODEL | Initial | Va Acc: {all_new_record['val']['acc']:.5f} | Loss: {all_new_record['val']['loss_avg']:.5f}"
    )
    f_model.train()
    f_model_linear.train()

    # f_model train
    for i in range(1, args.f_model_train_epoch + 1):
        for it in range(len(loader_train)):
            try:
                inputs, targets, sens_attrs = next(train_iter)
            except:
                train_iter = iter(loader_train)
                inputs, targets, sens_attrs = next(train_iter)
            inputs, targets = inputs.to(args.device), targets.float().to(args.device)
            # supervised loss
            feat = f_model(inputs)
            ops = f_model_linear(feat)
            loss = f_model_criterion(ops, targets)
            if it % 100 == 0 or it == (len(loader_train) - 1):
                logger.info(
                    f"F_MODEL | Epoch [{i}/{args.f_model_train_epoch}] | Loss [{it}/{len(loader_train)}]: {loss:.4f}"
                )
            f_model_optimizer.zero_grad()
            f_model_linear_optimizer.zero_grad()
            loss.backward()
            f_model_optimizer.step()
            f_model_linear_optimizer.step()
        schedular.step()
        schedular_linear.step()

        # per epoch test
        f_model.eval()
        f_model_linear.eval()
        all_result, all_new_record = evaluate_f_model(f_model, f_model_linear, loaders)
        logger.info(
            f"F_MODEL | Epoch [{i}/{args.f_model_train_epoch}] | Tr Acc: {all_new_record['tr']['acc']:.5f} | Loss: {all_new_record['tr']['loss_avg']:.5f}"
        )
        logger.info(
            f"F_MODEL | Epoch [{i}/{args.f_model_train_epoch}] | Te Acc: {all_new_record['te']['acc']:.5f} | Loss: {all_new_record['te']['loss_avg']:.5f}"
        )
        logger.info(
            f"F_MODEL | Epoch [{i}/{args.f_model_train_epoch}] | Va Acc: {all_new_record['val']['acc']:.5f} | Loss: {all_new_record['val']['loss_avg']:.5f}"
        )
        f_model.train()
        f_model_linear.train()
        # best save
        if all_new_record["tr"]["acc"] >= best_tr_ap:
            best_f_model = f_model.state_dict()
            best_f_model_linear = f_model_linear.state_dict()
            best_tr_ap = all_new_record["tr"]["acc"]

    # save model
    model_dir = osp.join(args.data_root, "Models", args.key_info, str(args.seed))
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(best_f_model, osp.join(model_dir, "f_model.pt"))
    torch.save(best_f_model_linear, osp.join(model_dir, "f_model_linear.pt"))

    # get bias values on train/validation/test dataset
    f_model_test = ResNet18_Encoder(pretrained=True).to(args.device)
    f_model_linear_test = LinearModel().to(args.device)
    f_model_test.load_state_dict(torch.load(osp.join(model_dir, "f_model.pt")))
    f_model_linear_test.load_state_dict(
        torch.load(osp.join(model_dir, "f_model_linear.pt"))
    )
    f_model_test.eval()
    f_model_linear_test.eval()
    all_result, all_new_record = evaluate_f_model(
        f_model_test, f_model_linear_test, loaders
    )
    logger.info(
        f"F_MODEL SAVED | Tr Acc: {all_new_record['tr']['acc']:.5f} | Loss: {all_new_record['tr']['loss_avg']:.5f}"
    )
    logger.info(
        f"F_MODEL SAVED | Te Acc: {all_new_record['te']['acc']:.5f} | Loss: {all_new_record['te']['loss_avg']:.5f}"
    )
    logger.info(
        f"F_MODEL SAVED | Va Acc: {all_new_record['val']['acc']:.5f} | Loss: {all_new_record['val']['loss_avg']:.5f}"
    )

    df["train"]["bias_group"] = all_new_record["tr"]["bias"]
    df["test"]["bias_group"] = all_new_record["te"]["bias"]
    df["val"]["bias_group"] = all_new_record["val"]["bias"]
    df["train"]["bias"] = all_new_record["tr"]["bias"]
    df["test"]["bias"] = all_new_record["te"]["bias"]
    df["val"]["bias"] = all_new_record["val"]["bias"]
    df["train"]["y_score"] = all_new_record["tr"]["y_scores"]
    df["test"]["y_score"] = all_new_record["te"]["y_scores"]
    df["val"]["y_score"] = all_new_record["val"]["y_scores"]

    # save raw data dataframe
    with open(f"{model_dir}/data_frame_with_yscores.pickle", "wb") as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return df


##################################################
# ################## TOXIC #######################
##################################################

def f_model_train_toxic(df, args):
    loaders = f_model_loaders_load(df, "org", args)
    f_model = F_Model_TOXIC().to(args.device)
    f_model_criterion = nn.BCELoss()
    f_model_optimizer = optim.Adam(f_model.parameters(), lr=1e-4)
    schedular = optim.lr_scheduler.ExponentialLR(f_model_optimizer, gamma=0.9)
    loader_train = loaders["train"]
    train_iter = iter(loader_train)
    best_tr_ap = 0

    # INITIAL TEST
    f_model.eval()
    all_result, all_new_record = evaluate_f_model(
        f_model, None, loaders, print_all=True
    )
    logger.info(
        f"F_MODEL | Initial | Tr Acc: {all_new_record['tr']['acc']:.5f} | Loss: {all_new_record['tr']['loss_avg']:.5f}"
    )
    logger.info(
        f"F_MODEL | Initial | Te Acc: {all_new_record['te']['acc']:.5f} | Loss: {all_new_record['te']['loss_avg']:.5f}"
    )
    logger.info(
        f"F_MODEL | Initial | Va Acc: {all_new_record['val']['acc']:.5f} | Loss: {all_new_record['val']['loss_avg']:.5f}"
    )
    f_model.train()

    # f_model train
    for i in range(1, args.f_model_train_epoch + 1):
        for it in range(len(loader_train)):
            try:
                inputs, targets, sens_attrs = next(train_iter)
            except:
                train_iter = iter(loader_train)
                inputs, targets, sens_attrs = next(train_iter)
            inputs, targets = inputs.to(args.device), targets.float().to(args.device)
            # supervised loss
            ops = f_model(inputs)
            loss = f_model_criterion(ops, targets)
            if it % 100 == 0 or it == (len(loader_train) - 1):
                logger.info(
                    f"F_MODEL | Epoch [{i}/{args.f_model_train_epoch}] | Loss [{it}/{len(loader_train)}]: {loss:.4f}"
                )
            f_model_optimizer.zero_grad()
            loss.backward()
            f_model_optimizer.step()
        schedular.step()

        # per epoch evaluation
        f_model.eval()
        all_result, all_new_record = evaluate_f_model(f_model, None, loaders)
        logger.info(
            f"F_MODEL | Epoch [{i}/{args.f_model_train_epoch}] | Tr Acc: {all_new_record['tr']['acc']:.5f} | Loss: {all_new_record['tr']['loss_avg']:.5f}"
        )
        logger.info(
            f"F_MODEL | Epoch [{i}/{args.f_model_train_epoch}] | Te Acc: {all_new_record['te']['acc']:.5f} | Loss: {all_new_record['te']['loss_avg']:.5f}"
        )
        logger.info(
            f"F_MODEL | Epoch [{i}/{args.f_model_train_epoch}] | Va Acc: {all_new_record['val']['acc']:.5f} | Loss: {all_new_record['val']['loss_avg']:.5f}"
        )
        f_model.train()
        if all_new_record["tr"]["acc"] >= best_tr_ap:
            best_f_model = f_model.state_dict()
            best_tr_ap = all_new_record["tr"]["acc"]

    # save model
    model_dir = osp.join(args.data_root, "Models", args.key_info, str(args.seed))
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(best_f_model, osp.join(model_dir, "f_model.pt"))

    # get bias values on train/validation/test dataset
    f_model_test = F_Model_TOXIC().to(args.device)
    f_model_test.load_state_dict(torch.load(osp.join(model_dir, "f_model.pt")))
    f_model_test.eval()
    all_result, all_new_record = evaluate_f_model(
        f_model_test, None, loaders
    )
    logger.info(
        f"F_MODEL SAVED | Tr Acc: {all_new_record['tr']['acc']:.5f} | Loss: {all_new_record['tr']['loss_avg']:.5f}"
    )
    logger.info(
        f"F_MODEL SAVED | Te Acc: {all_new_record['te']['acc']:.5f} | Loss: {all_new_record['te']['loss_avg']:.5f}"
    )
    logger.info(
        f"F_MODEL SAVED | Va Acc: {all_new_record['val']['acc']:.5f} | Loss: {all_new_record['val']['loss_avg']:.5f}"
    )

    df["train"]["bias_group"] = all_new_record["tr"]["bias"]
    df["test"]["bias_group"] = all_new_record["te"]["bias"]
    df["val"]["bias_group"] = all_new_record["val"]["bias"]
    df["train"]["bias"] = all_new_record["tr"]["bias"]
    df["test"]["bias"] = all_new_record["te"]["bias"]
    df["val"]["bias"] = all_new_record["val"]["bias"]
    df["train"]["y_score"] = all_new_record["tr"]["y_scores"]
    df["test"]["y_score"] = all_new_record["te"]["y_scores"]
    df["val"]["y_score"] = all_new_record["val"]["y_scores"]

    with open(f"{model_dir}/data_frame_with_yscores.pickle", "wb") as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return df