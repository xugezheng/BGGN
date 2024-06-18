import torch
import torch.nn as nn
from torch import optim
import os
import os.path as osp
import logging
logger = logging.getLogger("intersectionalFair")

from utils.losses import weighted_mse_loss, cross_entropy
from utils.evaluation import evaluate_bias_predictor
    

def bias_predictor_train(loaders, pred, args):
    best_re_tr = 100000

    pred_optimizer = optim.Adam(pred.parameters(), lr=args.pred_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(pred_optimizer, step_size=10, gamma=0.8)

    # TO TEST
    loader_train = loaders["train_reduced"]
    train_iter = iter(loader_train)
    
    # predictor train
    for i in range(1, args.pred_train_epoch):
        logger.info(f"[L_f(x) TRAIN] Epoch {i}:")
        logger.info(f"[L_f(x) TRAIN] Current LR is {pred_optimizer.param_groups[0]['lr']}")
        for it in range(len(loader_train)):
            try:
                _, __, a_one_hot, bias, bias_group, weight, cls_label = next(train_iter)
            except:
                train_iter = iter(loader_train)
                _, __, a_one_hot, bias, bias_group, weight, cls_label = next(train_iter)
            
            # device merge
            a_one_hot, bias, bias_group = (
                a_one_hot.float().to(args.device),
                bias.float().to(args.device),
                bias_group.float().to(args.device),
            )
            weight = weight.float().to(args.device)
            cls_label = cls_label.long().to(args.device)
            
            # get pred
            bias_pred, cls_pred, _ = pred(a_one_hot)
            
            # calculate loss
            loss = 0.0 
            reg_loss = weighted_mse_loss(bias_pred, bias_group, weight)
            loss = reg_loss * args.pred_reg_loss_coef
            if cls_pred is not None:
                ce_loss = cross_entropy(cls_label, cls_pred, weight, args)
                ce_loss = ce_loss * args.pred_ce_loss_coef
                loss += ce_loss
            else:
                ce_loss = 0.0
            
            if it % 10 == 0:
                logger.info(
                    f"[L_f(x) TRAIN] | Epoch [{i}/{args.pred_train_epoch}] | Loss [{it}/{len(loader_train)}]: {loss.item():.4f} | Reg (* {args.pred_reg_loss_coef}): {reg_loss:.4f} | CE (* {args.pred_ce_loss_coef}): {ce_loss:.4f}"
                )
            
            pred_optimizer.zero_grad()
            loss.backward()
            pred_optimizer.step()
        scheduler.step()

        pred.eval()
        logger.info(f"[L_f(x) TRAIN] train set evaluation")
        re_tr_ind, re_tr_grp, re_cls_tr = evaluate_bias_predictor(
            pred, loaders["tr_reduced"], args.vae_eval_bias_thresh
        )
        logger.info(f"[L_f(x) TRAIN] test set evaluation")
        re_te_ind, re_te_grp, re_cls_te = evaluate_bias_predictor(
            pred, loaders["te_reduced"], args.vae_eval_bias_thresh
        )
        logger.info(f"[L_f(x) TRAIN] Evaluation Summary")
        logger.info(
            f"L_f(x) TRAIN] Summary | Epoch [{i}/{args.pred_train_epoch}] | Tr MAE (Grp): [{re_tr_grp:.5f}]"
        )
        logger.info(
            f"L_f(x) TRAIN] Summary | Epoch [{i}/{args.pred_train_epoch}] | Te MAE (Grp): [{re_te_grp:.5f}]"
        )
        logger.info(
            f"L_f(x) TRAIN] Summary | Epoch [{i}/{args.pred_train_epoch}] | Tr cls result: [{re_cls_tr:.5f}]"
        )
        logger.info(
            f"L_f(x) TRAIN] Summary | Epoch [{i}/{args.pred_train_epoch}] | Te cls result: [{re_cls_te:.5f}]"
        )
        pred.train()

        if re_tr_grp <= best_re_tr:
            best_bias_pred = pred.state_dict()
            best_re_tr = re_tr_grp

            # save model
            model_dir = osp.join(args.output_dir, "bias_pred_model")
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(best_bias_pred, osp.join(model_dir, f"{args.pred_type}_bias_pred.pt"))

    return pred