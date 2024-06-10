import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import os.path as osp
import copy
import pandas as pd

import logging

logger = logging.getLogger("intersectionalFair")

from vae_bias_utils import (
    grad_activate,
    grad_freeze,
    ten2vec,
    get_z_based_loss,
    get_trA_based_loss
)
from utils import save_dset_dict_2_df, df_2_lists
from evaluation import evaluate_genModel_multi_round
from evaluation_utils import compare_evaluations


def bias_vae_train(Model_en, Model_de, pred, df, args, loaders):
    '''
    while training:
    1) save all generated tr/te/new A and X, also dataset A to ./output/key_info/df/all
    2) save an overall evaluation information dataframe ./output/key_info/df/all_xxx.csv
    3) save all middle model
    4) print evaluation result 
    '''
    df_list = []
    better_than_vae = []
    others = []
    
    # ============ Decoder Back Up ============
    grad_activate(Model_de)
    Model_de_backup = copy.deepcopy(Model_de)
    grad_freeze(Model_de_backup)

    # ============ Feature Bank Prepare ============
    # bias prediction result prepare
    a_dicts = df_2_lists(df, args.sens_attr_ids)
    # Evaluation Result
    a_tr, gp_bias_tr = a_dicts["train"]  
    a_bank = torch.tensor(a_tr).long().to(args.device).unsqueeze(0)
    gp_bias_bank = torch.tensor(gp_bias_tr).float().to(args.device)
    with torch.no_grad():
        a_input = (
            F.one_hot(a_bank.long(), num_classes=2)
            .squeeze(0)
            .view(-1, args.class_dim * args.output_dim)
        )
        _, _, a_fea_bank = pred(a_input.float())

    # ============ Training Preperations ============
    # define a baeline to reduce the variance (without gradient)
    loss_en = torch.tensor(0).to(args.device)

    # Define optimizer (need to fine tuning)
    optimizer_en = optim.Adam(Model_en.parameters(), lr=args.en_vae_bias_lr)
    optimizer_de = optim.Adam(Model_de.parameters(), lr=args.de_vae_bias_lr)

    # train A to train bias vae
    loader_train = loaders["train_reduced_ft"]
    
    # ================================
    # evaluation before fine-tuning - VAE
    logger.info(f"[BGFT] | VAE Evaluation.")
    dataset_dict, init_best_evaluation_dict, init_merged_dict, init_df_list, df_index = evaluate_genModel_multi_round(
        args.gen_round, Model_de, pred, df, args, a_bank, a_fea_bank, gp_bias_bank, "epo_0"
    )
    # df save
    df_list += init_df_list
    # dataset sens attrs save
    if args.save_all_gen_result:
        df_all_dir = osp.join(args.output_dir, "df", args.key_info, "all")
        if not osp.exists(df_all_dir):
            os.makedirs(df_all_dir)
        save_dset_dict_2_df(dataset_dict, df_all_dir)
    # ================================
    
    
    logger.info(f"[BGFT] | Epoch [0/{args.vae_bias_epoch}] | VAE Evaluation Finished. Bias Fine-tuning starts...")
    # Fine-Tuning
    R_global = 0
    for epoch in range(1, args.vae_bias_epoch):
        epo_log_str = f'[BGFT] | Epoch [{epoch}/{args.vae_bias_epoch}] | Training Info: \n'
        loss = torch.tensor(0.0).to(args.device)
        # =====================
        # ====== Encoder ======
        # =====================
        # taining the encoder-decoder model through wake-sleep algorithms (or alternative optimization)
        if epoch % 3 == 0 and args.vae_bias_train_en != 0:
            # stage wake, fix generator
            grad_activate(Model_en)
            grad_freeze(Model_de)
            z_samples = torch.randn(args.batch_size, args.latent_dim).to(args.device)
            generated_samples = Model_de(z_samples)
            generated_2D = ten2vec(generated_samples, args.output_dim, args.class_dim)
            mu_phi, log_var = Model_en(generated_2D)
            loss_en = torch.mean(
                0.5 * torch.norm(z_samples - mu_phi) ** 2 / torch.exp(log_var) + log_var
            )  # w-disance version

            optimizer_en.zero_grad()
            loss_en.backward()
            optimizer_en.step()
            epo_log_str += f"[BGFT] | Epoch [{epoch}/{args.vae_bias_epoch}] | Encoder Total Loss (recon): {loss_en.item():.4f}\n"
        # =============================================
        # ====== Decoder fine tuning  - Sample Z ======
        # =============================================
        loss_z, loss_main, entropy_reg, negative_log_z_estimate, b, R_final, R_global= get_z_based_loss(Model_en, Model_de, pred, args, a_bank, a_fea_bank, gp_bias_bank, "bs_trA", "var_control", R_global)
        loss += loss_z
        
        epo_log_str += f"[BGFT] | Epoch [{epoch}/{args.vae_bias_epoch}] | Decoder Total Loss Z Sample (main/reg): {loss_z.item():.2f} ({loss_main.item():.2f}/{entropy_reg.item():.2f})\n"
        epo_log_str += f"[BGFT] | Epoch [{epoch}/{args.vae_bias_epoch}] | R all info (negLog/b): {R_final.detach().clone().mean().item():.2f} ({negative_log_z_estimate.detach().clone().mean().item():.2f}/{b.detach().clone().mean().item():.2f})\n"
        epo_log_str += f"[BGFT] | Epoch [{epoch}/{args.vae_bias_epoch}] | R global: {R_global.item():.4f}\n"
        # =============================================
        # ====== Decoder fine tuning  - Tr A Z =========
        # =============================================
        loss_tr, loss_main_tr, entropy_reg_tr, negative_log_z_estimate_tr, b_tr, R_tr = get_trA_based_loss(loader_train, Model_en, Model_de, args)
        loss += loss_tr
        
        epo_log_str += f"[BGFT] | Epoch [{epoch}/{args.vae_bias_epoch}] | Decoder Total Loss Train A (main/reg): {loss_tr.item():.2f} ({loss_main_tr.item():.2f}/{entropy_reg_tr.item():.2f})\n"
        epo_log_str += f"[BGFT] | Epoch [{epoch}/{args.vae_bias_epoch}] | R_tr info (negLog/b): {R_tr.detach().clone().mean().item():.2f} ({negative_log_z_estimate_tr.detach().clone().mean().item():.2f}/{b_tr.detach().clone().mean().item():.2f})\n"
            
        # =============================================
        # ====== Decoder fine tuning  - Baseline =========
        # =============================================
        loss_base, loss_main_base, entropy_reg_base, negative_log_z_estimate_base, b_base, R_base = get_z_based_loss(Model_en, Model_de, pred, args, a_bank, a_fea_bank, gp_bias_bank, "bs")
        loss += loss_base
        
        epo_log_str += f"[BGFT] | Epoch [{epoch}/{args.vae_bias_epoch}] | Baseline (bs mask) Fine-Tune (main/reg): {loss_base.item():.2f} ({loss_main_base.item():.2f}/{entropy_reg_base.item():.2f})\n"
        epo_log_str += f"[BGFT] | Epoch [{epoch}/{args.vae_bias_epoch}] | R_base info (negLog/b): {R_base.detach().clone().mean().item():.2f} ({negative_log_z_estimate_base.detach().clone().mean().item():.2f}/{b_base.detach().clone().mean().item():.2f})\n"


        optimizer_de.zero_grad()
        loss.backward()
        optimizer_de.step()

        # ============ Middle Evaluation ============
        if (epoch + 1) % 100 == 0:
            logger.info(epo_log_str)
            Model_de_backup.eval()
            Model_de.eval()
            pred.eval()
            logger.info(f"[BGFT] | BGGN Evaluation.")
            _, ft_best_evaluation_dict, ft_merged_dict, ft_df_list, ft_index = evaluate_genModel_multi_round(args.gen_round, Model_de, pred, df, args, a_bank, a_fea_bank, gp_bias_bank, f"epo_{str(epoch)}")
            replace_merge_flag = compare_evaluations(init_merged_dict, ft_merged_dict, print_flag=True)
            replace_best_flag = compare_evaluations(init_best_evaluation_dict, ft_best_evaluation_dict, print_flag=True)
            assert df_index == ft_index # make sure get same dataframe columns
            if replace_merge_flag or replace_best_flag:
                if replace_merge_flag:
                    ft_df_list[-1][2] = 1
                if replace_best_flag:
                    ft_df_list[-2][2] = 1
                better_than_vae.append(epoch)
                
                # save model
                logger.info(f"[BGFT] Saved Model Record | Epoch [{epoch}] Model Saved")
                encoder_model = Model_en.state_dict()
                decoder_model = Model_de.state_dict()
                model_dir = osp.join(args.output_dir, "vae_bias_model", args.key_info)
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(encoder_model, osp.join(model_dir, f"{epoch}_bias_encoder.pt"))
                torch.save(decoder_model, osp.join(model_dir, f"{epoch}_bias_decoder.pt"))
            else:
                others.append(epoch)
        
            df_list += ft_df_list
            Model_de.train()
            pred.train()
        # ==========================================

    logger.info(f"[BGFT] | Final Merged Evaluation Result")
    logger.info(f"[BGFT] | Better than VAE: {better_than_vae}; Others: {others}")
        
    # save summary csv - evaluation info dataframe
    df_dir_root = osp.join(args.output_dir, "df", args.key_info)
    if not osp.exists(df_dir_root):
        os.makedirs(df_dir_root)
    df = pd.DataFrame(data=df_list, columns=df_index)
    df.to_csv(osp.join(df_dir_root, f"all_{args.key_info}_{args.dset}.csv"))
        
    return Model_en, Model_de

