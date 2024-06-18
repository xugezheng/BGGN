import argparse
import pickle
import os
import os.path as osp
import torch
import yaml
import ast


from layers.network import Encoder, Decoder, pred_init
from data.loaders import a_bias_loaders_load
from utils.utils import seed_set, bias_value_df_generation
from utils.hypers import CELEBA_2_FEAS_ID, CELEBA_39_FEAS_ID, TOXIC_25_FEAS_ID

# main functions import
from engine.engine_bias_predictor import bias_predictor_train
from engine.engine_vae import vae_train
from engine.engine_vae_bias import bias_vae_train

# from evaluation import evaluate_a_bias_dataset
from utils.loggers import TxtLogger, set_logger
import logging

logger = logging.getLogger("intersectionalFair")


# MAIN code - bias vae training
def train(args):
    log_dir, log_file = set_logger(args.key_info, args.seed, args.output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger, fh, sh = TxtLogger(filename=osp.abspath(osp.join(log_dir, log_file)))
    logger.info(args)
    
    
    logger.info("=================== STEP 1: DATA PREPARATION ===================")
    if args.f_model_df_regroup_dir is None:
        logger.info("[DATA PREPARATION] f model result (df_yscores) loading ...")
        df_bias_path = osp.join(args.data_root, args.f_model_df_midlle_dir, 'all_data_frame_with_yscores.pickle')
        with open(
            f"{df_bias_path}",
            "rb",
        ) as handle:
            df_bias = pickle.load(handle)  # dict: train, test, val (3 dataframes)
        logger.info(f"[DATA PREPARATION] f model result (df_yscores) loaded! (FROM: {df_bias_path})")
        logger.info("[DATA PREPARATION] New Bias Dataset Generating")
        # dict: all, train, test (3 regrouped dataframes), x based, include per sample
        df_regroup = bias_value_df_generation(df_bias, args)  
        logger.info("[DATA PREPARATION] New Bias Dataset Saved!")
    else:
        logger.info("[REGROUPED DATA LOAD] Bias Dataset df_regroup loading ...")
        df_regroup_path = osp.join(args.output_dir, args.f_model_df_regroup_dir)
        with open(
            f"{df_regroup_path}", "rb"
        ) as handle:
            df_regroup = pickle.load(handle)
        logger.info(f"[REGROUPED DATA LOAD] Bias Dataset df_regroup loaded! (FROM: {df_regroup_path})")

    # loaders_bias (DICT): train, tr, te (values are dataloaders) 
    # [A BIAS DATALOADER] step
    logger.info("[A BIAS DATALOADER] a bias DATALOADER INIT")
    loaders_bias = a_bias_loaders_load(df_regroup, args)

   
    logger.info("=================== STEP 2: BIAS PREDICTOR L_f(a) ===================")
    pred = pred_init(args).to(args.device)
    if args.pred_dir is None:
        logger.info("[L_f(a) TRAIN] Bias Pred training ...")
        pred = bias_predictor_train(loaders_bias, pred, args)
        logger.info("[L_f(a) TRAIN] Bias Pred training is finished!")
    else:
        pred.load_state_dict(
            torch.load(
                osp.join(
                    args.output_dir,
                    "bias_pred_model",
                    f"{args.pred_type}_{args.pred_dir}",
                ),
                map_location=args.device,
            )
        )
        logger.info("[L_f(a) LOAD] Bias Predictor loaded")


    logger.info("=================== STEP 3: VAE Pre-Train ===================")
    encoder = Encoder(args.input_dim, args.latent_dim, args.hidden_dim).to(args.device)
    decoder = Decoder(
        args.latent_dim, args.hidden_dim, args.output_dim, args.class_dim
    ).to(args.device)
    if args.vae_dir is None:
        encoder, decoder = vae_train(
            loaders_bias, encoder, decoder, pred, df_regroup, args
        )
        logger.info("[VAE TRAIN] VAE trained and saved")
    else:
        encoder.load_state_dict(
            torch.load(
                osp.join(args.output_dir, args.vae_dir, "encoder.pt"),
                map_location=args.device,
            )
        )
        decoder.load_state_dict(
            torch.load(
                osp.join(args.output_dir, args.vae_dir, "decoder.pt"),
                map_location=args.device,
            )
        )
        logger.info("[VAE LOAD] VAE Loaded")

    # ==========================================================
    # ====================== VAE bias train ====================
    # ==========================================================
    logger.info("=================== STEP 4: Bias Guide Fine-Tuning ===================")
    encoder, decoder = bias_vae_train(
        encoder, decoder, pred, df_regroup, args, loaders_bias
    )

    logger.removeHandler(fh)  
    logger.removeHandler(sh)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="subGroupBias_BGGN")
    parser.add_argument(
        "--config",
        type=str,
        help="config file",
        default=r"./EXPS/toxic_train.yml",
    )
    parser.add_argument("--dset", default="celebA", choices=["celebA", "toxic"])
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument(
        "--key_info",
        type=str,
        help="name for log prefix and df sub folder name",
        default="test",
    )

    parser.add_argument("--data_root", type=str, default="./DATASOURCE/celebA")
    parser.add_argument("--output_dir", type=str, default="./output/celebA_test")
    parser.add_argument("--f_model_df_midlle_dir", type=str, default="Models/f_model_train")

    parser.add_argument(
        "--target_id",
        default=2,
        type=int,
        help="prediction index - 2:attractive/31:smile/33:wavy hair",
    )
    parser.add_argument(
        "--class_dim",
        default=2,
        type=int,
        help="per sensitive attribute's class number",
    )
    parser.add_argument(
        "--output_dim", default=20, type=int, help="sensitive attributes number"
    )

    # df regroup
    parser.add_argument("--te_par", type=float, default=0.3)
    parser.add_argument("--f_model_df_regroup_dir", type=str, default=None)
    parser.add_argument(
        "--f_model_df_regroup_norm",
        type=str,
        choices=["org", "minmax", "zscore"],
        default="pass",
    )

    # predictor train
    parser.add_argument(
        "--reweight", type=str, choices=["inverse", "sqrt_inv", "none"], default="none"
    )
    parser.add_argument("--pred_lr", type=float, default=1.0e-3)
    parser.add_argument("--pred_dir", type=str, default="bias_pred.pt")
    parser.add_argument(
        "--pred_type",
        type=str,
        choices=["tf_reg_cls", "tf_reg", "mlp_reg_cls", "mlp_reg"],
        default="tf_reg_cls",
    )
    parser.add_argument("--pred_train_epoch", type=int, default=60)
    parser.add_argument(
        "--pred_hidden_dim",
        default=512,
        type=int,
    )
    parser.add_argument("--pred_fea_dim", default=256, type=int)
    ## classification task
    parser.add_argument("--pred_cls_interval", type=float, default=0.1)
    parser.add_argument("--pred_cls_num", type=int, default=10)
    parser.add_argument("--pred_reg_loss_coef", type=float, default=0.01)
    parser.add_argument("--pred_ce_loss_coef", type=float, default=1)

    # vae train
    parser.add_argument("--vae_dir", type=str, default="vae_model")
    parser.add_argument(
        "--latent_dim", default=200, type=int, help="Dim of latent variable Z for vae"
    )
    parser.add_argument(
        "--hidden_dim", default=500, type=int, help="Dim of hidden layers for vae"
    )
    parser.add_argument("--vae_pretrain_epoch", type=int, default=5)
    parser.add_argument("--vae_train_kl_weight", type=float, default=0.1)
    parser.add_argument("--en_vae_lr", type=float, default=0.001)
    parser.add_argument("--de_vae_lr", type=float, default=0.001)
    

    # vae bias fine-tuning
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ft_batch_size", type=int, default=256)
    parser.add_argument("--z_resample", type=int, default=1, help="z sample times")
    parser.add_argument("--mask_bs_coef", type=float, default=1.0)
    
    parser.add_argument("--vae_bias_b", type=float, default=50000)
    parser.add_argument(
        "--vae_bias_pred_mode", type=str, choices=["nn", "gt_nn"], default="nn"
    )

    parser.add_argument("--vae_bias_epoch", type=int, default=1000)
    parser.add_argument("--vae_bias_reg_coef", type=float, default=1.0)
    parser.add_argument("--vae_bias_r_local", type=float, default=0.01)
    parser.add_argument("--vae_bias_r_global", type=float, default=0.01)
    parser.add_argument("--vae_bias_neglog", type=float, default=0.05)
    parser.add_argument("--vae_bias_train_en", type=int, default=1)
    parser.add_argument("--en_vae_bias_lr", type=float, default=2.0e-5)
    parser.add_argument("--de_vae_bias_lr", type=float, default=1.0e-5)
    

    # generation evaluation
    parser.add_argument("--evaluation_batch_num", type=int, default=100)
    parser.add_argument("--gen_round", type=int, default=5)
    parser.add_argument("--vae_eval_bias_thresh", type=float, default=0.4)
    ## precision at k
    parser.add_argument("--precision_k", type=int, default=1000, help="x based")
    ## recall at k
    parser.add_argument("--recall_k", type=int, default=200)
    ## rr at k
    parser.add_argument("--rr_k", type=int, default=5)
    parser.add_argument("--rr_k_ratio", type=float, default=0.05)
    ## dcg
    parser.add_argument("--dcg_k", type=int, default=20)
    parser.add_argument(
        "--dcg_gains", type=str, choices=["linear", "exponential"], default="linear"
    )
    parser.add_argument("--save_all_gen_result", type=ast.literal_eval, default=True)

    args = parser.parse_args()

    if args.config:
        cfg_dir = osp.abspath(osp.join(osp.dirname(__file__), args.config))
        opt = vars(args)
        args = yaml.load(open(cfg_dir), Loader=yaml.FullLoader)
        opt.update(args)
        args = argparse.Namespace(**opt)

    seed_set(args)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.dset == "celebA":
        # for attractive
        if args.target_id == 2:
            args.sens_attr_ids = CELEBA_2_FEAS_ID
        # for young
        elif args.target_id == 39:
            args.sens_attr_ids = CELEBA_39_FEAS_ID
    elif args.dset == "toxic":
        args.sens_attr_ids = TOXIC_25_FEAS_ID
        args.target_id = 0
    
    args.output_dim = len(args.sens_attr_ids)
    print(f"sens attrs [{args.output_dim}]: {args.sens_attr_ids}")
    args.input_dim = int(args.output_dim * args.class_dim)
    train(args)
