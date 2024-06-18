import argparse
import pickle
import os
import os.path as osp
import torch
import yaml

from utils.utils import seed_set
from utils.hypers import CELEBA_2_FEAS_ID, CELEBA_39_FEAS_ID
# main functions import
from engine.engine_f_model import f_model_train

# from evaluation import evaluate_a_bias_dataset
from utils.loggers import TxtLogger, set_logger
import logging

logger = logging.getLogger("intersectionalFair")


def train_f_model(args):
    log_dir, log_file = set_logger(args.key_info, args.seed, args.output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger, fh, sh = TxtLogger(filename=osp.abspath(osp.join(log_dir, log_file)))
    logger.info(args)

    # ==========================================================
    # ==================== data loader =========================
    # ==========================================================
    ## df: dict; 'train', 'val', 'test', are PANDAS.DATAFRAME
    logger.info("DATA loading ...")
    if args.dset == 'celebA':
        pickle_dir = f"{args.data_root}/Anno/data_frame.pickle"
    elif args.dset == 'toxic':
        pickle_dir = f"{args.data_root}/data_frame.pickle"
    else:
        logger.info(f"Unknown dset: {args.dset}")
        pickle_dir = ''
    with open(pickle_dir, "rb") as handle:
        df = pickle.load(handle)
    logger.info("DATA loaded!")

    # ==========================================================
    # ================== f_model train - (X, Y) ================
    # ==========================================================
    ## get f_model and corresponding y_score result
    logger.info(f"F MODEL training ... SEED: {args.seed}")
    # For f_model_train, save:
    # 1) dataset with bias value for each x
    # 2) f_model param for further evaluation
    _ = f_model_train(df, args)
    logger.info("F MODEL training done!")

    logger.removeHandler(fh)  # remove current file handler to avoid log file error
    logger.removeHandler(sh)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F Model Training")
    parser.add_argument(
        "--config",
        type=str,
        help="config file",
        default="./EXPS/celebA_f_model.yml",
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
    parser.add_argument(
        "--target_id",
        default=2,
        type=int,
        help="prediction index - 2:attractive/31:smile/33:wavy hair",
    )
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
        args.target_id = 2 # only for evaluation, target id or sens attrs do not influence f_model training or bias value dataset generation
        args.sens_attr_ids = CELEBA_2_FEAS_ID # only for evaluation, target id or sens attrs do not influence f_model training or bias value dataset generation
        for s in [2021, 2022, 2023, 2024, 2025]:
            args.seed = s
            train_f_model(args)
    elif args.dset == "toxic":
        # sens attr start from col 1, end with 34, in total 34; 0 is toxicity; 
        args.sens_attr_ids = list(range(1,35))  # can be different from the main training process. Only for f model post evaluation
        args.target_id = 0
        for s in [2021, 2022, 2023, 2024, 2025]:
            args.seed = s
            train_f_model(args)
