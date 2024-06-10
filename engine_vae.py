import torch
import torch.nn as nn
from torch import optim
import os
import os.path as osp
import logging
logger = logging.getLogger("intersectionalFair")

# VAE pretrain
from vae_bias_utils import reparameterize_latent
from evaluation import evaluate_genModel


def vae_train(loaders, encoder, decoder, pred, df_regroup, args):
    """
    loaders - bias loader
        train, tr, val, te
        per item:
            image(x)
            label[self.target_id] (y)
            a_one_hot ( 01 10 01 01 ... 10 ) - size (batch_size, sens_attrs * attrs_dim)
            bias (f_model prediction bias value)
    """

    vae_criterion = nn.MSELoss()
    vae_en_optimizer = optim.Adam(encoder.parameters(), lr=args.en_vae_lr)
    vae_de_optimizer = optim.Adam(decoder.parameters(), lr=args.de_vae_lr)
    loader_train = loaders["train_reduced"]
    train_iter = iter(loader_train)

    # predictor train
    for i in range(1, args.vae_pretrain_epoch):
        for it in range(len(loader_train)):
            try:
                _, __, a_one_hot, bias, bias_group, weight, cls_label = next(train_iter)
            except:
                train_iter = iter(loader_train)
                _, __, a_one_hot, bias, bias_group, weight, cls_label = next(train_iter)
            a_one_hot = a_one_hot.float().to(args.device)
            mu, log_var = encoder(a_one_hot)
            z = reparameterize_latent(mu, log_var, args)
            a_hat = decoder(z)  # [batch, output_dim, class_dim]. after softmax
            loss_recons = vae_criterion(
                a_hat, a_one_hot.view(-1, args.output_dim, args.class_dim)
            )
            loss_kl = torch.mean(
                -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
            )
            loss = loss_recons + args.vae_train_kl_weight * loss_kl
            if it % 10 == 0:
                logger.info(
                    f"[VAE TRAIN] tr loss | Epoch [{i}/{args.vae_pretrain_epoch}] | Loss(recon/kl) [{it}/{len(loader_train)}]: {loss:.4f}({loss_recons:.4f}/{args.vae_train_kl_weight} * {loss_kl:.4f})"
                )
            vae_en_optimizer.zero_grad()
            vae_de_optimizer.zero_grad()
            loss.backward()
            vae_en_optimizer.step()
            vae_de_optimizer.step()

        decoder.eval()
        pred.eval()
        evaluate_genModel(decoder, pred, df_regroup, args)
        decoder.train()
        pred.train()

    encoder_model = encoder.state_dict()
    decoder_model = decoder.state_dict()

    # save model
    model_dir = osp.join(args.output_dir, "vae_model")
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(encoder_model, osp.join(model_dir, "encoder.pt"))
    torch.save(decoder_model, osp.join(model_dir, "decoder.pt"))

    return encoder, decoder

