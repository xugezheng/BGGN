import torch
import numpy as np
from sklearn.metrics import average_precision_score
from collections import defaultdict
import torch.nn.functional as F
import random

import logging
logger = logging.getLogger("intersectionalFair")

# ======================================================
# VAE related
# ======================================================

# ======================================================
# VAE pretrain
# ======================================================
# sampling trick w.r.t. continous random variable Z (disgraded)
def reparameterize_latent(mu, logvar, args):
        # sampling from a generative model w.r.t Z
        std = torch.exp(0.5 * logvar).to(args.device)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    
# ======================================================
# VAE bias train
# ======================================================
# compute the log-negative-probability value of a (3d vectors), please note here is a positive value
def compute_log_softmax_distribution(vector_proba, vector_true, normal=True):
    if not normal:
        softmax_probs = F.softmax(vector_proba, dim=2)
    else:
        softmax_probs = vector_proba
    
    log_likelihoods = torch.sum(torch.log(softmax_probs+1e-9) * vector_true.float(), dim=2)
    log_likelihood_per_sample = torch.sum(log_likelihoods, dim=1)

    # return for each sample 
    return -1 * log_likelihood_per_sample

def compute_log_gaussian_nll_distribution(vector_proba, vector_true, args):
    '''
    expectation - input - first args - vector_proba
    sampled - target - second args - vector_true
    '''
    log_likelihood_per_sample = F.gaussian_nll_loss(vector_proba, vector_true, args.bias_vae_logvar, reduction='sum')
    return log_likelihood_per_sample

# given a multi-softmax-distribution, sample a combination (here we assume we have a data batch)   
def sample_from_multi_softmax_distribution(vector, class_dim, normal=True, one_hot=True):
    if not normal:
        softmax_probs = F.softmax(vector, dim=2)
    else:
        softmax_probs = vector
    
    num_of_batch = softmax_probs.size(0)


    # Reshape the 'probabilities' tensor to combine the first and second dimensions
    # The shape becomes (num_of_output * 5, 3)
    reshaped_probabilities = softmax_probs.view(-1, softmax_probs.size(-1))
    

    # Create a Categorical distribution for each tensor in the 'probabilities' batch
    categorical_dist = torch.distributions.Categorical(probs=reshaped_probabilities)
    
    # Sampling indices based on the probability distribution for the entire 'probabilities' tensor
    # The returned 'samples' tensor will have shape (num_of_output * 5, num_samples)
    samples = categorical_dist.sample((1,))
        
    # Reshape the 'samples' tensor to (num_of_output, 5, num_samples)
    sampled_indices = samples.view(num_of_batch, softmax_probs.size(1))
    
    if one_hot:
        sample_indices = F.one_hot(sampled_indices, num_classes=class_dim)
        sample_indices = sample_indices.to(torch.float)    
        
    return sample_indices




# compute the entropy, given a 3D tensor with normalized distribution
def compute_entropy(vector, normal=True, mask=None): 
    if not normal:
        softmax_probs = F.softmax(vector, dim=2)
    else:
        softmax_probs = vector
    
    entropy_values = -1 * torch.log(softmax_probs+1e-9) * softmax_probs

    # Sum along the last dimension (summing over the class probabilities)
    # The resulting 'entropy_sum' tensor will have shape (100, 5)
    entropy_sum = entropy_values.sum(dim=-1)

    # Compute the average entropy along the second dimension (averaging over the samples)
    # torch mean returns a scalar value
    if mask is None:
        average_entropy = torch.mean(entropy_sum.mean(dim=1))
    else:
        average_entropy = torch.mean(entropy_sum.mean(-1) * mask)

    return average_entropy


# convert vector tensor into 3D tensor
def vec2ten(vector, attr_dim, attri_range):
    return vector.view(-1, attr_dim, attri_range)

# flattern the 3D tensor into 2D tensor
def ten2vec(vector, attr_dim, attri_range):
    return vector.view(-1, attri_range * attr_dim) 

def grad_freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return None

def grad_activate(model):
    for param in model.parameters():
        param.requires_grad = True
    return None


# ==============================================
# ============== Bias Value Get    =============
# ==============================================
def get_bias(a_batch, pred, a_bank, a_fea_bank, gp_bias_bank, args):
    
    a_fea_bank = F.normalize(a_fea_bank, dim=-1)
    
    with torch.no_grad():
        a_batch_2d = a_batch.view(-1, args.class_dim * args.output_dim)
        nn_pred_result, nn_cls_result, batch_fea = pred(a_batch_2d)
        
        if args.vae_bias_pred_mode == 'nn':
            return nn_pred_result.view(-1,1).to(args.device)
        
        elif args.vae_bias_pred_mode == 'gt_nn':
            a_batch_vec = torch.argmax(a_batch, dim=-1).long()
            # calculate similarity and find the target a, to replace the bias value
            a_batch_vec = a_batch_vec.unsqueeze(1)
            dist = torch.abs(a_batch_vec - a_bank).sum(-1)
            batch_min_dist, batch_min_alist_index = torch.min(dist, axis=-1)
            replace_index = torch.where(batch_min_dist == 0)
            nn_pred_result[replace_index] = gp_bias_bank[batch_min_alist_index[replace_index]].view(-1)

            return torch.tensor(nn_pred_result).view(-1,1).to(args.device)
        
        elif args.vae_bias_pred_mode == 'gt':
            a_batch_vec = torch.argmax(a_batch, dim=-1).long()
            # calculate similarity and find the target a, to replace the bias value
            a_batch_vec = a_batch_vec.unsqueeze(1)
            dist = torch.abs(a_batch_vec - a_bank).sum(-1)
            batch_min_dist, batch_min_alist_index = torch.min(dist, axis=-1)
            gt_pred_result = gp_bias_bank[batch_min_alist_index]

            return torch.tensor(gt_pred_result).view(-1,1).to(args.device)
        else:
            return nn_pred_result

# ==============================================
# ============== Bias Vae Get Mask   ===========
# ==============================================
def get_mask(b, args, mask_type, dis_x=None, a_bank=None):
    if mask_type is None:
        mask = torch.ones_like(b)
    elif mask_type == 'bs':
        # Mask 1 - top batch size num
        _, topBs_idx = torch.topk(b.view(-1), args.batch_size)
        mask = torch.zeros_like(b)
        mask[topBs_idx] = 1
    elif mask_type == 'bs_trA' and a_bank is not None:
        # mask for A in tr
        a_batch_vec = torch.argmax(dis_x, dim=-1).long()
        a_batch_vec = a_batch_vec.unsqueeze(1)
        dist = torch.abs(a_batch_vec - a_bank).sum(-1) # a_bank has been unsqueezed
        batch_min_dist, batch_min_alist_index = torch.min(dist, axis=-1)
        genA_in_Tr_index = torch.where(batch_min_dist == 0)
        mask = torch.zeros_like(b).squeeze()
        mask[genA_in_Tr_index] = 1.0
        # mask for bs
        _, topBs_idx = torch.topk(b.view(-1), int(args.mask_bs_coef * args.batch_size))
        mask_bs = torch.zeros_like(b)
        mask_bs[topBs_idx] = 1
        # merge
        mask = mask.unsqueeze(1) * mask_bs
    return mask.to(args.device)

# ============================================================================
# ============== Z based Sampling Process and Calculate for Loss   ===========
# ============================================================================
def get_z_based_loss(Model_en, Model_de, pred, args, a_bank, a_fea_bank, gp_bias_bank, mask_type, loss_type=None, R_global = 0):
    grad_activate(Model_de)
    grad_freeze(Model_en)
    z_samples = torch.randn(args.batch_size * args.z_resample, args.latent_dim).to(
        args.device
    )
    # sample softmax data (w.r.t. 3D tensor, normalized)
    hat_x = Model_de(z_samples)
    # sampling discrete variable (one-hot-3D tensor) batch_size \time variable_dim \time one_hot_code
    dis_x = sample_from_multi_softmax_distribution(
        hat_x, args.class_dim, normal=True, one_hot=True
    )  # bs * 20 * 2
    with torch.no_grad():
        # ============ MAIN R ============
        # estimating bias value through predictor model
        x_view = dis_x.view(-1, args.class_dim * args.output_dim)  # bs * 40
        b = get_bias(dis_x, pred, a_bank, a_fea_bank, gp_bias_bank, args)
        mask = get_mask(
            b,
            args,
            mask_type=mask_type,
            dis_x=dis_x.detach().clone(),
            a_bank=a_bank,
        )
        mu_phi, log_var = Model_en(x_view)
        negative_log_z_estimate = (
            0.5 * torch.norm(z_samples - mu_phi) ** 2 / torch.exp(log_var)
            + log_var
        )
        negative_log_z_estimate = (
            args.vae_bias_neglog * negative_log_z_estimate * mask
        )
        b = args.vae_bias_b * b * mask
        R = negative_log_z_estimate + b  # -b
        
        if loss_type == 'var_control':
            z_samples_ghost = torch.randn(args.batch_size, args.latent_dim).to(
                    args.device
                )
            hat_x_ghost = Model_de(z_samples_ghost)
            dis_x_ghost = sample_from_multi_softmax_distribution(
                hat_x_ghost, args.class_dim, normal=True, one_hot=True
            )
            dis_x_ghost_view = dis_x_ghost.view(-1, args.class_dim * args.output_dim)
            b_ghost, _, __ = pred(dis_x_ghost_view)
            mu_phi_ghost, log_var_ghost = Model_en(dis_x_ghost_view)
            negative_log_z_ghost = (
                0.5
                * torch.norm(z_samples_ghost - mu_phi_ghost) ** 2
                / torch.exp(log_var_ghost)
                + log_var_ghost
            )
            R_ghost = (
                args.vae_bias_neglog * negative_log_z_ghost + args.vae_bias_b * b_ghost
            )
            R_local = torch.mean(R_ghost)
            # here updating by moving average variable
            R_global = (
                args.vae_bias_r_local * R_local + args.vae_bias_r_global * R_global
            )
            R_final = (R - R_global) * mask
        else:
            R_final = R
    
    loss_main = (
            torch.dot(
                R_final.view(-1),
                compute_log_softmax_distribution(vector_proba=hat_x, vector_true=dis_x),
            )
            / args.batch_size
        )
    entropy_reg = args.vae_bias_reg_coef * compute_entropy(hat_x, mask=mask)
    loss_org = loss_main + entropy_reg
    
    if loss_type == "var_control":
        return loss_org, loss_main, entropy_reg, negative_log_z_estimate, b, R_final, R_global
    else:
        return loss_org, loss_main, entropy_reg, negative_log_z_estimate, b, R_final

# ====================================================================================
# ============== train set based Sampling Process and Calculate for Loss   ===========
# ====================================================================================
def get_trA_based_loss(loader_train, Model_en, Model_de, args):
    train_iter = iter(loader_train)
    try:
        _, __, a_one_hot, bias, bias_group, weight, cls_label = next(train_iter)
    except:
        train_iter = iter(loader_train)
        _, __, a_one_hot, bias, bias_group, weight, cls_label = next(train_iter)

    a_one_hot = a_one_hot.float().to(args.device)
    b_tr = bias_group.float().to(args.device).view(-1, 1)
    mu_tr, log_var_tr = Model_en(a_one_hot)
    z_tr = reparameterize_latent(mu_tr, log_var_tr, args)
    hat_x_tr = Model_de(z_tr)  # [batch, output_dim, class_dim]. after softmax
    dis_x_tr = sample_from_multi_softmax_distribution(
        hat_x_tr, args.class_dim, normal=True, one_hot=True
    )

    with torch.no_grad():
        # ============ MAIN R ============
        # estimating bias value through predictor model
        x_view_tr = dis_x_tr.view(-1, args.class_dim * args.output_dim)
        mask_tr = get_mask(b_tr, args, mask_type="bs")
        mu_phi_1_tr, log_var_1_tr = Model_en(x_view_tr)
        negative_log_z_estimate_tr = (
            0.5 * torch.norm(z_tr - mu_phi_1_tr) ** 2 / torch.exp(log_var_1_tr)
            + log_var_1_tr
        )
        negative_log_z_estimate_tr = (
            args.vae_bias_neglog * negative_log_z_estimate_tr * mask_tr
        )
        b_tr = args.vae_bias_b * b_tr * mask_tr
        R_tr = negative_log_z_estimate_tr + b_tr  # -b
    
    loss_main_tr = (
        torch.dot(
            R_tr.view(-1),
            compute_log_softmax_distribution(
                vector_proba=hat_x_tr, vector_true=dis_x_tr
            ),
        )
        / args.batch_size
    )
    entropy_reg_tr = args.vae_bias_reg_coef * compute_entropy(
        hat_x_tr, mask=mask_tr
    )
    loss_tr = loss_main_tr + entropy_reg_tr
    return loss_tr, loss_main_tr, entropy_reg_tr, negative_log_z_estimate_tr, b_tr, R_tr