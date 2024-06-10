import torch
import torch.nn.functional as F
import torch.nn as nn

'''
https://github.com/YyzHarry/imbalanced-regression
'''
def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def cross_entropy(targets, output, weight, args, reduction='mean', smooth=0.1):
    if targets.size() != output.size():
        ones = torch.eye(args.pred_cls_num, device=output.device)
        targets_2d = torch.index_select(ones, dim=0, index=targets)
    else:
        targets_2d = targets
    
    pred = nn.Softmax(dim=-1)(output)
    
    if smooth > 0:
        targets_2d = (1 - smooth) * targets_2d + smooth / args.pred_cls_num
    
    if reduction == 'none':
        return torch.sum(- targets_2d * torch.log(pred + 1e-5) * weight.unsqueeze(-1), 1)
    elif reduction == 'mean':
        return torch.mean(torch.sum(- targets_2d * torch.log(pred + 1e-5) * weight.unsqueeze(-1), 1))