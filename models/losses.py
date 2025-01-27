#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from functools import partial
import torch
import torch.nn as nn

from pytorchvideo.losses.soft_target_cross_entropy import (
    SoftTargetCrossEntropyLoss,
)


class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, dummy_labels=None):
        targets = torch.zeros(inputs.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss(reduction=self.reduction).cuda()(
            inputs, targets
        )
        return loss


class MultipleMSELoss(nn.Module):
    """
    Compute multiple mse losses and return their average.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MultipleMSELoss, self).__init__()
        self.mse_func = nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        loss_sum = 0.0
        multi_loss = []
        for xt, yt in zip(x, y):
            if isinstance(yt, (tuple,)):
                if len(yt) == 2:
                    yt, wt = yt
                    lt = "mse"
                elif len(yt) == 3:
                    yt, wt, lt = yt
                else:
                    raise NotImplementedError
            else:
                wt, lt = 1.0, "mse"
            if lt == "mse":
                loss = self.mse_func(xt, yt)
            else:
                raise NotImplementedError
            loss_sum += loss * wt
            multi_loss.append(loss)
        return loss_sum, multi_loss


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        smooth_target = target * confidence + 0.5 * self.smoothing
        loss = nn.BCEWithLogitsLoss(reduction='none')(x, smooth_target)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class MarginRankingLoss(nn.Module):
    def __init__(self, margin=0.5, mode='inter'):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.mode = mode

    def forward(self, x, target):
        assert x.shape[:-1] == target.shape
        if self.mode == 'inter':
            return self.cal_inter(x, target)
        elif self.mode == 'intra':
            return self.cal_intra(x, target)
        else:
            raise NotImplementedError

    def cal_inter(self, x, target):
        "TransRank"
        margin_sum = 0
        margin_cnt = 0
        for vid in range(x.shape[0]):
            for tid in range(x.shape[2]):
                t_cid = torch.where(target[vid] == tid)
                for cid in range(x.shape[1]):
                    if cid == t_cid: continue
                    else:
                        loss = max(0, x[vid, cid, tid] - x[vid, t_cid, tid] + self.margin)
                        margin_sum = margin_sum + loss
                        margin_cnt = margin_cnt + 1
        return margin_sum / margin_cnt

    def cal_intra(self, x, target):
        margin_sum = 0
        margin_cnt = 0
        for vid in range(x.shape[0]):
            for cid in range(x.shape[1]):
                t_tid = target[vid, cid]
                for tid in range(x.shape[2]):
                    if tid == t_tid: continue
                    else:
                        loss = max(0, x[vid, cid, tid] - x[vid, cid, t_tid] + self.margin)
                        margin_sum = margin_sum + loss
                        margin_cnt = margin_cnt + 1
        return margin_sum / margin_cnt


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "smoothing_bce_logit": LabelSmoothingBCEWithLogitsLoss,
    "soft_cross_entropy": partial(
        SoftTargetCrossEntropyLoss, normalize_targets=False
    ),
    "contrastive_loss": ContrastiveLoss,
    "mse": nn.MSELoss,
    "multi_mse": MultipleMSELoss,
    "margin": MarginRankingLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
