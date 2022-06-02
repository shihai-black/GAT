# -*- coding: utf-8 -*-
# @author:caojinlei
# @file: utils.py
# @time: 2021/11/01
import torch.distributed as dist
import torch
from torch import nn
from torch.nn import functional as F


def time_diff(t_end, t_start):
    """
    计算时间差。t_end, t_start are datetime format, so use deltatime
    Parameters
    ----------
    t_end
    t_start

    Returns
    -------
    """
    diff_sec = (t_end - t_start).seconds
    diff_min, rest_sec = divmod(diff_sec, 60)
    diff_hrs, rest_min = divmod(diff_min, 60)
    return (diff_hrs, rest_min, rest_sec)


def cleanup():
    dist.destroy_process_group()


class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert 0 < alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] = (1 - alpha)

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def fill_zero(x, return_type=float):
    if x == '""':
        if return_type == float:
            return 0.0
        else:
            return 0
    else:
        if return_type == float:
            return float(x)
        else:
            return int(x)


if __name__ == '__main__':
    pred = torch.randn((3, 5))
    print(f"pred:{pred}")

    label = torch.tensor([2, 3, 4])
    print(f"label: {label}")
    loss_fn = focal_loss(alpha=0.25, gamma=2, num_classes=5)
    loss = loss_fn(pred, label)
    print(loss)

    loss_fn2 = focal_loss(alpha=[1, 2, 3, 3, 0.25], gamma=2, num_classes=5)
    loss2 = loss_fn2(pred, label)
    print("cpu loss2 --->", loss2)  #
