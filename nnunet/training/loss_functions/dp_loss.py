#    Copyright 2022 Insitute of Medical Informatics, Luebeck
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
import torch.nn as nn
import numpy as np

class MultipleOutputLoss2_FlexibleArgs(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2_FlexibleArgs, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y, *args):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0], *args)
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i], *args)
        return l

def ce_loss_separate_classes(logits, targets):
    targets = torch.nn.functional.one_hot(targets).permute(0,3,1,2)
    loss = -targets*nn.functional.log_softmax(logits, 1)
    # return loss.sum(dim=1).mean() # This would return conventional CE loss term
    return loss

class DPLoss(nn.Module):
    def __init__(self, threeD, num_classes):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super().__init__()
        self.n_dims =  (-3,-2,-1) if threeD else (-2,-1)
        self.threeD = threeD
        self.num_classes = num_classes

    def get_fixed_weighting(self, target):
        gt_num = torch.zeros(target.shape[0], self.num_classes, device=target.device)
        for bt_idx, batch_member in enumerate(target):
            class_ids, counts = batch_member.argmax(0).unique(return_counts=True)
            gt_num[bt_idx, class_ids] = counts.float()
        return (gt_num+np.exp(1)).log()+np.exp(1)

    def get_p_pred_num(self, net_output):
        # Return positive predicted count per class
        p_pred_num = torch.zeros(net_output.shape[0], self.num_classes, device=net_output.device)
        for bt_idx, batch_member in enumerate(net_output):
            class_ids, counts = batch_member.argmax(0).unique(return_counts=True)
            p_pred_num[bt_idx, class_ids] = counts.float()
        return p_pred_num

    def forward(self, net_output, target, bare_weight):
        p_pred_num = self.get_p_pred_num(net_output)
        one_hot_target = torch.nn.functional.one_hot(target.squeeze().long(), num_classes=self.num_classes).permute(0,3,1,2)
        dp_loss = ce_loss_separate_classes(net_output, target.squeeze().long())
        dp_loss = dp_loss.mean(self.n_dims)

        weight = torch.sigmoid(bare_weight)
        weight = weight/weight.mean(dim=0)
        weight = weight/self.get_fixed_weighting(target)

        if not self.threeD:
            risk_regularization = -weight*p_pred_num/(net_output.shape[-2]*net_output.shape[-1])
        else:
            risk_regularization = -weight*p_pred_num/(net_output.shape[-3]*net_output.shape[-2]*net_output.shape[-1])

        return (dp_loss*weight)[:,1:].sum() + risk_regularization[:,1:].sum()