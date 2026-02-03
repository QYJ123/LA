import torch
import torch.nn as nn
import math
from ..builder import LOSSES
from .utils import weighted_loss

from .smooth_l1_loss import smooth_l1_loss,l1_loss

@weighted_loss
def l1_cp_loss(pred,target,beta=1.0):
    """L1 loss

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    e_center = ((pred[:,0]-target[:,0])**2+(pred[:,1]-target[:,1])**2).sqrt()
    e_shape = (pred[:,2]-target[:,2]).abs()+(pred[:,3]-target[:,3]).abs()+(pred[:,4]-target[:,4]).abs()
    loss = (e_center**2+1)*(e_shape**2+1)-1

    return loss





@LOSSES.register_module()
class SmoothL1CPLoss(nn.Module):
    """Smooth L1 loss

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1CPLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_cp_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        
        total_loss = loss_bbox

        return total_loss

