import torch
import torch.nn as nn
import math
from ..builder import LOSSES
from .utils import weighted_loss

from .smooth_l1_loss import smooth_l1_loss,l1_loss


@weighted_loss
def diag_loss(pred,target,anchors):
    """L1 loss

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    pw = anchors[:, 2]-anchors[:, 0]
    ph = anchors[:, 3]-anchors[:, 1]
    assert pred.size() == target.size() and target.numel() > 0

    #D1 = (0.5*pw*pred[:,0].exp())**2+(ph*pred[:,1].exp()*pred[:,3]/2)**2,std effect del 0.5;
    D1 = ((pw*pred[:,0].exp())**2+(ph*pred[:,3]*pred[:,1].exp())**2).sqrt()
    D2 = ((ph*pred[:,1].exp())**2+(pw*pred[:,2]*pred[:,0].exp())**2).sqrt()
    Dg = ((pw*target[:,0].exp())**2+(ph*target[:,3]*target[:,1].exp())**2).sqrt()
    loss = ((D1-Dg)**2+(D2-Dg)**2)/Dg**2
    
    return loss

@LOSSES.register_module()
class SmoothL1IMPROVEDLoss(nn.Module):
    """Smooth L1 loss

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0, w=0.1):
        super(SmoothL1IMPROVEDLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.w =w

    def forward(self,
                pred,
                target,
                weight=None,
                anchors=None,
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

        loss_bbox = self.loss_weight*smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        
        loss_diagonal_line = self.loss_weight*diag_loss(
            pred[:,2:6],
            target[:,2:6],
            anchors= anchors,
            weight= weight[:, 2:6].max(dim=1)[0],
            reduction= reduction,
            avg_factor= avg_factor,**kwargs)

        loss = self.w*loss_diagonal_line+loss_bbox
        
        return loss
