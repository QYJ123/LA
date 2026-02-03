import numpy as np
import torch

from ..base_bbox_coder import BaseBBoxCoder
from mmdet.core.bbox.transforms_obb import obb2hbb, obb2poly, rectpoly2obb
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class MidpointOffsetheadCoder(BaseBBoxCoder):

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        encoded_bboxes = bbox2delta_sp(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta_sp2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                       wh_ratio_clip)
        return decoded_bboxes


def bbox2delta_sp(proposals, gt,
                  means=(0., 0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1., 1.)):
    assert proposals.size(0) == gt.size(0)

    proposals = proposals.float()
    gt = gt.float()

    ptheta = proposals[..., 4]
    
    proposal_hbb, proposal_poly = obb2hbb(proposals), obb2poly(proposals)
    px = (proposal_hbb[..., 0] + proposal_hbb[..., 2]) * 0.5
    py = (proposal_hbb[..., 1] + proposal_hbb[..., 3]) * 0.5
    pw = proposal_hbb[..., 2] - proposal_hbb[..., 0]
    ph = proposal_hbb[..., 3] - proposal_hbb[..., 1]

    x_coor, y_coor = proposal_poly[:, 0::2], proposal_poly[:, 1::2]
    y_min, _ = torch.min(y_coor, dim=1, keepdim=True)
    x_max, _ = torch.max(x_coor, dim=1, keepdim=True)

    _x_coor = x_coor.clone()
    _x_coor[torch.abs(y_coor-y_min) > 0.1] = -1000
    pa, _ = torch.max(_x_coor, dim=1)

    _y_coor = y_coor.clone()
    _y_coor[torch.abs(x_coor-x_max) > 0.1] = -1000
    pb, _ = torch.max(_y_coor, dim=1)
    
    hbb, poly = obb2hbb(gt), obb2poly(gt)
    gx = (hbb[..., 0] + hbb[..., 2]) * 0.5
    gy = (hbb[..., 1] + hbb[..., 3]) * 0.5
    gw = hbb[..., 2] - hbb[..., 0]
    gh = hbb[..., 3] - hbb[..., 1]

    x_coor, y_coor = poly[:, 0::2], poly[:, 1::2]
    y_min, _ = torch.min(y_coor, dim=1, keepdim=True)
    x_max, _ = torch.max(x_coor, dim=1, keepdim=True)

    _x_coor = x_coor.clone()
    _x_coor[torch.abs(y_coor-y_min) > 0.1] = -1000
    ga, _ = torch.max(_x_coor, dim=1)

    _y_coor = y_coor.clone()
    _y_coor[torch.abs(x_coor-x_max) > 0.1] = -1000
    gb, _ = torch.max(_y_coor, dim=1)

    dx = (torch.cos(-ptheta)*(gx-px)+torch.sin(-ptheta)*(gy-py)) / pw
    dy = (-torch.sin(-ptheta)*(gx-px)+torch.cos(-ptheta)*(gy-py)) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    da = (ga - gx) / gw-((pa - px) / pw)
    db = (gb - gy) / gh-((pb - py) / ph)
    
    deltas = torch.stack([dx, dy, dw, dh, da, db], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta_sp2bbox(rois, deltas,
                  means=(0., 0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1., 1.),
                  wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 6)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 6)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::6]
    dy = denorm_deltas[:, 1::6]
    dw = denorm_deltas[:, 2::6]
    dh = denorm_deltas[:, 3::6]
    da = denorm_deltas[:, 4::6]
    db = denorm_deltas[:, 5::6]
   
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    proposals = rois
    ptheta = (rois[..., 4]).unsqueeze(1)
    proposal_hbb, proposal_poly = obb2hbb(proposals), obb2poly(proposals)
    px = ((proposal_hbb[..., 0] + proposal_hbb[..., 2]) * 0.5).unsqueeze(1)
    py = ((proposal_hbb[..., 1] + proposal_hbb[..., 3]) * 0.5).unsqueeze(1)
    pw = (proposal_hbb[..., 2] - proposal_hbb[..., 0]).unsqueeze(1)
    ph = (proposal_hbb[..., 3] - proposal_hbb[..., 1]).unsqueeze(1)

    x_coor, y_coor = proposal_poly[:, 0::2], proposal_poly[:, 1::2]
    y_min, _ = torch.min(y_coor, dim=1, keepdim=True)
    x_max, _ = torch.max(x_coor, dim=1, keepdim=True)

    _x_coor = x_coor.clone()
    _x_coor[torch.abs(y_coor-y_min) > 0.1] = -1000
    pa, _ = torch.max(_x_coor, dim=1)

    _y_coor = y_coor.clone()
    _y_coor[torch.abs(x_coor-x_max) > 0.1] = -1000
    pb, _ = torch.max(_y_coor, dim=1)
    pa = pa.unsqueeze(1)
    pb = pb.unsqueeze(1)
    
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi

    gx = dx*pw*torch.cos(-ptheta) - dy*ph*torch.sin(-ptheta) + px
    gy = dx*pw*torch.sin(-ptheta) + dy*ph*torch.cos(-ptheta) + py

    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    
    da = (da+((pa - px) / pw)).clamp(min=-0.5, max=0.5)
    db = (db+((pb - py) / ph)).clamp(min=-0.5, max=0.5)
    
    ga = gx + da * gw
    _ga = gx - da * gw
    gb = gy + db * gh
    _gb = gy - db * gh
    polys = torch.stack([ga, y1, x2, gb, _ga, y2, x1, _gb], dim=-1)
    
    center = torch.stack([gx, gy, gx, gy, gx, gy, gx, gy], dim=-1)
    center_polys = polys - center
    diag_len = torch.sqrt(
        torch.square(center_polys[..., 0::2]) + torch.square(center_polys[..., 1::2]))
    max_diag_len, _ = torch.max(diag_len, dim=-1, keepdim=True)
    diag_scale_factor = max_diag_len / diag_len
    center_polys = center_polys * diag_scale_factor.repeat_interleave(2, dim=-1)
    rectpolys = center_polys + center

    obboxes = rectpoly2obb(rectpolys).flatten(-2)

    return obboxes
