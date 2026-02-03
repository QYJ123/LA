import numpy as np
import torch

from ..base_bbox_coder import BaseBBoxCoder
from mmdet.core.bbox.transforms_obb import obb2hbb, obb2poly, rectpoly2obb,poly2obb
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class GlidingVertexRpnCoders(BaseBBoxCoder):

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0., 0.,0.,0.),
                 target_stds=(1., 1., 1., 1., 1., 1.,1.,1.)):
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
                  means=(0., 0., 0., 0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1., 1., 1., 1.)):
    assert proposals.size(0) == gt.size(0)

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]
    
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

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    da = (ga - hbb[..., 0]) / gw
    db = (gb - hbb[..., 1]) / gh
    dc = da
    dd = db

    deltas = torch.stack([dx, dy, dw, dh, da, db, dc, dd], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta_sp2bbox(rois, deltas,
                  means=(0., 0., 0., 0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1., 1., 1., 1.),
                  wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 8)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 8)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::8]
    dy = denorm_deltas[:, 1::8]
    dw = denorm_deltas[:, 2::8]
    dh = denorm_deltas[:, 3::8]
    da = denorm_deltas[:, 4::8]
    db = denorm_deltas[:, 5::8]
    dc = denorm_deltas[:, 6::8]
    dd = denorm_deltas[:, 7::8]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy

    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    da = da.clamp(min=-0.5, max=0.5)
    db = db.clamp(min=-0.5, max=0.5)
    dc = dc.clamp(min=-0.5, max=0.5)
    dd = dd.clamp(min=-0.5, max=0.5)
    ga  = x1 + gw *da
    gb  = y1 + gh *db
    _ga = x1 + gw*(1-dc)
    _gb = y1 + gh*(1-dd)
    polys = torch.stack([ga, y1, x2, gb, _ga, y2, x1, _gb], dim=-1)
    
    obboxes = poly2obb(polys).flatten(-2)
    return obboxes
