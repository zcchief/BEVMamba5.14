import torch
from mmdet3d.registry import TASK_UTILS as MMDET3D_TASK_UTILS
from mmdet.registry import TASK_UTILS
from mmengine.structures import InstanceData


def _unwrap_to_tensor(x, field_candidates=('bboxes', 'scores', 'labels')):
    """Accept Tensor or InstanceData and return tensor payload."""
    if torch.is_tensor(x):
        return x
    if isinstance(x, InstanceData):
        for name in field_candidates:
            v = getattr(x, name, None)
            if v is not None:
                return v
    raise TypeError(f'Unsupported input type for match cost: {type(x)}')


@MMDET3D_TASK_UTILS.register_module()
@TASK_UTILS.register_module()
class BBox3DL1Cost:
    """BBox3DL1Cost.

    Args:
        weight (int | float, optional): loss_weight.
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """Compute pairwise L1 cost matrix."""
        bbox_pred = _unwrap_to_tensor(bbox_pred, ('bboxes', 'scores'))
        gt_bboxes = _unwrap_to_tensor(gt_bboxes, ('bboxes',))
        if gt_bboxes.device != bbox_pred.device:
            gt_bboxes = gt_bboxes.to(bbox_pred.device)

        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss."""
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss.sum(-1)

@MMDET3D_TASK_UTILS.register_module()
@TASK_UTILS.register_module()
class SmoothL1Cost:
    """SmoothL1Cost.

    Args:
        weight (int | float, optional): loss weight.
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, input, target):
        """Compute pairwise smooth L1 cost matrix."""
        input = _unwrap_to_tensor(input, ('bboxes', 'scores'))
        target = _unwrap_to_tensor(target, ('bboxes',))
        if target.device != input.device:
            target = target.to(input.device)
        n1, c = input.shape
        n2, c = target.shape
        input = input.contiguous().view(n1, c)[:, None, :]
        target = target.contiguous().view(n2, c)[None, :, :]
        cost = smooth_l1_loss(input, target)
        return cost * self.weight
