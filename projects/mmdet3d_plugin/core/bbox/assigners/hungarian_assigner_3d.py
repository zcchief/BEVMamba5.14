import torch
from scipy.optimize import linear_sum_assignment
from mmengine.structures import InstanceData
from mmdet.models.task_modules import AssignResult, BaseAssigner
from mmdet.registry import TASK_UTILS
from mmdet3d.registry import TASK_UTILS as MMDET3D_TASK_UTILS
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

@MMDET3D_TASK_UTILS.register_module()
@TASK_UTILS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pc_range=None):
        try:
            self.cls_cost = MMDET3D_TASK_UTILS.build(cls_cost)
            self.reg_cost = MMDET3D_TASK_UTILS.build(reg_cost)
            self.iou_cost = MMDET3D_TASK_UTILS.build(iou_cost)
        except Exception:
            self.cls_cost = TASK_UTILS.build(cls_cost)
            self.reg_cost = TASK_UTILS.build(reg_cost)
            self.iou_cost = TASK_UTILS.build(iou_cost)

        self.pc_range = pc_range

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs."""
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        if gt_labels.device != bbox_pred.device:
            gt_labels = gt_labels.to(bbox_pred.device)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute weighted costs
        try:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
        except Exception:
            pred_instances = InstanceData(scores=cls_pred)
            gt_instances = InstanceData(labels=gt_labels)
            cls_cost = self.cls_cost(pred_instances, gt_instances)
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        try:
            reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
        except Exception:
            pred_instances = InstanceData(bboxes=bbox_pred[:, :8])
            gt_instances = InstanceData(bboxes=normalized_gt_bboxes[:, :8])
            reg_cost = self.reg_cost(pred_instances, gt_instances)

        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU
        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
