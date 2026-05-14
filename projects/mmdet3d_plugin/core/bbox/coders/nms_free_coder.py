import torch

from mmdet.models.task_modules import BaseBBoxCoder
from mmdet.registry import TASK_UTILS
from mmdet3d.registry import TASK_UTILS as MMDET3D_TASK_UTILS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

@MMDET3D_TASK_UTILS.register_module()
@TASK_UTILS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector."""

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes for one sample."""
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is None:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        post_center_range = torch.as_tensor(self.post_center_range, device=scores.device)
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(1)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(1)

        if self.score_threshold is not None:
            mask &= thresh_mask

        boxes3d = final_box_preds[mask]
        scores = final_scores[mask]
        labels = final_preds[mask]

        predictions_dict = {
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels
        }
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes for a batch."""
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]

        batch_size = all_cls_scores.size(0)
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list

