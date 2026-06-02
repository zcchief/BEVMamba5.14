import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.structures import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import copy
from projects.mmdet3d_plugin.bevformer.modules.Augment_Temporal import Temporal_bev_queue, Mamba3DModel
import inspect
from mmdet.registry import MODELS as MMDET_MODELS
import os


for _name in ('ResNet', 'FPN'):
    if _name not in MODELS.module_dict and _name in MMDET_MODELS.module_dict:
        MODELS.register_module(name=_name, module=MMDET_MODELS.module_dict[_name])

@MMDET_MODELS.register_module()
@MODELS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 use_bev_queue=False,
                 bev_queue_length=3,
                 bev_queue_rotate_center=None,
                 use_mamba_temporal_aug=False,
                 mamba_warmup_frames=3,
                 mamba_temporal_cfg=None
                 ):

        if pretrained is not None and isinstance(img_backbone, dict):
            img_backbone = copy.deepcopy(img_backbone)
            if isinstance(pretrained, dict) and pretrained.get('img', None) is not None:
                img_backbone.setdefault(
                    'init_cfg', dict(type='Pretrained', checkpoint=pretrained['img']))
            elif isinstance(pretrained, str):
                img_backbone.setdefault(
                    'init_cfg', dict(type='Pretrained', checkpoint=pretrained))

        init_kwargs = dict(
            pts_voxel_layer=pts_voxel_layer,
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            img_neck=img_neck,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        # Avoid forwarding legacy optional args that are not configured.
        # In newer mmdet3d versions these `None` keys can be propagated to
        # Base3DDetector and trigger unexpected keyword errors.
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        # Filter kwargs by current MVXTwoStageDetector signature to support
        # both old and new mmdet3d versions.
        sig = inspect.signature(MVXTwoStageDetector.__init__)
        if pretrained is not None and 'pretrained' in sig.parameters:
            init_kwargs['pretrained'] = pretrained
        supports_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if supports_var_kw:
            super(BEVFormer, self).__init__(**init_kwargs)
        else:
            valid_kwargs = {
                k: v for k, v in init_kwargs.items() if k in sig.parameters and k != 'self'
            }
            super(BEVFormer, self).__init__(**valid_kwargs)

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.use_bev_queue = use_bev_queue
        self.use_mamba_temporal_aug = use_mamba_temporal_aug
        self.mamba_warmup_frames = mamba_warmup_frames

        self.bev_temporal_queue = None
        if self.use_bev_queue:
            embed_dims = getattr(self.pts_bbox_head, 'embed_dims', 256)
            self.bev_temporal_queue = Temporal_bev_queue(
                mixer_cls=lambda _dim: nn.Identity(),
                max_length=bev_queue_length,
                rotate_center=bev_queue_rotate_center,
                rotate_prev_bev=True,
                embed_dims=embed_dims,
                bev_h=self.pts_bbox_head.bev_h,
                bev_w=self.pts_bbox_head.bev_w,
            )

        self.temporal_mamba = None
        if self.use_mamba_temporal_aug:
            cfg = dict(bev_h=self.pts_bbox_head.bev_h, bev_w=self.pts_bbox_head.bev_w)
            if mamba_temporal_cfg is not None:
                cfg.update(mamba_temporal_cfg)
            self.temporal_mamba = Mamba3DModel(**cfg)

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def _to_hwbs(self, bev):
        """Convert BEV to [hw, bs, c] for temporal queue processing."""
        bev_h = self.pts_bbox_head.bev_h
        bev_w = self.pts_bbox_head.bev_w
        hw = bev_h * bev_w
        if bev.dim() != 3:
            raise ValueError(f'Expected 3D BEV tensor, got shape {tuple(bev.shape)}.')
        if bev.shape[0] == hw:
            return bev, 'hwbs'
        if bev.shape[1] == hw:
            return bev.permute(1, 0, 2).contiguous(), 'bshwc'
        raise ValueError(f'Unsupported BEV shape {tuple(bev.shape)} for bev_h*bev_w={hw}.')

    def _restore_prev_bev_format(self, bev_hwbs, src_format):
        """Restore BEV tensor format after temporal processing."""
        if src_format == 'hwbs':
            return bev_hwbs
        if src_format == 'bshwc':
            return bev_hwbs.permute(1, 0, 2).contiguous()
        raise ValueError(f'Unsupported source format: {src_format}.')

    def _build_temporal_aug_prev_bev(self, prev_bev, img_metas):
        """Update queue and optionally produce Mamba-enhanced prev_bev."""
        if not self.use_bev_queue or self.bev_temporal_queue is None or prev_bev is None:
            return prev_bev

        prev_bev_hwbs, src_format = self._to_hwbs(prev_bev)
        self.bev_temporal_queue.update(prev_bev_hwbs, img_metas)

        if (not self.use_mamba_temporal_aug) or self.temporal_mamba is None:
            return prev_bev

        if len(self.bev_temporal_queue.queue) < self.mamba_warmup_frames:
            return prev_bev

        aligned = self.bev_temporal_queue.get_aligned_queue(
            img_metas, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w)
        if len(aligned) == 0:
            return prev_bev

        merged = self.bev_temporal_queue.aligned_process(
            aligned,
            bev_h=self.pts_bbox_head.bev_h,
            bev_w=self.pts_bbox_head.bev_w,
            hybrid_positions=True
        )
        enhanced_prev_bev = self.temporal_mamba(merged, return_bev=True)
        enhanced_prev_bev = enhanced_prev_bev.permute(1, 0, 2).contiguous()  # [hw, bs, c]
        fused_prev_bev = enhanced_prev_bev + prev_bev_hwbs
        return self._restore_prev_bev_format(fused_prev_bev, src_format)

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self,
                inputs=None,
                data_samples=None,
                mode='tensor',
                return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if os.getenv('DEBUG_BEVFORMER_FILE', '0') == '1' and not hasattr(self, '_debug_file_printed'):
            print('[BEVFormer.debug] loaded from:', __file__)
            print('[BEVFormer.debug] env flags:',
                  'DEBUG_MMENGINE_MAP_GT3D=', os.getenv('DEBUG_MMENGINE_MAP_GT3D', ''),
                  'DEBUG_FORWARD_TRAIN_GT3D=', os.getenv('DEBUG_FORWARD_TRAIN_GT3D', ''))
            self._debug_file_printed = True

        if return_loss is not None:
            mode = 'loss' if return_loss else 'predict'

        if mode == 'loss':
            if inputs is not None or data_samples is not None:
                kwargs = self._format_mmengine_train_args(inputs, data_samples, **kwargs)
            return self.forward_train(**kwargs)
        if mode == 'predict':
            if inputs is not None:
                kwargs.setdefault('img', inputs['img'] if isinstance(inputs, dict) and 'img' in inputs else inputs)
            if data_samples is not None:
                kwargs.setdefault('img_metas', data_samples)
            return self.forward_test(**kwargs)
        if mode == 'tensor':
            if inputs is not None:
                kwargs.setdefault('img', inputs['img'] if isinstance(inputs, dict) and 'img' in inputs else inputs)
            return self.forward_dummy(kwargs.get('img'))
        raise RuntimeError(f'Invalid mode "{mode}".')

    @staticmethod
    def _format_mmengine_train_args(inputs, data_samples, **kwargs):
        """Convert MMEngine batch dict to BEVFormer legacy train kwargs."""
        if inputs is not None:
            kwargs.setdefault('img', inputs['img'] if isinstance(inputs, dict) and 'img' in inputs else inputs)
        if data_samples is None:
            return kwargs

        def _set_if_absent_or_none(key, value):
            if key not in kwargs or kwargs[key] is None:
                kwargs[key]=value

        img_metas = []
        gt_bboxes_3d, gt_labels_3d = [], []
        gt_bboxes, gt_labels, gt_bboxes_ignore = [], [], []

        for sample in data_samples:
            meta = sample.metainfo if hasattr(sample, 'metainfo') else sample
            queue_metas = meta.get('queue_metas', None) if isinstance(meta, dict) else None
            img_metas.append(queue_metas if queue_metas is not None else meta)

            if hasattr(sample, 'gt_instances_3d'):
                ins3d = sample.gt_instances_3d
                b3d = getattr(ins3d, 'bboxes_3d', None)
                if b3d is None:
                    b3d = getattr(ins3d, 'bboxes', None)
                l3d = getattr(ins3d, 'labels_3d', None)
                if l3d is None:
                    l3d = getattr(ins3d, 'labels', None)
                if b3d is not None:
                    gt_bboxes_3d.append(b3d)
                if l3d is not None:
                    gt_labels_3d.append(l3d)
            if hasattr(sample, 'gt_bboxes_3d') and not hasattr(sample, 'gt_instances_3d'):
                gt_bboxes_3d.append(sample.gt_bboxes_3d)
            if hasattr(sample, 'gt_labels_3d') and not hasattr(sample, 'gt_instances_3d'):
                gt_labels_3d.append(sample.gt_labels_3d)

            if hasattr(sample, 'gt_instances'):
                gt_bboxes.append(sample.gt_instances.bboxes)
                gt_labels.append(sample.gt_instances.labels)
            if hasattr(sample, 'ignored_instances'):
                gt_bboxes_ignore.append(sample.ignored_instances.bboxes)

        _set_if_absent_or_none('img_metas', img_metas)
        if gt_bboxes_3d:
                kwargs.setdefault('gt_bboxes_3d', gt_bboxes_3d)
        if gt_labels_3d:
                kwargs.setdefault('gt_labels_3d', gt_labels_3d)
        if gt_bboxes:
                kwargs.setdefault('gt_bboxes', gt_bboxes)
        if gt_labels:
                kwargs.setdefault('gt_labels', gt_labels)
        if gt_bboxes_ignore:
                kwargs.setdefault('gt_bboxes_ignore', gt_bboxes_ignore)
        if gt_bboxes_3d and not gt_labels_3d:
            raise ValueError(
                'Found gt_instances_3d but no 3D labels in data_samples. '
                'Expected one of: labels_3d or labels.')

        if os.getenv('DEBUG_MMENGINE_MAP_GT3D', '0') == '1':
            mapped = kwargs.get('gt_labels_3d', None)
            try:
                mapped_len = len(mapped) if mapped is not None else 'None'
            except Exception:
                mapped_len = 'N/A'
            has_none = isinstance(mapped, list) and any(x is None for x in mapped)
            print('[BEVFormer._format_mmengine_train_args] gt_labels_3d type=', type(mapped),
                  'len=', mapped_len, 'contains_none=', has_none,
                  'keys=', sorted(list(kwargs.keys())))

        return kwargs


    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively without Mamba enhancement.

        The temporal queue is filled with per-frame history BEV in sample order.
        Mamba fusion is deferred to current keyframe generation.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            if self.use_bev_queue and self.bev_temporal_queue is not None:
                self.bev_temporal_queue.reset()
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
                if self.use_bev_queue and self.bev_temporal_queue is not None and prev_bev is not None:
                    prev_bev_hwbs, _ = self._to_hwbs(prev_bev)
                    self.bev_temporal_queue.update(prev_bev_hwbs, img_metas)
            self.train()
            return prev_bev

    @staticmethod
    def _normalize_train_img(img):
        """Normalize train img to tensor shape [B, len_queue, num_cams, C, H, W]."""
        if isinstance(img, list):
            if len(img) == 0:
                raise ValueError('Received empty image list in forward_train.')
            if not all(isinstance(x, torch.Tensor) for x in img):
                raise TypeError('Expected each element in img list to be torch.Tensor.')
            img = torch.stack(img, dim=0)
        if not isinstance(img, torch.Tensor):
            raise TypeError(f'Expected img to be tensor or list[tensor], got {type(img)}.')
        if img.dim() != 6:
            raise ValueError(
                'Expected img with 6 dims [B, len_queue, num_cams, C, H, W], '
                f'got {tuple(img.shape)}. Please fix dataset/collate so len_queue is explicit.')
        if img.shape[-1] in (1, 3) and img.shape[3] not in (1, 3):
            img = img.permute(0, 1, 2, 5, 3, 4).contiguous()

        if img.shape[3] not in (1, 3):
            raise ValueError(
                'Expected channel dimension at index 3 to be 1 or 3 after normalization, '
                f'got shape {tuple(img.shape)}.')
        return img


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        img = self._normalize_train_img(img)
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        else:
            prev_bev = self._build_temporal_aug_prev_bev(prev_bev, img_metas)

        if os.getenv('DEBUG_FORWARD_TRAIN_GT3D', '0') == '1':
            has_none = isinstance(gt_labels_3d, list) and any(x is None for x in gt_labels_3d)
            try:
                gt_len = len(gt_labels_3d)
            except Exception:
                gt_len = 'N/A'
            print('[BEVFormer.forward_train] type(gt_labels_3d)=', type(gt_labels_3d),
                  'len(gt_labels_3d)=', gt_len,
                  'contains_none=', has_none)

        if isinstance(gt_labels_3d, list):
            norm_labels = []
            for lb in gt_labels_3d:
                if lb is None:
                    norm_labels.append(None)
                elif torch.is_tensor(lb):
                    norm_labels.append(lb)
                else:
                    norm_labels.append(torch.as_tensor(lb))
            gt_labels_3d = norm_labels

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        if self.use_bev_queue and self.bev_temporal_queue is not None:
            self.bev_temporal_queue.reset()
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
            if self.use_bev_queue:
                self.bev_temporal_queue.reset()
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None
            if self.use_bev_queue:
                self.bev_temporal_queue.reset()

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = self._build_temporal_aug_prev_bev(new_prev_bev, img_metas[0])
        return bbox_results

    def get_bev_queue(self, img_metas):
        """Return aligned BEV queue for the current timestamp."""
        return self.bev_temporal_queue.get_aligned_queue(
            img_metas, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w)

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list