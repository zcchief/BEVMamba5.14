import copy
import random
from os import path as osp
from typing import List, Optional, Union

import numpy as np
import torch
from mmengine.fileio import load
from mmdet.registry import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from mmdet3d.registry import DATASETS as MMDET3D_DATASETS
from .nuscnes_eval import NuScenesEval_custom

@MMDET3D_DATASETS.register_module()
@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    """NuScenes Dataset (MMEngine-compatible).

    This dataset supports temporal queue loading and custom evaluation.
    """

    def __init__(self,
                 queue_length: int = 4,
                 bev_size: tuple = (200, 200),
                 overlap_test: bool = False,
                 *args,
                 **kwargs):
        # mmdet3d 1.x compatibility: legacy configs pass `classes=...`.
        # In newer MMEngine-style datasets this should be in `metainfo`.
        classes = kwargs.pop('classes', None)
        if classes is not None:
            metainfo = copy.deepcopy(kwargs.get('metainfo', {}))
            metainfo.setdefault('classes', classes)
            kwargs['metainfo'] = metainfo


        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.bev_size = bev_size
        self.overlap_test = overlap_test
        if not hasattr(self, 'CLASSES'):
            self.CLASSES = self.metainfo.get('classes', None)

    def load_data_list(self):
        """Load and parse annotation file with backward compatibility.

        Supported formats:
        1) MMEngine new format:
           {'metainfo': {...}, 'data_list': [...]}
        2) Legacy format:
           {'metadata': {...}, 'infos': [...]}
        3) Legacy pure list:
           [info0, info1, ...]

        Returns:
            List[dict]: parsed data list (each item is parse_data_info output).
        """
        ann = load(self.ann_file)

        def _merge_file_meta(file_meta):
            if isinstance(file_meta, dict):
                for k, v in file_meta.items():
                    self.metainfo.setdefault(k, v)

        def _ensure_list(x, name):
            if not isinstance(x, list):
                raise TypeError(f'Expected {name} to be list, got {type(x)}')
            return x

        def _parse_all(raw_list):
            parsed = []
            for i, raw in enumerate(raw_list):
                if raw is None:
                    continue
                try:
                    item = self.parse_data_info(raw)
                except Exception as e:
                    raise RuntimeError(
                        f'Failed to parse ann item at index {i} from {self.ann_file}: {e}'
                    ) from e
                if item is not None:
                    parsed.append(item)
            return parsed

        # Case 1: new MMEngine format
        if isinstance(ann, dict) and 'data_list' in ann:
            _merge_file_meta(ann.get('metainfo', {}))
            raw_list = _ensure_list(ann['data_list'], 'ann["data_list"]')
            return _parse_all(raw_list)

        # Case 2: legacy dict format
        if isinstance(ann, dict) and 'infos' in ann:
            _merge_file_meta(ann.get('metadata', {}))
            raw_list = _ensure_list(ann['infos'], 'ann["infos"]')
            return _parse_all(raw_list)

        # Case 3: legacy pure list
        if isinstance(ann, list):
            return _parse_all(ann)

        raise ValueError(
            'Unsupported annotation format in '
            f'{self.ann_file}. Expected one of: '
            '{metainfo,data_list}, {metadata,infos}, or list.'
        )

    def parse_data_info(self, info: dict) -> dict:
        """Parse raw info and enrich temporal/camera metadata."""
        info = copy.deepcopy(info)

        if 'ego2global_translation' not in info or 'ego2global_rotation' not in info:
            ego2global = info.get('ego2global', {})
            if isinstance(ego2global, dict):
                info.setdefault('ego2global_translation', ego2global.get('translation'))
                info.setdefault('ego2global_rotation', ego2global.get('rotation'))

        if 'lidar2ego_translation' not in info or 'lidar2ego_rotation' not in info:
            lidar2ego = info.get('lidar2ego', {})
            if isinstance(lidar2ego, dict):
                info.setdefault('lidar2ego_translation', lidar2ego.get('translation'))
                info.setdefault('lidar2ego_rotation', lidar2ego.get('rotation'))

        # 兼容旧 nuScenes info：cams -> images
        if 'images' not in info and 'cams' in info:
            info['images'] = {}
            for cam_name, cam_info in info['cams'].items():
                info['images'][cam_name] = dict(
                    img_path=cam_info.get('data_path', ''),
                    cam2img=cam_info.get('cam_intrinsic', None),
                    sensor2lidar_rotation=cam_info.get('sensor2lidar_rotation', None),
                    sensor2lidar_translation=cam_info.get('sensor2lidar_translation', None),
                )

        if 'cams' not in info and 'images' in info:
            info['cams'] = {}
            for cam_name, img_info in info['images'].items():
                cam2img = img_info.get('cam2img', None)
                if isinstance(cam2img, list):
                    cam2img = np.asarray(cam2img)
                info['cams'][cam_name] = dict(
                    data_path=img_info.get('img_path', ''),
                    cam_intrinsic=cam2img,
                    sensor2lidar_rotation=img_info.get('sensor2lidar_rotation', None),
                    sensor2lidar_translation=img_info.get('sensor2lidar_translation', None),
                )

        input_dict = super().parse_data_info(info)

        input_dict['ego2global_translation'] = info.get('ego2global_translation')
        input_dict['ego2global_rotation'] = info.get('ego2global_rotation')
        input_dict['prev_idx'] = info.get('prev', None)
        input_dict['next_idx'] = info.get('next', None)
        input_dict['scene_token'] = info.get('scene_token')
        input_dict['can_bus'] = info.get('can_bus', None)
        input_dict['frame_idx'] = info.get('frame_idx', None)
        timestamp = info.get('timestamp', 0)
        input_dict['timestamp'] = timestamp / 1e6 if timestamp > 1e5 else timestamp


        if self.modality.get('use_camera', False):
            cam_intrinsics = []
            lidar2cam_rts = []
            images_dict = {}
            lidar2img = []
            for cam_name, cam_info in info['cams'].items():
                # 优先使用已存在的 lidar2cam 字段（可以与新格式 pkl 兼容）
                if 'lidar2cam' in cam_info:
                    lidar2cam_data = cam_info['lidar2cam']
                    if isinstance(lidar2cam_data, dict):
                        # 字典形式：{'rotation': ..., 'translation': ...}
                        rotation = np.array(lidar2cam_data['rotation'])
                        translation = np.array(lidar2cam_data['translation'])
                        lidar2cam_r = np.linalg.inv(rotation)
                        lidar2cam_t = translation @ lidar2cam_r.T
                        lidar2cam_rt = np.eye(4)
                        lidar2cam_rt[:3, :3] = lidar2cam_r.T
                        lidar2cam_rt[3, :3] = -lidar2cam_t
                        lidar2cam_matrix = lidar2cam_rt.T
                    else:
                        lidar2cam_matrix = np.array(lidar2cam_data)
                else:
                    # 回退到从 sensor2lidar 计算（原始逻辑）
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    lidar2cam_matrix = lidar2cam_rt.T

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

                images_dict[cam_name] = {
                    'img_path': cam_info['data_path'],
                    'cam2img': cam_info['cam_intrinsic'],
                    'lidar2cam': lidar2cam_matrix,
                }
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_matrix)


            for cam_intrinsic, lidar2cam in zip(cam_intrinsics, lidar2cam_rts):
                # 注意乘法顺序：一般 lidar2img = intrinsic @ lidar2cam
                # 具体根据你的坐标系可能需调整，但 BEVFormer 官方代码通常为 intrinsic @ lidar2cam
                lidar2img.append(cam_intrinsic @ lidar2cam)

            input_dict['lidar2img'] = lidar2img
            input_dict['images'] = images_dict
            input_dict['cam_intrinsic'] = cam_intrinsics
            input_dict['lidar2cam'] = lidar2cam_rts

        if input_dict.get('can_bus') is not None :
            rotation = Quaternion(input_dict['ego2global_rotation'])
            translation = input_dict['ego2global_translation']
            can_bus = input_dict['can_bus']
            can_bus[:3] = translation
            can_bus[3:7] = rotation
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle

        return input_dict

    def prepare_data(self, idx: int) -> Union[dict, None]:
        """Prepare data for training or testing."""
        if self.test_mode:
            return super().prepare_data(idx)

        queue = []
        index_list = list(range(idx - self.queue_length, idx))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(idx)

        prev_scene_token = None
        prev_pos = None
        prev_angle = None

        for i in index_list:
            i = max(0, i)
            data_info = self.get_data_info(i)
            if data_info is None:
                return None

            data = self.pipeline(data_info)
            if data is None:
                return None

            if self.filter_empty_gt:
                data_sample = data.get('data_samples', None)
                gt_instances_3d = getattr(data_sample, 'gt_instances_3d', None)
                labels_3d = getattr(gt_instances_3d, 'labels_3d', None)
                if labels_3d is not None and not (labels_3d != -1).any():
                    return None

            data_sample = data['data_samples']
            scene_token = data_sample.metainfo.get('scene_token', None) if hasattr(data_sample, 'metainfo') else None

            if scene_token != prev_scene_token:
                data_sample.set_metainfo({'prev_bev_exists': False})
                prev_scene_token = scene_token
                prev_pos = copy.deepcopy(data_sample.metainfo['can_bus'][:3])
                prev_angle = copy.deepcopy(data_sample.metainfo['can_bus'][-1])
                # can_bus 的相对位姿置零操作也需要更新回 metainfo
                data_sample.metainfo['can_bus'][:3] = 0
                data_sample.metainfo['can_bus'][-1] = 0
            else:
                data_sample.set_metainfo({'prev_bev_exists': True})
                tmp_pos = copy.deepcopy(data_sample.metainfo['can_bus'][:3])
                tmp_angle = copy.deepcopy(data_sample.metainfo['can_bus'][-1])
                data_sample.metainfo['can_bus'][:3] -= prev_pos
                data_sample.metainfo['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

            queue.append(data)
            #print("=== After append, data's prev_bev_exists:",
                  #data['data_samples'].metainfo.get('prev_bev_exists'))

        return self._fuse_queue_data(queue)

    def _fuse_queue_data(self, queue: List[dict]) -> Optional[dict]:
        """Fuse queued frames into one temporal training sample."""
        if not queue:
            return None

        imgs_list = []
        for data in queue:
            inputs = data['inputs']
            img=inputs['img'] if isinstance(inputs, dict) else inputs
            if isinstance(img,list):
                img=torch.stack([torch.as_tensor(x) for x in img])
            else:
                img = torch.as_tensor(img)
            imgs_list.append(img)

        fused_data = queue[-1].copy()
        if isinstance(fused_data['inputs'], dict):
            fused_data['inputs'] = fused_data['inputs'].copy()
            fused_data['inputs']['img'] = torch.stack(imgs_list)
        else:
            fused_data['inputs'] = torch.stack(imgs_list)

        queue_metas = []
        for i, data in enumerate(queue):
            data_sample = data['data_samples']
            meta = copy.deepcopy(data_sample.metainfo if hasattr(data_sample, 'metainfo') else data_sample)
            meta['queue_idx'] = i
            queue_metas.append(meta)

        fused_data['data_samples'].set_metainfo({'queue_metas': queue_metas})
        return fused_data

    def __getitem__(self, idx: int) -> dict:
        """Get one sample; retry when filtered out in training mode."""
        if self.test_mode:
            return self.prepare_data(idx)

        while True:
            data = self.prepare_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _evaluate_single(self,
                         result_path: str,
                         logger: Optional[object] = None,
                         metric: str = 'bbox',
                         result_name: str = 'pts_bbox') -> dict:
        """Evaluation for a single model in nuScenes protocol."""
        from nuscenes import NuScenes

        self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
                             verbose=True)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }

        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos)
        self.nusc_eval.main(plot_examples=0, render_curves=False)

        metrics = load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = float(f'{v:.4f}')
            for k, v in metrics['label_tp_errors'][name].items():
                detail[f'{metric_prefix}/{name}_{k}'] = float(f'{v:.4f}')
            for k, v in metrics['tp_errors'].items():
                detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = float(f'{v:.4f}')
        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
        return detail
