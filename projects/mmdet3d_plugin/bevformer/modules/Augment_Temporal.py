from collections import deque
from einops import rearrange
import numpy as np
import torch
from pyquaternion import Quaternion
from torch import nn
from torchvision.transforms.functional import rotate
from mmcv.cnn.bricks.drop import build_dropout
from mamba_ssm import Mamba
from torch import Tensor
from typing import Optional
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from functools import partial
#from .ssm2d import Block2D,Mamba2D,SplitHead2D
Block2D = Mamba2D = SplitHead2D = nn.Identity
from einops import rearrange
from mmengine.logging import MMLogger
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmcv.cnn import build_norm_layer
from mmengine.utils import to_2tuple
from typing import Tuple


class Temporal_bev_queue(nn.Module):
    """Maintain a FIFO queue of BEV features and align them to current ego pose.

    This helper is designed for the BEVFormer pipeline: store the latest BEV
    features, rotate the previous 3 time steps to the current coordinate
    frame using NuScenes ego-pose (yaw) from ``img_metas``, and evict the
    oldest element when the queue is full.
    """

    def __init__(self,mixer_cls,
                 max_length=3,
                 rotate_center=None,
                 rotate_prev_bev=True,
                 embed_dims=None,
                 bev_h=None,
                 bev_w=None,
                 use_hybrid_pos_encoding=False,
                 temporal_mlp_hidden=128,
                 pose_mlp_hidden=128,
                 norm_cls=nn.LayerNorm, fused_add_norm=False,
                 residual_in_fp32=False,reverse=False,
                 transpose=False,split_head=False,
                drop_path_rate=0.0,drop_rate=0.0,use_mlp=False,):
        super().__init__()
        self.max_length = max_length
        self.rotate_center = rotate_center
        self.rotate_prev_bev = rotate_prev_bev
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.use_hybrid_pos_encoding = use_hybrid_pos_encoding
        if self.use_hybrid_pos_encoding:
            if self.embed_dims is None or self.bev_h is None or self.bev_w is None:
                raise ValueError('Hybrid positional encoding requires embed_dims, bev_h, and bev_w.')
            self.temporal_mlp = self._build_mlp(1, temporal_mlp_hidden, self.embed_dims)
            self.pose_mlp = self._build_mlp(16, pose_mlp_hidden, self.embed_dims)
        self.queue = deque()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(embed_dims)
        self.norm = norm_cls(embed_dims)
        self.split_head = split_head
        self.reverse = reverse
        self.transpose = transpose
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        )
        self.dropout = build_dropout(
            dict(type='Dropout', drop_prob=drop_rate)
        )
        if use_mlp:
            self.ffn = SwiGLUFFNFused(
                    embed_dims=embed_dims,
                    feedforward_channels=int(embed_dims*4),
                    layer_scale_init_value=0.0)
            self.ln2 = build_norm_layer(dict(type='LN'), embed_dims)
        else:
            self.ffn = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def reset(self):
        """Clear the queue."""
        self.queue.clear()

    def _extract_yaw(self, img_metas, device, dtype):
        yaw = [meta['can_bus'][-1] for meta in img_metas]
        return torch.as_tensor(yaw, device=device, dtype=dtype)

    def _extract_timestamps(self, img_metas, device, dtype):
        timestamps = [meta.get('timestamp', 0.0) for meta in img_metas]
        return torch.as_tensor(timestamps, device=device, dtype=dtype)

    def _extract_ego_pose(self, img_metas, device, dtype):
        poses = []
        for meta in img_metas:
            translation = np.array(meta.get('ego2global_translation', [0.0, 0.0, 0.0]), dtype=np.float32)
            rotation = meta.get('ego2global_rotation', [1.0, 0.0, 0.0, 0.0])
            rotation_matrix = Quaternion(rotation).rotation_matrix
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rotation_matrix
            pose[:3, 3] = translation
            poses.append(pose)
        poses = torch.as_tensor(np.stack(poses, axis=0), device=device, dtype=dtype)
        return poses

    def _build_mlp(self, in_dim, hidden_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )


    def get_aligned_queue(self, img_metas, bev_h, bev_w, return_hybrid_pos=False):
        """Rotate queued BEV features into the current coordinate frame.

        Args:
            img_metas (list[dict]): Metadata for the current time step.
            bev_h (int): BEV height.
            bev_w (int): BEV width.

        Returns:
            list[Tensor] | tuple[list[Tensor], list[Tensor]]: Aligned BEV
            features in FIFO order. If ``return_hybrid_pos`` is True, also
            returns the hybrid positional encodings for each BEV.
        """
        if not self.queue:
            return ([], []) if return_hybrid_pos else []

        device = self.queue[0][0].device
        dtype = self.queue[0][0].dtype
        current_yaw = self._extract_yaw(img_metas, device, dtype)
        current_timestamp = self._extract_timestamps(img_metas, device, dtype)
        current_pose = self._extract_ego_pose(img_metas, device, dtype)

        aligned = []
        hybrid_positions = []
        for bev, past_yaw, past_timestamp, past_pose in self.queue:
            if self.rotate_prev_bev:
                bs = bev.shape[1]
                for i in range(bs):
                    rotation_angle = (current_yaw[i] - past_yaw[i]).item()
                    tmp_prev_bev = bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(
                        tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    bev[:, i] = tmp_prev_bev[:, 0]
            if self.use_hybrid_pos_encoding:
                temporal_delta = (current_timestamp - past_timestamp).unsqueeze(-1)
                temporal_pos = self.temporal_mlp(temporal_delta).unsqueeze(0)
                relative_pose = torch.linalg.inv(current_pose) @ past_pose
                pose_flat = relative_pose.reshape(relative_pose.shape[0], -1)
                pose_pos = self.pose_mlp(pose_flat).unsqueeze(0)
                hybrid_pos = temporal_pos + pose_pos
                if return_hybrid_pos:
                    hybrid_positions.append(hybrid_pos)
                elif return_hybrid_pos:
                    raise RuntimeError('Hybrid positional encoding is disabled on this queue.')
                aligned.append(bev)
        return (aligned, hybrid_positions) if return_hybrid_pos else aligned

    def aligned_process(self, aligned, hybrid_positions=True):
                """Stack/reorder aligned BEV list into a 5D temporal tensor.

                Args:
                    aligned (list[Tensor]): Each tensor has shape ``[bev_h * bev_w, bs, c]``.
                    bev_h (int): BEV height.
                    bev_w (int): BEV width.
                    order (str): Temporal-spatial order for the last 3 dims. Supported:
                        ``'t h w'`` -> ``[bs, c, t, h, w]``;
                        ``'t w h'`` -> ``[bs, c, t, w, h]``;
                        ``'w h t'`` -> ``[bs, c, w, h, t]``.
                    hybrid_positions (list[Tensor] | None): Optional list where each
                        item is ``[1, bs, c]``. When provided, it is added to merged
                        channel features before spatial reshape.

                Returns:
                    Tensor: Reordered tensor in the target ``order``.
                """

                if len(aligned) == 0:
                    raise ValueError('aligned must contain at least one tensor.')
                merged = torch.stack([bev.permute(1, 2, 0).contiguous() for bev in aligned], dim=0)
                t, bs, c, hw = merged.shape
                if hybrid_positions is not None:
                    if len(hybrid_positions) != t:
                        raise ValueError(f'hybrid_positions length ({len(hybrid_positions)}) must match t ({t}).')
                    hybrid_merged = torch.stack([pos.squeeze(0) for pos in hybrid_positions], dim=0)
                    if hybrid_merged.shape != (t, bs, c):
                        raise ValueError(
                            f'hybrid_positions stacked shape must be {(t, bs, c)}, but got {tuple(hybrid_merged.shape)}.')
                    merged = merged + hybrid_merged.unsqueeze(-1)
                return merged


    def forward(self, bev, img_metas, bev_h=None, bev_w=None, order='t h w',
                return_stacked=True, update_queue=True):
        """Run temporal queue read-align-fuse as a single callable entry.

        Workflow:
            1) Read queued historical BEV and align them to current frame.
            2) Optionally stack/reorder aligned history to a 5D tensor.
            3) When hybrid positional encoding is enabled, add ``hybrid_pos``
               to the channel dimension on merged stacked features.
            4) Optionally update queue with current BEV for next time step.

        Args:
            bev (Tensor): Current BEV feature used as reference frame and to
                update queue. Expected shape ``[bev_h * bev_w, bs, c]`` or
                ``[bs, bev_h * bev_w, c]``.
            img_metas (list[dict]): Metadata of current frame.
            bev_h (int, optional): BEV height. Falls back to ``self.bev_h``.
            bev_w (int, optional): BEV width. Falls back to ``self.bev_w``.
            order (str): Output order used by ``stack_aligned``.
            return_stacked (bool): If True, return stacked 5D tensor when
                aligned history is not empty.
            update_queue (bool): If True, push current ``bev`` to queue.

        Returns:
            dict: {
                'aligned': list[Tensor],
                'stacked': Tensor | None,
                'hybrid_positions': list[Tensor] | None
            }
        """
        bev_h = self.bev_h if bev_h is None else bev_h
        bev_w = self.bev_w if bev_w is None else bev_w
        if bev_h is None or bev_w is None:
            raise ValueError('bev_h and bev_w must be provided either in args or constructor.')

        if self.use_hybrid_pos_encoding:
            aligned, hybrid_positions = self.get_aligned_queue(
                img_metas, bev_h, bev_w, return_hybrid_pos=True)
        else:
            aligned = self.get_aligned_queue(img_metas, bev_h, bev_w, return_hybrid_pos=False)
            hybrid_positions = None

        stacked = None
        if return_stacked and len(aligned) > 0:
            stacked = self.aligned_process(
                aligned, bev_h=bev_h, bev_w=bev_w, order=order, hybrid_positions=hybrid_positions)

        if update_queue:
            self.update(bev, img_metas)

        return {'aligned': aligned, 'stacked': stacked, 'hybrid_positions': hybrid_positions}

    def update(self, bev, img_metas):
        """Insert the latest BEV feature and drop the oldest if needed.

        Args:
            bev (Tensor): Current BEV feature.
            img_metas (list[dict]): Metadata for the current time step.
        """
        yaw = self._extract_yaw(img_metas, bev.device, bev.dtype)
        timestamp = self._extract_timestamps(img_metas, bev.device, bev.dtype)
        ego_pose = self._extract_ego_pose(img_metas, bev.device, bev.dtype)
        self.queue.append((bev.detach(), yaw.detach(), timestamp.detach(), ego_pose.detach()))
        while len(self.queue) > self.max_length:
            self.queue.popleft()

class Block(nn.Module):
    def __init__(self,dim, mixer_cls,bev_w,bev_h,norm_cls=nn.LayerNorm,fused_add_norm=False, residual_in_fp32=False,reverse=False,
        transpose=False,split_head=False,
        drop_path_rate=0.0,drop_rate=0.0,use_mlp=False,
    ):

        super().__init__()
        self.bev_w = bev_w
        self.bev_h = bev_h
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.split_head = split_head
        self.reverse = reverse
        self.transpose = transpose
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        )
        self.dropout = build_dropout(
            dict(type='Dropout', drop_prob=drop_rate)
        )
        if use_mlp:
            self.ffn = SwiGLUFFNFused(
                embed_dims=dim,
                feedforward_channels=int(dim * 4),
                layer_scale_init_value=0.0)
            self.ln2 = build_norm_layer(dict(type='LN'), dim)
        else:
            self.ffn = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, x, residual=None, order='t h w', inference_params=None):
        t, bs, c, hw = x.shape
        bev_h = self.bev_h
        bev_w = self.bev_w
        tgt_order = f'bs c ({order})'
        if hw != bev_h * bev_w:
            raise ValueError(f'Expected bev_h*bev_w={bev_h * bev_w}, but got {hw}.')

        x = x.reshape(t, bs, c, bev_h, bev_w)
        base = x.permute(1, 2, 3, 4, 0).contiguous()  # [bs, c, h, w, t]
        base = rearrange(base, 'bs c h w t -> bs c (h w t)')
        ssm_queue = rearrange(base, f'bs c (h w t) -> {tgt_order}', t=t, h=bev_h, w=bev_w)

        # reverse=False 时也要有默认值
        hidden_states = ssm_queue
        if self.reverse:
            hidden_states = ssm_queue.flip(2)
            if residual is not None:
                residual = residual.flip(2)

        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
        hidden_states, residual = fused_add_norm_fn(
            hidden_states,
            self.norm.weight,
            self.norm.bias,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
            eps=self.norm.eps,
        )

        hidden_states = self.drop_path(self.mixer(hidden_states, inference_params=inference_params))
        if self.ffn is not None:
            hidden_states = self.ffn(self.ln2(hidden_states), identity=hidden_states)

        if self.reverse:
            hidden_states = hidden_states.flip(2)
            if residual is not None:
                residual = residual.flip(2)

        # 输出统一回 canonical 格式
        hidden_states = rearrange(hidden_states, f'{tgt_order} -> bs c (h w t)', t=t, h=bev_h, w=bev_w)
        return hidden_states

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def create_block(
            d_model,
            bev_w,
            bev_h,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            rms_norm=False,
            residual_in_fp32=False,
            fused_add_norm=False,
            layer_idx=None,
            device=None,
            dtype=None,
            reverse=None,
            drop_rate=0.1,
            drop_path_rate=0.1,
            use_mlp=False,
            transpose=False,
    ):
        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )

        block = Block(
                d_model,
                mixer_cls,
                bev_w=bev_w,
                bev_h=bev_h,
                norm_cls=norm_cls,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                reverse=reverse,
                transpose=transpose,
                drop_rate=drop_rate,
                use_mlp=use_mlp,
                drop_path_rate=drop_path_rate,
            )
        block.layer_idx = layer_idx
        return block


class Mamba3DModel(BaseModule):
    arch_zoo = {
        **dict.fromkeys(
            ['t', 'tiny'], {
                'embed_dims': 256,
                'num_layers': 2,
                'num_heads': 8,
                'feedforward_channels': 256 * 2,
            }),
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 256,
                'num_layers': 3,
                'num_heads': 8,
                'feedforward_channels': 256 * 2,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 256,
                'num_layers': 4,
                'num_heads': 8,
                'feedforward_channels': 256 * 2,
            }),
    }


    def __init__(self,
                 arch='base',
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 norm_cfg_2=dict(type='LN', eps=1e-6),
                 final_norm=True,

                 frozen_stages=-1,

                 layer_scale_init_value=0.,

                 layer_cfgs=dict(),
                 pre_norm=False,
                 init_cfg=None,

                 inflate_len=True,

                 has_transpose=True,
                 fused_add_norm=True,
                 use_mlp=False,
                 split_head=False,
                 pretrained=None,

                 dt_scale=0.0,
                 dt_scale_tmp=0.0,

                 update_interval=None,
                 copy_weight=False,
                 factorization=None,

                 n_dim_pos=4,
                 d_state=16,
                 num_classes=400,

                 bev_h=None,
                 bev_w=None,
                 **kwargs):
        super(Mamba3DModel, self).__init__(init_cfg)
        self.pretrained = pretrained
        self.n_dim_pos = n_dim_pos
        self.factorization = factorization
        self.inflate_len = inflate_len
        self.update_interval = update_interval
        self.copy_weight = copy_weight
        self.out_type = 'cls_token'
        if pretrained:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        if self.inflate_len == 4:
            self.num_layers = self.arch_settings['num_layers'] * 4
        elif self.inflate_len:
            self.num_layers = self.arch_settings['num_layers'] * 3

        else:
            self.num_layers = self.arch_settings['num_layers'] * (2 if not use_mlp else 1)

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.bev_h = bev_h
        self.bev_w = bev_w
        if self.bev_h is None or self.bev_w is None:
            raise ValueError('Mamba3DModel requires bev_h and bev_w in kwargs.')

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        ssm_cfg = {"d_state": d_state}

        if dt_scale > 0:
            ssm_cfg['dt_scale'] = dt_scale
        for i in range(self.num_layers):
            block_ssm_cfg = dict(ssm_cfg)
            if dt_scale_tmp > 0 and (i // 2) % 3 == 2:
                block_ssm_cfg['dt_scale'] = dt_scale_tmp
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                layer_scale_init_value=layer_scale_init_value,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            # self.layers.append(TransformerEncoderLayer(**_layer_cfg))
            self.layers.append(
                create_block(
                    d_model=self.embed_dims,
                    bev_w=self.bev_w,
                    bev_h=self.bev_h,
                    ssm_cfg=block_ssm_cfg,
                    fused_add_norm=fused_add_norm,
                    residual_in_fp32=True,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    reverse=(not split_head) and (i % 2) > 0,
                    transpose=(not split_head) and has_transpose and (i % 4) >= 2,
                    use_mlp=use_mlp,
                    rms_norm=False,
                )
            )
        self.frozen_stages = frozen_stages
        if pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.embed_dims)
        else:
            self.pre_norm = nn.Identity()

        self.final_norm = final_norm
        if self.out_type == 'avg_featmap':
            self.ln1 = nn.Identity()
            self.ln2 = build_norm_layer(norm_cfg_2, self.embed_dims)
        elif final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)
        # if self.out_type == 'avg_featmap':

        # freeze pre-norm
        for param in self.pre_norm.parameters():
            param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers):
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

            if self.out_type == 'avg_featmap':
                self.ln2.eval()
                for param in self.ln2.parameters():
                    param.requires_grad = False

    def forward(self, x, return_bev=False):
        if x.dim() != 4:
            raise ValueError(f'Expected x shape [t, bs, c, hw], but got {tuple(x.shape)}.')

        x = self.drop_after_pos(x)
        x = self.pre_norm(x)
        residual = None
        orders = (
            't h w',
            't w h',
            'w h t'
        )

        if self.update_interval:
            raw_x = x
            for i, blk in enumerate(self.layers):
                z = i // 2
                d = z % len(orders)
                x = x + blk(raw_x, order=orders[d])
                if (i + 1) % self.update_interval == 0 or i == len(self.layers) - 1:
                    raw_x = x
                if i == len(self.layers) - 1:
                    x = (x + residual) if residual is not None else x
                if i == len(self.layers) - 1 and self.final_norm:
                    x = self.ln1(x)

        else:
            for i, blk in enumerate(self.layers):
                z = i // 2
                d = z % len(orders)
                x = blk(x, order=orders[d])
                if i == len(self.layers) - 1:
                    x = (x + residual) if residual is not None else x
                if i == len(self.layers) - 1 and self.final_norm:
                    x = self.ln1(x)

        if return_bev:
            # Return the current-step enhanced BEV feature as [bs, hw, c].
            return x[-1].permute(0, 2, 1).contiguous()

        # x: [t, bs, c, hw] -> [bs, c]
        x = x.mean(dim=(0, 3))
        x = self.head(x)
        return x

    def count_parameters(self, model=None):
        if model is None:
            model = self
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        self.total_parms = total_params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}

    def get_num_layers(self):
        return len(self.layers)
