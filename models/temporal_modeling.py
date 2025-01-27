import os

import torch
import torch.nn as nn

import open_clip
import json

from .build import MODEL_REGISTRY
from models.contrastive import Normalize

import logging
logger = logging.getLogger(__name__)


def load_spatial_model(cfg):
    logger.info(f"Try loading spatial encoder weights from path: {cfg.MODEL.SPATIAL_MODEL_PATH}")
    weight_path = os.path.join(cfg.MODEL.SPATIAL_MODEL_PATH, 'open_clip_pytorch_model.bin')
    config_path = os.path.join(cfg.MODEL.SPATIAL_MODEL_PATH, 'open_clip_config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)['model_cfg']
    logger.info(f"Spatial encoder config:\n{config}")
    assert cfg.MODEL.SPATIAL_EMBEDDING_DIM == config['vision_cfg']['width'], \
        f"{cfg.MODEL.SPATIAL_EMBEDDING_DIM}, {config['vision_cfg']['width']}"

    # HACK
    if not cfg.TM.FROM_SCRATCH:
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.MODEL.SPATIAL_MODEL_ARCH,
            pretrained=weight_path,
            precision='bf16' if cfg.TM.ENABLE_BF16 else 'fp32',
            **config,
        )
    else:
        model = open_clip.create_model(
            cfg.MODEL.SPATIAL_MODEL_ARCH,
            **config,
        )

    spatial_encoder = model.visual

    return spatial_encoder


def load_temporal_model(cfg, arch_type, chn, ks):
    if arch_type == 'conv3d':
        temporal_encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=chn,
                out_channels=cfg.MODEL.TEMPORAL_HIDDEN_DIM,
                kernel_size=(ks, ks, ks),
                stride=(1, 1, 1),
                padding=0 if ks == 1 else 1,
                bias=True
            ),
            nn.SiLU(),
            nn.Conv3d(
                in_channels=cfg.MODEL.TEMPORAL_HIDDEN_DIM,
                out_channels=cfg.MODEL.TEMPORAL_EMBEDDING_DIM,
                kernel_size=(ks, ks, ks),
                stride=(1, 1, 1),
                padding=0 if ks == 1 else 1,
                bias=True
            )
        )
    elif arch_type == 'resnet3d':
        from models.resnet_helper import ResStage
        from models.batchnorm_helper import get_norm
        norm_module = get_norm(cfg)
        temporal_encoder = ResStage(
            dim_in=[chn],
            dim_out=[cfg.MODEL.TEMPORAL_EMBEDDING_DIM],
            dim_inner=[cfg.MODEL.TEMPORAL_HIDDEN_DIM],
            temp_kernel_sizes=[[1]],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[cfg.TM.NUM_BLOCKS],
            num_groups=[1],
            num_block_temp_kernel=[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=norm_module,
        )
    elif arch_type == 'video_swin':
        from models.swin_transformer import BasicLayer
        assert chn == cfg.MODEL.TEMPORAL_EMBEDDING_DIM
        temporal_encoder = BasicLayer(
            dim=chn,
            depth=cfg.TM.NUM_BLOCKS,
            num_heads=cfg.TM.NUM_HEADS,
            window_size=(4, 7, 7),
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False
        )
    elif arch_type == 'tfmer_enc':
        temporal_encoder = torch.nn.TransformerEncoderLayer(
            d_model=chn,
            nhead=cfg.TM.NUM_HEADS,
            dim_feedforward=cfg.MODEL.TEMPORAL_HIDDEN_DIM,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )

    else:
        raise NotImplementedError

    return temporal_encoder


def load_head(cfg, dim_in, num_classes):
    from .head_helper import ResNetBasicHead
    if cfg.SSL.NUM_MLP_LAYERS == 1:
        head = nn.Linear(dim_in, num_classes, bias=True)
    else:
        head = ResNetBasicHead(
            dim_in=[dim_in],
            num_classes=num_classes,
            pool_size=[None],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func="none",
            cfg=cfg,
        )
    return head


@MODEL_REGISTRY.register()
class TemporalModel(nn.Module):
    def __init__(self, cfg):
        super(TemporalModel, self).__init__()
        self.from_scratch = cfg.TM.FROM_SCRATCH
        self.enable_bf16 = cfg.TM.ENABLE_BF16
        self.task = cfg.SSL.TASK
        self.skip_temporal_modeling = cfg.TM.SKIP_TM
        self.temporal_cat = cfg.TM.TEMPORAL_CAT
        self.linear_proj = cfg.TM.LINEAR_PROJ
        self.linear_probing = cfg.LINEAR_PROBING

        spatial_encoder = load_spatial_model(cfg)
        if not self.from_scratch:
            self.spatial_encoder = spatial_encoder.eval().requires_grad_(False)
        else:
            self.spatial_encoder = spatial_encoder

        ks = 3
        if cfg.TM.SPATIAL_POOL_DIM is not None:
            self.sp_p = cfg.TM.SPATIAL_POOL_DIM
            self.pool_spatial = nn.AdaptiveAvgPool2d((self.sp_p, self.sp_p))
            if self.sp_p == 1:
                ks = 1

        chn = cfg.MODEL.SPATIAL_EMBEDDING_DIM
        if cfg.TM.CHANNEL_POOL_DIM is not None:
            self.ch_p = cfg.TM.CHANNEL_POOL_DIM
            self.pool_channel = nn.AdaptiveAvgPool1d(self.ch_p)
            chn = self.ch_p

        self.t_arch = cfg.TM.TEMPORAL_ARCH
        if not self.skip_temporal_modeling:
            self.temporal_encoder = load_temporal_model(cfg, self.t_arch, chn, ks)

        if self.temporal_cat:
            t_dim = cfg.MODEL.TEMPORAL_EMBEDDING_DIM // cfg.DATA.NUM_FRAMES
            if not self.linear_proj:
                self.pool = nn.AdaptiveAvgPool3d((t_dim, 1, 1))
            else:
                self.pool = nn.Linear(cfg.MODEL.TEMPORAL_EMBEDDING_DIM, t_dim)
                self.pool_spatial = nn.AdaptiveAvgPool2d((1, 1))

        self.head_cls = load_head(
            cfg,
            dim_in=cfg.MODEL.TEMPORAL_EMBEDDING_DIM,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

        if "byol" in self.task:
            self.mmt = cfg.CONTRASTIVE.MOMENTUM
            self.T = cfg.CONTRASTIVE.T

            self.temporal_encoder_hist = load_temporal_model(
                cfg, self.t_arch, chn, ks
            ).eval().requires_grad_(False)      # stop gradient

            self.head_projector = load_head(
                cfg,
                dim_in=cfg.MODEL.TEMPORAL_EMBEDDING_DIM,
                num_classes=cfg.CONTRASTIVE.DIM,
            )

            self.head_projector_hist = load_head(
                cfg,
                dim_in=cfg.MODEL.TEMPORAL_EMBEDDING_DIM,
                num_classes=cfg.CONTRASTIVE.DIM,
            ).eval().requires_grad_(False)      # stop gradient

            self.head_predictor = load_head(
                cfg,
                dim_in=cfg.CONTRASTIVE.DIM,
                num_classes=cfg.CONTRASTIVE.DIM,
            )

            self.l2_norm = Normalize(dim=1)

    @torch.no_grad()
    def _update_history(self):
        # momentum update
        m = self.mmt
        dist_temporal_encoder, dist_head_projector = {}, {}
        for name, p in self.temporal_encoder.named_parameters():
            dist_temporal_encoder[name] = p
        for name, p in self.head_projector.named_parameters():
            dist_head_projector[name] = p

        if not hasattr(self, 'init_flag'):
            setattr(self, 'init_flag', True)
            logger.info(f"EMA Models Initializing.")
            for name, p in self.temporal_encoder_hist.named_parameters():
                p.data.copy_(dist_temporal_encoder[name].data)
            for name, p in self.head_projector_hist.named_parameters():
                p.data.copy_(dist_head_projector[name].data)

        for name, p in self.temporal_encoder_hist.named_parameters():
            p.data = dist_temporal_encoder[name].data * (1.0 - m) + p.data * m
        for name, p in self.head_projector_hist.named_parameters():
            p.data = dist_head_projector[name].data * (1.0 - m) + p.data * m

    def spatial_forward(self, x):
        """
        x.shape: [bs, ch, h, w]
        """
        bs, ch, _, _ = x.shape
        if self.enable_bf16:
            x = x.type(torch.bfloat16)

        if not self.from_scratch:
            with torch.no_grad():
                _, tokens = self.spatial_encoder(x)
        else:
            _, tokens = self.spatial_encoder(x)
        # [bs, L, D]

        hw, D = int(tokens.shape[1] ** 0.5), tokens.shape[2]

        spatial_feats = tokens.reshape(bs, hw, hw, D).float()
        # [bs, nh, nw, D]

        if hasattr(self, 'ch_p') and spatial_feats.shape[-1] != self.ch_p:
            spatial_feats = self.pool_channel(spatial_feats.flatten(0, 2))
            spatial_feats = spatial_feats.reshape(bs, hw, hw, self.ch_p)

        spatial_feats = spatial_feats.permute(0, 3, 1, 2).contiguous()
        D = spatial_feats.shape[1]
        # [bs, D, nh, nw]

        if hasattr(self, 'sp_p') and spatial_feats.shape[-1] != self.sp_p:
            spatial_feats = self.pool_spatial(spatial_feats.flatten(0, 1))
            spatial_feats = spatial_feats.reshape(bs, D, self.sp_p, self.sp_p)

        spatial_feats = spatial_feats.permute(0, 2, 3, 1).contiguous()
        # [bs, nh, nw, D]
        return spatial_feats

    def temporal_forward(self, x, online=True):
        """
        x.shape: [bs, D, nf, nh, nw]
        """
        bs, D, nf, nh, nw = x.shape

        if online:
            encoder = self.temporal_encoder
        else:
            encoder = self.temporal_encoder_hist

        if self.t_arch == 'resnet3d':
            x = encoder(x.unsqueeze(0))[0]
        elif self.t_arch == 'tfmer_enc':
            x = x.permute(0, 2, 3, 4, 1).flatten(1, 3)
            x = encoder(x).reshape(bs, nf, nh, nw, D)
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        else:
            x = encoder(x)
        # [bs, nD, nt, nh, nw]

        return x

    def head_bridge(self, feat):
        bs = feat.shape[0]
        if self.temporal_cat:
            if not self.linear_proj:
                feat = feat.permute(0, 2, 1, 3, 4)
                feat = self.pool(feat).reshape(bs, -1)
            else:
                feat = feat.permute(0, 2, 3, 4, 1)
                feat = self.pool(feat).permute(0, 4, 1, 2, 3)
                feat = self.pool_spatial(feat).reshape(bs, -1)
        return feat

    def backbone_forward(self, inp):
        """
        inp: list of x.shape: [nv, nc, ch, nf, h, w]
        """
        feats, keys = [], []
        for x in inp:
            nv, nc, ch, nf, h, w = x.shape
            x = x.permute(0, 1, 3, 2, 4, 5).flatten(0, 2)
            # [nv * nc * nf, ch, h, w]

            x = self.spatial_forward(x)

            _, nh, nw, D = x.shape
            x = x.reshape(nv, nc, nf, nh, nw, D)
            x = x.permute(0, 1, 5, 2, 3, 4).flatten(0, 1).contiguous()
            # [nv * nc, D, nf, nh, nw]

            if not self.skip_temporal_modeling:
                feat = self.temporal_forward(x)
            else:
                feat = x

            feat = self.head_bridge(feat)
            feats.append(feat)

            if 'byol' in self.task:
                with torch.no_grad():
                    key = self.temporal_forward(x, online=False)
                key = self.head_bridge(key)
                keys.append(key)

        return feats, keys

    def classify_forward(self, inp):
        logits = []
        for x in inp:
            if self.linear_probing:
                x = nn.functional.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.shape[0], -1)
                logits.append(self.head_cls(x))
            else:
                logits.append(self.head_cls([x]))
        return logits

    def contrast_forward(self, feats, keys):
        assert len(feats) == len(keys) == 2         # HACK: Only support 2 positive samples for now.
        keys = keys[::-1]

        loss = 0.
        for feat, key in zip(feats, keys):
            feat = self.head_projector(feat)
            q = self.head_predictor(feat)

            with torch.no_grad():
                k = self.head_projector_hist(key)

            q = self.l2_norm(q)
            k = self.l2_norm(k)

            sim = torch.einsum("nc,nc->n", [q, k])
            sim /= self.T

            loss += -sim.mean()
        loss = loss / len(feats) + 1.0 / self.T

        return loss

    def forward(self, inp):
        """
        inp: list of x.shape: [nv, nc, ch, nf, h, w]
        """
        cls_logits, contrast_loss, mvm_loss = None, None, None
        feats, keys = self.backbone_forward(inp)

        if 'speed' in self.task or 'action' in self.task:
            cls_logits = self.classify_forward(feats)

        if 'byol' in self.task:
            self._update_history()
            contrast_loss = self.contrast_forward(feats, keys)

        return [cls_logits, contrast_loss, mvm_loss]

