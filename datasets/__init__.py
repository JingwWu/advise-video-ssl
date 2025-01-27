import torch
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)

import datasets.video_dataset
import datasets.sampling as spl
import datasets.augmentation as aug
import datasets.transform as _trans
import datasets.utils as utils

from utils.misc import seed_worker


def build_aug(cfg, mode='train'):
    if mode == 'train':
        if cfg.AUG.TYPE == 'simple':
            aug_list = [
                aug.Resize(w=cfg.AUG.RESIZE[0], h=cfg.AUG.RESIZE[1]),
                aug.RandomCrop(k=cfg.AUG.TARGET_SIZE),
                aug.RandomColorJitter(
                    p=cfg.AUG.COLOR[0], b=cfg.AUG.COLOR[1], c=cfg.AUG.COLOR[2], s=cfg.AUG.COLOR[3], h=cfg.AUG.COLOR[4]
                ),
            ]
            aug_func = aug.AugsWarper(aug_list)

        elif cfg.AUG.TYPE == 'OnlyCrop':
            RandResizeCrop = _trans.RandomResizedCropAndInterpolation(
                size=cfg.AUG.TARGET_SIZE,
                scale=(cfg.AUG.MIN_AREA, 1.0),
                ratio=cfg.AUG.RAND_CROP_RATIO,
                interpolation="bilinear",
            )
            aug_func = aug.AugsWarper([RandResizeCrop,])

        elif cfg.AUG.TYPE == 'clip_aug':
            RandResizeCrop = aug.ClipRandomResizedCrop(
                size=cfg.AUG.TARGET_SIZE,
                scale=(cfg.AUG.MIN_AREA, 1.0),
                ratio=cfg.AUG.RAND_CROP_RATIO,
                jitter=cfg.AUG.CAMERA_SHAKE,
                shift=cfg.AUG.CAMERA_SHIFT,
                zoom=cfg.AUG.ZOOM,
                brightness=cfg.AUG.COLOR_BRI,
                saturation=cfg.AUG.COLOR_SAT,
                white_blan=cfg.AUG.WHITE_BALANCE,
                interpolation="bilinear",
            )
            aug_func = aug.AugsWarper([RandResizeCrop,], aug_mode='clip',)

        elif cfg.AUG.TYPE == 'none':
            aug_func = aug.AugsWarper([], aug_mode='clip',)

        elif cfg.AUG.TYPE == 'aa':
            RandResizeCrop = _trans.RandomResizedCropAndInterpolation(
                size=cfg.AUG.TARGET_SIZE,
                scale=(cfg.AUG.MIN_AREA, cfg.AUG.MAX_AREA),
                ratio=cfg.AUG.RAND_CROP_RATIO,
                interpolation="bilinear",
            )
            if cfg.AUG.AA_TYPE is not None:
                aug_transform = _trans.create_random_augment(
                    input_size=cfg.AUG.TARGET_SIZE,
                    auto_augment=cfg.AUG.AA_TYPE,
                    interpolation=cfg.AUG.INTERPOLATION,
                )
                aug_func = aug.AugsWarper([RandResizeCrop, aug_transform])

            else:
                raise NotImplementedError

        elif cfg.AUG.TYPE == 'rbyol':
            # only used in STDynamicsSampling, see datasets/sampling.py
            ByolAug = rbyol_aug(cfg)
            aug_func = aug.AugsWarper(ByolAug, aug_mode='none',)

        else:
            raise NotImplementedError

        return aug_func

    elif mode == 'speed':
        RandResizeCrop = _trans.RandomResizedCropAndInterpolation(
            size=cfg.VAL.TARGET_SIZE,
            scale=(cfg.VAL.MIN_AREA, 1.0),
            ratio=cfg.VAL.RAND_CROP_RATIO,
            interpolation="bilinear",
        )
        aug_func = aug.AugsWarper([RandResizeCrop,])

        return aug_func

    elif mode == 'action':
        ShortSideScale = _trans.ShortSideScale(target_size=cfg.INFER.RES)
        RandCropTensor = aug.RandomCrop(k=cfg.INFER.RES)
        aug_func = aug.AugsWarper([ShortSideScale, RandCropTensor,])

        return aug_func

    else:
        raise NotImplementedError


def build_spl_func(cfg, aug_func, mode='train'):
    aug_mode = getattr(aug_func, 'aug_mode')

    if mode == 'train':
        if cfg.TASK in ['speed']:
            stp_gen = spl.SpeedStampsGenerator(
                nums=cfg.DATA.NUM_CLIP,
                length=cfg.DATA.NUM_FRAMES,
                stride=cfg.DATA.STRIDE,
                method=cfg.SSL.METHOD,
                jitter=cfg.SSL.JITTER,
            )
            spl_func = spl.SpeedSampling(
                stps_gen=stp_gen,
                aug_func=aug_func,
                speed_range=cfg.SSL.RANGE,
                aug_mode=aug_mode,
            )
        elif cfg.TASK in ['tmodeling']:
            stp_gen = spl.IntervalsStampsGenerator(
                nums=cfg.DATA.NUM_CLIP,
                length=cfg.DATA.NUM_FRAMES,
                stride=cfg.DATA.STRIDE,
                method=cfg.SSL.METHOD,
                speed_range=cfg.SSL.RANGE,
                jitter=cfg.SSL.JITTER,
                intervals=None,
            )
            spl_func = spl.TemporalModelingSampling(
                stps_gen=stp_gen,
                aug_func=aug_func,
                aug_mode=aug_mode,
            )
        elif cfg.TASK in ['action_recog']:
            assert cfg.SSL.RANGE == [1,]
            stp_gen = spl.SpeedStampsGenerator(
                nums=cfg.DATA.NUM_CLIP,
                length=cfg.DATA.NUM_FRAMES,
                stride=cfg.DATA.STRIDE,
                method=cfg.SSL.METHOD,
                jitter=cfg.SSL.JITTER,
            )
            spl_func = spl.SpeedSampling(
                stps_gen=stp_gen,
                aug_func=aug_func,
                speed_range=cfg.SSL.RANGE,
                aug_mode=aug_mode,
            )
        else:
            raise NotImplementedError

    elif mode == 'speed':
        stp_gen = spl.SpeedStampsValGenerator(
            length=cfg.VAL.NUM_FRAME,
            stride=cfg.VAL.STRIDE,
            jitter=cfg.VAL.JITTER,
        )
        spl_func = spl.SpeedValSampling(
            stps_gen=stp_gen,
            aug_func=aug_func,
            speed_range=cfg.VAL.RANGE,
            aug_mode=aug_mode,
        )

    elif mode == 'action':
        stp_gen = spl.DuplicSpeedStampsGenerator(
            nums=cfg.INFER.NUM_CLIPS,
            length=cfg.INFER.NUM_FRAMES,
            stride=cfg.INFER.STRIDE,
            method=cfg.INFER.SAMPLE_METHOD,
            jitter=cfg.INFER.JITTER,
            duplic=cfg.INFER.NUM_CROPS,
        )
        spl_func = spl.SpeedSampling(
            stps_gen=stp_gen,
            aug_func=aug_func,
            speed_range=cfg.SSL.RANGE,
            aug_mode=aug_mode,
        )
    else:
        raise NotImplementedError

    return spl_func


def build_dataset(cfg, mode='train'):
    if mode == 'train':
        # get augmentation
        aug_func = build_aug(cfg)

        # get sampling function
        spl_func = build_spl_func(cfg, aug_func)

        # get dataset
        if cfg.DATA.DATASET in ['ucf-101', 'kinetics', 'diving', 'something']:
            vid_dataset = video_dataset.SamplingDataset(
                name=cfg.DATA.DATASET,
                data_dir=cfg.DATA.DATADIR,
                label_dir=cfg.DATA.LABELDIR,
                split_name=cfg.DATA.SPLITFILE,
                spl_func=spl_func,
            )
        else:
            raise NotImplementedError
        logger.info(
            f"Constructing training dataset with {len(vid_dataset)} samples.")

    elif mode in ['speed', 'action']:   # build validation or inference dataset
        # get augmentation
        aug_func = build_aug(cfg, mode)

        # get sampling function
        spl_func = build_spl_func(cfg, aug_func, mode)

        # get dataset
        if mode == 'speed':
            vid_dataset = video_dataset.SamplingDataset(
                name=cfg.DATA.DATASET,
                data_dir=cfg.VAL.DATADIR,
                label_dir=cfg.VAL.LABELDIR,
                split_name=cfg.VAL.SPLITFILE,
                spl_func=spl_func,
            )
            logger.info(
                f"Constructing {mode} validation dataset with {len(vid_dataset)} samples."
            )

        elif mode == 'action':
            vid_dataset = video_dataset.SamplingDataset(
                name=cfg.DATA.DATASET,
                data_dir=cfg.INFER.DATADIR,
                label_dir=cfg.INFER.LABELDIR,
                split_name=cfg.INFER.SPLITFILE,
                spl_func=spl_func,
            )
            logger.info(
                f"Constructing {mode} inference dataset with {len(vid_dataset)} samples."
            )
    else:
        raise NotImplementedError

    return vid_dataset


def build_loader(cfg, mode='train'):
    # get dataset
    dataset = build_dataset(cfg, mode)

    # work with DDP
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=(True if mode == 'train' else False),
    )
    g = torch.Generator()
    g.manual_seed(cfg.TRAIN.SEED)

    # get data loader
    if mode == 'train':
        loader = DataLoader(
            dataset,
            batch_size=cfg.DATA.BATCHSIZE_PER_GPU,
            persistent_workers=False,
            num_workers=cfg.DATA.WORKERS,
            sampler=data_sampler,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    elif mode == 'speed':
        loader = DataLoader(
            dataset,
            batch_size=cfg.VAL.BATCHSIZE_PER_GPU,
            persistent_workers=False,
            num_workers=cfg.VAL.WORKERS,
            sampler=data_sampler,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g,
        )

    elif mode == 'action':
        loader = DataLoader(
            dataset,
            batch_size=cfg.INFER.BATCHSIZE_PER_GPU,
            persistent_workers=False,
            num_workers=cfg.INFER.WORKERS,
            sampler=data_sampler,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g,
        )

    else:
        raise NotImplementedError

    return loader


class rbyol_aug:
    def __init__(self, cfg):
        self.bri_con_sat = [
            cfg.AUG.COLOR[1], cfg.AUG.COLOR[2], cfg.AUG.COLOR[3],
        ]
        self.hue = cfg.AUG.COLOR[4]
        self.p_convert_gray = cfg.AUG.GRAYSCALE
        self.moco_v2_aug = True
        self.gaussan_sigma_min = cfg.AUG.SSL_BLUR_SIGMA_MIN
        self.gaussan_sigma_max = cfg.AUG.SSL_BLUR_SIGMA_MAX
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.spatial_idx = -1
        self.min_scale = cfg.AUG.RESIZE[1]
        self.max_scale = cfg.AUG.RESIZE[0]
        self.crop_size = cfg.AUG.TARGET_SIZE
        self.random_horizontal_flip = cfg.AUG.RANDOM_FLIP
        self.inverse_uniform_sampling = cfg.AUG.INV_UNIFORM_SAMPLE
        self.aspect_ratio = [*cfg.AUG.RAND_CROP_RATIO]
        self.scale = [cfg.AUG.MIN_AREA, cfg.AUG.MAX_AREA]
        self.motion_shift = cfg.AUG.TRAIN_JITTER_MOTION_SHIFT

    def __call__(self, clip):
        clip = clip.permute(0, 2, 3, 1)
        clip = _trans.color_jitter_video_ssl(
            clip,
            bri_con_sat=self.bri_con_sat,
            hue=self.hue,
            p_convert_gray=self.p_convert_gray,
            moco_v2_aug=self.moco_v2_aug,
            gaussan_sigma_min=self.gaussan_sigma_min,
            gaussan_sigma_max=self.gaussan_sigma_max,
        )

        clip = utils.tensor_normalize(
            clip, self.mean, self.std,
        )

        clip = clip.permute(3, 0, 1, 2)
        clip = utils.spatial_sampling(
            clip,
            spatial_idx=self.spatial_idx,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            crop_size=self.crop_size,
            random_horizontal_flip=self.random_horizontal_flip,
            inverse_uniform_sampling=self.inverse_uniform_sampling,
            aspect_ratio=self.aspect_ratio,
            scale=self.scale,
            motion_shift=self.motion_shift,
        )
        clip = clip.permute(1, 0, 2, 3)
        return clip
