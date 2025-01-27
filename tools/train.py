import torch

from datasets import build_loader
from models import build_model
from utils.solver import (
        build_optimizer,
        update_lr,
        get_grad_norm
)
from utils.misc import (
        set_seed,
        checkpoint_manager,
        training_helper,
        count_parameters,
        training_resume,
        TopKAccuracyCalculator,
)
from models.losses import _LOSSES
from models.contrastive import contrastive_forward

# import pynvml

import logging
logger = logging.getLogger(__name__)


#=========================== USER ZONE ===============================

def post_process(cfg, data, infos):
    if cfg.MODEL.MODEL_NAME == 'TemporalModel':
        assert data.ndim == 6, f"data.ndim: {data.ndim}"
        if cfg.TRAIN.SEQUENTIAL:
            data = [[x.unsqueeze(1)] for x in data.unbind(1)]
            infos_list = []
            for idx in range(cfg.DATA.NUM_CLIP):
                infos_new = {}
                for k, v in infos.items():
                    if isinstance(v, list):
                        infos_new[k] = v
                    elif isinstance(v, torch.Tensor):
                        if v.ndim == 1:
                            infos_new[k] = v
                        else:
                            infos_new[k] = v[:, idx, ...].unsqueeze(1)
                    else:
                        infos_new[k] = v
                infos_list.append(infos_new)
        else:
            data = [[data]]
            infos_list = [infos]


    return data, infos_list

def model_forward(cfg, model, data, infos, _tr):
    if cfg.MODEL.MODEL_NAME == 'ResNet':
        # in a slowpath manner
        data = [data.cuda(non_blocking=True).flatten(0, 1)]
        outputs = model(data)

        return outputs, model

    elif cfg.MODEL.MODEL_NAME == 'ContrastiveModel':
        data = [[data[:, idx, ::].cuda(non_blocking=True)] for idx in range(data.shape[1])]
        index = infos['item_id'].cuda(non_blocking=True)
        time = None         # just work with byol
        epoch_exact = _tr['curr_ep'] + (_tr['it'] + 1) / _tr['epoch_iters']
        (
            model,
            preds,
            partial_loss,
            perform_backward,
        ) = contrastive_forward(
            model, cfg, data, index, time, epoch_exact, None
        )

        return (preds, partial_loss), model

    elif cfg.MODEL.MODEL_NAME == 'TemporalModel':
        data = [x.cuda(non_blocking=True) for x in data]
        if data[0].shape[-1] == 112:
            b, v, c, f, h, w = data[0].shape
            data[0] = data[0].permute(0, 1, 3, 2, 4, 5)
            data[0] = torch.nn.functional.interpolate(data[0].flatten(0, 2), scale_factor=2, mode='bilinear', align_corners=False).view(b, v, f, c, h*2, w*2)
            data[0] = data[0].permute(0, 1, 3, 2, 4, 5)
        outputs = model(data)

        return outputs, model

    else:
        raise NotImplementedError


def metrics_init(cfg):
    metrics = {}
    for stat, met in zip(cfg.SSL.STAT, cfg.SSL.METRIC):
        if met == 'ce':
            metrics[stat] = _LOSSES['cross_entropy'](label_smoothing = cfg.SSL.SMOOTHING)
        elif met == 'bce_logit':
            metrics[stat] = _LOSSES['bce_logit']()
        elif met == 'smoothing_bce_logit':
            metrics[stat] = _LOSSES['smoothing_bce_logit'](smoothing=cfg.SSL.SMOOTHING)
        elif met == 'margin':
            mode = 'intra'
            metrics[stat] = _LOSSES['margin'](margin=cfg.SSL.MARGIN, mode=mode)
        elif met == 'mse':
            metrics[stat] = _LOSSES['mse']()
        elif met == 'contrastive_loss':
            metrics[stat] = _LOSSES['contrastive_loss']()
        elif met == 'acc@1':
            metrics[stat] = TopKAccuracyCalculator(k=1)
        elif met == 'acc@5':
            metrics[stat] = TopKAccuracyCalculator(k=5)
        elif met == 'none':
            metrics[stat] = None
        else:
            raise NotImplementedError
    return metrics


def output_handler(cfg, output, infos, metrics):
    if cfg.MODEL.MODEL_NAME == 'ContrastiveModel':
        preds, partial_loss = output

        return {'loss_byol': partial_loss + 1.0,}

    if cfg.TASK == 'speed':
        assert cfg.SSL.METRIC[0] in ['ce', 'bce_logit', 'smoothing_bce_logit', 'margin']
        spd_label = torch.stack(infos['spd_label']).permute(1, 0).flatten(0, 1).cuda(non_blocking=True)

    elif cfg.TASK == 'tmodeling':
        spd_label = infos['speeds'].flatten(0, 1)
        spd_label = torch.where(spd_label == 1, 0, spd_label)
        spd_label = torch.where(spd_label == 2, 1, spd_label)
        spd_label = torch.where(spd_label == 4, 2, spd_label)
        spd_label = torch.where(spd_label == 8, 3, spd_label).cuda(non_blocking=True)

    elif cfg.TASK == 'action_recog':
        assert cfg.SSL.METRIC[0] == 'ce'
        cls_label = torch.stack(infos['cls_id']).permute(1, 0).flatten(0, 1).cuda(non_blocking=True)

    else:
        raise NotImplementedError

    if cfg.TASK in ['speed']:
        if cfg.SSL.METRIC[0] == 'ce':
            loss_spd = metrics['loss_spd'](output, spd_label)

        elif cfg.SSL.METRIC[0] in ['bce_logit', 'smoothing_bce_logit']:
            bce_label = torch.nn.functional.one_hot(spd_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
            loss_spd = metrics['loss_spd'](output, bce_label)

        elif cfg.SSL.METRIC[0] == 'margin':
            rs_output = output.reshape(cfg.DATA.BATCHSIZE_PER_GPU, cfg.DATA.NUM_CLIP, -1)
            rs_spd_label = spd_label.reshape(cfg.DATA.BATCHSIZE_PER_GPU, cfg.DATA.NUM_CLIP)
            loss_spd = metrics['loss_spd'](rs_output, rs_spd_label)

        else:
            raise NotImplementedError

        acc_spd = metrics['acc_spd'](output, spd_label)

        return {'loss_spd': loss_spd,
                'acc_spd': acc_spd,}

    elif cfg.TASK == 'tmodeling':
        loss_spd, acc_spd = 0., 0.
        if 'speed' in cfg.SSL.TASK:
            assert cfg.SSL.METRIC[0] == 'ce'
            logits = output[0]
            spd_label = [spd_label]

            for out, label in zip(logits, spd_label):
                loss_spd += metrics['loss_spd'](out, label)
            loss_spd /= len(logits)


            for out, label in zip(logits, spd_label):
                acc_spd += metrics['acc_spd'](out, label)
            acc_spd /= len(logits)

        return {'loss_spd': loss_spd,
                'acc_spd': acc_spd,}

    elif cfg.TASK == 'action_recog':
        if cfg.MODEL.MODEL_NAME == 'TemporalModel':
            output = output[0][0]
        if not cfg.TRAIN.SEQUENTIAL:
            cls_label = cls_label.unsqueeze(1).expand(cfg.DATA.BATCHSIZE_PER_GPU, cfg.DATA.NUM_CLIP).flatten(0, 1)
        loss = metrics['loss'](output, cls_label)

        acc_1 = metrics['acc_1'](output, cls_label)
        acc_5 = metrics['acc_5'](output, cls_label)

        return {'loss': loss,
                'acc_1': acc_1, 'acc_5': acc_5,}

    else:
        raise NotImplementedError


def loss_backward(cfg, results):
    if cfg.MODEL.MODEL_NAME == 'ContrastiveModel':
        # `loss.backward` has already been called at contrastive_forward
        return

    if cfg.TASK in ['speed']:
        loss = results['loss_spd']

    elif cfg.TASK in ['action_recog']:
        loss = results['loss']

    elif cfg.TASK == 'tmodeling':
        loss = 0.
        if 'speed' in cfg.SSL.TASK:
            loss += results['loss_spd']

    else:
        raise NotImplementedError

    loss.backward()


def save_best_criteria(summary, task):
    if task in ['speed']:
        return {'item': 'loss',
                'value': summary['loss_spd'],}

    elif task == 'tmodeling':
        return {'item': 'loss',
                'value': summary['loss_spd'],}

    elif task == 'action_recog':
        return {'item': 'loss',
                'value': summary['loss'],}

    else:
        raise NotImplementedError


def update_lr_criteria(_tr, task):
    """used for plateau"""
    if task == 'tmodeling':
        return getattr(_tr, 'loss_spd').avg


#=========================== USER ZONE ===============================


def train_one_epoch(cfg, _tr, model, loader, optimizer, handle = None):
    # env configs
    torch.set_grad_enabled(True)
    model.train()

    # setup metrics
    metrics = metrics_init(cfg)
    # ITER LOOP
    for it, (data, infos) in enumerate(loader):
        data_list, infos_list = post_process(cfg, data, infos)

        _tr.iter_step(it)
        optimizer.zero_grad()
        for data, infos in zip(data_list, infos_list):
            # model forward
            outputs, model = model_forward(cfg, model, data, infos, _tr)

            # handle outputs
            results = output_handler(cfg, outputs, infos, metrics)

            # update losses meters
            _tr.update_meters(results)

            # loss backward
            loss_backward(cfg, results)

        # optimizer pre-process
        update_lr(cfg, _tr, optimizer, update_lr_criteria)

        # optimizer post-process
        optimizer.step()
        get_grad_norm(_tr, model)

        if handle is not None:
            used_mem = pynvml.nvmlDeviceGetMemoryInfo(handle) # type: ignore
            print(f"Used memory: {used_mem.used / 1024 ** 2} MB")

        _tr.iter_end()


def trainer(cfg):
    set_seed(cfg.TRAIN.SEED + cfg.RANK)
    # print configs
    logger.info(cfg)

    # setup managers
    _ckpt = checkpoint_manager(cfg, save_best_criteria)
    _tr = training_helper(cfg, tb='pt')

    # DATA
    loader = build_loader(cfg)
    _tr['epoch_iters'] = len(loader)
    _tr['total_iters'] = len(loader) * cfg.SOLVER.MAX_EPOCH

    # MODEL
    model = build_model(cfg)
    if cfg.LINEAR_PROBING:
        for name, param in model.named_parameters():
            if 'temporal' in name:
                param.requires_grad = False
    logger.info(model)
    _tr['#param'] = count_parameters(model)

    # SOLVER
    optimizer = build_optimizer(cfg, model)

    # RESUME
    if cfg.TRAIN.RESUME or \
            (cfg.TRAIN.AUTO_RESUME and _ckpt.has_checkpoint()):
        training_resume(cfg, _tr, _ckpt, model, optimizer)

    # AMP
    if cfg.TRAIN.AMP:
        pass

    # training init
    handle = None
    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(cfg.LOCAL_RANK)

    _tr.train_start()
    # MAIN LOOP
    for curr_ep in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.MAX_EPOCH):
        set_seed(cfg.TRAIN.SEED + cfg.RANK)
        # update ep infos
        _tr.epoch_start(curr_ep)
        # train one epoch
        train_one_epoch(cfg, _tr, model, loader, optimizer, handle)

        _tr.epoch_end()
        _ckpt(_tr, model, optimizer)

    # training ends
    _tr.train_end()
    return model
