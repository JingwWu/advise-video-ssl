import os

import torch
import torch.distributed as dist

import logging
logger = logging.getLogger(__name__)

from datasets import build_loader
from models import build_model
from utils.misc import (
        set_seed_strict,
        model_init,
        count_parameters,
        )

def inference(cfg, model, infer_type):
    assert infer_type == 'action'
    logger.info(f"{'>'*10} Inference starts, type: {infer_type} {'<'*10}")

    set_seed_strict(cfg.INFER.SEED)
    logger.info(f"Inference configs:\n{cfg.INFER}")

    # DATA
    loader = build_loader(cfg, infer_type)
    ttl_iter = len(loader)

    # MODEL
    if cfg.MODELDATA is not None:
        assert cfg.RESUME is None
        model = build_model(cfg)
        cfg.RESUME = os.path.join(
            cfg.MODELDATA, "checkpoints/Model_Best.pth")
        model_init(cfg, model)
    elif model is None and cfg.RESUME is not None:
        model = build_model(cfg)
        model_init(cfg, model)
    else:
        raise NotImplementedError

    # no grad
    torch.set_grad_enabled(False)
    model.eval()

    correct, total = 0, 0
    torch.distributed.barrier()
    for it, (data, infos) in enumerate(loader):
        if cfg.MODEL.MODEL_NAME == 'ResNet':
            data = [data.cuda(non_blocking=True).flatten(0, 1)]
        if cfg.MODEL.MODEL_NAME == 'TemporalModel':
            data = [data.cuda(non_blocking=True)]
            if data[0].shape[-1] == 112:
                b, v, c, f, h, w = data[0].shape
                data[0] = data[0].permute(0, 1, 3, 2, 4, 5)
                data[0] = torch.nn.functional.interpolate(data[0].flatten(0, 2), scale_factor=2, mode='bilinear', align_corners=False).view(b, v, f, c, h*2, w*2)
                data[0] = data[0].permute(0, 1, 3, 2, 4, 5)

        outputs = model(data)

        if cfg.MODEL.MODEL_NAME == 'TemporalModel':
            outputs = outputs[0][0]

        logits = outputs.mean(dim=0, keepdim=True)
        label = torch.stack(infos['cls_id'])

        _, predicted = torch.topk(logits, 1, dim=1)
        if label.item() == predicted.item():
            correct += 1
        total += 1

        logger.info(f"Inference [{(it + 1):04d}/{ttl_iter:04d}]")
        torch.distributed.barrier()

    correct = torch.Tensor([correct]).cuda(non_blocking=True)
    total = torch.Tensor([total]).cuda(non_blocking=True)

    correct_gather = [torch.zeros_like(correct) for _ in range(dist.get_world_size())]
    dist.all_gather(correct_gather, correct)
    correct_all = torch.stack(correct_gather).sum().cpu()

    total_gather = [torch.zeros_like(total) for _ in range(dist.get_world_size())]
    dist.all_gather(total_gather, total)
    total_all = torch.stack(total_gather).sum().cpu()

    logger.info(f"Acc@1: {(correct_all * 100. / total_all):.2f}")

