import os

import torch
import torch.distributed as dist

import numpy as np
np.set_printoptions(suppress=True)

import logging
logger = logging.getLogger(__name__)

from datasets import build_loader
from models import build_model
from utils.misc import (
        set_seed_strict,
        model_init,
        count_parameters,
        )

def validation(cfg, model, val_type):
    assert val_type == 'speed'
    logger.info(f"{'>'*10} Validation starts, type: {val_type} {'<'*10}")

    set_seed_strict(0)
    logger.info(f"Validation configs:\n{cfg.VAL}")

    # DATA
    loader = build_loader(cfg, val_type)
    ttl_iter = len(loader)

    # MODEL
    if model is None:
        model = build_model(cfg)
        model_init(cfg, model)

    # VALIDATION
    torch.set_grad_enabled(False)
    model.eval()

    confuse_met = torch.zeros(len(cfg.VAL.RANGE), len(cfg.VAL.RANGE)).cuda(non_blocking=True)
    torch.distributed.barrier()
    for it, (data, infos) in enumerate(loader):
        # in a slowpath manner
        data = [data.cuda(non_blocking=True).flatten(0, 1)]
        outputs = model(data)

        spd_label = torch.stack(infos['spd_label']).permute(1, 0).flatten(0, 1)
        spd_label = spd_label.cuda(non_blocking=True)

        _, predicted = torch.topk(outputs, 1, dim=1)
        for idx in range(spd_label.shape[0]):
            confuse_met[spd_label[idx], predicted[idx]] += 1

        if it == 0 or (it + 1) % 50 == 0 or (it + 1) == ttl_iter:
            logger.info(f"Progress: [{(it + 1):04d}/{ttl_iter:04d}]")

        torch.distributed.barrier()

    # gather all
    g_data = [torch.zeros_like(confuse_met) for _ in range(dist.get_world_size())]
    dist.all_gather(g_data, confuse_met)
    g_data = torch.stack(g_data).sum(dim=0).cpu()

    # post process
    confuse_met = g_data.numpy()
    ratios = np.diag(confuse_met) / np.sum(confuse_met, axis=1)
    logger.info(f"Confuse Metrics:\n{confuse_met}\nAcc: {ratios}")
    np.savetxt(os.path.join(cfg.LOGDIR, 'Metrics.csv'), confuse_met, delimiter=',')
