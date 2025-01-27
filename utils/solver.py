import models.optimizer as optim

import logging
logger = logging.getLogger(__name__)


def adjust_base_lr(cfg):
    lr_old = (cfg.SOLVER.BASE_LR, cfg.SOLVER.COSINE_END_LR, cfg.SOLVER.WARMUP_START_LR)
    scale = cfg.DATA.BATCHSIZE_PER_GPU * cfg.NUM_WORLDS / 64.
    cfg.SOLVER.BASE_LR *= scale
    cfg.SOLVER.COSINE_END_LR *= scale
    cfg.SOLVER.WARMUP_START_LR *= scale
    lr_new = (cfg.SOLVER.BASE_LR, cfg.SOLVER.COSINE_END_LR, cfg.SOLVER.WARMUP_START_LR)
    logger.info("Adjusting BASE_LR, COSINE_END_LR, WARMUP_START_LR from {} to {}".format(lr_old, lr_new,))
    logger.info("Base on BATCHSIZE_PER_GPU: {}; NUM_WORLDS: {}; Scale: {}".format(
        cfg.DATA.BATCHSIZE_PER_GPU, cfg.NUM_WORLDS, scale))


def build_optimizer(cfg, model):
    adjust_base_lr(cfg)

    grad_info = "Checking `requires_grad` flag..."
    for p_name, p in model.named_parameters():
        grad_info += f"\n{p_name.ljust(85)}| requires_grad: {p.requires_grad}"
    logger.info(grad_info)

    # create optimizer
    if cfg.SOLVER.OPTIMIZING_METHOD == 'simple_sgd':
        assert cfg.SOLVER.LARS_ON == False
        assert cfg.SOLVER.DAMPENING == 0.0
        from torch.optim import SGD
        optimizer = SGD(
            params=model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == 'simple_adamw':
        assert cfg.SOLVER.LARS_ON == False
        from torch.optim import AdamW
        optimizer = AdamW(
            params=model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            amsgrad=False,
        )

    else:
        optimizer = optim.construct_optimizer(model, cfg)
    logger.info(optimizer)

    return optimizer


def update_lr(cfg, _tr, optimizer, criteria):
    # get new lr
    epoch_exact = _tr['curr_ep'] + (_tr['it'] + 1) / _tr['epoch_iters']
    if cfg.SOLVER.LR_POLICY == 'cosine':
        lr = optim.get_epoch_lr(epoch_exact, cfg)

    elif cfg.SOLVER.LR_POLICY == 'plateau':
        if epoch_exact <= cfg.SOLVER.WARMUP_EPOCHS:
            lr_start, lr_end = cfg.SOLVER.WARMUP_START_LR, cfg.SOLVER.BASE_LR
            alpha = (lr_end - lr_start) / cfg.SOLVER.WARMUP_EPOCHS

            lr = epoch_exact * alpha + lr_start

        if epoch_exact > cfg.SOLVER.WARMUP_EPOCHS:
            if _tr['it'] + 1 == _tr['epoch_iters']:
                if not hasattr(_tr, 'lr_scheduler'):
                    from torch.optim.lr_scheduler import ReduceLROnPlateau
                    setattr(
                        _tr,
                        'lr_scheduler',
                        ReduceLROnPlateau(
                            optimizer=optimizer,
                            mode='min',
                            factor=0.5,
                            patience=5,
                            threshold=0.0001,
                        ),
                    )
                val = criteria(_tr, cfg.TASK)
                _tr.lr_scheduler.step(val)
            lr = optimizer.param_groups[0]['lr']
    _tr['curr_lr'] = lr

    # set new lr
    if cfg.SOLVER.LR_POLICY == 'cosine':
        if cfg.SOLVER.OPTIMIZING_METHOD in ['simple_sgd', 'simple_adamw']:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            optim.set_lr(optimizer, lr)

    elif cfg.SOLVER.LR_POLICY == 'plateau':
        if epoch_exact <= cfg.SOLVER.WARMUP_EPOCHS:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            pass

    else:
        raise NotImplementedError


def get_grad_norm(_tr, model):
    grad_norm = optim.get_grad_norm_(model.parameters()).cpu()
    _tr['grad_norm'].update(grad_norm)

