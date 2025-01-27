import os

from tools.train import trainer
from tools.validate import validation
from tools.inference import inference
from utils.parser import parse_args, load_config

from configs.defaults import assert_and_infer_cfg
from utils.distributed import (
        distributed_init,
        launch_job
        )
from utils.misc import (
        set_seed,
        get_current_commit,
        setup_logger
        )

import logging
logger = logging.getLogger(__name__)


def train_worker(cfg):
    # DDP init
    distributed_init(cfg)

    # setup logger
    setup_logger(cfg.RANK, os.path.join(cfg.LOGDIR, 'log.txt'))

    model = None
    # train
    if 'train' in cfg.PIPELINE:
        model = trainer(cfg)
    else:
        logger.info(f"No training in pipeline; {cfg.PIPELINE}")

    # validate
    for phase in cfg.PIPELINE:
        if phase.endswith('val'):
            val_type = phase.split('_')[0]
            validation(cfg, model, val_type)

    # inference
    for phase in cfg.PIPELINE:
        if phase.endswith('infer'):
            infer_type = phase.split('_')[0]
            inference(cfg, model, infer_type)

    logger.info(f"All works done; {cfg.PIPELINE}")


def main(cfg):
    # set seed
    set_seed(cfg.TRAIN.SEED)

    # launch job
    launch_job(cfg, train_worker)


if __name__ == '__main__':
    args = parse_args()
    print("config files: {}".format(args.cfg_files))

    # load configs
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

    # log commit SHA
    try:
        current_commit = get_current_commit()
        cfg.COMMIT_SHA = current_commit
    except:
        pass

    main(cfg)

