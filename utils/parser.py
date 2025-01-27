import argparse
from configs.defaults import get_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='SpeedPro')
    parser.add_argument(
            "--cfg",
            dest="cfg_files",
            help="Path to the config files",
            default=None, 
            nargs="+",
    )
    parser.add_argument(
            "--output", 
            default=None, 
            type=str
    )
    parser.add_argument(
            "--opts",
            help="See slowfast/config/defaults.py for all options",
            default=None,
            nargs=argparse.REMAINDER,
    )

    return parser.parse_args()


def load_config(args, path_to_config=None):
    # Setup cfg.
    cfg = get_cfg()

    # Load config from cfg.
    if path_to_config is not None:
        cfg.merge_from_file(path_to_config)

    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "cfg_files"):
        cfg.CONFIG = args.cfg_files
    
    if hasattr(args, "output"):
        cfg.LOGDIR = args.output

    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id

    return cfg
