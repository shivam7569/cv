from LeNet.parser import get_parser
from configs.config import CfgNode as CN, get_cfg
from global_params import Global

def add_LeNet_config(cfg):

    cfg.LeNet = CN()
    cfg.LeNet.TESTING = "test_variable"

def setup_config():
    args = get_parser().parse_args()
    cfg = get_cfg()
    add_LeNet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    return cfg