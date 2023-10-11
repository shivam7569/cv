from fvcore.common.config import CfgNode as _CfgNode
from iopath.common.file_io import PathManager as PathManagerBase

from configs.model_CONFIGS import *
from src.cv_parser import get_parser

PathManager = PathManagerBase()

class CfgNode(_CfgNode):

    @classmethod
    def _open_cfg(cls, filename):
        return PathManager.open(filename, 'r')
    
    def merge_from_file(self, cfg_filename, allow_unsafe = True):
        
        assert PathManager.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist"
        loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        self.merge_from_other_cfg(loaded_cfg)

    def dump(self, *args, **kwargs):
        return super().dump(*args, **kwargs)


def get_cfg() -> CfgNode:
    from configs.defaults import _C

    return _C.clone()

def setup_config():
    args = get_parser().parse_args()
    cfg = get_cfg()
    ModelConfigs.addModelConfigs(cfg, args.model_name)
    cfg.freeze()

    return cfg

class ModelConfigs:

    @classmethod
    def addModelConfigs(cls, cfg, model_name: str):
        if model_name.lower() == "lenet":
            LeNetConfig(cfg)
        elif model_name.lower() == "alexnet":
            AlexNetConfig(cfg)
        elif model_name.lower() == "vgg16":
            VGG16Config(cfg)