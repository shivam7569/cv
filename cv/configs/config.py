from fvcore.common.config import CfgNode as _CfgNode
from iopath.common.file_io import PathManager as PathManagerBase

from cv.configs import *
from cv.src.cv_parser import get_parser

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
    from cv.configs.defaults import _C

    return _C.clone()

def setup_config(args=None, default=False):

    cfg = get_cfg()
    
    if default: return cfg

    if args is None: args = get_parser().parse_args()
    cfg.num_gpus = len(args.gpu_devices.split(","))

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
        elif model_name.lower() == "inception":
            InceptionConfig(cfg)
        elif model_name.lower() == "densenet":
            DenseNetConfig(cfg)
        elif model_name.lower() == "resnet":
            ResNetConfig(cfg)
        elif model_name.lower() == "inceptionv2":
            Inceptionv2Config(cfg)
        elif model_name.lower() == "inceptionv3":
            Inceptionv3Config(cfg)
        elif model_name.lower() == "inceptionv4":
            Inceptionv4Config(cfg)
        elif model_name.lower() == "resnext":
            ResNeXtConfig(cfg)
        elif model_name.lower() == "mobilenet":
            MobileNetConfig(cfg)
        elif model_name.lower() == "mobilenetv2":
            MobileNetv2Config(cfg)
        elif model_name.lower() == "senet":
            SENetConfig(cfg)
        elif model_name.lower() == "rcnn":
            RCNNConfig(cfg)
        elif model_name.lower() == "sppnet":
            SPPNetConfig(cfg)
        elif model_name.lower() == "vit":
            ViTConfig(cfg)
        elif model_name.lower() == "convnext":
            ConvNeXtConfig(cfg)
        elif model_name.lower() == "deit":
            DeiTConfig(cfg)
        elif model_name.lower() == "resnetsb":
            ResNetSBConfig(cfg)
        elif model_name.lower() == "t2t_vit":
            T2T_ViTConfig(cfg)
        elif model_name.lower() == "tnt":
            TNTConfig(cfg)
        elif model_name.lower() == "bot_vit":
            BoT_ViTConfig(cfg)
        elif model_name.lower() == "hpool_vit":
            HPool_ViTConfig(cfg)
        elif model_name.lower() == "convit":
            ConViTConfig(cfg)
        elif model_name.lower() == "deepvit":
            DeepViTConfig(cfg)
        elif model_name.lower() == "ceit":
            CeiTConfig(cfg)
        elif model_name.lower() == "swint":
            SwinTConfig(cfg)
        elif model_name.lower() == "fasterrcnn":
            FasterRCNNConfig(cfg)
        elif model_name.lower() == "fcn":
            FCNConfig(cfg)
        elif model_name.lower() == "unet":
            UNetConfig(cfg)
        elif model_name.lower() == "deeplabv1":
            DeepLabv1Config(cfg)
        elif model_name.lower() == "segnet":
            SegNetConfig(cfg)