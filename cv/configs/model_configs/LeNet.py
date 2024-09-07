def LeNetConfig(cfg):

    from cv.configs.config import CfgNode as CN

    cfg.LeNet = CN()
    cfg.LOGGING.NAME = "LeNet"
    cfg.METRICS.NAME = "LeNet"
    cfg.CHECKPOINT.BASENAME = "LeNet"
    cfg.TENSORBOARD.BASENAME = "LeNet"

    cfg.LeNet.TRANSFORMS = CN()
    cfg.LeNet.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="Resize", params=dict(size=(224, 224))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.LeNet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="Resize", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.LeNet.OPTIMIZER = CN()
    cfg.LeNet.OPTIMIZER.NAME = "SGD"
    cfg.LeNet.OPTIMIZER.PARAMS = CN()
    cfg.LeNet.OPTIMIZER.PARAMS.lr = 0.01
    cfg.LeNet.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.LeNet.OPTIMIZER.PARAMS.weight_decay = 0.0001

    cfg.LeNet.LOSS = CN()
    cfg.LeNet.LOSS.NAME = "CrossEntropyLoss"
    cfg.LeNet.LOSS.PARAMS = CN()
    cfg.LeNet.LOSS.PARAMS.weight = None
    cfg.LeNet.LOSS.PARAMS.size_average = None
    cfg.LeNet.LOSS.PARAMS.ignore_index = -100
    cfg.LeNet.LOSS.PARAMS.reduce = None
    cfg.LeNet.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = "L2"
    cfg.REGULARIZATION.STRENGTH = 0.001