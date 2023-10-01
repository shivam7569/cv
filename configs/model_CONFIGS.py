def LeNetConfig(cfg):

    from configs.config import CfgNode as CN

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
    cfg.LeNet.OPTIMIZER.NAME = "Adam"
    cfg.LeNet.OPTIMIZER.PARAMS = CN()
    cfg.LeNet.OPTIMIZER.PARAMS.lr = 0.001
    cfg.LeNet.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.LeNet.OPTIMIZER.PARAMS.eps = 1e-08

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

def AlexNetConfig(cfg):

    from configs.config import CfgNode as CN

    cfg.AlexNet = CN()
    cfg.LOGGING.NAME = "AlexNet"
    cfg.METRICS.NAME = "AlexNet"
    cfg.CHECKPOINT.BASENAME = "AlexNet"
    cfg.TENSORBOARD.BASENAME = "AlexNet"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="alexNetresize"),
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="alexNetresize"),
    ]

    cfg.AlexNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.batch_size = 128
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.num_workers = 16
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.AlexNet.DATALOADER_VAL_PARAMS = CN()
    cfg.AlexNet.DATALOADER_VAL_PARAMS.batch_size = 32
    cfg.AlexNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.AlexNet.DATALOADER_VAL_PARAMS.num_workers = 16
    cfg.AlexNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.AlexNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.AlexNet.TRANSFORMS = CN()
    cfg.AlexNet.TRANSFORMS.TRAIN = [
        dict(name="FancyPCA", params=dict(alpha_std=0.1)),
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[1.0, 1.0, 1.0]))
    ]
    cfg.AlexNet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[1.0, 1.0, 1.0]))
    ]

    cfg.AlexNet.OPTIMIZER = CN()
    cfg.AlexNet.OPTIMIZER.NAME = "SGD"
    cfg.AlexNet.OPTIMIZER.PARAMS = CN()
    cfg.AlexNet.OPTIMIZER.PARAMS.lr = 0.01
    cfg.AlexNet.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.AlexNet.OPTIMIZER.PARAMS.weight_decay = 0.0005

    cfg.AlexNet.LR_SCHEDULER = CN()
    cfg.AlexNet.LR_SCHEDULER.NAME = "ReduceLROnPlateau"
    cfg.AlexNet.LR_SCHEDULER.PARAMS = CN()
    cfg.AlexNet.LR_SCHEDULER.PARAMS.mode = "min"
    cfg.AlexNet.LR_SCHEDULER.PARAMS.factor = 0.1
    cfg.AlexNet.LR_SCHEDULER.PARAMS.patience = 10
    cfg.AlexNet.LR_SCHEDULER.PARAMS.threshold = 0.0001
    cfg.AlexNet.LR_SCHEDULER.PARAMS.threshold_mode = "rel"
    cfg.AlexNet.LR_SCHEDULER.PARAMS.cooldown = 0
    cfg.AlexNet.LR_SCHEDULER.PARAMS.min_lr = 0.0001
    cfg.AlexNet.LR_SCHEDULER.PARAMS.eps = 1e-08
    cfg.AlexNet.LR_SCHEDULER.PARAMS.verbose = True

    cfg.AlexNet.LOSS = CN()
    cfg.AlexNet.LOSS.NAME = "CrossEntropyLoss"
    cfg.AlexNet.LOSS.PARAMS = CN()
    cfg.AlexNet.LOSS.PARAMS.weight = None
    cfg.AlexNet.LOSS.PARAMS.size_average = None
    cfg.AlexNet.LOSS.PARAMS.ignore_index = -100
    cfg.AlexNet.LOSS.PARAMS.reduce = None
    cfg.AlexNet.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0