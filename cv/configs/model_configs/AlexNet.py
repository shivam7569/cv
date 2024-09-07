def AlexNetConfig(cfg):

    from cv.configs.config import CfgNode as CN

    cfg.ASYNC_TRAINING = True
    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.AlexNet = CN()
    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "AlexNet"
    cfg.METRICS.NAME = "AlexNet"
    cfg.CHECKPOINT.BASENAME = "AlexNet"
    cfg.TENSORBOARD.BASENAME = "AlexNet"
    cfg.CHECKPOINT.TREE = False

    cfg.AlexNet.PARAMS = CN()
    cfg.AlexNet.PARAMS.num_classes = 1000
    cfg.AlexNet.PARAMS.in_channels = 3

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 100

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="alexNetresize", params=dict())
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="alexNetresize", params=dict())
    ]

    cfg.AlexNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.batch_size = 128
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.AlexNet.DATALOADER_VAL_PARAMS = CN()
    cfg.AlexNet.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.AlexNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.AlexNet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.AlexNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.AlexNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.AlexNet.TRANSFORMS = CN()
    cfg.AlexNet.TRANSFORMS.TRAIN = [
        dict(name="FancyPCA", params=dict(alpha_std=0.1, p=0.5)),
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.AlexNet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.AlexNet.OPTIMIZER = CN()
    cfg.AlexNet.OPTIMIZER.NAME = "SGD"
    cfg.AlexNet.OPTIMIZER.PARAMS = CN()
    cfg.AlexNet.OPTIMIZER.PARAMS.lr = 0.01 * 3
    cfg.AlexNet.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.AlexNet.OPTIMIZER.PARAMS.weight_decay = 0.0005

    cfg.AlexNet.LR_SCHEDULER = CN()
    cfg.AlexNet.LR_SCHEDULER.NAME = "StepLR"
    cfg.AlexNet.LR_SCHEDULER.PARAMS = CN()
    cfg.AlexNet.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.AlexNet.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.AlexNet.LR_SCHEDULER.PARAMS.verbose = False

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