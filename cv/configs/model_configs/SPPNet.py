def SPPNetConfig(cfg):

    from cv.configs.config import CfgNode as CN

    cfg.SPPNet = CN()
    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "SPPNet"
    cfg.METRICS.NAME = "SPPNet"
    cfg.CHECKPOINT.BASENAME = "SPPNet"
    cfg.CHECKPOINT.TREE = False
    cfg.TENSORBOARD.BASENAME = "SPPNet"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="alexNetresize", params=dict())
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="alexNetresize", params=dict())
    ]

    cfg.COLLATE_FN = CN()
    cfg.COLLATE_FN.PROCESS = "RandomSize"
    cfg.COLLATE_FN.SIZES = [180, 224, 227, 256]

    cfg.SPPNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.SPPNet.DATALOADER_TRAIN_PARAMS.batch_size = 128
    cfg.SPPNet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.SPPNet.DATALOADER_TRAIN_PARAMS.num_workers = 16
    cfg.SPPNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.SPPNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.SPPNet.DATALOADER_VAL_PARAMS = CN()
    cfg.SPPNet.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.SPPNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.SPPNet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.SPPNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.SPPNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.SPPNet.TRANSFORMS = CN()
    cfg.SPPNet.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.SPPNet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.SPPNet.OPTIMIZER = CN()
    cfg.SPPNet.OPTIMIZER.NAME = "SGD"
    cfg.SPPNet.OPTIMIZER.PARAMS = CN()
    cfg.SPPNet.OPTIMIZER.PARAMS.lr = 0.01
    cfg.SPPNet.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.SPPNet.OPTIMIZER.PARAMS.weight_decay = 0.0005

    cfg.SPPNet.LR_SCHEDULER = CN()
    cfg.SPPNet.LR_SCHEDULER.NAME = "StepLR"
    cfg.SPPNet.LR_SCHEDULER.PARAMS = CN()
    cfg.SPPNet.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.SPPNet.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.SPPNet.LR_SCHEDULER.PARAMS.verbose = False

    cfg.SPPNet.LOSS = CN()
    cfg.SPPNet.LOSS.NAME = "CrossEntropyLoss"
    cfg.SPPNet.LOSS.PARAMS = CN()
    cfg.SPPNet.LOSS.PARAMS.weight = None
    cfg.SPPNet.LOSS.PARAMS.size_average = None
    cfg.SPPNet.LOSS.PARAMS.ignore_index = -100
    cfg.SPPNet.LOSS.PARAMS.reduce = None
    cfg.SPPNet.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0