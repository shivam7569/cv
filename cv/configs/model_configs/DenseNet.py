def DenseNetConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.DenseNet = CN()

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "DenseNet"
    cfg.METRICS.NAME = "DenseNet"
    cfg.CHECKPOINT.BASENAME = "DenseNet"
    cfg.TENSORBOARD.BASENAME = "DenseNet"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.DenseNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.DenseNet.DATALOADER_TRAIN_PARAMS.batch_size = 64
    cfg.DenseNet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.DenseNet.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.DenseNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.DenseNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.DenseNet.DATALOADER_VAL_PARAMS = CN()
    cfg.DenseNet.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.DenseNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.DenseNet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.DenseNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.DenseNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.DenseNet.TRANSFORMS = CN()
    cfg.DenseNet.TRANSFORMS.TRAIN = [
        dict(name="FancyPCA", params=dict(alpha_std=0.1, p=0.5)),
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=224)),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=0.4, contrast=0.4, saturation=0.4))
            ], p=0.5
        )),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.DenseNet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.DenseNet.OPTIMIZER = CN()
    cfg.DenseNet.OPTIMIZER.NAME = "SGD"
    cfg.DenseNet.OPTIMIZER.PARAMS = CN()
    cfg.DenseNet.OPTIMIZER.PARAMS.lr = 0.1
    cfg.DenseNet.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.DenseNet.OPTIMIZER.PARAMS.weight_decay = 1e-4

    cfg.DenseNet.LR_SCHEDULER = CN()
    cfg.DenseNet.LR_SCHEDULER.NAME = "StepLR"
    cfg.DenseNet.LR_SCHEDULER.PARAMS = CN()
    cfg.DenseNet.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.DenseNet.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.DenseNet.LR_SCHEDULER.PARAMS.verbose = False

    cfg.DenseNet.LOSS = CN()
    cfg.DenseNet.LOSS.NAME = "CrossEntropyLoss"
    cfg.DenseNet.LOSS.PARAMS = CN()
    cfg.DenseNet.LOSS.PARAMS.weight = None
    cfg.DenseNet.LOSS.PARAMS.size_average = None
    cfg.DenseNet.LOSS.PARAMS.ignore_index = -100
    cfg.DenseNet.LOSS.PARAMS.reduce = None
    cfg.DenseNet.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0