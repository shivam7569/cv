def VGG16Config(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.ASYNC_TRAINING = True
    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.VGG16 = CN()
    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "VGG16"
    cfg.METRICS.NAME = "VGG16"
    cfg.CHECKPOINT.BASENAME = "VGG16"
    cfg.TENSORBOARD.BASENAME = "VGG16"

    cfg.VGG16.PARAMS = CN()
    cfg.VGG16.PARAMS.num_classes = 1000
    cfg.VGG16.PARAMS.in_channels = 3

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 150

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="vgg16resize", params=dict(s_min=256, s_max=512))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.VGG16.DATALOADER_TRAIN_PARAMS = CN()
    cfg.VGG16.DATALOADER_TRAIN_PARAMS.batch_size = 128
    cfg.VGG16.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.VGG16.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.VGG16.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.VGG16.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.VGG16.DATALOADER_VAL_PARAMS = CN()
    cfg.VGG16.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.VGG16.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.VGG16.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.VGG16.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.VGG16.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.VGG16.TRANSFORMS = CN()
    cfg.VGG16.TRANSFORMS.TRAIN = [
        dict(name="FancyPCA", params=dict(alpha_std=0.1, p=0.5)),
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.VGG16.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.VGG16.OPTIMIZER = CN()
    cfg.VGG16.OPTIMIZER.NAME = "SGD"
    cfg.VGG16.OPTIMIZER.PARAMS = CN()
    cfg.VGG16.OPTIMIZER.PARAMS.lr = 0.01
    cfg.VGG16.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.VGG16.OPTIMIZER.PARAMS.weight_decay = 1e-4

    cfg.VGG16.LR_SCHEDULER = CN()
    cfg.VGG16.LR_SCHEDULER.NAME = "StepLR"
    cfg.VGG16.LR_SCHEDULER.PARAMS = CN()
    cfg.VGG16.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.VGG16.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.VGG16.LR_SCHEDULER.PARAMS.verbose = False

    cfg.VGG16.LOSS = CN()
    cfg.VGG16.LOSS.NAME = "CrossEntropyLoss"
    cfg.VGG16.LOSS.PARAMS = CN()
    cfg.VGG16.LOSS.PARAMS.weight = None
    cfg.VGG16.LOSS.PARAMS.size_average = None
    cfg.VGG16.LOSS.PARAMS.ignore_index = -100
    cfg.VGG16.LOSS.PARAMS.reduce = None
    cfg.VGG16.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0