import math


def FCNConfig(cfg):

    from cv.configs.config import CfgNode as CN

    cfg.FCN = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = False
    cfg.SAVE_FIRST_SAMPLE = True
    cfg.WRITE_TENSORBOARD_GRAPH = False

    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "FCN"
    cfg.METRICS.NAME = "FCN"
    cfg.CHECKPOINT.BASENAME = "FCN"
    cfg.TENSORBOARD.BASENAME = "FCN"
    cfg.PROFILER.BASENAME = "FCN/Profiling"

    cfg.FCN.PARAMS = CN()
    cfg.FCN.PARAMS.backbone_name = "VGG16"
    cfg.FCN.PARAMS.backbone_params = CN()
    cfg.FCN.PARAMS.backbone_params.num_classes = 1000
    cfg.FCN.PARAMS.num_classes = 81

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 500
    cfg.TRAIN.PARAMS.gradient_accumulation = False
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = None
    cfg.TRAIN.PARAMS.gradient_clipping = None
    cfg.TRAIN.PARAMS.exponential_moving_average = None
    cfg.TRAIN.PARAMS.updateStochasticDepthRate = None

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256)),
        dict(func="mask_to_img_size", params=dict())
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256)),
        dict(func="mask_to_img_size", params=dict())
    ]

    cfg.FCN.DATALOADER_TRAIN_PARAMS = CN()
    cfg.FCN.DATALOADER_TRAIN_PARAMS.batch_size = 8
    cfg.FCN.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.FCN.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.FCN.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.FCN.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.FCN.DATALOADER_VAL_PARAMS = CN()
    cfg.FCN.DATALOADER_VAL_PARAMS.batch_size = 8
    cfg.FCN.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.FCN.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.FCN.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.FCN.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.FCN.TRANSFORMS = CN()

    cfg.FCN.TRANSFORMS.TRAIN = [
        dict(name="SegmentationToPILImage", params=dict()),
        dict(name="SegmentationRandomCrop", params=dict(size=(224, 224))),
        dict(name="SegmentationHorizontalFlip", params=dict(p=0.5)),
        dict(name="SegmentationToTensor", params=dict()),
        dict(name="SegmentationNormalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.FCN.TRANSFORMS.VAL = [
        dict(name="SegmentationToPILImage", params=dict()),
        dict(name="SegmentationCenterCrop", params=dict(size=(224, 224))),
        dict(name="SegmentationHorizontalFlip", params=dict(p=0.5)),
        dict(name="SegmentationToTensor", params=dict()),
        dict(name="SegmentationNormalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.FCN.OPTIMIZER = CN()
    cfg.FCN.OPTIMIZER.NAME = "SGD"
    cfg.FCN.OPTIMIZER.PARAMS = CN()
    cfg.FCN.OPTIMIZER.PARAMS.lr = 1e-4
    cfg.FCN.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.FCN.OPTIMIZER.PARAMS.weight_decay = 1e-5

    cfg.FCN.LR_SCHEDULER = CN()
    cfg.FCN.LR_SCHEDULER.NAME = "CosineAnnealingWarmRestarts"
    cfg.FCN.LR_SCHEDULER.PARAMS = CN()
    cfg.FCN.LR_SCHEDULER.PARAMS.T_0 = 20
    cfg.FCN.LR_SCHEDULER.PARAMS.T_mult = 2
    cfg.FCN.LR_SCHEDULER.PARAMS.eta_min = 1e-5
    cfg.FCN.LR_SCHEDULER.PARAMS.last_epoch = -1
    
    cfg.FCN.LOSS = CN()
    cfg.FCN.LOSS.NAME = "DiceLoss"
    cfg.FCN.LOSS.PARAMS = CN()
    cfg.FCN.LOSS.PARAMS.num_classes = 81
    cfg.FCN.LOSS.PARAMS.ignore_index = -1
    cfg.FCN.LOSS.PARAMS.reduction = "mean"
    cfg.FCN.LOSS.PARAMS.log_loss = True
    cfg.FCN.LOSS.PARAMS.smooth = 0.0
    cfg.FCN.LOSS.PARAMS.classes =   None

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0
