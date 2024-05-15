import math


def UNetConfig(cfg):

    from configs.config import CfgNode as CN

    cfg.UNet = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = False
    cfg.SAVE_FIRST_SAMPLE = True
    cfg.WRITE_TENSORBOARD_GRAPH = True

    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "UNet"
    cfg.METRICS.NAME = "UNet"
    cfg.CHECKPOINT.BASENAME = "UNet"
    cfg.TENSORBOARD.BASENAME = "UNet"
    cfg.PROFILER.BASENAME = "UNet/Profiling"

    cfg.UNet.PARAMS = CN()
    cfg.UNet.PARAMS.channels = [64, 128, 256, 512, 1024]
    cfg.UNet.PARAMS.in_channels = 3
    cfg.UNet.PARAMS.num_classes = 81
    cfg.UNet.PARAMS.dropout = 0.5
    cfg.UNet.PARAMS.retain_size = True

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
        dict(func="resizeWithAspectRatio", params=dict(size=600)),
        dict(func="mask_to_img_size", params=dict())
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=600)),
        dict(func="mask_to_img_size", params=dict())
    ]

    cfg.UNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.UNet.DATALOADER_TRAIN_PARAMS.batch_size = 1
    cfg.UNet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.UNet.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.UNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.UNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.UNet.DATALOADER_VAL_PARAMS = CN()
    cfg.UNet.DATALOADER_VAL_PARAMS.batch_size = 1
    cfg.UNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.UNet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.UNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.UNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.UNet.TRANSFORMS = CN()

    cfg.UNet.TRANSFORMS.TRAIN = [
        dict(name="SegmentationToPILImage", params=dict()),
        dict(name="SegmentationElasticTransform", params=dict(p=0.1, alpha=50.0, sigma=5.0)),
        dict(name="SegmentationRandomCrop", params=dict(size=(572, 572))),
        dict(name="SegmentationHorizontalFlip", params=dict(p=0.5)),
        dict(name="SegmentationToTensor", params=dict()),
        dict(name="SegmentationNormalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.UNet.TRANSFORMS.VAL = [
        dict(name="SegmentationToPILImage", params=dict()),
        dict(name="SegmentationCenterCrop", params=dict(size=(572, 572))),
        dict(name="SegmentationHorizontalFlip", params=dict(p=0.5)),
        dict(name="SegmentationToTensor", params=dict()),
        dict(name="SegmentationNormalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.UNet.OPTIMIZER = CN()
    cfg.UNet.OPTIMIZER.NAME = "SGD"
    cfg.UNet.OPTIMIZER.PARAMS = CN()
    cfg.UNet.OPTIMIZER.PARAMS.lr = 1e-4 * math.sqrt(cfg.num_gpus) 
    cfg.UNet.OPTIMIZER.PARAMS.momentum = 0.99
    cfg.UNet.OPTIMIZER.PARAMS.weight_decay = 5**-4
    
    cfg.UNet.LOSS = CN()
    cfg.UNet.LOSS.NAME = "CrossEntropyLoss"
    cfg.UNet.LOSS.PARAMS = CN()
    cfg.UNet.LOSS.PARAMS.size_average = None
    cfg.UNet.LOSS.PARAMS.ignore_index = -100
    cfg.UNet.LOSS.PARAMS.reduce = None
    cfg.UNet.LOSS.PARAMS.reduction = "mean"
    # cfg.UNet.LOSS.PARAMS.class_weightage_method = "inverse_frequency"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0
