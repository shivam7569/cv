def BoT_ViTConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.BoT_ViT = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = False
    cfg.SAVE_FIRST_SAMPLE = False
    cfg.num_gpus = 3

    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "BoT_ViT"
    cfg.METRICS.NAME = "BoT_ViT"
    cfg.CHECKPOINT.BASENAME = "BoT_ViT"
    cfg.TENSORBOARD.BASENAME = "BoT_ViT"
    cfg.PROFILER.BASENAME = "BoT_ViT/Exp"

    cfg.BoT_ViT.PARAMS = CN()
    cfg.BoT_ViT.PARAMS.num_classes = 1000
    cfg.BoT_ViT.PARAMS.mhsa_num_heads = 16
    cfg.BoT_ViT.PARAMS.attention_dropout = 0.0
    cfg.BoT_ViT.PARAMS.in_channels = 3

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 250
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 4096
    cfg.TRAIN.PARAMS.gradient_clipping = None
    cfg.TRAIN.PARAMS.exponential_moving_average = None

    cfg.DATA_MIXING.enabled = False
    cfg.DATA_MIXING.mixup_alpha = 0.2
    cfg.DATA_MIXING.cutmix_alpha = 1.0
    cfg.DATA_MIXING.cutmix_minmax = None
    cfg.DATA_MIXING.prob = 0.8
    cfg.DATA_MIXING.switch_prob = 0.5
    cfg.DATA_MIXING.mode = "batch"
    cfg.DATA_MIXING.correct_lam = True
    cfg.DATA_MIXING.label_smoothing = 0.0
    cfg.DATA_MIXING.one_hot_targets = False
    cfg.DATA_MIXING.num_classes = 1000

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=1184))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=1184))
    ]

    cfg.BoT_ViT.DATALOADER_TRAIN_PARAMS = CN()
    cfg.BoT_ViT.DATALOADER_TRAIN_PARAMS.batch_size = 1
    cfg.BoT_ViT.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.BoT_ViT.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.BoT_ViT.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.BoT_ViT.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.BoT_ViT.DATALOADER_VAL_PARAMS = CN()
    cfg.BoT_ViT.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.BoT_ViT.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.BoT_ViT.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.BoT_ViT.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.BoT_ViT.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.BoT_ViT.TRANSFORMS = CN()

    cfg.BoT_ViT.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=(1024, 1024))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="RandomAugmentation", params=dict(m=10, n=2, mstd=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.BoT_ViT.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(1024, 1024))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.BoT_ViT.OPTIMIZER = CN()
    cfg.BoT_ViT.OPTIMIZER.NAME = "AdamW"
    cfg.BoT_ViT.OPTIMIZER.PARAMS = CN()
    cfg.BoT_ViT.OPTIMIZER.PARAMS.lr = 1e-3 * cfg.num_gpus
    cfg.BoT_ViT.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.BoT_ViT.OPTIMIZER.PARAMS.weight_decay = 4e-5

    cfg.BoT_ViT.LR_SCHEDULER = CN()
    cfg.BoT_ViT.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.BoT_ViT.LR_SCHEDULER.PARAMS = CN()
    cfg.BoT_ViT.LR_SCHEDULER.PARAMS.lr_initial = 0.0
    cfg.BoT_ViT.LR_SCHEDULER.PARAMS.lr_final = cfg.BoT_ViT.OPTIMIZER.PARAMS.lr
    cfg.BoT_ViT.LR_SCHEDULER.PARAMS.warmup_epochs = 5
    cfg.BoT_ViT.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.BoT_ViT.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.BoT_ViT.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingWarmRestarts"
    cfg.BoT_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.BoT_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_0 = 5
    cfg.BoT_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_mult = 3
    cfg.BoT_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 0.0
    cfg.BoT_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.BoT_ViT.LOSS = CN()
    cfg.BoT_ViT.LOSS.NAME = "CrossEntropyLoss"
    cfg.BoT_ViT.LOSS.PARAMS = CN()
    cfg.BoT_ViT.LOSS.PARAMS.weight = None
    cfg.BoT_ViT.LOSS.PARAMS.size_average = None
    cfg.BoT_ViT.LOSS.PARAMS.ignore_index = -100
    cfg.BoT_ViT.LOSS.PARAMS.reduce = None
    cfg.BoT_ViT.LOSS.PARAMS.reduction = "mean"
    cfg.BoT_ViT.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0