import math


def SwinTConfig(cfg):

    from cv.configs.config import CfgNode as CN

    cfg.SwinT = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = False
    cfg.SAVE_FIRST_SAMPLE = False
    cfg.WRITE_TENSORBOARD_GRAPH = False

    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "SwinT"
    cfg.METRICS.NAME = "SwinT"
    cfg.CHECKPOINT.BASENAME = "SwinT"
    cfg.TENSORBOARD.BASENAME = "SwinT"
    cfg.PROFILER.BASENAME = "SwinT/Profiling"

    cfg.SwinT.PARAMS = CN()
    cfg.SwinT.PARAMS.embed_c = 128
    cfg.SwinT.PARAMS.patch_size = 4
    cfg.SwinT.PARAMS.window_size = 7
    cfg.SwinT.PARAMS.d_ff_ratio = 4
    cfg.SwinT.PARAMS.num_heads_per_stage = [4, 8, 16, 32]
    cfg.SwinT.PARAMS.shift = 2
    cfg.SwinT.PARAMS.patch_merge_size = 2
    cfg.SwinT.PARAMS.stage_embed_dim_ratio = 2
    cfg.SwinT.PARAMS.num_blocks = [2, 2, 18, 2]
    cfg.SwinT.PARAMS.classifier_mlp_d = 2048
    cfg.SwinT.PARAMS.encoder_dropout = 0.0
    cfg.SwinT.PARAMS.attention_dropout = 0.0
    cfg.SwinT.PARAMS.projection_dropout = 0.0
    cfg.SwinT.PARAMS.classifier_dropout = 0.0
    cfg.SwinT.PARAMS.global_aggregate = "avg"
    cfg.SwinT.PARAMS.image_size = 224
    cfg.SwinT.PARAMS.patchify_technique = "convolutional"
    cfg.SwinT.PARAMS.stodepth = True
    cfg.SwinT.PARAMS.stodepth_mp = 0.1
    cfg.SwinT.PARAMS.layer_scale = 1e-6
    cfg.SwinT.PARAMS.qkv_bias = False
    cfg.SwinT.PARAMS.in_channels = 3
    cfg.SwinT.PARAMS.num_classes = 1000

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 1000
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 1024
    cfg.TRAIN.PARAMS.gradient_clipping = None
    cfg.TRAIN.PARAMS.exponential_moving_average = None
    cfg.TRAIN.PARAMS.updateStochasticDepthRate = None

    cfg.DATA_MIXING.enabled = True
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
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=224 if cfg.SwinT.PARAMS.image_size == 192 else 256))
    ]

    cfg.SwinT.DATALOADER_TRAIN_PARAMS = CN()
    cfg.SwinT.DATALOADER_TRAIN_PARAMS.batch_size = 16
    cfg.SwinT.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.SwinT.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.SwinT.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.SwinT.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.SwinT.DATALOADER_VAL_PARAMS = CN()
    cfg.SwinT.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.SwinT.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.SwinT.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.SwinT.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.SwinT.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.SwinT.TRANSFORMS = CN()

    cfg.SwinT.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=(cfg.SwinT.PARAMS.image_size, cfg.SwinT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="RandomAugmentation", params=dict(m=9, n=2, mstd=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
        dict(name="RandomCutOut", params=dict(probability=0.25, mode="pixel", device="cpu"))
    ]

    cfg.SwinT.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(cfg.SwinT.PARAMS.image_size, cfg.SwinT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.SwinT.OPTIMIZER = CN()
    cfg.SwinT.OPTIMIZER.NAME = "AdamW"
    cfg.SwinT.OPTIMIZER.PARAMS = CN()
    cfg.SwinT.OPTIMIZER.PARAMS.lr = 1e-3 * math.sqrt(cfg.num_gpus)
    cfg.SwinT.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.SwinT.OPTIMIZER.PARAMS.weight_decay = 0.05

    cfg.SwinT.LR_SCHEDULER = CN()
    cfg.SwinT.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.SwinT.LR_SCHEDULER.PARAMS = CN()
    cfg.SwinT.LR_SCHEDULER.PARAMS.lr_initial = 1e-8
    cfg.SwinT.LR_SCHEDULER.PARAMS.lr_final = cfg.SwinT.OPTIMIZER.PARAMS.lr
    cfg.SwinT.LR_SCHEDULER.PARAMS.warmup_epochs = 20
    cfg.SwinT.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.SwinT.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.SwinT.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingWarmRestarts"
    cfg.SwinT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.SwinT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_0 = 15
    cfg.SwinT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_mult = 4
    cfg.SwinT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 1e-8
    cfg.SwinT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.SwinT.LOSS = CN()
    cfg.SwinT.LOSS.NAME = "CrossEntropyLoss"
    cfg.SwinT.LOSS.PARAMS = CN()
    cfg.SwinT.LOSS.PARAMS.weight = None
    cfg.SwinT.LOSS.PARAMS.size_average = None
    cfg.SwinT.LOSS.PARAMS.ignore_index = -100
    cfg.SwinT.LOSS.PARAMS.reduce = None
    cfg.SwinT.LOSS.PARAMS.reduction = "mean"
    cfg.SwinT.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0
