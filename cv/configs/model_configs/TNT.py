def TNTConfig(cfg):

    from cv.configs.config import CfgNode as CN

    cfg.TNT = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = True

    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "TNT"
    cfg.METRICS.NAME = "TNT"
    cfg.CHECKPOINT.BASENAME = "TNT"
    cfg.TENSORBOARD.BASENAME = "TNT"
    cfg.PROFILER.BASENAME = "TNT/Profiling"

    cfg.TNT.PARAMS = CN()
    cfg.TNT.PARAMS.image_size = 224
    cfg.TNT.PARAMS.patch_size = 16
    cfg.TNT.PARAMS.pixel_size = 4
    cfg.TNT.PARAMS.patch_embed = 640
    cfg.TNT.PARAMS.pixel_embed = 40
    cfg.TNT.PARAMS.patch_d_ff = 2560
    cfg.TNT.PARAMS.pixel_d_ff = 160
    cfg.TNT.PARAMS.patch_num_heads = 10
    cfg.TNT.PARAMS.pixel_num_heads = 4
    cfg.TNT.PARAMS.num_blocks = 12
    cfg.TNT.PARAMS.patch_encoder_dropout = 0.1
    cfg.TNT.PARAMS.patch_attention_dropout = 0.0
    cfg.TNT.PARAMS.patch_projection_dropout = 0.0
    cfg.TNT.PARAMS.patch_ln_order = "residual"
    cfg.TNT.PARAMS.patch_stodepth = True
    cfg.TNT.PARAMS.patch_stodepth_mp = 0.1
    cfg.TNT.PARAMS.patch_layer_scale = None
    cfg.TNT.PARAMS.patch_se_block = (cfg.TNT.PARAMS.image_size // cfg.TNT.PARAMS.patch_size) ** 2 + 1
    cfg.TNT.PARAMS.patch_se_points = "msa"
    cfg.TNT.PARAMS.patch_qkv_bias = False
    cfg.TNT.PARAMS.patch_in_dims = None
    cfg.TNT.PARAMS.pixel_encoder_dropout = 0.0
    cfg.TNT.PARAMS.pixel_attention_dropout = 0.0
    cfg.TNT.PARAMS.pixel_projection_dropout = 0.0
    cfg.TNT.PARAMS.pixel_ln_order = "pre"
    cfg.TNT.PARAMS.pixel_stodepth = False
    cfg.TNT.PARAMS.pixel_stodepth_mp = None
    cfg.TNT.PARAMS.pixel_layer_scale = None
    cfg.TNT.PARAMS.pixel_se_block = (cfg.TNT.PARAMS.patch_size // cfg.TNT.PARAMS.pixel_size) ** 2
    cfg.TNT.PARAMS.pixel_se_points = "msa"
    cfg.TNT.PARAMS.pixel_qkv_bias = False
    cfg.TNT.PARAMS.pixel_in_dims = None
    cfg.TNT.PARAMS.patchify_technique = "linear"
    cfg.TNT.PARAMS.classifier_mlp_d = 2048
    cfg.TNT.PARAMS.classifier_dropout = 0.0
    cfg.TNT.PARAMS.num_classes = 1000
    cfg.TNT.PARAMS.in_channels = 3

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 1000
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 1024
    cfg.TRAIN.PARAMS.gradient_clipping = ("norm", 1)
    cfg.TRAIN.PARAMS.exponential_moving_average = CN()
    cfg.TRAIN.PARAMS.exponential_moving_average.beta = 0.999
    cfg.TRAIN.PARAMS.exponential_moving_average.warmup_steps = 0
    cfg.TRAIN.PARAMS.exponential_moving_average.decay_period = 1
    cfg.TRAIN.PARAMS.exponential_moving_average.decay_method = "constant"
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
        dict(func="resizeWithAspectRatio", params=dict(size=224 if cfg.TNT.PARAMS.image_size == 192 else 256))
    ]

    cfg.TNT.DATALOADER_TRAIN_PARAMS = CN()
    cfg.TNT.DATALOADER_TRAIN_PARAMS.batch_size = 16
    cfg.TNT.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.TNT.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.TNT.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.TNT.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.TNT.DATALOADER_VAL_PARAMS = CN()
    cfg.TNT.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.TNT.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.TNT.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.TNT.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.TNT.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.TNT.TRANSFORMS = CN()

    cfg.DeiT.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=(cfg.TNT.PARAMS.image_size, cfg.TNT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="RandomAugmentation", params=dict(m=9, n=2, mstd=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
        dict(name="RandomCutOut", params=dict(probability=0.25, mode="pixel", device="cpu"))
    ]

    cfg.TNT.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(cfg.TNT.PARAMS.image_size, cfg.TNT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.TNT.OPTIMIZER = CN()
    cfg.TNT.OPTIMIZER.NAME = "AdamW"
    cfg.TNT.OPTIMIZER.PARAMS = CN()
    cfg.TNT.OPTIMIZER.PARAMS.lr = 5e-4 * cfg.num_gpus 
    cfg.TNT.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.TNT.OPTIMIZER.PARAMS.weight_decay = 5e-2

    cfg.TNT.LR_SCHEDULER = CN()
    cfg.TNT.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.TNT.LR_SCHEDULER.PARAMS = CN()
    cfg.TNT.LR_SCHEDULER.PARAMS.lr_initial = 1e-8
    cfg.TNT.LR_SCHEDULER.PARAMS.lr_final = cfg.TNT.OPTIMIZER.PARAMS.lr
    cfg.TNT.LR_SCHEDULER.PARAMS.warmup_epochs = 5
    cfg.TNT.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.TNT.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.TNT.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingWarmRestarts"
    cfg.TNT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.TNT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_0 = 15
    cfg.TNT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_mult = 3
    cfg.TNT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 1e-8
    cfg.TNT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.TNT.LOSS = CN()
    cfg.TNT.LOSS.NAME = "CrossEntropyLoss"
    cfg.TNT.LOSS.PARAMS = CN()
    cfg.TNT.LOSS.PARAMS.weight = None
    cfg.TNT.LOSS.PARAMS.size_average = None
    cfg.TNT.LOSS.PARAMS.ignore_index = -100
    cfg.TNT.LOSS.PARAMS.reduce = None
    cfg.TNT.LOSS.PARAMS.reduction = "mean"
    cfg.TNT.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0
