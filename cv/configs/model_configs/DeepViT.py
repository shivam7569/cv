def DeepViTConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.DeepViT = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = True

    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "DeepViT"
    cfg.METRICS.NAME = "DeepViT"
    cfg.CHECKPOINT.BASENAME = "DeepViT"
    cfg.TENSORBOARD.BASENAME = "DeepViT"
    cfg.PROFILER.BASENAME = "DeepViT/Exp"
    cfg.CHECKPOINT.TREE = False

    cfg.DeepViT.PARAMS = CN()
    cfg.DeepViT.PARAMS.num_classes = 1000
    cfg.DeepViT.PARAMS.d_model = 384
    cfg.DeepViT.PARAMS.image_size = 224
    cfg.DeepViT.PARAMS.patch_size = 16
    cfg.DeepViT.PARAMS.classifier_mlp_d = 2048
    cfg.DeepViT.PARAMS.encoder_mlp_d = 384 * 3
    cfg.DeepViT.PARAMS.encoder_num_heads = 12
    cfg.DeepViT.PARAMS.num_encoder_blocks = 32
    cfg.DeepViT.PARAMS.dropout = 0.0
    cfg.DeepViT.PARAMS.encoder_dropout = 0.0
    cfg.DeepViT.PARAMS.encoder_attention_dropout = 0.0
    cfg.DeepViT.PARAMS.encoder_projection_dropout = 0.0
    cfg.DeepViT.PARAMS.patchify_technique = "linear"
    cfg.DeepViT.PARAMS.stochastic_depth = True
    cfg.DeepViT.PARAMS.stochastic_depth_mp = 0.1
    cfg.DeepViT.PARAMS.layer_scale = 1e-4
    cfg.DeepViT.PARAMS.ln_order = "residual"
    cfg.DeepViT.PARAMS.re_attention = True
    cfg.DeepViT.PARAMS.in_channels = 3

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 500
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 256
    cfg.TRAIN.PARAMS.gradient_clipping = ("norm", 1)
    cfg.TRAIN.PARAMS.exponential_moving_average = CN()
    cfg.TRAIN.PARAMS.exponential_moving_average.beta = 0.999
    cfg.TRAIN.PARAMS.exponential_moving_average.warmup_steps = 0
    cfg.TRAIN.PARAMS.exponential_moving_average.decay_period = 1
    cfg.TRAIN.PARAMS.exponential_moving_average.decay_method = "constant"
    cfg.TRAIN.PARAMS.updateStochasticDepthRate = None

    cfg.DATA_MIXING.enabled = True
    cfg.DATA_MIXING.mixup_alpha = 0.8
    cfg.DATA_MIXING.cutmix_alpha = 1.0
    cfg.DATA_MIXING.cutmix_minmax = None
    cfg.DATA_MIXING.prob = 0.7
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
        dict(func="resizeWithAspectRatio", params=dict(size=224 if cfg.DeepViT.PARAMS.image_size == 192 else 256))
    ]

    cfg.DeepViT.DATALOADER_TRAIN_PARAMS = CN()
    cfg.DeepViT.DATALOADER_TRAIN_PARAMS.batch_size = 16
    cfg.DeepViT.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.DeepViT.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.DeepViT.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.DeepViT.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.DeepViT.DATALOADER_VAL_PARAMS = CN()
    cfg.DeepViT.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.DeepViT.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.DeepViT.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.DeepViT.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.DeepViT.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.DeepViT.TRANSFORMS = CN()

    cfg.DeepViT.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=(cfg.DeepViT.PARAMS.image_size, cfg.DeepViT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ThreeAugment", params=dict(p=0.5)),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=0.3, contrast=0.3, saturation=0.3))
            ], p=0.5
        )),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.DeepViT.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(cfg.DeepViT.PARAMS.image_size, cfg.DeepViT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.DeepViT.OPTIMIZER = CN()
    cfg.DeepViT.OPTIMIZER.NAME = "AdamW"
    cfg.DeepViT.OPTIMIZER.PARAMS = CN()
    cfg.DeepViT.OPTIMIZER.PARAMS.lr = 5e-4 * cfg.num_gpus 
    cfg.DeepViT.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.DeepViT.OPTIMIZER.PARAMS.weight_decay = 0.05

    cfg.DeepViT.LR_SCHEDULER = CN()
    cfg.DeepViT.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.DeepViT.LR_SCHEDULER.PARAMS = CN()
    cfg.DeepViT.LR_SCHEDULER.PARAMS.lr_initial = 1e-8
    cfg.DeepViT.LR_SCHEDULER.PARAMS.lr_final = cfg.DeepViT.OPTIMIZER.PARAMS.lr
    cfg.DeepViT.LR_SCHEDULER.PARAMS.warmup_epochs = 3
    cfg.DeepViT.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.DeepViT.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.DeepViT.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingWarmRestarts"
    cfg.DeepViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.DeepViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_0 = 15
    cfg.DeepViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_mult = 3
    cfg.DeepViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 1e-8
    cfg.DeepViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.DeepViT.LOSS = CN()
    cfg.DeepViT.LOSS.NAME = "CrossEntropyLoss"
    cfg.DeepViT.LOSS.PARAMS = CN()
    cfg.DeepViT.LOSS.PARAMS.weight = None
    cfg.DeepViT.LOSS.PARAMS.size_average = None
    cfg.DeepViT.LOSS.PARAMS.ignore_index = -100
    cfg.DeepViT.LOSS.PARAMS.reduce = None
    cfg.DeepViT.LOSS.PARAMS.reduction = "mean"
    cfg.DeepViT.LOSS.PARAMS.label_smoothing = 0.0

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0