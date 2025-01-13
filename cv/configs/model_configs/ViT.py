def ViTConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.ViT = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = True

    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "ViT"
    cfg.METRICS.NAME = "ViT"
    cfg.CHECKPOINT.BASENAME = "ViT"
    cfg.TENSORBOARD.BASENAME = "ViT"
    cfg.PROFILER.BASENAME = "ViT/Exp"
    cfg.CHECKPOINT.TREE = False

    cfg.ViT.PARAMS = CN()
    cfg.ViT.PARAMS.num_classes = 1000
    cfg.ViT.PARAMS.d_model = 768
    cfg.ViT.PARAMS.image_size = 192
    cfg.ViT.PARAMS.patch_size = 16
    cfg.ViT.PARAMS.classifier_mlp_d = 2048
    cfg.ViT.PARAMS.encoder_mlp_d = 768 * 4
    cfg.ViT.PARAMS.encoder_num_heads = 12
    cfg.ViT.PARAMS.num_encoder_blocks = 12
    cfg.ViT.PARAMS.registers = 16
    cfg.ViT.PARAMS.dropout = 0.0
    cfg.ViT.PARAMS.encoder_dropout = 0.0
    cfg.ViT.PARAMS.encoder_attention_dropout = 0.0
    cfg.ViT.PARAMS.encoder_projection_dropout = 0.0
    cfg.ViT.PARAMS.patchify_technique = "convolutional"
    cfg.ViT.PARAMS.stochastic_depth = True
    cfg.ViT.PARAMS.stochastic_depth_mp = 0.1
    cfg.ViT.PARAMS.layer_scale = 1e-4
    cfg.ViT.PARAMS.ln_order = "residual"
    cfg.ViT.PARAMS.in_channels = 3
    cfg.ViT.PARAMS.classifier = True

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 600
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 2048
    cfg.TRAIN.PARAMS.gradient_clipping = ("norm", 1)
    cfg.TRAIN.PARAMS.exponential_moving_average = CN()
    cfg.TRAIN.PARAMS.exponential_moving_average.beta = 0.9999
    cfg.TRAIN.PARAMS.exponential_moving_average.update_every = 1
    cfg.TRAIN.PARAMS.exponential_moving_average.update_after_step = 5
    cfg.TRAIN.PARAMS.exponential_moving_average.inv_gamma = 1.0
    cfg.TRAIN.PARAMS.exponential_moving_average.power = 11/10
    cfg.TRAIN.PARAMS.exponential_moving_average.use_foreach = True
    cfg.TRAIN.PARAMS.exponential_moving_average.include_online_model = False
    cfg.TRAIN.PARAMS.exponential_moving_average.update_model_with_ema_every = 32
    cfg.TRAIN.PARAMS.exponential_moving_average.update_model_with_ema_beta = 0.0
    cfg.TRAIN.PARAMS.updateStochasticDepthRate = [dict(
        step_epochs=200, k=0.05
    )]

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
        dict(func="resizeWithAspectRatio", params=dict(size=224 if cfg.ViT.PARAMS.image_size == 192 else 256))
    ]

    cfg.ViT.DATALOADER_TRAIN_PARAMS = CN()
    cfg.ViT.DATALOADER_TRAIN_PARAMS.batch_size = 32
    cfg.ViT.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.ViT.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.ViT.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.ViT.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.ViT.DATALOADER_VAL_PARAMS = CN()
    cfg.ViT.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.ViT.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.ViT.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.ViT.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.ViT.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.ViT.TRANSFORMS = CN()

    cfg.ViT.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=(cfg.ViT.PARAMS.image_size, cfg.ViT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ThreeAugment", params=dict(p=0.5)),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
            ], p=0.5
        )),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.ViT.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(cfg.ViT.PARAMS.image_size, cfg.ViT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.ViT.OPTIMIZER = CN()
    cfg.ViT.OPTIMIZER.NAME = "Lamb"
    cfg.ViT.OPTIMIZER.PARAMS = CN()
    cfg.ViT.OPTIMIZER.PARAMS.lr = 3e-3
    cfg.ViT.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.ViT.OPTIMIZER.PARAMS.weight_decay = 0.05

    cfg.ViT.LR_SCHEDULER = CN()
    cfg.ViT.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.ViT.LR_SCHEDULER.PARAMS = CN()
    cfg.ViT.LR_SCHEDULER.PARAMS.lr_initial = 1e-8
    cfg.ViT.LR_SCHEDULER.PARAMS.lr_final = cfg.ViT.OPTIMIZER.PARAMS.lr
    cfg.ViT.LR_SCHEDULER.PARAMS.warmup_epochs = 5
    cfg.ViT.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.ViT.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.ViT.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingLR"
    cfg.ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_max = 595
    cfg.ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 1e-8
    cfg.ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.ViT.LOSS = CN()
    cfg.ViT.LOSS.NAME = "CrossEntropyLoss"
    cfg.ViT.LOSS.PARAMS = CN()
    cfg.ViT.LOSS.PARAMS.weight = None
    cfg.ViT.LOSS.PARAMS.size_average = None
    cfg.ViT.LOSS.PARAMS.ignore_index = -100
    cfg.ViT.LOSS.PARAMS.reduce = None
    cfg.ViT.LOSS.PARAMS.reduction = "mean"
    cfg.ViT.LOSS.PARAMS.label_smoothing = 0.0

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0