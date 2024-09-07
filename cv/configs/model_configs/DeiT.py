import math

def DeiTConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.DeiT = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = True

    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "DeiT"
    cfg.METRICS.NAME = "DeiT"
    cfg.CHECKPOINT.BASENAME = "DeiT"
    cfg.TENSORBOARD.BASENAME = "DeiT"
    cfg.PROFILER.BASENAME = "DeiT/Exp"
    cfg.CHECKPOINT.TREE = False

    cfg.DeiT.PARAMS = CN()
    cfg.DeiT.PARAMS.num_classes = 1000
    cfg.DeiT.PARAMS.d_model = 768
    cfg.DeiT.PARAMS.image_size = 224
    cfg.DeiT.PARAMS.patch_size = 16
    cfg.DeiT.PARAMS.classifier_mlp_d = 2048
    cfg.DeiT.PARAMS.encoder_mlp_d = 768 * 4
    cfg.DeiT.PARAMS.encoder_num_heads = 12
    cfg.DeiT.PARAMS.num_encoder_blocks = 12
    cfg.DeiT.PARAMS.dropout = 0.1
    cfg.DeiT.PARAMS.encoder_dropout = 0.1
    cfg.DeiT.PARAMS.encoder_attention_dropout = 0.0
    cfg.DeiT.PARAMS.patchify_technique = "linear"
    cfg.DeiT.PARAMS.stochastic_depth = True
    cfg.DeiT.PARAMS.stochastic_depth_mp = 0.1
    cfg.DeiT.PARAMS.layer_scale = 1e-6
    cfg.DeiT.PARAMS.return_logits_type = "fusion"
    cfg.DeiT.PARAMS.teacher_model_name = "ConvNeXt"
    cfg.DeiT.PARAMS.in_channels = 3

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 500
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 1024
    cfg.TRAIN.PARAMS.gradient_clipping = None
    cfg.TRAIN.PARAMS.exponential_moving_average = CN()
    cfg.TRAIN.PARAMS.exponential_moving_average.beta = 0.999
    cfg.TRAIN.PARAMS.exponential_moving_average.warmup_steps = 0
    cfg.TRAIN.PARAMS.exponential_moving_average.decay_period = 1
    cfg.TRAIN.PARAMS.exponential_moving_average.decay_method = "constant"

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
        dict(func="resizeWithAspectRatio", params=dict(size=224 if cfg.DeiT.PARAMS.image_size == 192 else 256))
    ]

    cfg.DeiT.DATALOADER_TRAIN_PARAMS = CN()
    cfg.DeiT.DATALOADER_TRAIN_PARAMS.batch_size = 32
    cfg.DeiT.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.DeiT.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.DeiT.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.DeiT.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.DeiT.DATALOADER_VAL_PARAMS = CN()
    cfg.DeiT.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.DeiT.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.DeiT.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.DeiT.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.DeiT.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.DeiT.TRANSFORMS = CN()

    cfg.DeiT.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=(cfg.DeiT.PARAMS.image_size, cfg.DeiT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="RandomAugmentation", params=dict(m=9, n=2, mstd=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
        dict(name="RandomCutOut", params=dict(probability=0.25, mode="pixel", device="cpu"))
    ]

    cfg.DeiT.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(cfg.DeiT.PARAMS.image_size, cfg.DeiT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.DeiT.OPTIMIZER = CN()
    cfg.DeiT.OPTIMIZER.NAME = "AdamW"
    cfg.DeiT.OPTIMIZER.PARAMS = CN()
    cfg.DeiT.OPTIMIZER.PARAMS.lr = 3e-3
    cfg.DeiT.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.DeiT.OPTIMIZER.PARAMS.weight_decay = 0.05

    cfg.DeiT.LR_SCHEDULER = CN()
    cfg.DeiT.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.DeiT.LR_SCHEDULER.PARAMS = CN()
    cfg.DeiT.LR_SCHEDULER.PARAMS.lr_initial = 0.0
    cfg.DeiT.LR_SCHEDULER.PARAMS.lr_final = cfg.DeiT.OPTIMIZER.PARAMS.lr
    cfg.DeiT.LR_SCHEDULER.PARAMS.warmup_epochs = 5
    cfg.DeiT.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.DeiT.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.DeiT.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingWarmRestarts"
    cfg.DeiT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.DeiT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_0 = 15
    cfg.DeiT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_mult = 3
    cfg.DeiT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 0.0
    cfg.DeiT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.DeiT.LOSS = CN()
    cfg.DeiT.LOSS.NAME = "DeiT_Distillation_Loss"
    cfg.DeiT.LOSS.PARAMS = CN()
    cfg.DeiT.LOSS.PARAMS.distillation_kind = "hard"
    cfg.DeiT.LOSS.PARAMS.ce_loss_params = CN()
    cfg.DeiT.LOSS.PARAMS.ce_loss_params.size_average = None
    cfg.DeiT.LOSS.PARAMS.ce_loss_params.ignore_index = -100
    cfg.DeiT.LOSS.PARAMS.ce_loss_params.reduce = None
    cfg.DeiT.LOSS.PARAMS.ce_loss_params.reduction = "mean"
    cfg.DeiT.LOSS.PARAMS.ce_loss_params.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0