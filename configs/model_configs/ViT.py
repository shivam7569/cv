import math


def ViTConfig(cfg):
    from configs.config import CfgNode as CN

    cfg.ViT = CN()

    cfg.ASYNC_TRAINING = True

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
    cfg.ViT.PARAMS.image_size = 224
    cfg.ViT.PARAMS.patch_size = 16
    cfg.ViT.PARAMS.classifier_mlp_d = 3072 # (4 * d_model)
    cfg.ViT.PARAMS.encoder_mlp_d = 2048
    cfg.ViT.PARAMS.encoder_num_heads = 12
    cfg.ViT.PARAMS.num_encoder_blocks = 12
    cfg.ViT.PARAMS.dropout = 0.1
    cfg.ViT.PARAMS.encoder_dropout = 0.1
    cfg.ViT.PARAMS.encoder_attention_dropout = None
    cfg.ViT.PARAMS.patchify_technique = "linear"
    cfg.ViT.PARAMS.stochastic_depth = True
    cfg.ViT.PARAMS.stochastic_depth_mp = 0.1
    cfg.ViT.PARAMS.in_channels = 3

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 500
    cfg.TRAIN.PARAMS.gradient_accumulation = False
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 4096
    cfg.TRAIN.PARAMS.gradient_clipping = ("norm", 1)
    cfg.TRAIN.PARAMS.exponential_moving_average = None
    cfg.TRAIN.PARAMS.MixUp = [
        ("alpha", 0.5), ("prob", 0.75)
    ]
    cfg.TRAIN.PARAMS.CutMix = None

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
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
        dict(name="RandAugment", params=dict(num_ops=2, magnitude=15)),
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
    cfg.ViT.OPTIMIZER.NAME = "Adam"
    cfg.ViT.OPTIMIZER.PARAMS = CN()
    cfg.ViT.OPTIMIZER.PARAMS.lr = 3e-3 * math.sqrt(cfg.ViT.DATALOADER_TRAIN_PARAMS.batch_size/4096)
    cfg.ViT.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.ViT.OPTIMIZER.PARAMS.weight_decay = 0.3

    cfg.ViT.LR_SCHEDULER = CN()
    cfg.ViT.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.ViT.LR_SCHEDULER.PARAMS = CN()
    cfg.ViT.LR_SCHEDULER.PARAMS.lr_initial = 1e-6
    cfg.ViT.LR_SCHEDULER.PARAMS.lr_final = 3e-3 * math.sqrt(cfg.ViT.DATALOADER_TRAIN_PARAMS.batch_size/4096)
    cfg.ViT.LR_SCHEDULER.PARAMS.warmup_epochs = 30
    cfg.ViT.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.ViT.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.ViT.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingWarmRestarts"
    cfg.ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_0 = 15
    cfg.ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_mult = 2
    cfg.ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 1e-6
    cfg.ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.ViT.LOSS = CN()
    cfg.ViT.LOSS.NAME = "CrossEntropyLoss"
    cfg.ViT.LOSS.PARAMS = CN()
    cfg.ViT.LOSS.PARAMS.weight = None
    cfg.ViT.LOSS.PARAMS.size_average = None
    cfg.ViT.LOSS.PARAMS.ignore_index = -100
    cfg.ViT.LOSS.PARAMS.reduce = None
    cfg.ViT.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0