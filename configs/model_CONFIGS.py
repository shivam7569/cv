def LeNetConfig(cfg):

    from configs.config import CfgNode as CN

    cfg.LeNet = CN()
    cfg.LOGGING.NAME = "LeNet"
    cfg.METRICS.NAME = "LeNet"
    cfg.CHECKPOINT.BASENAME = "LeNet"
    cfg.TENSORBOARD.BASENAME = "LeNet"

    cfg.LeNet.TRANSFORMS = CN()
    cfg.LeNet.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="Resize", params=dict(size=(224, 224))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.LeNet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="Resize", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.LeNet.OPTIMIZER = CN()
    cfg.LeNet.OPTIMIZER.NAME = "Adam"
    cfg.LeNet.OPTIMIZER.PARAMS = CN()
    cfg.LeNet.OPTIMIZER.PARAMS.lr = 0.001
    cfg.LeNet.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.LeNet.OPTIMIZER.PARAMS.eps = 1e-08

    cfg.LeNet.LOSS = CN()
    cfg.LeNet.LOSS.NAME = "CrossEntropyLoss"
    cfg.LeNet.LOSS.PARAMS = CN()
    cfg.LeNet.LOSS.PARAMS.weight = None
    cfg.LeNet.LOSS.PARAMS.size_average = None
    cfg.LeNet.LOSS.PARAMS.ignore_index = -100
    cfg.LeNet.LOSS.PARAMS.reduce = None
    cfg.LeNet.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = "L2"
    cfg.REGULARIZATION.STRENGTH = 0.001

def AlexNetConfig(cfg):

    from configs.config import CfgNode as CN

    cfg.AlexNet = CN()
    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "AlexNet"
    cfg.METRICS.NAME = "AlexNet"
    cfg.CHECKPOINT.BASENAME = "AlexNet"
    cfg.TENSORBOARD.BASENAME = "AlexNet"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="alexNetresize", params=dict())
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="alexNetresize", params=dict())
    ]

    cfg.AlexNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.batch_size = 128
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.num_workers = 16
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.AlexNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.AlexNet.DATALOADER_VAL_PARAMS = CN()
    cfg.AlexNet.DATALOADER_VAL_PARAMS.batch_size = 32
    cfg.AlexNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.AlexNet.DATALOADER_VAL_PARAMS.num_workers = 16
    cfg.AlexNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.AlexNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.AlexNet.TRANSFORMS = CN()
    cfg.AlexNet.TRANSFORMS.TRAIN = [
        dict(name="FancyPCA", params=dict(alpha_std=0.1)),
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.AlexNet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.AlexNet.OPTIMIZER = CN()
    cfg.AlexNet.OPTIMIZER.NAME = "SGD"
    cfg.AlexNet.OPTIMIZER.PARAMS = CN()
    cfg.AlexNet.OPTIMIZER.PARAMS.lr = 0.01
    cfg.AlexNet.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.AlexNet.OPTIMIZER.PARAMS.weight_decay = 0.0005

    cfg.AlexNet.LR_SCHEDULER = CN()
    cfg.AlexNet.LR_SCHEDULER.NAME = "StepLR"
    cfg.AlexNet.LR_SCHEDULER.PARAMS = CN()
    cfg.AlexNet.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.AlexNet.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.AlexNet.LR_SCHEDULER.PARAMS.verbose = False

    cfg.AlexNet.LOSS = CN()
    cfg.AlexNet.LOSS.NAME = "CrossEntropyLoss"
    cfg.AlexNet.LOSS.PARAMS = CN()
    cfg.AlexNet.LOSS.PARAMS.weight = None
    cfg.AlexNet.LOSS.PARAMS.size_average = None
    cfg.AlexNet.LOSS.PARAMS.ignore_index = -100
    cfg.AlexNet.LOSS.PARAMS.reduce = None
    cfg.AlexNet.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0

def VGG16Config(cfg):
    from configs.config import CfgNode as CN

    cfg.VGG16 = CN()
    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "VGG16"
    cfg.METRICS.NAME = "VGG16"
    cfg.CHECKPOINT.BASENAME = "VGG16"
    cfg.TENSORBOARD.BASENAME = "VGG16"

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
    cfg.VGG16.DATALOADER_TRAIN_PARAMS.batch_size = 32
    cfg.VGG16.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.VGG16.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.VGG16.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.VGG16.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.VGG16.DATALOADER_VAL_PARAMS = CN()
    cfg.VGG16.DATALOADER_VAL_PARAMS.batch_size = 32
    cfg.VGG16.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.VGG16.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.VGG16.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.VGG16.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.VGG16.TRANSFORMS = CN()
    cfg.VGG16.TRANSFORMS.TRAIN = [
        dict(name="FancyPCA", params=dict(alpha_std=0.1)),
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
    cfg.VGG16.OPTIMIZER.PARAMS.weight_decay = 0.0005

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

def InceptionConfig(cfg):
    from configs.config import CfgNode as CN

    cfg.Inception = CN()

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "Inception"
    cfg.METRICS.NAME = "Inception"
    cfg.CHECKPOINT.BASENAME = "Inception"
    cfg.TENSORBOARD.BASENAME = "Inception"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="inceptionv1resize", params=dict(aspect_ratio_min=3/4, aspect_ratio_max=4/3))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.Inception.DATALOADER_TRAIN_PARAMS = CN()
    cfg.Inception.DATALOADER_TRAIN_PARAMS.batch_size = 128
    cfg.Inception.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.Inception.DATALOADER_TRAIN_PARAMS.num_workers = 16
    cfg.Inception.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.Inception.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.Inception.DATALOADER_VAL_PARAMS = CN()
    cfg.Inception.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.Inception.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.Inception.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.Inception.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.Inception.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.Inception.TRANSFORMS = CN()
    cfg.Inception.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="ColorJitter", params=dict(brightness=[0, 1], contrast=[0, 1], saturation=[0, 1], hue=0.3)),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.Inception.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.Inception.OPTIMIZER = CN()
    cfg.Inception.OPTIMIZER.NAME = "SGD"
    cfg.Inception.OPTIMIZER.PARAMS = CN()
    cfg.Inception.OPTIMIZER.PARAMS.lr = 0.01
    cfg.Inception.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.Inception.OPTIMIZER.PARAMS.weight_decay = 0.0005

    cfg.Inception.LR_SCHEDULER = CN()
    cfg.Inception.LR_SCHEDULER.NAME = "MultiStepLR"
    cfg.Inception.LR_SCHEDULER.PARAMS = CN()
    cfg.Inception.LR_SCHEDULER.PARAMS.milestones = list(range(0, 450, 8))
    cfg.Inception.LR_SCHEDULER.PARAMS.gamma = 0.96
    cfg.Inception.LR_SCHEDULER.PARAMS.verbose = False

    cfg.Inception.LOSS = CN()
    cfg.Inception.LOSS.NAME = "CrossEntropyLoss"
    cfg.Inception.LOSS.PARAMS = CN()
    cfg.Inception.LOSS.PARAMS.weight = None
    cfg.Inception.LOSS.PARAMS.size_average = None
    cfg.Inception.LOSS.PARAMS.ignore_index = -100
    cfg.Inception.LOSS.PARAMS.reduce = None
    cfg.Inception.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0

def DenseNetConfig(cfg):
    from configs.config import CfgNode as CN

    cfg.DenseNet = CN()

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "DenseNet"
    cfg.METRICS.NAME = "DenseNet"
    cfg.CHECKPOINT.BASENAME = "DenseNet"
    cfg.TENSORBOARD.BASENAME = "DenseNet"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.DenseNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.DenseNet.DATALOADER_TRAIN_PARAMS.batch_size = 64
    cfg.DenseNet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.DenseNet.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.DenseNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.DenseNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.DenseNet.DATALOADER_VAL_PARAMS = CN()
    cfg.DenseNet.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.DenseNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.DenseNet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.DenseNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.DenseNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.DenseNet.TRANSFORMS = CN()
    cfg.DenseNet.TRANSFORMS.TRAIN = [
        dict(name="FancyPCA", params=dict(alpha_std=0.1)),
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=224)),
        dict(name="ColorJitter", params=dict(brightness=0.4, contrast=0.4, saturation=0.4)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.DenseNet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.DenseNet.OPTIMIZER = CN()
    cfg.DenseNet.OPTIMIZER.NAME = "SGD"
    cfg.DenseNet.OPTIMIZER.PARAMS = CN()
    cfg.DenseNet.OPTIMIZER.PARAMS.lr = 0.1
    cfg.DenseNet.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.DenseNet.OPTIMIZER.PARAMS.weight_decay = 1e-4

    cfg.DenseNet.LR_SCHEDULER = CN()
    cfg.DenseNet.LR_SCHEDULER.NAME = "StepLR"
    cfg.DenseNet.LR_SCHEDULER.PARAMS = CN()
    cfg.DenseNet.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.DenseNet.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.DenseNet.LR_SCHEDULER.PARAMS.verbose = False

    cfg.DenseNet.LOSS = CN()
    cfg.DenseNet.LOSS.NAME = "CrossEntropyLoss"
    cfg.DenseNet.LOSS.PARAMS = CN()
    cfg.DenseNet.LOSS.PARAMS.weight = None
    cfg.DenseNet.LOSS.PARAMS.size_average = None
    cfg.DenseNet.LOSS.PARAMS.ignore_index = -100
    cfg.DenseNet.LOSS.PARAMS.reduce = None
    cfg.DenseNet.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0