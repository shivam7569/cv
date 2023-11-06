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
    cfg.LeNet.OPTIMIZER.NAME = "SGD"
    cfg.LeNet.OPTIMIZER.PARAMS = CN()
    cfg.LeNet.OPTIMIZER.PARAMS.lr = 0.01
    cfg.LeNet.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.LeNet.OPTIMIZER.PARAMS.weight_decay = 0.0001

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
    cfg.AlexNet.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.AlexNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.AlexNet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.AlexNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.AlexNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.AlexNet.TRANSFORMS = CN()
    cfg.AlexNet.TRANSFORMS.TRAIN = [
        dict(name="FancyPCA", params=dict(alpha_std=0.1, p=0.5)),
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
        dict(name="FancyPCA", params=dict(alpha_std=0.1, p=0.5)),
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
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=[0, 1], contrast=[0, 1], saturation=[0, 1], hue=0.3))
            ], p=0.5
        )),
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
    cfg.Inception.LR_SCHEDULER.NAME = "StepLR"
    cfg.Inception.LR_SCHEDULER.PARAMS = CN()
    cfg.Inception.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.Inception.LR_SCHEDULER.PARAMS.gamma = 0.1
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
        dict(name="FancyPCA", params=dict(alpha_std=0.1, p=0.5)),
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=224)),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=0.4, contrast=0.4, saturation=0.4))
            ], p=0.5
        )),
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

def ResNetConfig(cfg):
    from configs.config import CfgNode as CN

    cfg.ResNet = CN()

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "ResNet"
    cfg.METRICS.NAME = "ResNet"
    cfg.CHECKPOINT.BASENAME = "ResNet"
    cfg.TENSORBOARD.BASENAME = "ResNet"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="vgg16resize", params=dict(s_min=256, s_max=480))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.ResNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.batch_size = 64
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.num_workers = 16
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.ResNet.DATALOADER_VAL_PARAMS = CN()
    cfg.ResNet.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.ResNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.ResNet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.ResNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.ResNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.ResNet.TRANSFORMS = CN()
    cfg.ResNet.TRANSFORMS.TRAIN = [
        dict(name="FancyPCA", params=dict(alpha_std=0.1, p=0.5)),
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=0.4, contrast=0.4, saturation=0.4))
            ], p=0.5
        )),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.ResNet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.ResNet.OPTIMIZER = CN()
    cfg.ResNet.OPTIMIZER.NAME = "SGD"
    cfg.ResNet.OPTIMIZER.PARAMS = CN()
    cfg.ResNet.OPTIMIZER.PARAMS.lr = 0.1
    cfg.ResNet.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.ResNet.OPTIMIZER.PARAMS.weight_decay = 1e-4

    cfg.ResNet.LR_SCHEDULER = CN()
    cfg.ResNet.LR_SCHEDULER.NAME = "StepLR"
    cfg.ResNet.LR_SCHEDULER.PARAMS = CN()
    cfg.ResNet.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.ResNet.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.ResNet.LR_SCHEDULER.PARAMS.verbose = False

    cfg.ResNet.LOSS = CN()
    cfg.ResNet.LOSS.NAME = "CrossEntropyLoss"
    cfg.ResNet.LOSS.PARAMS = CN()
    cfg.ResNet.LOSS.PARAMS.weight = None
    cfg.ResNet.LOSS.PARAMS.size_average = None
    cfg.ResNet.LOSS.PARAMS.ignore_index = -100
    cfg.ResNet.LOSS.PARAMS.reduce = None
    cfg.ResNet.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0

def Inceptionv2Config(cfg):
    from configs.config import CfgNode as CN

    cfg.Inceptionv2 = CN()

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "Inceptionv2"
    cfg.METRICS.NAME = "Inceptionv2"
    cfg.CHECKPOINT.BASENAME = "Inceptionv2"
    cfg.TENSORBOARD.BASENAME = "Inceptionv2"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="inceptionv2resize", params=dict(aspect_ratio_min=3/4, aspect_ratio_max=4/3))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=340))
    ]

    cfg.Inceptionv2.DATALOADER_TRAIN_PARAMS = CN()
    cfg.Inceptionv2.DATALOADER_TRAIN_PARAMS.batch_size = 32
    cfg.Inceptionv2.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.Inceptionv2.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.Inceptionv2.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.Inceptionv2.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.Inceptionv2.DATALOADER_VAL_PARAMS = CN()
    cfg.Inceptionv2.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.Inceptionv2.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.Inceptionv2.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.Inceptionv2.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.Inceptionv2.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.Inceptionv2.TRANSFORMS = CN()
    cfg.Inceptionv2.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=[0, 1], contrast=[0, 1], saturation=[0, 1], hue=0.3))
            ], p=0.5
        )),
        dict(name="RandomCrop", params=dict(size=(299, 299), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.Inceptionv2.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(299, 299))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.Inceptionv2.OPTIMIZER = CN()
    cfg.Inceptionv2.OPTIMIZER.NAME = "SGD"
    cfg.Inceptionv2.OPTIMIZER.PARAMS = CN()
    cfg.Inceptionv2.OPTIMIZER.PARAMS.lr = 0.01
    cfg.Inceptionv2.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.Inceptionv2.OPTIMIZER.PARAMS.weight_decay = 0.0005

    cfg.Inceptionv2.LR_SCHEDULER = CN()
    cfg.Inceptionv2.LR_SCHEDULER.NAME = "StepLR"
    cfg.Inceptionv2.LR_SCHEDULER.PARAMS = CN()
    cfg.Inceptionv2.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.Inceptionv2.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.Inceptionv2.LR_SCHEDULER.PARAMS.verbose = False

    cfg.Inceptionv2.LOSS = CN()
    cfg.Inceptionv2.LOSS.NAME = "CrossEntropyLoss"
    cfg.Inceptionv2.LOSS.PARAMS = CN()
    cfg.Inceptionv2.LOSS.PARAMS.weight = None
    cfg.Inceptionv2.LOSS.PARAMS.size_average = None
    cfg.Inceptionv2.LOSS.PARAMS.ignore_index = -100
    cfg.Inceptionv2.LOSS.PARAMS.reduce = None
    cfg.Inceptionv2.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0

def Inceptionv3Config(cfg):
    from configs.config import CfgNode as CN

    cfg.Inceptionv3 = CN()

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "Inceptionv3"
    cfg.METRICS.NAME = "Inceptionv3"
    cfg.CHECKPOINT.BASENAME = "Inceptionv3"
    cfg.TENSORBOARD.BASENAME = "Inceptionv3"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="inceptionv2resize", params=dict(aspect_ratio_min=3/4, aspect_ratio_max=4/3))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=340))
    ]

    cfg.Inceptionv3.DATALOADER_TRAIN_PARAMS = CN()
    cfg.Inceptionv3.DATALOADER_TRAIN_PARAMS.batch_size = 32
    cfg.Inceptionv3.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.Inceptionv3.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.Inceptionv3.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.Inceptionv3.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.Inceptionv3.DATALOADER_VAL_PARAMS = CN()
    cfg.Inceptionv3.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.Inceptionv3.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.Inceptionv3.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.Inceptionv3.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.Inceptionv3.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.Inceptionv3.TRANSFORMS = CN()
    cfg.Inceptionv3.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=[0, 1], contrast=[0, 1], saturation=[0, 1], hue=0.3))
            ], p=0.5
        )),
        dict(name="RandomCrop", params=dict(size=(299, 299), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.Inceptionv3.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(299, 299))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.Inceptionv3.OPTIMIZER = CN()
    cfg.Inceptionv3.OPTIMIZER.NAME = "RMSprop"
    cfg.Inceptionv3.OPTIMIZER.PARAMS = CN()
    cfg.Inceptionv3.OPTIMIZER.PARAMS.lr = 0.045
    cfg.Inceptionv3.OPTIMIZER.PARAMS.momentum = 0.009
    cfg.Inceptionv3.OPTIMIZER.PARAMS.weight_decay = 0.0005

    cfg.Inceptionv3.LR_SCHEDULER = CN()
    cfg.Inceptionv3.LR_SCHEDULER.NAME = "ExponentialLR"
    cfg.Inceptionv3.LR_SCHEDULER.PARAMS = CN()
    cfg.Inceptionv3.LR_SCHEDULER.PARAMS.gamma = 0.94
    cfg.Inceptionv3.LR_SCHEDULER.PARAMS.verbose = False

    cfg.Inceptionv3.LOSS = CN()
    cfg.Inceptionv3.LOSS.NAME = "CrossEntropyLoss"
    cfg.Inceptionv3.LOSS.PARAMS = CN()
    cfg.Inceptionv3.LOSS.PARAMS.weight = None
    cfg.Inceptionv3.LOSS.PARAMS.size_average = None
    cfg.Inceptionv3.LOSS.PARAMS.ignore_index = -100
    cfg.Inceptionv3.LOSS.PARAMS.reduce = None
    cfg.Inceptionv3.LOSS.PARAMS.reduction = "mean"
    cfg.Inceptionv3.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0

def ResNeXtConfig(cfg):
    from configs.config import CfgNode as CN

    cfg.ResNeXt = CN()

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "ResNeXt"
    cfg.METRICS.NAME = "ResNeXt"
    cfg.CHECKPOINT.BASENAME = "ResNeXt"
    cfg.TENSORBOARD.BASENAME = "ResNeXt"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="inceptionv1resize", params=dict(aspect_ratio_min=3/4, aspect_ratio_max=4/3))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.ResNeXt.DATALOADER_TRAIN_PARAMS = CN()
    cfg.ResNeXt.DATALOADER_TRAIN_PARAMS.batch_size = 128
    cfg.ResNeXt.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.ResNeXt.DATALOADER_TRAIN_PARAMS.num_workers = 16
    cfg.ResNeXt.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.ResNeXt.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.ResNeXt.DATALOADER_VAL_PARAMS = CN()
    cfg.ResNeXt.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.ResNeXt.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.ResNeXt.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.ResNeXt.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.ResNeXt.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.ResNeXt.TRANSFORMS = CN()
    cfg.ResNeXt.TRANSFORMS.TRAIN = [
        dict(name="FancyPCA", params=dict(alpha_std=0.1, p=0.5)),
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=0.4, contrast=0.4, saturation=0.4))
            ], p=0.5
        )),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.ResNeXt.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.ResNeXt.OPTIMIZER = CN()
    cfg.ResNeXt.OPTIMIZER.NAME = "SGD"
    cfg.ResNeXt.OPTIMIZER.PARAMS = CN()
    cfg.ResNeXt.OPTIMIZER.PARAMS.lr = 0.1
    cfg.ResNeXt.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.ResNeXt.OPTIMIZER.PARAMS.weight_decay = 1e-4

    cfg.ResNeXt.LR_SCHEDULER = CN()
    cfg.ResNeXt.LR_SCHEDULER.NAME = "StepLR"
    cfg.ResNeXt.LR_SCHEDULER.PARAMS = CN()
    cfg.ResNeXt.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.ResNeXt.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.ResNeXt.LR_SCHEDULER.PARAMS.verbose = False

    cfg.ResNeXt.LOSS = CN()
    cfg.ResNeXt.LOSS.NAME = "CrossEntropyLoss"
    cfg.ResNeXt.LOSS.PARAMS = CN()
    cfg.ResNeXt.LOSS.PARAMS.weight = None
    cfg.ResNeXt.LOSS.PARAMS.size_average = None
    cfg.ResNeXt.LOSS.PARAMS.ignore_index = -100
    cfg.ResNeXt.LOSS.PARAMS.reduce = None
    cfg.ResNeXt.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0

def MobileNetConfig(cfg):
    from configs.config import CfgNode as CN

    cfg.MobileNet = CN()

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "MobileNet"
    cfg.METRICS.NAME = "MobileNet"
    cfg.CHECKPOINT.BASENAME = "MobileNet"
    cfg.TENSORBOARD.BASENAME = "MobileNet"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.MobileNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.MobileNet.DATALOADER_TRAIN_PARAMS.batch_size = 128
    cfg.MobileNet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.MobileNet.DATALOADER_TRAIN_PARAMS.num_workers = 16
    cfg.MobileNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.MobileNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.MobileNet.DATALOADER_VAL_PARAMS = CN()
    cfg.MobileNet.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.MobileNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.MobileNet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.MobileNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.MobileNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.MobileNet.TRANSFORMS = CN()
    cfg.MobileNet.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.MobileNet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.MobileNet.OPTIMIZER = CN()
    cfg.MobileNet.OPTIMIZER.NAME = "RMSprop"
    cfg.MobileNet.OPTIMIZER.PARAMS = CN()
    cfg.MobileNet.OPTIMIZER.PARAMS.lr = 0.1
    cfg.MobileNet.OPTIMIZER.PARAMS.alpha = 0.99
    cfg.MobileNet.OPTIMIZER.PARAMS.eps = 1e-8
    cfg.MobileNet.OPTIMIZER.PARAMS.weight_decay = 0
    cfg.MobileNet.OPTIMIZER.PARAMS.momentum = 0

    cfg.MobileNet.LR_SCHEDULER = CN()
    cfg.MobileNet.LR_SCHEDULER.NAME = "StepLR"
    cfg.MobileNet.LR_SCHEDULER.PARAMS = CN()
    cfg.MobileNet.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.MobileNet.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.MobileNet.LR_SCHEDULER.PARAMS.verbose = False

    cfg.MobileNet.LOSS = CN()
    cfg.MobileNet.LOSS.NAME = "CrossEntropyLoss"
    cfg.MobileNet.LOSS.PARAMS = CN()
    cfg.MobileNet.LOSS.PARAMS.weight = None
    cfg.MobileNet.LOSS.PARAMS.size_average = None
    cfg.MobileNet.LOSS.PARAMS.ignore_index = -100
    cfg.MobileNet.LOSS.PARAMS.reduce = None
    cfg.MobileNet.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0