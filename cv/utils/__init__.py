class MetaWrapper(type):
    def __repr__(cls):
        if hasattr(cls, "__class_repr__"):
            return getattr(cls, "__class_repr__")()
        else:
            return super(MetaWrapper, cls).__repr__()

from cv.utils.global_params import Global