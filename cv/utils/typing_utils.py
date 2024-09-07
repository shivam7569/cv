import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

matplotlib.use('Agg')

class ClassRegistry(type):

    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)
        cls.REGISTRY[name] = new_class

        return new_class

class VariableRegistry:

    REGISTER = {}

    @classmethod
    def addVar(cls, var_name, var_value):
        cls.REGISTER[var_name] = var_value

def draw_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(include_values=False, cmap="cividis", colorbar=False)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    figure = plt.gcf()

    return figure