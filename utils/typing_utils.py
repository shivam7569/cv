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