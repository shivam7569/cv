class Container:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.setAttribute(k, v)

    def setAttribute(self, name, val):
        setattr(self, name, val)

