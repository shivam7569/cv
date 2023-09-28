import os

def check_dir(path, create=False):

    exists = os.path.exists(path)

    if not exists and create:
        os.mkdir(path)

    return exists