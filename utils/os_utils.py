import os
import shutil

def check_dir(path, create=True, forcedCreate=False):

    exists = os.path.exists(path) and os.path.isdir(path)

    if not exists and create:
        os.mkdir(path)
    if exists and forcedCreate:
        shutil.rmtree(path)
        os.mkdir(path)

    return exists

def check_file(path, remove=False):

    exists = os.path.exists(path) and os.path.isfile(path)

    if exists and remove:
        os.remove(path)

    return exists