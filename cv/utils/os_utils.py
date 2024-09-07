import os
import shutil

def check_dir(path, create=True, forcedCreate=False, tree=False):

    exists = os.path.exists(path) and os.path.isdir(path)

    if not exists and create:
        if tree:
            os.makedirs(path, exist_ok=True)
        else:
            try:
                os.mkdir(path)
            except FileNotFoundError:
                os.makedirs(path, exist_ok=True)
    if exists and forcedCreate:
        shutil.rmtree(path)
        if tree:
            os.makedirs(path, exist_ok=True)
        else:
            try:
                os.mkdir(path)
            except FileNotFoundError:
                os.makedirs(path, exist_ok=True)

    return exists

def check_file(path, remove=False):

    exists = os.path.exists(path) and os.path.isfile(path)

    if exists and remove:
        os.remove(path)

    return exists
