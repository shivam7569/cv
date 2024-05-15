import os
import shutil
import socket

def check_dir(path, create=True, forcedCreate=False, tree=False):

    exists = os.path.exists(path) and os.path.isdir(path)

    if not exists and create:
        if tree:
            os.makedirs(path, exist_ok=True)
        else:
            os.mkdir(path)
    if exists and forcedCreate:
        shutil.rmtree(path)
        if tree:
            os.makedirs(path, exist_ok=True)
        else:
            os.mkdir(path)

    return exists

def check_file(path, remove=False):

    exists = os.path.exists(path) and os.path.isfile(path)

    if exists and remove:
        os.remove(path)

    return exists

def find_free_port(port=12322, max_port=12382):

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port < max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError('no free ports')
