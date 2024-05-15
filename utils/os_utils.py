import os
import random
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

def find_free_port():

    while True:
        try:
            port = random.choice(range(12310, 12380))
            sock = socket.socket()
            sock.bind(("", port))

            return port
        except: 
            port = random.choice(range(12310, 12380))
