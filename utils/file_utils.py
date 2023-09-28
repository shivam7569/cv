def read_txt(path):
    with open(path, "r") as f:
        data = [i.strip() for i in f.readlines()]

    return data