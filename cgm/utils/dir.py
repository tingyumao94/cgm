import os.path


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_directory_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    dir = os.path.dirname(path)
    return dir, name