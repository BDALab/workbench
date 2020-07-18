import os


def ensure_directory(path):
    """Ensure that the directory(/ies) of the <path> exists"""
    dirs = os.path.dirname(path)
    if dirs and not os.path.exists(dirs):
        os.makedirs(dirs)
