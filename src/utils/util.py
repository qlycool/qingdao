import os
import time
from datetime import datetime


def format_time(time_str: str):
    if type(time_str) != 'str':
        return time_str
    try:
        return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return datetime.strptime(time_str, '%Y-%m-%d')


def get_current_time() -> str:
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def create_dir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)
