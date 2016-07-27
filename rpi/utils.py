import datetime
import inspect
import os


def current_method_name():
    return inspect.stack()[1][3]


def parent_method_name():
    return inspect.stack()[2][3]


def touch(fname, times=None):
    fhandle = open(fname, 'a')
    try:
        os.utime(fname, times)
    finally:
        fhandle.close()


def time_in_range(start, end, x):
    today = datetime.date.today()
    start = datetime.datetime.combine(today, start)
    end = datetime.datetime.combine(today, end)
    x = datetime.datetime.combine(today, x)
    if end <= start:
        end += datetime.timedelta(1)  # tomorrow!
    if x <= start:
        x += datetime.timedelta(1)  # tomorrow!
    return start <= x <= end


def make_sure_path_exist(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
