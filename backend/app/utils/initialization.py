"""Serialize lazy shared-service initialization when sync routes run in worker threads."""
from functools import wraps
from threading import RLock


def serialized_initialization(function):
    lock = RLock()

    @wraps(function)
    def wrapped(*args, **kwargs):
        with lock:
            return function(*args, **kwargs)

    return wrapped
