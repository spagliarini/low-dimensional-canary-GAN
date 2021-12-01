import joblib
from functools import wraps
import inspect
import collections


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def is_iterable(obj):
    if isinstance(obj, str):
        return False
    return isinstance(obj, collections.abc.Iterable)
