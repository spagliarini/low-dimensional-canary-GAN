import joblib
from functools import wraps
import inspect
import collections   

import joblib

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


def parallel_on_demand(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        
        default_args = get_default_args(func)
        
        n_jobs = default_args["workers"]
        backend = default_args["backend"]
        
        seq_args = [args] if len(args) <= 1 else args
        
        all_iterables = True
        for arg in seq_args:
            if not(is_iterable(arg)):
                all_iterables = False

        if all_iterables:
            zip_args = zip(*seq_args)
            
            if n_jobs is not None:
                loop = joblib.Parallel(n_jobs=n_jobs, backend=backend)
                
                def index_function(index, *args, **kwargs):
                    return index, func(*args, **kwargs)
                
                delayed_func = joblib.delayed(index_function)
                responses = loop(delayed_func(i, *a, **kwargs) for i, a in enumerate(zip_args))
                responses = [r[1] for r in sorted(responses, key=lambda x: x[0])]
                
            else:
                responses = [func(*arg, **kwargs) for arg in zip_args]
        else:
            print(args)
            responses = [func(*args, **kwargs)]

        return responses

    return wrapper