
from multiprocessing import Pool
import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    # result = mpp.IMapIterator(self._cache)
    result = mpp.IMapIterator(self) # istarmap.py for Python 3.8+
    self._taskqueue.put((self._guarded_task_generation(
        result._job, mpp.starmapstar, task_batches), result._set_length))
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap
from tqdm import tqdm

def multi_func(func, max_process_num, total, desc, unzip, *args, print_progress=True):
    """
    `multi_func` is a function that takes in a function `func`, a number of processes `max_process_num`,
    a total number of tasks `total`, a description `desc`, a boolean `unzip`, and a tuple of arguments
    `args`, and returns the result of applying `func` to each element of `args` in parallel
    
    Args:
      func: the function to be executed
      max_process_num: the maximum number of processes to use
      total: the total number of tasks
      desc: the description of the progress bar
      unzip: if the result of func is a tuple, unzip it
    """

    # args is a tuple
    assert len(args) > 0 and len(args[0]) > 0
    multi_param = len(args) > 1
    args = list(zip(*args)) if multi_param else args[0]

    with Pool(max_process_num) as p:
        map_func = p.istarmap if multi_param else p.imap
        if print_progress:
            res = list(tqdm(map_func(func, args), total=total, desc=desc))
        else:
            res = list(map_func(func, args))
        
    return tuple([list(x) for x in zip(*res)]) if (unzip and isinstance(res[0], tuple)) else res