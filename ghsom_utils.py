"""Utility functions for gsom and ghsom"""

from sys import stdout
from time import time
from numpy import (sqrt, dot, apply_along_axis, isfinite, exp)
import jsonpickle


def _incremental_index_verbose(m):
    """Yields numbers from 0 to m-1 printing the status on the stdout."""
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    stdout.write(progress)
    beginning = time()
    for i in range(m):
        yield i
        it_per_sec = (i+1) / (time() - beginning)
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i+1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100*(i+1)/m)
        progress += ' - {it_per_sec:4.2f} it/s'.format(it_per_sec=it_per_sec)
        stdout.write(progress)

def qe_sum_1d(x):
    """Calculate euclidean norm.
    """
    return sqrt(dot(x, x.T))

def qe_sum_1d_nan(x):
    """Calculate euclidean norm. Nans in x are filtered out and the norm
    is scaled based on how many nans exist
    """
    mask = isfinite(x)
    x = x[mask]
    if len(x) == 0:
        return 1e6
    else:
        return sqrt(dot(x, x.T)) * sqrt(1 / (len(x) / mask.size))

def qe_sum_2d(x):
    """Calculate euclidean norm for each row in x
    """
    return sum(apply_along_axis(qe_sum_1d, 1, x))


def asymptotic_decay(learning_rate, t, max_iter):
    """Decay function of the learning process.
    Parameters
    ----------
    learning_rate : float
        current learning rate.

    t : int
        current iteration.

    max_iter : int
        maximum number of iterations for the training.
    """
    return learning_rate / (1+t/(max_iter/2))

def exponential_decay(learning_rate, t, max_iter):
    """Decay function of the learning process.
    Parameters
    ----------
    learning_rate : float
        current learning rate.

    t : int
        current iteration.

    max_iter : int
        maximum number of iterations for the training.
    """
    return learning_rate*exp(0.05 * ((t/max_iter)*100) * -1)

def save_model(obj, filepath):
    f = open(filepath, 'w')
    json_obj = jsonpickle.encode(obj)
    f.write(json_obj)
    f.close()

def load_model(filepath):
    f = open(filepath, 'r')
    json_str = f.read()
    obj = jsonpickle.decode(json_str)
    return obj