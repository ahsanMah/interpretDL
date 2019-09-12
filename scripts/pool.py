from multiprocessing import Pool, TimeoutError
from multiprocessing import get_context
import time
import os

def f(x):
    time.sleep(1)
    return x*x

print("__name is:", __name__)

if __name__ == '__main__':
    with get_context("spawn").Pool(4) as pool:
        res = pool.map(f,range(4))