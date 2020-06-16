from numba import njit, jit, vectorize, float64, cuda
import numpy as np
import time

def foo1(x):
    lst = []
    for i in range(x):
        lst.append(np.arange(i))
    return lst


# @jit(nopython=True, target="cpu")
# @vectorize()
@njit
def foo2(x):
    lst = []
    for i in range(x):
        lst.append(np.arange(i))
    return lst

@vectorize([float64], target="cuda")
def f(x, y):
    return x + y


start = time.time()
# foo1(4)
foo1(1000)
print("without GPU:", time.time()-start)    

start = time.time()
# foo2(4)
foo2(1000)
print("with GPU:", time.time()-start) 

a = np.arange(6)
print(a)
print(f(a, a))
