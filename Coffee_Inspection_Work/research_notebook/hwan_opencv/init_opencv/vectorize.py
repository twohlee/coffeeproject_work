import numpy as np
from numba import vectorize, guvectorize, int64, float64
from timeit import default_timer as timer


@guvectorize(['void(int32[:,:], int32[:,:], int32[:,:])',
                'void(float32[:,:], float32[:,:], float32[:,:])'],
                '(x, y),(x, y)->(x, y)')
def add_2d_array(a, b, c):
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            c[i, j] = a[i, j] + b[i, j]

i1 = np.ones(shape=(3, 3), dtype=np.float32)
i2 = np.ones(shape=(3, 3), dtype=np.float32)
# c = add_2d_array(i1, i2)
c = add_2d_array(i1, add_2d_array(i1,i2))
print(c)

#------------------------------------------------------

# @guvectorize([(float64[:], float64[:])], '(n)->(n)')
# def init_values(invals, outvals):
#     invals[0] = 6.5
#     outvals[0] = 4.2

# invals = np.zeros(shape=(3, 3), dtype=np.float64)
# print(invals)
# outvals = init_values(invals)
# print(invals)
# print(outvals)


#------------------------------------------------------

# @guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)')
# def g(x, y, res):
#     res = np.append(res, 1)
#     # for i in range(x.shape[0]):
#     #     res[i] = x[i] + y

# a = np.arange(5)
# print(a)
# print( np.append(a, 2) )
# print(g(a, 2))


#------------------------------------------------------

# @vectorize(['float32(float32, float32, float32)','float64(float64, float64, float64)'], target="cuda")
# def saxpy_cuda(scala, a, b):
#     return scala * a + b

# @vectorize(["float32(float32,float32,float32),"], target='parallel')
# def saxpy_host(scala, a, b):
#     return scala * a + b    


# scala = 2.0
# np.random.seed(2019)
# print("size \t\t CUDA \t\t CPU")
# for i in range(16, 20):
#     N = 1 << i
#     print(N)
#     a = np.random.rand(N).astype(np.float32)
#     b = np.random.rand(N).astype(np.float32)
#     c = np.zeros(N, dtype=np.float32)

#     # warm-up
#     c = saxpy_cuda(scala, a, b)

#     # measuring execution time
#     start = timer()
#     c = saxpy_host(scala, a, b)
#     elapsed_time_host = (timer() - start) * 1e3

#     start = timer()
#     c = saxpy_cuda(scala, a, b)
#     elapsed_time_cuda = (timer() - start) * 1e3
#     print("[%d]: \t%.3f ms \t %.3f ms" % (N, elapsed_time_cuda, elapsed_time_host))
    