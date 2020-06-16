from numba import jit, cuda, njit 
import numpy as np 
from numba.typed import List
# to measure exec time 
from timeit import default_timer as timer    
  
# normal function to run on cpu 
def func(a):                                 
    for i in range(100): 
        qw =1
        a[i]+= 1      
        # print(qw)
  
# function optimized to run on gpu  
# @jit(target ="cuda")                          
@njit                          
def func2(a): 
    for i in range(100): 

        qw =1
        a[i]+= 1
        # print(qw)
if __name__=="__main__": 
    n = 100 
    a = np.ones(n, dtype = np.float64) 
    b = np.ones(n, dtype = np.float32) 
      
    start = timer() 
    func(a) 
    print("without GPU:", timer()-start)     
      
    start = timer() 
    func2(a) 
    print("with GPU:", timer()-start) 