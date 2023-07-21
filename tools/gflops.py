import time
import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import numpy as np

As = []
Bs = []
M = 100
K = 320
N = 1280
warm = 1000
for i in range(warm):
    As.append(np.random.rand(M, K))
    Bs.append(np.random.rand(K, N))
for i in range(warm):
    C = np.matmul(As[i], Bs[i])

loop = 1000
start = time.time()
for i in range(loop):
    C += np.matmul(As[i], Bs[i])
end = time.time()
cost = end - start
gflop = 2*M*K*N/1024/1024/1024
print('Numpy Matmul GFlop:', gflop, 'of M, K, N:', M, K, N)
print('Cost seconds:', cost)
print('Numpy Matmul GFlop/s:', gflop / cost * loop)
