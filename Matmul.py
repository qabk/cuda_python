from __future__ import division
import numpy as np
import math
import numba
from numba import cuda, float32
import time
import matplotlib.pyplot as plt
TPB = 10

@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array trong shared memory
    # size và data type phải được biết ở thời điểm compile
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()
    C[x, y] = tmp

def run_matmul():
    A = np.random.rand(1000,2000).astype(np.float32)
    B = np.random.rand(2000, 1000).astype(np.float32)
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    C_global_mem = cuda.device_array((100,36))
    threadsperblock = (TPB, TPB)
    print(TPB)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[1]))
    blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    mat_cpu = []
    mat_gpu = []

    for i in range(30):
        start = time.time()
        fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
        res = C_global_mem.copy_to_host()
        mat_gpu.append(time.time()-start)
        start = time.time()
        C = np.dot(A,B)
        mat_cpu.append(time.time()-start)

    idx = np.arange(0,30,1)
    fig, axs = plt.subplots()
    axs.plot(idx,np.array(mat_cpu),'b',label = "cpu")
    axs.plot(idx,np.array(mat_gpu),'r',label = "gpu")
    plt.show()

if __name__ =="__main__":
    run_matmul()