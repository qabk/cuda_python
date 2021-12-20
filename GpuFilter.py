from __future__ import division
import numpy as np
import math
import numba
from numba import cuda
import cv2
import matplotlib.pyplot as plt
import time

"""
hàm tính các bộ lọc thông thường như
Gauss, trung bình, v.v...
"""
@cuda.jit
def corrolation(Input, fil, Output):
    row, col = cuda.grid(2)
    rn = math.floor(fil.shape[0]/2)
    if (rn < row < Output.shape[0]-rn) and  (rn < col < Output.shape[1]-rn):
        tmp = 0.
        for i in range(-rn,rn+1):
            for j in range(-rn,rn+1):
                if Output.shape[0] > row+i >= 0 and Output.shape[1] > col+j >= 0 :
                    tmp += Input[row+i,col+j]*fil[i,j]
                
                    
        Output[row,col] = tmp
    else:
        Output[row,col] = Input[row,col]


"""
hàm tính bộ lọc trung vị
sử dụng sắp xếp chọn
"""
fil_size = 7
fil_size_flat = fil_size * fil_size
rn = math.floor(fil_size/2)
med = math.ceil(fil_size_flat/2)
@cuda.jit
def median_filter(Input,  Output):
    row, col = cuda.grid(2)
    windows = cuda.local.array(fil_size_flat, numba.int32)
    if (0 < row < Output.shape[0]-1) and  (0 < col < Output.shape[1]-1):
        index = 0
        "thêm các phần tử trong cửa sổ để sắp xếp"
        for i in range(row-rn, row+rn+1):
            for j in range(col-rn, col+rn+1):
                if i >= 0 and j >= 0 and i < Output.shape[0] and j < Output.shape[1]:
                    windows[index] = Input[i, j]
                else:
                    windows[index] = 0
                index += 1
        "sắp xếp chọn"
        for i in range(fil_size_flat-1):
            min_idx = i
            for j in range(1, fil_size_flat):
                if windows[min_idx] > windows[j]:
                    min_idx = j
                    windows[j], windows[min_idx] = windows[min_idx] , windows[j]           
        Output[row,col] = windows[med]

"""
so sánh tốc độ của bộ lọc trung vị
ở cv2 và bộ lọc trung vị cuda
"""
def run_filter(input_image):
    output_img = np.zeros_like(input_image)
    fil = np.ones((5,5))/25
    fil_global_mem = cuda.to_device(fil)
    res_med = np.zeros_like(input_image)
    res_avr =  np.zeros_like(input_image)

    """
    khởi tạo và cấp phát 2 array 
    input và output trong gpu
    """
    Input_global = cuda.to_device(input_image)
    Output_global = cuda.to_device(output_img)

    "thiết lập các thread, block, grid"
    threadsperblock = (32, 32)
    blockspergrid_x = int(math.ceil(input_image.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(input_image.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    """
    chạy thử và so sánh tốc độ trên 
    gpu và cpu trung bình(30 lần)
    cho ảnh với 2 TH 
    có io(chuyển từ gpu sang cpu) và 
    không có io
    """
    
    "TH1: không io"
    med_arr_gpu = []
    avr_arr_gpu = []
    med_arr_cpu = []
    avr_arr_cpu = []
    for i in range(30):
        start = time.time()
        median_filter[blockspergrid, threadsperblock](Input_global, Output_global)
        med_arr_gpu.append(time.time()-start)
        start = time.time()
        res_med = cv2.medianBlur(input_image,fil_size)
        med_arr_cpu.append(time.time()-start)
        start = time.time()
        corrolation[blockspergrid, threadsperblock](Input_global,fil_global_mem, Output_global)
        avr_arr_gpu.append(time.time()-start)
        start = time.time()
        res_avr = cv2.filter2D(input_image,-1,fil)
        avr_arr_cpu.append(time.time()-start)

    
    "TH2: có io"
    med_arr_gpu_io = []
    med_arr_cpu_io = []
    avr_arr_gpu_io = []
    avr_arr_cpu_io = []
    for i in range(30):
        start = time.time()
        median_filter[blockspergrid, threadsperblock](Input_global, Output_global)
        B = Output_global.copy_to_host()
        med_arr_gpu_io.append(time.time()-start)
        start = time.time()
        res = cv2.medianBlur(input_image,fil_size)
        med_arr_cpu_io.append(time.time()-start)

        start = time.time()
        corrolation[blockspergrid, threadsperblock](Input_global,fil_global_mem, Output_global)
        avr_arr_gpu_io.append(time.time()-start)
        start = time.time()
        res = cv2.filter2D(input_image,-1,fil)
        avr_arr_cpu_io.append(time.time()-start)

    cv2.imwrite("images/res_avr.jpg",res_avr)
    cv2.imwrite("images/res_med.jpg",res_med)
    seq = np.arange(0,30,1)
    figure, axis = plt.subplots(2,2)
    axis[0,0].plot(seq,np.array(med_arr_cpu),'b',label ="cpu")
    axis[0,0].plot(seq,np.array(med_arr_gpu),'r',label ="gpu")
    axis[0,0].set_title("Không có io cho bộ lọc trung vị")
    axis[0,0].legend(fancybox=True)

    axis[0,1].plot(seq,np.array(med_arr_cpu_io),'b',label ="cpu")
    axis[0,1].plot(seq,np.array(med_arr_gpu_io),'r',label ="gpu")
    axis[0,1].set_title("Có io cho bộ lọc trung vị")
    axis[0,1].legend(fancybox=True)

    axis[1,0].plot(seq,np.array(avr_arr_cpu),'b',label ="cpu")
    axis[1,0].plot(seq,np.array(avr_arr_gpu),'r',label ="gpu")
    axis[1,0].set_title("Không có io cho bộ lọc Blur")
    axis[1,0].legend(fancybox=True)

    axis[1,1].plot(seq,np.array(med_arr_cpu_io),'b',label ="cpu")
    axis[1,1].plot(seq,np.array(med_arr_gpu_io),'r',label ="gpu")
    axis[1,1].set_title("Có io cho bộ lọc Blur")
    axis[1,1].legend(fancybox=True)
    plt.show()

if __name__ == "__main__":
    img = cv2.imread(r"images/Median.jpg",cv2.IMREAD_GRAYSCALE)
    run_filter(img)



    


