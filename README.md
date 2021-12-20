# cuda_python
Project cài đặt 3 thuật toán gồm bộ lọc trung vị, bộ lọc trung bình và phép nhân ma trận. Cả 3 thuật toán được cài đặt trên GPU thông qua các API của Cuda và Numba, tốc độ thực thi phần cứng được so sánh trong 2 trường hợp chỉ tính toán trên GPU và tính toán trên GPU rồi trả về RAM CPU. Tốc độ được so sánh dựa trên phần cứng gồm RTX 2060 8GB và CPU i5 9400F 
## Bước 1: Cài đặt Cuda toolkit

### Cách 1: Cài đặt trên linux
```
sudo apt update 
sudo apt install nvidia-cuda-toolkit
```

### Cách 2: Cài đặt thông qua Anaconda
```
git clone https://github.com/qabk/cuda_python
cd cuda_python
conda env create -f py_cuda.yml
conda activate py_cuda
```

## Bước 2: Cài các gói thư viện cần thiết
```
pip install -r requirements.txt
```

## Bước 3: Chạy thử code
```
python GpuFilter.py
```

## Kết quả trên Gpu và Cpu
<p align="center">
  <img src="https://github.com/qabk/cuda_python/blob/main/images/Compare_speed.png"> 
</p>

## Tốc độ của Fast MatMul so với tốc độ của numpy
<p align="center">
  <img src="https://github.com/qabk/cuda_python/blob/main/images/mat_mul_res.png"> 
</p>
