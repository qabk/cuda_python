# cuda_python

## Bước 1: Cài đặt Cuda toolkit

### Cách 1: Cài đặt trên linux
```
sudo apt update <br/>
sudo apt install nvidia-cuda-toolkit<br/>
```

### Cách 2: Cài đặt thông qua Anaconda
```
git clone https://github.com/qabk/cuda_python<br/>
cd cuda_python<br/>
conda env create -f py_cuda.yml<br/>
conda activate py_cuda<br/>
```

## Bước 2: Cài các gói thư viện cần thiết
```
pip install -r requirements.txt<br/>
```

## Bước 3: Chạy thử code
```
python GpuFilter.py<br/>
```

## Kết quả trên Gpu và Cpu
<p align="center">
  <img src="https://github.com/qabk/cuda_python/blob/main/images/Compare_speed.png"> 
</p>

## Ảnh gốc và ảnh sau khi lọc trung bình và trung vị
<p align="center">
  <img src="https://github.com/qabk/cuda_python/blob/main/images/Median.jpg"> 
  <img src="https://github.com/qabk/cuda_python/blob/main/images/res_avr.jpg"> 
  <img src="https://github.com/qabk/cuda_python/blob/main/images/res_med.jpg"> 
</p>
