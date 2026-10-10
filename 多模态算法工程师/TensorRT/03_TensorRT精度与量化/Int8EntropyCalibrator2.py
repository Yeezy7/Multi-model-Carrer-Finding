import os 
import glob
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化 CUDA 上下文

class Int8EntropyCalibrator2(trt.IInt8EntropyCalibrator2):
    """标准校准器：逐 batch 把校准图像送入网络，收集激活直方图 → KL 选择 scale"""

    def __init__(self, calib_dir, cache_file, batch_size=8, input_shape=(3, 224, 224)):
        