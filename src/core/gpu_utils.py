#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU工具和管理器
支持CuPy和PyTorch的统一接口
"""

import numpy as np
from typing import Optional, Union, Any
import warnings

# 检测GPU可用性
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None


class GPUManager:
    """GPU统一管理器"""

    def __init__(self, prefer_pytorch: bool = True):
        """
        初始化GPU管理器

        Args:
            prefer_pytorch: 优先使用PyTorch（否则使用CuPy）
        """
        self.prefer_pytorch = prefer_pytorch
        self.cupy_available = CUPY_AVAILABLE
        self.torch_available = TORCH_AVAILABLE

        # 选择GPU后端
        if prefer_pytorch and TORCH_AVAILABLE:
            self.backend = 'pytorch'
            self._setup_pytorch()
        elif CUPY_AVAILABLE:
            self.backend = 'cupy'
            self._setup_cupy()
        else:
            self.backend = 'cpu'
            print("[WARNING] GPU不可用，使用CPU")

        print(f"[INFO] GPU后端: {self.backend}")

    def _setup_pytorch(self):
        """设置PyTorch GPU"""
        try:
            # 优化设置
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.set_float32_matmul_precision('high')

            # 内存管理
            torch.cuda.set_per_process_memory_fraction(0.7)
            torch.cuda.empty_cache()

            # 多进程设置
            torch.multiprocessing.set_sharing_strategy('file_system')

            self.device_name = torch.cuda.get_device_name(0)
            print(f"[GPU] PyTorch: {self.device_name}")

        except Exception as e:
            print(f"[WARNING] PyTorch GPU设置失败: {e}")
            self.backend = 'cpu'

    def _setup_cupy(self):
        """设置CuPy GPU"""
        try:
            # 内存池设置
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()

            # 获取内存信息
            free_memory, total_memory = cp.cuda.runtime.memGetInfo()
            memory_limit = int(free_memory * 0.7)
            mempool.set_limit(size=memory_limit)

            self.gpu_info = {
                'total_memory_gb': total_memory / (1024 ** 3),
                'free_memory_gb': free_memory / (1024 ** 3),
                'memory_limit_gb': memory_limit / (1024 ** 3)
            }

            print(f"[GPU] CuPy: {self.gpu_info['total_memory_gb']:.1f}GB总量, "
                  f"{self.gpu_info['memory_limit_gb']:.1f}GB限制")

        except Exception as e:
            print(f"[WARNING] CuPy GPU设置失败: {e}")
            self.backend = 'cpu'

    def to_gpu(self, array: np.ndarray) -> Union[cp.ndarray, torch.Tensor, np.ndarray]:
        """
        将numpy数组转移到GPU

        Args:
            array: numpy数组

        Returns:
            GPU数组（CuPy或PyTorch）或原数组（CPU）
        """
        if self.backend == 'pytorch':
            try:
                tensor = torch.from_numpy(array.copy())
                return tensor.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
            except Exception as e:
                warnings.warn(f"PyTorch GPU转移失败: {e}")
                return array

        elif self.backend == 'cupy':
            try:
                return cp.asarray(array, dtype=cp.float32)
            except Exception as e:
                warnings.warn(f"CuPy GPU转移失败: {e}")
                return array

        else:
            return array

    def to_cpu(self, gpu_array: Union[cp.ndarray, torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        将GPU数组转回CPU

        Args:
            gpu_array: GPU数组

        Returns:
            numpy数组
        """
        if isinstance(gpu_array, np.ndarray):
            return gpu_array

        if self.backend == 'pytorch' and torch.is_tensor(gpu_array):
            return gpu_array.cpu().numpy()

        elif self.backend == 'cupy' and isinstance(gpu_array, cp.ndarray):
            return cp.asnumpy(gpu_array)

        else:
            return np.asarray(gpu_array)

    def empty_cache(self):
        """清空GPU缓存"""
        if self.backend == 'pytorch':
            torch.cuda.empty_cache()
        elif self.backend == 'cupy':
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

    def get_memory_info(self) -> dict:
        """获取GPU内存信息"""
        if self.backend == 'pytorch':
            return {
                'allocated_gb': torch.cuda.memory_allocated() / (1024 ** 3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3)
            }
        elif self.backend == 'cupy':
            return self.gpu_info
        else:
            return {}

    def __repr__(self):
        return f"GPUManager(backend='{self.backend}')"