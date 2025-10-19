#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指纹相似性计算
支持Tanimoto、Dice等多种相似性度量
支持GPU加速（PyTorch/CuPy）
"""

import numpy as np
from typing import Union, Optional
import warnings

# GPU支持
TORCH_AVAILABLE = False
CUPY_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        DEVICE = torch.device('cuda')
except ImportError:
    pass

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    pass


class FingerprintSimilarity:
    """指纹相似性计算器"""

    def __init__(self, use_gpu: bool = True, gpu_backend: str = 'pytorch', metric: str = 'tanimoto'):
        """
        初始化指纹相似性计算器

        Args:
            use_gpu: 是否使用GPU加速
            gpu_backend: GPU后端 ('pytorch' 或 'cupy')
            metric: 相似性度量 ('tanimoto', 'dice', 'cosine')
        """
        self.metric = metric.lower()
        self.gpu_backend = gpu_backend.lower()

        # 确定是否使用GPU
        if use_gpu:
            if self.gpu_backend == 'pytorch' and TORCH_AVAILABLE:
                self.use_gpu = True
                self.backend = 'pytorch'
                print(f"[INFO] 指纹相似性计算使用GPU加速 (PyTorch, metric={self.metric})")
            elif self.gpu_backend == 'cupy' and CUPY_AVAILABLE:
                self.use_gpu = True
                self.backend = 'cupy'
                print(f"[INFO] 指纹相似性计算使用GPU加速 (CuPy, metric={self.metric})")
            else:
                self.use_gpu = False
                self.backend = 'cpu'
                print(f"[INFO] 指纹相似性计算使用CPU (GPU不可用, metric={self.metric})")
        else:
            self.use_gpu = False
            self.backend = 'cpu'
            print(f"[INFO] 指纹相似性计算使用CPU (metric={self.metric})")

    def compute_tanimoto_similarity(self,
                                    ref_fps: Union[np.ndarray, list],
                                    lib_fps: Union[np.ndarray, list]) -> np.ndarray:
        """
        计算Tanimoto相似性（主接口，兼容旧代码）

        Args:
            ref_fps: 参考指纹 (n_ref, fp_length)
            lib_fps: 库指纹 (n_lib, fp_length)

        Returns:
            相似性矩阵 (n_lib, n_ref)
        """
        return self.calculate_similarity(ref_fps, lib_fps)

    def calculate_similarity(self,
                             ref_fps: Union[np.ndarray, list],
                             lib_fps: Union[np.ndarray, list]) -> np.ndarray:
        """
        计算指纹相似性矩阵

        Args:
            ref_fps: 参考指纹 (n_ref, fp_length)
            lib_fps: 库指纹 (n_lib, fp_length)

        Returns:
            相似性矩阵 (n_lib, n_ref)
        """
        print(f"[DEBUG] 计算指纹相似性 (metric={self.metric}, backend={self.backend})...")

        # 转换为numpy数组
        ref_array, lib_array = self._prepare_arrays(ref_fps, lib_fps)

        if ref_array is None or lib_array is None:
            print("[ERROR] 指纹数组准备失败")
            n_lib = len(lib_fps) if isinstance(lib_fps, list) else lib_fps.shape[0]
            n_ref = len(ref_fps) if isinstance(ref_fps, list) else ref_fps.shape[0]
            return np.zeros((n_lib, n_ref), dtype=np.float32)

        # 根据度量类型计算相似性
        if self.metric == 'tanimoto':
            similarity = self._calculate_tanimoto(ref_array, lib_array)
        elif self.metric == 'dice':
            similarity = self._calculate_dice(ref_array, lib_array)
        elif self.metric == 'cosine':
            similarity = self._calculate_cosine(ref_array, lib_array)
        else:
            raise ValueError(f"未知的相似性度量: {self.metric}")

        # 验证结果
        self._validate_similarity(similarity)

        return similarity

    def _prepare_arrays(self, ref_fps, lib_fps):
        """准备指纹数组"""
        # 处理PyTorch张量
        if TORCH_AVAILABLE and torch.is_tensor(ref_fps):
            ref_fps = ref_fps.cpu().numpy()
        if TORCH_AVAILABLE and torch.is_tensor(lib_fps):
            lib_fps = lib_fps.cpu().numpy()

        # 处理CuPy数组
        if CUPY_AVAILABLE and isinstance(ref_fps, cp.ndarray):
            ref_fps = cp.asnumpy(ref_fps)
        if CUPY_AVAILABLE and isinstance(lib_fps, cp.ndarray):
            lib_fps = cp.asnumpy(lib_fps)

        # 处理列表
        if isinstance(ref_fps, list):
            valid_ref = [fp for fp in ref_fps if fp is not None]
            valid_lib = [fp for fp in lib_fps if fp is not None]

            if not valid_ref or not valid_lib:
                warnings.warn("没有有效指纹")
                return None, None

            ref_array = np.array(valid_ref, dtype=np.float32)
            lib_array = np.array(valid_lib, dtype=np.float32)
        else:
            ref_array = np.asarray(ref_fps, dtype=np.float32)
            lib_array = np.asarray(lib_fps, dtype=np.float32)

        print(f"[DEBUG] 指纹数组形状: ref={ref_array.shape}, lib={lib_array.shape}")

        return ref_array, lib_array

    def _calculate_tanimoto(self, ref_array: np.ndarray, lib_array: np.ndarray) -> np.ndarray:
        """计算Tanimoto相似性"""
        n_lib, n_ref = lib_array.shape[0], ref_array.shape[0]
        similarity_matrix = np.zeros((n_lib, n_ref), dtype=np.float32)

        # 向量化计算
        for i in range(n_lib):
            for j in range(n_ref):
                intersection = np.sum(np.minimum(lib_array[i], ref_array[j]))
                union = np.sum(np.maximum(lib_array[i], ref_array[j]))

                if union > 0:
                    similarity_matrix[i, j] = intersection / union
                else:
                    similarity_matrix[i, j] = 0.0

        return similarity_matrix

    def _calculate_dice(self, ref_array: np.ndarray, lib_array: np.ndarray) -> np.ndarray:
        """计算Dice相似性"""
        n_lib, n_ref = lib_array.shape[0], ref_array.shape[0]
        similarity_matrix = np.zeros((n_lib, n_ref), dtype=np.float32)

        for i in range(n_lib):
            for j in range(n_ref):
                intersection = np.sum(np.minimum(lib_array[i], ref_array[j]))
                sum_both = np.sum(lib_array[i]) + np.sum(ref_array[j])

                if sum_both > 0:
                    similarity_matrix[i, j] = 2 * intersection / sum_both
                else:
                    similarity_matrix[i, j] = 0.0

        return similarity_matrix

    def _calculate_cosine(self, ref_array: np.ndarray, lib_array: np.ndarray) -> np.ndarray:
        """计算余弦相似性"""
        # 归一化
        ref_norm = ref_array / (np.linalg.norm(ref_array, axis=1, keepdims=True) + 1e-8)
        lib_norm = lib_array / (np.linalg.norm(lib_array, axis=1, keepdims=True) + 1e-8)

        # 计算余弦相似性
        similarity = np.dot(lib_norm, ref_norm.T)

        return similarity.astype(np.float32)

    def _validate_similarity(self, similarity: np.ndarray):
        """验证相似性矩阵"""
        print(f"[DEBUG] 相似性矩阵形状: {similarity.shape}")
        print(f"[DEBUG] 相似性范围: {similarity.min():.6f} - {similarity.max():.6f}")
        print(f"[DEBUG] 相似性平均值: {similarity.mean():.6f}")
        print(f"[DEBUG] 相似性中位数: {np.median(similarity):.6f}")
        print(f"[DEBUG] 非零相似性比例: {(similarity > 0).sum() / similarity.size * 100:.2f}%")

        if similarity.max() == 0:
            warnings.warn("所有相似性都是0！")

    def __repr__(self):
        if self.use_gpu:
            backend_str = f"GPU-{self.backend}"
        else:
            backend_str = "CPU"
        return f"FingerprintSimilarity(backend={backend_str}, metric={self.metric})"