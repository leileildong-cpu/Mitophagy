#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相似性聚合器
整合多维相似性得分
"""

import numpy as np
from typing import Dict, Optional
import warnings


class SimilarityAggregator:
    """相似性聚合器"""

    def __init__(self, config: dict):
        """
        初始化聚合器

        Args:
            config: 配置字典
        """
        self.config = config

        # 权重配置
        similarity_config = config.get("similarity", {})
        self.w_1d = similarity_config.get("w_1d", 0.35)
        self.w_2d = similarity_config.get("w_2d", 0.40)
        self.w_3d = similarity_config.get("w_3d", 0.25)

        # 验证权重和为1
        total_weight = self.w_1d + self.w_2d + self.w_3d
        if abs(total_weight - 1.0) > 1e-6:
            warnings.warn(f"权重和不为1: {total_weight}, 自动归一化")
            self.w_1d /= total_weight
            self.w_2d /= total_weight
            self.w_3d /= total_weight

        print(f"[INFO] 相似性聚合权重: 1D={self.w_1d:.3f}, 2D={self.w_2d:.3f}, 3D={self.w_3d:.3f}")

    def aggregate(self, similarities: Dict, normalize: bool = False) -> Dict:
        """
        聚合多维相似性

        Args:
            similarities: 相似性字典 {
                '1d_similarity': np.ndarray (n_lib, n_ref),
                '2d_similarity': np.ndarray (n_lib, n_ref),
                '3d_similarity': np.ndarray (n_lib, n_ref)
            }
            normalize: 是否归一化各维度相似性

        Returns:
            聚合结果 {
                'combined_scores': np.ndarray (n_lib,),
                'individual_scores': Dict
            }
        """
        print("[INFO] 聚合多维相似性...")

        sim_1d = self._to_numpy(similarities['1d_similarity'])
        sim_2d = self._to_numpy(similarities['2d_similarity'])
        sim_3d = self._to_numpy(similarities['3d_similarity'])

        # 输入验证
        self._validate_input(sim_1d, sim_2d, sim_3d)

        # 取每个库分子与所有参考分子的最大相似性
        max_sim_1d = np.max(sim_1d, axis=1)
        max_sim_2d = np.max(sim_2d, axis=1)
        max_sim_3d = np.max(sim_3d, axis=1)

        print(f"[DEBUG] 最大相似性范围:")
        print(f"  - 1D: {max_sim_1d.min():.4f} - {max_sim_1d.max():.4f}")
        print(f"  - 2D: {max_sim_2d.min():.4f} - {max_sim_2d.max():.4f}")
        print(f"  - 3D: {max_sim_3d.min():.4f} - {max_sim_3d.max():.4f}")

        # 归一化（可选）
        if normalize:
            max_sim_1d = self._minmax_normalize(max_sim_1d)
            max_sim_2d = self._minmax_normalize(max_sim_2d)
            max_sim_3d = self._minmax_normalize(max_sim_3d)
            print("[DEBUG] 已应用Min-Max归一化")

        # 加权组合
        combined_scores = (
                self.w_1d * max_sim_1d +
                self.w_2d * max_sim_2d +
                self.w_3d * max_sim_3d
        )

        print(f"[DEBUG] 组合得分范围: {combined_scores.min():.4f} - {combined_scores.max():.4f}")
        print(f"[DEBUG] 组合得分平均: {combined_scores.mean():.4f}")
        print(f"[DEBUG] 组合得分中位数: {np.median(combined_scores):.4f}")

        # 详细分解（前10个）
        self._print_score_breakdown(max_sim_1d, max_sim_2d, max_sim_3d, combined_scores)

        return {
            'combined_scores': combined_scores,
            'individual_scores': {
                '1d_max': max_sim_1d,
                '2d_max': max_sim_2d,
                '3d_max': max_sim_3d,
                '1d_norm': self._minmax_normalize(max_sim_1d) if not normalize else max_sim_1d,
                '2d_norm': self._minmax_normalize(max_sim_2d) if not normalize else max_sim_2d,
                '3d_norm': self._minmax_normalize(max_sim_3d) if not normalize else max_sim_3d,
            }
        }

    def aggregate_similarities(self, similarities: Dict, normalize: bool = False) -> Dict:
        """
        聚合相似性（主接口方法，调用 aggregate）

        Args:
            similarities: 相似性字典
            normalize: 是否归一化

        Returns:
            聚合结果字典
        """
        return self.aggregate(similarities, normalize)
    def _to_numpy(self, array):
        """转换为numpy数组"""
        if array is None:
            raise ValueError("相似性数组不能为None")

        try:
            import torch
            if torch.is_tensor(array):
                return array.cpu().numpy()
        except ImportError:
            pass

        return np.asarray(array, dtype=np.float32)

    def _validate_input(self, sim_1d: np.ndarray, sim_2d: np.ndarray, sim_3d: np.ndarray):
        """验证输入相似性矩阵"""
        print(f"[DEBUG] 输入相似性矩阵形状:")
        print(f"  - 1D: {sim_1d.shape}")
        print(f"  - 2D: {sim_2d.shape}")
        print(f"  - 3D: {sim_3d.shape}")

        # 检查形状一致性
        if not (sim_1d.shape == sim_2d.shape == sim_3d.shape):
            raise ValueError(f"相似性矩阵形状不一致: 1D={sim_1d.shape}, 2D={sim_2d.shape}, 3D={sim_3d.shape}")

        print(f"[DEBUG] 输入相似性范围:")
        print(f"  - 1D: {sim_1d.min():.6f} - {sim_1d.max():.6f}")
        print(f"  - 2D: {sim_2d.min():.6f} - {sim_2d.max():.6f}")
        print(f"  - 3D: {sim_3d.min():.6f} - {sim_3d.max():.6f}")

    def _minmax_normalize(self, arr: np.ndarray) -> np.ndarray:
        """Min-Max归一化"""
        if len(arr) == 0:
            return arr

        min_val, max_val = np.min(arr), np.max(arr)
        if max_val - min_val < 1e-12:
            return arr.copy()

        return (arr - min_val) / (max_val - min_val)

    def _print_score_breakdown(self, max_sim_1d, max_sim_2d, max_sim_3d, combined_scores):
        """打印得分分解"""
        print(f"[DEBUG] 前10个化合物得分分解:")
        for i in range(min(10, len(combined_scores))):
            print(f"  [{i}] "
                  f"1D={max_sim_1d[i]:.4f}*{self.w_1d:.2f} + "
                  f"2D={max_sim_2d[i]:.4f}*{self.w_2d:.2f} + "
                  f"3D={max_sim_3d[i]:.4f}*{self.w_3d:.2f} = "
                  f"{combined_scores[i]:.4f}")

    def __repr__(self):
        return f"SimilarityAggregator(w_1d={self.w_1d:.3f}, w_2d={self.w_2d:.3f}, w_3d={self.w_3d:.3f})"