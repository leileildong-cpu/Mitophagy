#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速分子特征缓存系统
支持LRU策略、线程安全和统计信息
"""

import threading
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple
from rdkit import Chem
import numpy as np


class FastMolecularFeatureCache:
    """分子特征缓存系统"""

    def __init__(self, max_size: int = 10000):
        """
        初始化缓存系统

        Args:
            max_size: 最大缓存条目数
        """
        self.cache = {}
        self.max_size = max_size
        self.access_count = defaultdict(int)
        self.lock = threading.Lock()

        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    def get_key(self, mol: Chem.Mol) -> Optional[str]:
        """
        生成分子的唯一缓存键

        Args:
            mol: RDKit分子对象

        Returns:
            缓存键字符串，失败返回None
        """
        if mol is None:
            return None

        try:
            # 使用canonical SMILES作为键
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return smiles
        except Exception:
            return None

    def get(self, mol: Chem.Mol, feature_type: str) -> Optional[Any]:
        """
        获取缓存的特征

        Args:
            mol: RDKit分子对象
            feature_type: 特征类型（如'morgan_fp', 'descriptors'）

        Returns:
            缓存的特征，不存在返回None
        """
        key = self.get_key(mol)
        if key is None:
            return None

        cache_key = f"{key}_{feature_type}"

        with self.lock:
            if cache_key in self.cache:
                self.access_count[cache_key] += 1
                self.stats['hits'] += 1
                return self.cache[cache_key]
            else:
                self.stats['misses'] += 1

        return None

    def set(self, mol: Chem.Mol, feature_type: str, features: Any):
        """
        设置缓存

        Args:
            mol: RDKit分子对象
            feature_type: 特征类型
            features: 特征数据
        """
        key = self.get_key(mol)
        if key is None:
            return

        cache_key = f"{key}_{feature_type}"

        with self.lock:
            # 缓存满时清理最少使用的条目（LRU策略）
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            # 存储特征
            try:
                if isinstance(features, dict):
                    self.cache[cache_key] = features.copy()
                elif isinstance(features, np.ndarray):
                    self.cache[cache_key] = features.copy()
                else:
                    self.cache[cache_key] = features

                self.access_count[cache_key] = 1

            except Exception as e:
                print(f"[WARNING] 缓存设置失败: {e}")

    def _evict_lru(self):
        """清理最少使用的缓存条目"""
        # 计算要清理的数量（10%）
        evict_count = max(1, self.max_size // 10)

        # 按访问次数排序
        sorted_keys = sorted(self.cache.keys(), key=lambda k: self.access_count.get(k, 0))

        # 清理最少使用的
        for k in sorted_keys[:evict_count]:
            if k in self.cache:
                del self.cache[k]
                self.stats['evictions'] += 1
            if k in self.access_count:
                del self.access_count[k]

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        with self.lock:
            total_accesses = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_accesses * 100) if total_accesses > 0 else 0

            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'usage_percent': len(self.cache) / self.max_size * 100 if self.max_size > 0 else 0,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self.stats['evictions']
            }

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()
            self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}

    def warm_up(self, molecules: list, feature_extractors: dict):
        """
        预热缓存 - 批量提取特征

        Args:
            molecules: 分子列表
            feature_extractors: 特征提取函数字典 {'feature_type': extractor_func}
        """
        print(f"[INFO] 预热缓存，处理 {len(molecules)} 个分子...")

        for mol in molecules:
            for feature_type, extractor in feature_extractors.items():
                try:
                    features = extractor(mol)
                    self.set(mol, feature_type, features)
                except Exception as e:
                    continue

        stats = self.get_stats()
        print(f"[INFO] 缓存预热完成: {stats['size']} 条目")

    def __repr__(self):
        stats = self.get_stats()
        return (f"FastMolecularFeatureCache(size={stats['size']}/{stats['max_size']}, "
                f"hit_rate={stats['hit_rate']:.1f}%)")