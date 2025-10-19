#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLRP3抑制剂超高速虚拟筛选系统 - 完整版
整合原有4000行代码的所有功能 + NLRP3特异性优化
版本: v3.0 Complete Edition
"""

import os

os.environ['RDK_PICKLE_PROTOCOL'] = '2'

import warnings

warnings.filterwarnings('ignore', message='.*Pickling.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import argparse
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Union, Iterator
from datetime import datetime
import pickle
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial, lru_cache
import threading
from collections import defaultdict
import gc
import psutil
import logging
from logging.handlers import RotatingFileHandler

warnings.filterwarnings('ignore')

# 核心依赖
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# GPU加速相关
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    from cupyx.scipy.spatial.distance import cdist as cp_cdist

    GPU_AVAILABLE = True
    print("[INFO] GPU加速已启用 (CuPy)")
except ImportError:
    import numpy as cp

    GPU_AVAILABLE = False
    print("[WARNING] GPU不可用，使用CPU计算")

# PyTorch GPU支持
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = torch.cuda.is_available() if torch.cuda.is_available() else False
    if TORCH_AVAILABLE:
        print(f"[INFO] PyTorch GPU可用: {torch.cuda.get_device_name(0)}")
        DEVICE = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('high')
    else:
        print("[INFO] PyTorch使用CPU")
        DEVICE = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch不可用")

# 机器学习
try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn不可用，AI功能禁用")

# Mol2vec（论文核心）
try:
    from mol2vec.features import mol2alt_sentence, sentences2vec
    from gensim.models import word2vec

    MOL2VEC_AVAILABLE = True
except ImportError:
    MOL2VEC_AVAILABLE = False
    print("[WARNING] mol2vec不可用，使用Morgan指纹替代")

# 3D形状相似性
try:
    from rdkit.Chem import rdShapeHelpers
    from rdkit.Chem.rdMolAlign import AlignMol

    SHAPE3D_AVAILABLE = True
except ImportError:
    SHAPE3D_AVAILABLE = False
    print("[WARNING] 3D形状相似性不可用")

# 分子标准化
try:
    from rdkit.Chem import rdMolStandardize as rdMS
except ImportError:
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize as rdMS
    except ImportError:
        rdMS = None


# ========================================
# 第一部分：核心基础类（保留原版所有功能）
# ========================================

class OptimizedCPUThreadPool:
    """优化的CPU多线程管理器 - 更智能的任务调度（原版保留）"""

    def __init__(self, max_workers: Optional[int] = None):
        cpu_count = os.cpu_count() or 1
        available_memory = psutil.virtual_memory().available / (1024 ** 3)

        if max_workers is None:
            if available_memory > 16:
                self.max_workers = min(32, cpu_count * 2)
            elif available_memory > 8:
                self.max_workers = min(16, cpu_count + 4)
            else:
                self.max_workers = min(8, cpu_count)
        else:
            self.max_workers = max_workers

        print(f"[INFO] 优化CPU线程池: {self.max_workers} 个工作线程 (可用内存: {available_memory:.1f}GB)")

        self.process_pool = None
        self.thread_pool = None

    def map_parallel(self, func, items, desc="处理中", use_processes=False, batch_size=None,
                     early_stop_threshold=None):
        """增强的并行映射函数"""
        if len(items) < 50:
            return [func(item) for item in tqdm(items, desc=desc)]

        if batch_size is None:
            batch_size = max(1, len(items) // (self.max_workers * 4))

        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        results = []
        with executor_class(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                if len(batch) == 1:
                    future = executor.submit(func, batch[0])
                else:
                    batch_func = partial(self._batch_process, func)
                    future = executor.submit(batch_func, batch)
                futures.append(future)

            with tqdm(total=len(items), desc=f"{desc} (批处理)") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        if isinstance(batch_results, list):
                            results.extend(batch_results)
                            pbar.update(len(batch_results))
                        else:
                            results.append(batch_results)
                            pbar.update(1)

                        if early_stop_threshold and len(results) >= early_stop_threshold:
                            break

                    except Exception as e:
                        print(f"[WARNING] 批处理任务失败: {e}")
                        continue

        return results[:len(items)]

    def _batch_process(self, func, batch):
        """批处理函数"""
        return [func(item) for item in batch]

    def __del__(self):
        """清理资源"""
        if self.process_pool:
            self.process_pool.shutdown(wait=False)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)


class FastMolecularFeatureCache:
    """分子特征缓存系统（原版保留）"""

    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = defaultdict(int)
        self.lock = threading.Lock()

    def get_key(self, mol):
        """生成分子的缓存键"""
        if mol is None:
            return None
        try:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return smiles
        except Exception:
            return None

    def get(self, mol, feature_type):
        """获取缓存的特征"""
        key = self.get_key(mol)
        if key is None:
            return None

        cache_key = f"{key}_{feature_type}"

        with self.lock:
            if cache_key in self.cache:
                self.access_count[cache_key] += 1
                return self.cache[cache_key]

        return None

    def set(self, mol, feature_type, features):
        """设置缓存"""
        key = self.get_key(mol)
        if key is None:
            return

        cache_key = f"{key}_{feature_type}"

        with self.lock:
            # 缓存满时清理最少使用的
            if len(self.cache) >= self.max_size:
                sorted_keys = sorted(self.cache.keys(), key=lambda k: self.access_count.get(k, 0))
                for k in sorted_keys[:self.max_size // 10]:
                    if k in self.cache:
                        del self.cache[k]
                    if k in self.access_count:
                        del self.access_count[k]

            # 确保features是可序列化的
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

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'usage': len(self.cache) / self.max_size * 100 if self.max_size > 0 else 0
            }

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()


# ========================================
# 第二部分：NLRP3特异性模块（新增）
# ========================================

# 导入NLRP3模块（假设已有这些文件）
try:
    from nlrp3_filters import NLRP3SpecificFilters, Brenk_Filter
    from molecular_docking import MolecularDocking

    NLRP3_MODULES_AVAILABLE = True
except ImportError:
    NLRP3_MODULES_AVAILABLE = False
    print("[WARNING] NLRP3模块不可用")

# ... 继续Part 2 ...