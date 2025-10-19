#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超高速GPU加速 + 多线程CPU优化的论文复现版虚拟筛选系统
基于"AI-Assisted Discovery of Mitophagy-Inducing Compounds"
包含：多表示学习 + GPU加速 + CPU多线程 + 完整论文逻辑 + 速度优化
版本: v2.0 Ultra Fast
"""
import os
os.environ['RDK_PICKLE_PROTOCOL'] = '2'  # 设置pickle协议版本

import warnings
warnings.filterwarnings('ignore', message='.*Pickling.*')  # 忽略pickle警告
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
        # 优化PyTorch设置
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('high')  # 启用TensorFloat-32
    else:
        print("[INFO] PyTorch使用CPU")
        DEVICE = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch不可用")

# 机器学习
try:
    from sklearn.cluster import MiniBatchKMeans  # 替换KMeans为更快的MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors  # 用于快速相似性搜索

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


class OptimizedCPUThreadPool:
    """优化的CPU多线程管理器 - 更智能的任务调度"""

    def __init__(self, max_workers: Optional[int] = None):
        # 动态计算最优线程数
        cpu_count = os.cpu_count() or 1
        available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB

        # 基于CPU和内存动态调整线程数
        if max_workers is None:
            if available_memory > 16:  # 大内存系统
                self.max_workers = min(32, cpu_count * 2)
            elif available_memory > 8:  # 中等内存
                self.max_workers = min(16, cpu_count + 4)
            else:  # 小内存系统
                self.max_workers = min(8, cpu_count)
        else:
            self.max_workers = max_workers

        print(f"[INFO] 优化CPU线程池: {self.max_workers} 个工作线程 (可用内存: {available_memory:.1f}GB)")

        self.process_pool = None
        self.thread_pool = None

    def map_parallel(self, func, items, desc="处理中", use_processes=False, batch_size=None,
                     early_stop_threshold=None):
        """增强的并行映射函数 - 支持批处理和早停"""
        if len(items) < 50:  # 小数据集直接串行处理
            return [func(item) for item in tqdm(items, desc=desc)]

        # 动态批处理
        if batch_size is None:
            batch_size = max(1, len(items) // (self.max_workers * 4))

        # 选择执行器
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        results = []
        with executor_class(max_workers=self.max_workers) as executor:
            # 批量提交任务
            futures = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                if len(batch) == 1:
                    future = executor.submit(func, batch[0])
                else:
                    # 批处理函数
                    batch_func = partial(self._batch_process, func)
                    future = executor.submit(batch_func, batch)
                futures.append(future)

            # 收集结果，支持早停
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

                        # 早停检查
                        if early_stop_threshold and len(results) >= early_stop_threshold:
                            break

                    except Exception as e:
                        print(f"[WARNING] 批处理任务失败: {e}")
                        continue

        return results[:len(items)]  # 确保返回正确数量的结果

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
    """分子特征缓存系统 - 修复版"""

    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = defaultdict(int)

    def get_key(self, mol):
        """生成分子的缓存键 - 修复版"""
        if mol is None:
            return None
        try:
            # 使用SMILES作为键，而不是分子对象
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return smiles
        except Exception as e:
            return None

    def get(self, mol, feature_type):
        """获取缓存的特征"""
        key = self.get_key(mol)
        if key is None:
            return None

        cache_key = f"{key}_{feature_type}"
        if cache_key in self.cache:
            self.access_count[cache_key] += 1
            return self.cache[cache_key]
        return None

    def set(self, mol, feature_type, features):
        """设置缓存 - 修复版"""
        key = self.get_key(mol)
        if key is None:
            return

        cache_key = f"{key}_{feature_type}"

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
            # 如果features是字典，转换为可哈希的形式
            if isinstance(features, dict):
                # 不直接缓存字典，而是缓存其副本
                self.cache[cache_key] = features.copy()
            elif isinstance(features, np.ndarray):
                self.cache[cache_key] = features.copy()
            else:
                self.cache[cache_key] = features

            self.access_count[cache_key] = 1
        except Exception as e:
            print(f"[WARNING] 缓存设置失败: {e}")


class UltraFastVirtualScreening:
    """超高速GPU加速 + 多线程CPU优化的虚拟筛选系统"""

    def __init__(self, config: dict):
        self.config = self._validate_and_setup_config(config)
        self.cpu_pool = OptimizedCPUThreadPool()
        self.feature_cache = FastMolecularFeatureCache(max_size=20000)
        self.setup_gpu()
        self.setup_logging()
        self._precomputed_models = {}

        # 性能监控
        self.perf_stats = {
            'preprocessing_time': 0,
            'feature_extraction_time': 0,
            'similarity_computation_time': 0,
            'filtering_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def _validate_and_setup_config(self, config: dict) -> dict:
        """验证和设置配置默认值"""

        # 确保filters配置存在
        if "filters" not in config:
            config["filters"] = {}

        # 设置过滤器默认值
        filters_defaults = {
            "pains": True,
            "cns_filter": True,
            "cns_rules": {
                "mw_min": 130,  # 分子量最小值
                "mw_max": 725,  # 分子量最大值
                "logp_min": -7,  # LogP最小值
                "logp_max": 5.5,  # LogP最大值
                "hbd_max": 7,  # 氢键供体最大数
                "hba_max": 12,  # 氢键受体最大数
                "tpsa_max": 200,  # 拓扑极性表面积最大值
                "rotb_max": 11  # 可旋转键最大数
            }
        }

        for key, default_value in filters_defaults.items():
            if key not in config["filters"]:
                config["filters"][key] = default_value

        # 确保similarity配置存在
        if "similarity" not in config:
            config["similarity"] = {}

        similarity_defaults = {
            "w_1d": 0.4,
            "w_2d": 0.4,
            "w_3d": 0.2
        }

        for key, default_value in similarity_defaults.items():
            if key not in config["similarity"]:
                config["similarity"][key] = default_value

        # 确保ai_model配置存在
        if "ai_model" not in config:
            config["ai_model"] = {}

        ai_model_defaults = {
            "use_clustering": False,
            "use_outlier_filter": False,
            "similarity_threshold": 0.75
        }

        for key, default_value in ai_model_defaults.items():
            if key not in config["ai_model"]:
                config["ai_model"][key] = default_value

        # 确保performance配置存在
        if "performance" not in config:
            config["performance"] = {}

        performance_defaults = {
            "chunk_size": 10000,
            "use_streaming": False,
            "max_results": 50000
        }

        for key, default_value in performance_defaults.items():
            if key not in config["performance"]:
                config["performance"][key] = default_value

        # 确保fingerprints配置存在
        if "fingerprints" not in config:
            config["fingerprints"] = {}

        fingerprints_defaults = {
            "morgan_radius": 2,
            "morgan_nbits": 2048
        }

        for key, default_value in fingerprints_defaults.items():
            if key not in config["fingerprints"]:
                config["fingerprints"][key] = default_value

        # 确保输出配置存在
        if "output" not in config:
            config["output"] = {}

        output_defaults = {
            "dir": "results",
            "hits_csv": "virtual_screening_hits.csv",
            "report_txt": "screening_report.txt"
        }

        for key, default_value in output_defaults.items():
            if key not in config["output"]:
                config["output"][key] = default_value

        return config

    def setup_gpu(self):
        """优化的GPU环境设置 - 修复内存管理"""
        global GPU_AVAILABLE, TORCH_AVAILABLE

        if GPU_AVAILABLE:
            try:
                import cupy as cp
                device_count = cp.cuda.runtime.getDeviceCount()
                current_device = cp.cuda.device.get_device_id()

                # 设置内存池策略 - 更保守的设置
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()

                # 获取内存信息
                free_memory, total_memory = cp.cuda.runtime.memGetInfo()
                # 使用更保守的内存限制（70%而不是85%）
                memory_limit = int(free_memory * 0.7)
                mempool.set_limit(size=memory_limit)

                self.gpu_info = {
                    'device_count': device_count,
                    'current_device': current_device,
                    'total_memory': total_memory / (1024 ** 3),
                    'free_memory': free_memory / (1024 ** 3),
                    'memory_limit': memory_limit / (1024 ** 3)
                }

                print(f"[GPU] CuPy设置: {self.gpu_info['total_memory']:.1f}GB总量, "
                      f"{self.gpu_info['memory_limit']:.1f}GB限制")

            except Exception as e:
                print(f"[WARNING] CuPy GPU设置失败: {e}")
                GPU_AVAILABLE = False

        # PyTorch GPU优化设置
        if TORCH_AVAILABLE:
            try:
                # 设置PyTorch内存管理
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.set_num_threads(min(4, self.cpu_pool.max_workers // 4))

                if torch.cuda.is_available():
                    # 更保守的内存设置
                    torch.cuda.set_per_process_memory_fraction(0.7)  # 从0.9改为0.7
                    torch.cuda.empty_cache()

                    # 禁用内存固定以避免警告
                    torch.multiprocessing.set_sharing_strategy('file_system')

                    device_name = torch.cuda.get_device_name(0)
                    print(f"[GPU] PyTorch设置: {device_name}")

            except Exception as e:
                print(f"[WARNING] PyTorch GPU设置失败: {e}")
                TORCH_AVAILABLE = False

        # PyTorch GPU优化设置
        if TORCH_AVAILABLE:
            try:
                # 更激进的GPU设置
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.set_num_threads(min(4, self.cpu_pool.max_workers // 4))

                if torch.cuda.is_available():
                    # 使用更多显存
                    torch.cuda.set_per_process_memory_fraction(0.9)
                    torch.cuda.empty_cache()

                    # 启用内存预分配
                    torch.cuda.set_per_process_memory_fraction(0.9)

                    device_name = torch.cuda.get_device_name(0)
                    print(f"[GPU] PyTorch优化设置: {device_name}")

            except Exception as e:
                print(f"[WARNING] PyTorch GPU优化失败: {e}")
                TORCH_AVAILABLE = False



    def setup_logging(self):
        """设置日志系统"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_dir = self.config["output"]["dir"]
        self.output_dir = f"{original_dir}_ultrafast_{timestamp}"
        self.config["output"]["dir"] = self.output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # 保存配置文件副本
        config_backup = os.path.join(self.output_dir, "config_used.yaml")
        with open(config_backup, 'w') as f:
            yaml.dump(self.config, f)

    def load_and_preprocess_data_streaming(self) -> Tuple[pd.DataFrame, Iterator]:
        """流式数据加载和预处理 - 处理超大数据集"""
        print("[INFO] 启动流式数据加载...")
        start_time = time.time()

        # 读取参考数据（通常较小）
        ref_df = pd.read_csv(self.config["data"]["references_csv"])
        print(f"[INFO] 参考分子: {len(ref_df)}")

        # 验证必需列
        required_cols = {"id", "name", "smiles"}
        for df, name in [(ref_df, "references")]:
            if not required_cols.issubset(set(df.columns)):
                raise ValueError(f"{name} 缺少必需列: {required_cols - set(df.columns)}")

        # 预处理参考分子
        ref_df = self._preprocess_molecules_parallel(ref_df, "参考分子")

        # 流式处理库文件
        def library_stream():
            """库文件流式生成器"""
            chunk_size = self.config.get("performance", {}).get("chunk_size", 10000)

            for chunk in pd.read_csv(self.config["data"]["library_csv"], chunksize=chunk_size):
                if not required_cols.issubset(set(chunk.columns)):
                    continue

                # 预处理chunk
                processed_chunk = self._preprocess_molecules_parallel(chunk, f"库分子块")
                if len(processed_chunk) > 0:
                    yield processed_chunk

        self.perf_stats['preprocessing_time'] = time.time() - start_time
        return ref_df, library_stream()

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """标准数据加载（向后兼容）"""
        print("[INFO] 加载和预处理数据（优化版）...")
        start_time = time.time()

        # 读取数据
        ref_df = pd.read_csv(self.config["data"]["references_csv"])
        lib_df = pd.read_csv(self.config["data"]["library_csv"])

        print(f"[INFO] 原始数据: 参考分子={len(ref_df)}, 库分子={len(lib_df)}")

        # 验证必需列
        required_cols = {"id", "name", "smiles"}
        for df, name in [(ref_df, "references"), (lib_df, "library")]:
            if not required_cols.issubset(set(df.columns)):
                raise ValueError(f"{name} 缺少必需列: {required_cols - set(df.columns)}")

        # 并行分子预处理（优化版）
        ref_df = self._preprocess_molecules_parallel_optimized(ref_df, "参考分子")
        lib_df = self._preprocess_molecules_parallel_optimized(lib_df, "库分子")

        if len(ref_df) == 0 or len(lib_df) == 0:
            raise ValueError("预处理后没有有效分子")

        print(f"[INFO] 有效分子: 参考={len(ref_df)}, 库={len(lib_df)}")
        self.perf_stats['preprocessing_time'] = time.time() - start_time
        return ref_df, lib_df

    def _preprocess_molecules_parallel_optimized(self, df: pd.DataFrame, desc: str) -> pd.DataFrame:
        """优化的并行分子预处理"""
        print(f"[INFO] 预处理{desc}（优化版）...")

        # 准备数据
        smiles_data = [(idx, row['smiles'], row) for idx, row in df.iterrows()]

        # 优化的并行标准化 - 使用更大的批处理
        batch_size = max(100, len(smiles_data) // (self.cpu_pool.max_workers * 2))

        standardize_func = partial(self._standardize_molecule_with_data_cached)
        valid_mols = self.cpu_pool.map_parallel(
            standardize_func,
            smiles_data,
            desc=f"标准化{desc}",
            use_processes=True,
            batch_size=batch_size
        )

        # 过滤有效结果（向量化操作）
        valid_mols = [mol for mol in valid_mols if mol is not None]

        if not valid_mols:
            return pd.DataFrame()

        result_df = pd.DataFrame(valid_mols)

        # 去重（优化版）
        before_dedup = len(result_df)
        # 使用hash加速去重
        result_df['smiles_hash'] = result_df['canonical_smiles'].apply(hash)
        result_df = result_df.drop_duplicates(subset=['smiles_hash']).reset_index(drop=True)
        result_df = result_df.drop('smiles_hash', axis=1)
        after_dedup = len(result_df)

        if before_dedup != after_dedup:
            print(f"[INFO] {desc}去重: {before_dedup} → {after_dedup}")

        return result_df

    def _preprocess_molecules_parallel(self, df: pd.DataFrame, desc: str) -> pd.DataFrame:
        """向后兼容的预处理方法"""
        return self._preprocess_molecules_parallel_optimized(df, desc)

    @lru_cache(maxsize=10000)
    def _standardize_molecule_cached(self, smiles: str) -> Optional[str]:
        """缓存的分子标准化"""
        mol = self._standardize_molecule(smiles)
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None

    def _standardize_molecule_with_data_cached(self, data: Tuple) -> Optional[Dict]:
        """使用缓存的分子标准化"""
        idx, smiles, row = data

        # 检查缓存
        canonical_smiles = self._standardize_molecule_cached(smiles)

        if canonical_smiles:
            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol:
                self.perf_stats['cache_hits'] += 1
                new_row = row.to_dict()
                new_row['mol'] = mol
                new_row['canonical_smiles'] = canonical_smiles
                return new_row

        self.perf_stats['cache_misses'] += 1
        return None

    def _standardize_molecule_with_data(self, data: Tuple) -> Optional[Dict]:
        """向后兼容方法"""
        return self._standardize_molecule_with_data_cached(data)

    def _standardize_molecule(self, smiles: str) -> Optional[Chem.Mol]:
        """优化的分子标准化"""
        if not isinstance(smiles, str) or not smiles.strip():
            return None

        try:
            # 基础解析
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return None

            # 快速有效性检查
            if mol.GetNumAtoms() == 0 or mol.GetNumAtoms() > 150:  # 原子数过滤
                return None

            # RDKit标准化
            Chem.SanitizeMol(mol)

            # 简化的标准化流程（跳过耗时的步骤）
            if rdMS is not None:
                try:
                    # 只进行必要的标准化
                    normalizer = rdMS.Normalizer()
                    mol = normalizer.normalize(mol)
                except:
                    pass

            # 通过canonical SMILES重建
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return Chem.MolFromSmiles(canonical_smiles)

        except Exception:
            return None

    def extract_molecular_features_optimized(self, df: pd.DataFrame, desc: str) -> Dict[str, np.ndarray]:
        """优化的分子特征提取 - 支持缓存和批处理"""
        print(f"[INFO] 提取{desc}特征（超高速版）...")
        start_time = time.time()

        molecules = df['mol'].tolist()
        n_mols = len(molecules)

        # 预加载模型
        if 'mol2vec_model' not in self._precomputed_models:
            self._precomputed_models['mol2vec_model'] = self._load_mol2vec_model()

        if 'morgan_params' not in self._precomputed_models:
            self._precomputed_models['morgan_params'] = self.config["fingerprints"]

        if 'pharm_factory' not in self._precomputed_models:
            self._precomputed_models['pharm_factory'] = Gobbi_Pharm2D.factory

        # 批量特征提取（更大的批处理）
        batch_size = max(500, n_mols // (self.cpu_pool.max_workers))

        extract_func = partial(
            self._extract_batch_features_cached,
            morgan_params=self._precomputed_models['morgan_params'],
            pharm_factory=self._precomputed_models['pharm_factory'],
            mol2vec_model=self._precomputed_models['mol2vec_model']
        )

        # 分批处理
        all_features = []
        for i in range(0, n_mols, batch_size):
            batch_mols = molecules[i:i + batch_size]
            batch_features = self.cpu_pool.map_parallel(
                extract_func,
                [batch_mols],  # 单个批次作为参数
                desc=f"{desc}特征提取 批次{i // batch_size + 1}",
                use_processes=False
            )
            if batch_features:
                all_features.extend(batch_features[0])  # 提取批次结果

        # 整理特征
        features = {
            'morgan_fps': [],
            'pharm2d_fps': [],
            'mol2vec_vecs': [],
            'descriptors': [],
            'molecules': molecules
        }

        for feat_dict in all_features:
            for key in ['morgan_fps', 'pharm2d_fps', 'mol2vec_vecs', 'descriptors']:
                features[key].append(feat_dict.get(key))

        # GPU转移（优化版）
        if GPU_AVAILABLE or TORCH_AVAILABLE:
            features = self._transfer_to_gpu_ultra_fast(features)

        self.perf_stats['feature_extraction_time'] += time.time() - start_time
        return features

    # def extract_molecular_features_optimized(self, df: pd.DataFrame, desc: str) -> Dict[str, np.ndarray]:
    #     """极简版特征提取 - 用于快速测试"""
    #     print(f"[INFO] {desc}极简版特征提取...")
    #     start_time = time.time()
    #
    #     molecules = df['mol'].tolist()
    #     n_mols = len(molecules)
    #
    #     print(f"[DEBUG] 处理 {n_mols} 个分子")
    #
    #     # 只提取最基本的Morgan指纹
    #     features = {
    #         'morgan_fps': [],
    #         'pharm2d_fps': [None] * n_mols,  # 全部设为None，跳过药效团计算
    #         'mol2vec_vecs': [None] * n_mols,  # 全部设为None，跳过mol2vec计算
    #         'descriptors': [],  # 只计算基本描述符
    #         'molecules': molecules
    #     }
    #
    #     # 固定参数，避免从配置读取
    #     morgan_radius = 2
    #     morgan_nbits = 1024  # 减小尺寸提高速度
    #
    #     print("[DEBUG] 开始处理Morgan指纹和基本描述符...")
    #
    #     for i, mol in enumerate(molecules):
    #         # 进度显示
    #         if i % 500 == 0 and i > 0:
    #             progress = (i / n_mols) * 100
    #             elapsed = time.time() - start_time
    #             speed = i / elapsed if elapsed > 0 else 0
    #             print(f"[DEBUG] 进度: {i}/{n_mols} ({progress:.1f}%) - 速度: {speed:.1f} 分子/秒")
    #
    #         if mol is not None:
    #             try:
    #                 # Morgan指纹
    #                 fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
    #                     mol,
    #                     radius=morgan_radius,
    #                     nBits=morgan_nbits
    #                 )
    #                 features['morgan_fps'].append(np.array(fp, dtype=np.uint8))
    #
    #                 # 基本描述符（只计算最重要的几个）
    #                 try:
    #                     desc_dict = {
    #                         'mw': float(Descriptors.MolWt(mol)),
    #                         'logp': float(Crippen.MolLogP(mol)),
    #                         'hbd': float(Lipinski.NumHDonors(mol)),
    #                         'hba': float(Lipinski.NumHAcceptors(mol)),
    #                     }
    #                     features['descriptors'].append(desc_dict)
    #                 except Exception as e:
    #                     print(f"[WARNING] 分子 {i} 描述符计算失败: {e}")
    #                     features['descriptors'].append({
    #                         'mw': 0.0, 'logp': 0.0, 'hbd': 0.0, 'hba': 0.0
    #                     })
    #
    #             except Exception as e:
    #                 print(f"[WARNING] 分子 {i} Morgan指纹计算失败: {e}")
    #                 features['morgan_fps'].append(None)
    #                 features['descriptors'].append({
    #                     'mw': 0.0, 'logp': 0.0, 'hbd': 0.0, 'hba': 0.0
    #                 })
    #         else:
    #             # 处理空分子
    #             features['morgan_fps'].append(None)
    #             features['descriptors'].append({
    #                 'mw': 0.0, 'logp': 0.0, 'hbd': 0.0, 'hba': 0.0
    #             })
    #
    #     elapsed_time = time.time() - start_time
    #     print(f"[INFO] {desc}极简版完成 - 耗时: {elapsed_time:.2f}秒")
    #     print(f"[INFO] 处理速度: {n_mols / elapsed_time:.1f} 分子/秒")
    #
    #     # 不进行GPU转移，直接返回CPU版本
    #     return features



    def extract_molecular_features(self, df: pd.DataFrame, desc: str) -> Dict[str, np.ndarray]:
        """向后兼容的特征提取方法"""
        return self.extract_molecular_features_optimized(df, desc)

    def _extract_batch_features_cached(self, molecules_batch: List[Chem.Mol],
                                       morgan_params: dict, pharm_factory, mol2vec_model) -> List[Dict]:
        """批量特征提取（使用缓存）"""
        results = []

        for mol in molecules_batch:
            # 检查缓存
            cached_features = {}
            cache_keys = ['morgan_fps', 'pharm2d_fps', 'mol2vec_vecs', 'descriptors']

            for feature_type in cache_keys:
                cached = self.feature_cache.get(mol, feature_type)
                if cached is not None:
                    cached_features[feature_type] = cached

            # 计算缺失的特征
            if len(cached_features) == len(cache_keys):
                # 全部命中缓存
                results.append(cached_features)
                self.perf_stats['cache_hits'] += 1
            else:
                # 计算特征
                features = self._extract_single_molecule_features_fast(
                    mol, morgan_params, pharm_factory, mol2vec_model
                )

                # 更新缓存
                for feature_type, feature_value in features.items():
                    if feature_type in cache_keys:
                        self.feature_cache.set(mol, feature_type, feature_value)

                results.append(features)
                self.perf_stats['cache_misses'] += 1

        return results

    def _extract_single_molecule_features_fast(self, mol: Chem.Mol, morgan_params: dict,
                                               pharm_factory, mol2vec_model) -> Dict:
        """优化的单分子特征提取"""
        result = {}

        if mol is None:
            return {
                'morgan_fps': None,
                'pharm2d_fps': None,
                'mol2vec_vecs': None,
                'descriptors': {}
            }

        try:
            # 优先使用更快的特征
            if mol2vec_model and MOL2VEC_AVAILABLE:
                mol2vec_vec = self._compute_mol2vec_features_fast(mol, mol2vec_model)
                result['mol2vec_vecs'] = mol2vec_vec
                result['morgan_fps'] = None
            else:
                # 使用缓存的Morgan指纹计算
                morgan_fp = self._compute_morgan_fingerprint_fast(mol, morgan_params)
                result['morgan_fps'] = morgan_fp
                result['mol2vec_vecs'] = None

            # 药效团指纹（优化版）
            pharm_fp = self._compute_pharmacophore_fingerprint_fast(mol, pharm_factory)
            result['pharm2d_fps'] = pharm_fp

            # 分子描述符（只计算关键描述符）
            descriptors = self._compute_essential_descriptors_fast(mol)
            result['descriptors'] = descriptors

        except Exception as e:
            print(f"[WARNING] 快速特征提取失败: {e}")
            result = {
                'morgan_fps': None,
                'pharm2d_fps': None,
                'mol2vec_vecs': None,
                'descriptors': {}
            }

        return result

    def _extract_single_molecule_features(self, mol: Chem.Mol, morgan_params: dict,
                                          pharm_factory, mol2vec_model) -> Dict:
        """向后兼容方法"""
        return self._extract_single_molecule_features_fast(mol, morgan_params, pharm_factory, mol2vec_model)

    @lru_cache(maxsize=5000)
    def _compute_morgan_fingerprint_fast(self, mol: Chem.Mol, params_tuple) -> Optional[np.ndarray]:
        """缓存的快速Morgan指纹计算"""
        if mol is None:
            return None

        try:
            # 将params转换为可哈希的元组用于缓存
            if isinstance(params_tuple, dict):
                radius = params_tuple["morgan_radius"]
                nbits = params_tuple["morgan_nbits"]
            else:
                radius, nbits = params_tuple

            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
            return np.array(fp, dtype=np.uint8)  # 使用更小的数据类型
        except Exception:
            return None

    def _compute_morgan_fingerprint(self, mol: Chem.Mol, params: dict) -> Optional[np.ndarray]:
        """向后兼容方法"""
        return self._compute_morgan_fingerprint_fast(mol, (params["morgan_radius"], params["morgan_nbits"]))

    def _compute_mol2vec_features_fast(self, mol: Chem.Mol, model: object) -> Optional[np.ndarray]:
        """优化的Mol2vec特征计算"""
        if mol is None or model is None:
            return None

        try:
            # 使用更小的radius以加速
            sentence = mol2alt_sentence(mol, radius=1)

            # 兼容不同版本的Gensim（优化版）
            try:
                vec = sentences2vec([sentence], model.wv, unseen='UNK')
            except AttributeError:
                vec = sentences2vec([sentence], model, unseen='UNK')

            return vec[0].astype(np.float32) if len(vec) > 0 else None  # 使用float32节省内存
        except Exception:
            return None

    def _compute_mol2vec_features(self, mol: Chem.Mol, model: object) -> Optional[np.ndarray]:
        """向后兼容方法"""
        return self._compute_mol2vec_features_fast(mol, model)

    def _compute_pharmacophore_fingerprint_fast(self, mol: Chem.Mol, factory) -> Optional[np.ndarray]:
        """优化的药效团指纹计算"""
        if mol is None:
            return None

        try:
            fp = Generate.Gen2DFingerprint(mol, factory)
            return np.array(fp, dtype=np.uint8)  # 使用更小的数据类型
        except Exception:
            return np.zeros(2048, dtype=np.uint8)  # 返回零指纹

    def _compute_pharmacophore_fingerprint(self, mol: Chem.Mol, factory) -> Optional[np.ndarray]:
        """向后兼容方法"""
        return self._compute_pharmacophore_fingerprint_fast(mol, factory)

    def _compute_essential_descriptors_fast(self, mol: Chem.Mol) -> Dict[str, float]:
        """计算关键分子描述符（优化：只计算必要的）"""
        if mol is None:
            return {}

        try:
            # 只计算最重要的描述符以提高速度
            desc = {
                # Lipinski五项（必须）
                'mw': float(Descriptors.MolWt(mol)),
                'logp': float(Crippen.MolLogP(mol)),
                'hbd': float(Lipinski.NumHDonors(mol)),
                'hba': float(Lipinski.NumHAcceptors(mol)),
                'rotb': float(Lipinski.NumRotatableBonds(mol)),

                # 关键拓扑描述符
                'tpsa': float(rdMolDescriptors.CalcTPSA(mol)),
                'heavy_atoms': float(mol.GetNumHeavyAtoms()),
                'aromatic_rings': float(rdMolDescriptors.CalcNumAromaticRings(mol)),

                # 药物相似性（可能耗时，可选）
                'qed': float(Descriptors.qed(mol)) if hasattr(Descriptors, 'qed') else 0.0,
            }

            return desc

        except Exception:
            return {}

    def _compute_comprehensive_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """向后兼容方法"""
        return self._compute_essential_descriptors_fast(mol)

    def _load_mol2vec_model(self) -> Optional[object]:
        """加载Mol2vec模型（带缓存）"""
        model_path = self.config.get("mol2vec", {}).get("model_path")
        if not model_path or not os.path.exists(model_path) or not MOL2VEC_AVAILABLE:
            return None

        try:
            model = word2vec.Word2Vec.load(model_path)
            print(f"[INFO] Mol2vec模型已加载: {model_path}")
            return model
        except Exception as e:
            print(f"[WARNING] Mol2vec模型加载失败: {e}")
            return None

    def _transfer_to_gpu_ultra_fast(self, features: Dict) -> Dict:
        """超高速GPU转移（优化内存管理）"""
        if not (GPU_AVAILABLE or TORCH_AVAILABLE):
            return features

        try:
            gpu_features = {}

            for key, value in features.items():
                if key == 'molecules':
                    gpu_features[key] = value
                    continue

                if not isinstance(value, list) or len(value) == 0:
                    gpu_features[key] = value
                    continue

                # 预处理和向量化
                if key == 'descriptors':
                    gpu_features[key] = self._process_descriptors_gpu(value)
                elif key in ['morgan_fps', 'pharm2d_fps']:
                    gpu_features[key] = self._process_fingerprints_gpu(value, key)
                elif key == 'mol2vec_vecs':
                    gpu_features[key] = self._process_mol2vec_gpu(value)
                else:
                    gpu_features[key] = value

            return gpu_features

        except Exception as e:
            print(f"[WARNING] 超高速GPU转移失败，使用CPU: {e}")
            return features

    def _transfer_to_gpu_optimized(self, features: Dict) -> Dict:
        """向后兼容方法"""
        return self._transfer_to_gpu_ultra_fast(features)

    def _process_descriptors_gpu(self, descriptors_list):
        """GPU描述符处理 - 修复返回类型问题"""
        try:
            # 快速矩阵化
            matrices = []
            all_keys = set()

            # 收集所有键
            for desc_dict in descriptors_list:
                if isinstance(desc_dict, dict):
                    all_keys.update(desc_dict.keys())

            all_keys = sorted(list(all_keys))

            # 构建矩阵
            for desc_dict in descriptors_list:
                if isinstance(desc_dict, dict) and desc_dict:
                    row = [float(desc_dict.get(key, 0.0)) for key in all_keys]
                else:
                    row = [0.0] * len(all_keys)
                matrices.append(row)

            if matrices:
                matrix = np.array(matrices, dtype=np.float32)

                if TORCH_AVAILABLE:
                    # 修复：直接返回numpy数组而不是张量，避免后续判断问题
                    tensor = torch.from_numpy(matrix.copy())
                    gpu_tensor = tensor.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
                    # 立即转回CPU numpy数组，避免张量传播
                    return gpu_tensor.cpu().numpy()
                elif GPU_AVAILABLE:
                    gpu_array = cp.asarray(matrix, dtype=cp.float32)
                    # 立即转回CPU numpy数组
                    return cp.asnumpy(gpu_array)
                else:
                    return matrix
            else:
                return descriptors_list

        except Exception as e:
            print(f"[WARNING] GPU描述符处理失败: {e}")
            return descriptors_list

    def _process_fingerprints_gpu(self, fps_list, fp_type):
        """GPU指纹处理 - 修复内存固定问题"""
        try:
            valid_fps = []
            fp_length = 2048  # 默认长度

            for fp in fps_list:
                if fp is not None and hasattr(fp, '__len__'):
                    fp_array = np.array(fp, dtype=np.float32)
                    if len(fp_array) > 0:
                        fp_length = len(fp_array)
                        break

            # 构建矩阵
            for fp in fps_list:
                if fp is not None and hasattr(fp, '__len__'):
                    fp_array = np.array(fp, dtype=np.float32)
                    valid_fps.append(fp_array)
                else:
                    valid_fps.append(np.zeros(fp_length, dtype=np.float32))

            if valid_fps:
                matrix = np.array(valid_fps, dtype=np.float32)

                if TORCH_AVAILABLE:
                    # 修复：使用安全的转换方式
                    tensor = torch.from_numpy(matrix.copy())
                    return tensor.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
                elif GPU_AVAILABLE:
                    return cp.asarray(matrix, dtype=cp.float32)
                else:
                    return matrix
            else:
                return fps_list

        except Exception as e:
            print(f"[WARNING] GPU指纹处理失败: {e}")
            return fps_list

    def _process_mol2vec_gpu(self, vecs_list):
        """GPU Mol2vec处理 - 修复内存固定问题"""
        try:
            valid_vecs = []
            vec_length = 300  # 默认长度

            for vec in vecs_list:
                if vec is not None and hasattr(vec, '__len__'):
                    vec_array = np.array(vec, dtype=np.float32)
                    if len(vec_array) > 0:
                        vec_length = len(vec_array)
                        break

            # 构建矩阵
            for vec in vecs_list:
                if vec is not None and hasattr(vec, '__len__'):
                    vec_array = np.array(vec, dtype=np.float32)
                    valid_vecs.append(vec_array)
                else:
                    valid_vecs.append(np.zeros(vec_length, dtype=np.float32))

            if valid_vecs:
                matrix = np.array(valid_vecs, dtype=np.float32)

                if TORCH_AVAILABLE:
                    # 修复：使用安全的转换方式
                    tensor = torch.from_numpy(matrix.copy())
                    return tensor.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
                elif GPU_AVAILABLE:
                    return cp.asarray(matrix, dtype=cp.float32)
                else:
                    return matrix
            else:
                return vecs_list

        except Exception as e:
            print(f"[WARNING] GPU Mol2vec处理失败: {e}")
            return vecs_list

    def compute_multi_dimensional_similarity_ultra_fast(self, ref_features: Dict, lib_features: Dict) -> Dict:
        """超高速多维相似性计算 - 完全修复版"""
        print("[INFO] 计算多维相似性（超高速版）...")
        start_time = time.time()

        n_lib = len(lib_features['molecules'])
        n_ref = len(ref_features['molecules'])

        # ========== 安全的调试输出 ==========
        print(f"\n[DEBUG] ========== 1D Morgan指纹诊断 ==========")
        print(f"[DEBUG] 参考分子数: {n_ref}")
        print(f"[DEBUG] 库分子数: {n_lib}")

        ref_morgan = ref_features.get('morgan_fps', [])
        lib_morgan = lib_features.get('morgan_fps', [])

        # 安全的长度检查函数
        def safe_len(obj):
            """安全地获取对象长度"""
            try:
                if obj is None:
                    return 0
                if TORCH_AVAILABLE and torch.is_tensor(obj):
                    return obj.shape[0] if len(obj.shape) > 0 else 0
                if hasattr(obj, '__len__'):
                    return len(obj)
                if hasattr(obj, 'shape'):
                    return obj.shape[0]
                return 0
            except:
                return 0

        print(f"[DEBUG] 参考Morgan指纹列表长度: {safe_len(ref_morgan)}")
        print(f"[DEBUG] 库Morgan指纹列表长度: {safe_len(lib_morgan)}")

        # 安全地检查第一个指纹
        def safe_check_first(fps, name):
            """安全地检查第一个指纹"""
            try:
                if fps is None:
                    print(f"[DEBUG] {name}为None")
                    return

                # PyTorch张量
                if TORCH_AVAILABLE and torch.is_tensor(fps):
                    if fps.numel() > 0:
                        print(f"[DEBUG] {name}类型: PyTorch Tensor, 形状: {fps.shape}")
                        return
                    else:
                        print(f"[DEBUG] {name}为空张量")
                        return

                # 列表/数组
                length = safe_len(fps)
                if length > 0:
                    print(f"[DEBUG] {name}类型: {type(fps)}, 长度: {length}")
                    # 尝试获取第一个元素的信息
                    try:
                        if hasattr(fps, '__getitem__'):
                            first = fps[0]
                            if first is not None:
                                print(f"[DEBUG] {name}第一个元素类型: {type(first)}")
                    except:
                        pass
                else:
                    print(f"[DEBUG] {name}为空")
            except Exception as e:
                print(f"[DEBUG] {name}检查失败: {e}")

        safe_check_first(ref_morgan, "参考Morgan指纹")
        safe_check_first(lib_morgan, "库Morgan指纹")

        print(f"[DEBUG] ==========================================\n")
        # ==========================================

        # 异步计算不同维度的相似性
        similarity_futures = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            # 1D相似性
            if self._has_valid_features(ref_features, 'mol2vec_vecs'):
                print("[DEBUG] 使用mol2vec计算1D相似性")
                similarity_futures['1d'] = executor.submit(
                    self._compute_mol2vec_similarity_ultra_fast,
                    ref_features['mol2vec_vecs'],
                    lib_features['mol2vec_vecs']
                )
            else:
                print("[DEBUG] 使用Morgan指纹计算1D相似性")
                similarity_futures['1d'] = executor.submit(
                    self._compute_fingerprint_similarity_ultra_fast,
                    ref_features['morgan_fps'],
                    lib_features['morgan_fps']
                )

            # 2D相似性
            similarity_futures['2d'] = executor.submit(
                self._compute_fingerprint_similarity_ultra_fast,
                ref_features['pharm2d_fps'],
                lib_features['pharm2d_fps']
            )

            # 3D相似性（如果启用）
            if self.config["similarity"]["w_3d"] > 0:
                similarity_futures['3d'] = executor.submit(
                    self._compute_3d_similarity_fast,
                    ref_features,
                    lib_features
                )
            else:
                similarity_futures['3d'] = executor.submit(
                    lambda: np.zeros((n_lib, n_ref), dtype=np.float32)
                )

        # 收集结果
        similarities = {}
        for dim, future in similarity_futures.items():
            try:
                similarities[f'{dim}_similarity'] = future.result()
            except Exception as e:
                print(f"[WARNING] {dim}D相似性计算失败: {e}")
                import traceback
                traceback.print_exc()
                similarities[f'{dim}_similarity'] = np.zeros((n_lib, n_ref), dtype=np.float32)

        self.perf_stats['similarity_computation_time'] = time.time() - start_time
        return similarities

    # def compute_multi_dimensional_similarity_ultra_fast(self, ref_features: Dict, lib_features: Dict) -> Dict:
    #     """极简版多维相似性计算 - 只使用Morgan指纹"""
    #     print("[INFO] 计算相似性（极简版 - 仅Morgan指纹）...")
    #     start_time = time.time()
    #
    #     n_lib = len(lib_features['molecules'])
    #     n_ref = len(ref_features['molecules'])
    #
    #     print(f"[DEBUG] 库分子: {n_lib}, 参考分子: {n_ref}")
    #
    #     # 只计算Morgan指纹相似性
    #     try:
    #         morgan_similarity = self._compute_fingerprint_similarity_minimal(
    #             ref_features['morgan_fps'],
    #             lib_features['morgan_fps']
    #         )
    #
    #         # 其他相似性设为零
    #         zero_similarity = np.zeros((n_lib, n_ref), dtype=np.float32)
    #
    #         similarities = {
    #             '1d_similarity': morgan_similarity,  # 使用Morgan作为1D
    #             '2d_similarity': zero_similarity,  # 跳过2D
    #             '3d_similarity': zero_similarity  # 跳过3D
    #         }
    #
    #         elapsed_time = time.time() - start_time
    #         print(f"[INFO] 相似性计算完成 - 耗时: {elapsed_time:.2f}秒")
    #
    #         return similarities
    #
    #     except Exception as e:
    #         print(f"[ERROR] 相似性计算失败: {e}")
    #         # 返回全零矩阵
    #         zero_similarity = np.zeros((n_lib, n_ref), dtype=np.float32)
    #         return {
    #             '1d_similarity': zero_similarity,
    #             '2d_similarity': zero_similarity,
    #             '3d_similarity': zero_similarity
    #         }

    def _compute_fingerprint_similarity_minimal(self, ref_fps: List, lib_fps: List) -> np.ndarray:
        """极简版指纹相似性计算 - CPU向量化"""
        print("[DEBUG] 开始指纹相似性计算...")

        try:
            # 提取有效指纹
            valid_ref_fps = []
            valid_lib_fps = []

            for fp in ref_fps:
                if fp is not None:
                    valid_ref_fps.append(np.array(fp, dtype=np.float32))
                else:
                    valid_ref_fps.append(np.zeros(1024, dtype=np.float32))

            for fp in lib_fps:
                if fp is not None:
                    valid_lib_fps.append(np.array(fp, dtype=np.float32))
                else:
                    valid_lib_fps.append(np.zeros(1024, dtype=np.float32))

            if not valid_ref_fps or not valid_lib_fps:
                print("[WARNING] 没有有效指纹，返回零矩阵")
                return np.zeros((len(lib_fps), len(ref_fps)), dtype=np.float32)

            # 转换为numpy数组
            ref_matrix = np.array(valid_ref_fps, dtype=np.float32)
            lib_matrix = np.array(valid_lib_fps, dtype=np.float32)

            print(f"[DEBUG] 参考矩阵: {ref_matrix.shape}, 库矩阵: {lib_matrix.shape}")

            # 简化的Tanimoto相似性计算（CPU向量化）
            print("[DEBUG] 计算Tanimoto相似性...")

            # 使用批处理避免内存问题
            batch_size = 1000
            n_lib = lib_matrix.shape[0]
            n_ref = ref_matrix.shape[0]

            similarity = np.zeros((n_lib, n_ref), dtype=np.float32)

            for i in range(0, n_lib, batch_size):
                end_i = min(i + batch_size, n_lib)
                lib_batch = lib_matrix[i:end_i]

                if i % (batch_size * 5) == 0:
                    progress = (i / n_lib) * 100
                    print(f"[DEBUG] 相似性计算进度: {progress:.1f}%")

                # 计算交集
                intersection = np.dot(lib_batch, ref_matrix.T)

                # 计算并集
                lib_sum = np.sum(lib_batch, axis=1, keepdims=True)
                ref_sum = np.sum(ref_matrix, axis=1, keepdims=True)
                union = lib_sum + ref_sum.T - intersection

                # Tanimoto系数
                batch_similarity = intersection / (union + 1e-8)
                similarity[i:end_i] = batch_similarity

            print("[DEBUG] 指纹相似性计算完成")
            return similarity

        except Exception as e:
            print(f"[ERROR] 指纹相似性计算失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回零矩阵
            return np.zeros((len(lib_fps), len(ref_fps)), dtype=np.float32)

    def compute_multi_dimensional_similarity(self, ref_features: Dict, lib_features: Dict) -> Dict:
        """向后兼容方法"""
        return self.compute_multi_dimensional_similarity_ultra_fast(ref_features, lib_features)

    def _has_valid_features(self, features: Dict, feature_type: str) -> bool:
        """检查是否有有效特征"""
        if feature_type not in features:
            return False

        feature_list = features[feature_type]
        if not isinstance(feature_list, list):
            return False

        return any(f is not None for f in feature_list)

    def _compute_mol2vec_similarity_ultra_fast(self, ref_vecs, lib_vecs) -> np.ndarray:
        """超高速Mol2vec相似性计算"""
        # PyTorch GPU计算（最优先）
        if TORCH_AVAILABLE and torch.is_tensor(ref_vecs) and torch.is_tensor(lib_vecs):
            try:
                # 使用PyTorch的高度优化矩阵运算
                with torch.no_grad():
                    # 计算余弦相似性：(A @ B.T) / (||A|| * ||B||)
                    ref_norm = F.normalize(ref_vecs, p=2, dim=1)
                    lib_norm = F.normalize(lib_vecs, p=2, dim=1)

                    # 批量矩阵乘法
                    similarity = torch.mm(lib_norm, ref_norm.T)

                    return similarity.cpu().numpy()

            except Exception as e:
                print(f"[WARNING] PyTorch超高速计算失败: {e}")

        # CuPy GPU计算（备选）
        if GPU_AVAILABLE:
            try:
                if not (hasattr(ref_vecs, '__array__') and hasattr(lib_vecs, '__array__')):
                    # 转换为numpy数组
                    ref_array = np.array([v for v in ref_vecs if v is not None], dtype=np.float32)
                    lib_array = np.array([v for v in lib_vecs if v is not None], dtype=np.float32)
                else:
                    ref_array = np.asarray(ref_vecs)
                    lib_array = np.asarray(lib_vecs)

                ref_gpu = cp.asarray(ref_array, dtype=cp.float32)
                lib_gpu = cp.asarray(lib_array, dtype=cp.float32)

                # 余弦相似性计算
                ref_norm = cp.linalg.norm(ref_gpu, axis=1, keepdims=True)
                lib_norm = cp.linalg.norm(lib_gpu, axis=1, keepdims=True)

                ref_normalized = ref_gpu / (ref_norm + 1e-8)
                lib_normalized = lib_gpu / (lib_norm + 1e-8)

                similarity = cp.dot(lib_normalized, ref_normalized.T)
                return cp.asnumpy(similarity)

            except Exception as e:
                print(f"[WARNING] CuPy超高速计算失败: {e}")

        # CPU向量化计算（回退）
        return self._compute_mol2vec_similarity_cpu_vectorized(ref_vecs, lib_vecs)

    def _compute_mol2vec_similarity_gpu(self, ref_vecs, lib_vecs) -> np.ndarray:
        """向后兼容方法"""
        return self._compute_mol2vec_similarity_ultra_fast(ref_vecs, lib_vecs)

    def _compute_mol2vec_similarity_cpu_vectorized(self, ref_vecs, lib_vecs) -> np.ndarray:
        """CPU向量化Mol2vec相似性计算"""
        try:
            # 提取有效向量
            valid_ref = [v for v in ref_vecs if v is not None]
            valid_lib = [v for v in lib_vecs if v is not None]

            if not valid_ref or not valid_lib:
                return np.zeros((len(lib_vecs), len(ref_vecs)), dtype=np.float32)

            ref_array = np.array(valid_ref, dtype=np.float32)
            lib_array = np.array(valid_lib, dtype=np.float32)

            # 使用sklearn的优化余弦相似性
            similarity = cosine_similarity(lib_array, ref_array).astype(np.float32)

            return similarity

        except Exception as e:
            print(f"[WARNING] CPU向量化计算失败: {e}")
            return np.zeros((len(lib_vecs), len(ref_vecs)), dtype=np.float32)

    def _compute_fingerprint_similarity_ultra_fast(self, ref_fps, lib_fps) -> np.ndarray:
        """
        超高速指纹相似性计算 - 完全修复版
        """
        print(f"[DEBUG] === 开始指纹相似性计算 ===")
        print(f"[DEBUG] 参考指纹数量: {len(ref_fps) if hasattr(ref_fps, '__len__') else 'unknown'}")
        print(f"[DEBUG] 库指纹数量: {len(lib_fps) if hasattr(lib_fps, '__len__') else 'unknown'}")

        # ========== 强制转换为numpy数组 ==========
        try:
            # 处理PyTorch张量
            if TORCH_AVAILABLE and torch.is_tensor(ref_fps):
                print("[DEBUG] 检测到PyTorch张量，转换为numpy")
                ref_fps = ref_fps.cpu().numpy()
                lib_fps = lib_fps.cpu().numpy()

            # 处理CuPy数组
            elif GPU_AVAILABLE:
                try:
                    import cupy as cp
                    if isinstance(ref_fps, cp.ndarray):
                        print("[DEBUG] 检测到CuPy数组，转换为numpy")
                        ref_fps = cp.asnumpy(ref_fps)
                        lib_fps = cp.asnumpy(lib_fps)
                except:
                    pass

            # 处理列表
            if isinstance(ref_fps, list):
                print("[DEBUG] 检测到列表，转换为numpy数组")
                # 过滤None值
                valid_ref = []
                for fp in ref_fps:
                    if fp is not None:
                        if hasattr(fp, '__array__'):
                            valid_ref.append(np.array(fp, dtype=np.float32))
                        else:
                            valid_ref.append(fp)

                valid_lib = []
                for fp in lib_fps:
                    if fp is not None:
                        if hasattr(fp, '__array__'):
                            valid_lib.append(np.array(fp, dtype=np.float32))
                        else:
                            valid_lib.append(fp)

                if not valid_ref or not valid_lib:
                    print("[ERROR] 过滤后没有有效指纹！")
                    return np.zeros((len(lib_fps), len(ref_fps)), dtype=np.float32)

                ref_array = np.array(valid_ref, dtype=np.float32)
                lib_array = np.array(valid_lib, dtype=np.float32)
            else:
                # 已经是数组
                ref_array = np.asarray(ref_fps, dtype=np.float32)
                lib_array = np.asarray(lib_fps, dtype=np.float32)

            print(f"[DEBUG] 转换后参考数组形状: {ref_array.shape}")
            print(f"[DEBUG] 转换后库数组形状: {lib_array.shape}")
            print(f"[DEBUG] 参考数组数据类型: {ref_array.dtype}")
            print(f"[DEBUG] 库数组数据类型: {lib_array.dtype}")

            # 检查数组内容
            print(f"[DEBUG] 参考数组样本（前5个的前10位）:")
            for i in range(min(5, len(ref_array))):
                print(f"  Ref {i}: {ref_array[i][:10]}")

            print(f"[DEBUG] 库数组样本（前5个的前10位）:")
            for i in range(min(5, len(lib_array))):
                print(f"  Lib {i}: {lib_array[i][:10]}")

            # ========== 计算Tanimoto相似性 ==========
            print("[DEBUG] 开始计算Tanimoto相似性...")

            # 方法1：逐对计算（最可靠）
            n_lib = lib_array.shape[0]
            n_ref = ref_array.shape[0]
            similarity_matrix = np.zeros((n_lib, n_ref), dtype=np.float32)

            for i in range(n_lib):
                for j in range(n_ref):
                    # Tanimoto系数 = 交集 / 并集
                    intersection = np.sum(np.minimum(lib_array[i], ref_array[j]))
                    union = np.sum(np.maximum(lib_array[i], ref_array[j]))

                    if union > 0:
                        similarity_matrix[i, j] = intersection / union
                    else:
                        similarity_matrix[i, j] = 0.0

            print(f"[DEBUG] 相似性矩阵形状: {similarity_matrix.shape}")
            print(f"[DEBUG] 相似性范围: {similarity_matrix.min():.6f} - {similarity_matrix.max():.6f}")
            print(f"[DEBUG] 相似性平均值: {similarity_matrix.mean():.6f}")
            print(f"[DEBUG] 相似性中位数: {np.median(similarity_matrix):.6f}")
            print(f"[DEBUG] 非零相似性比例: {(similarity_matrix > 0).sum() / similarity_matrix.size * 100:.2f}%")

            # 打印相似性矩阵的一部分
            print(f"[DEBUG] 相似性矩阵样本（前5x5）:")
            print(similarity_matrix[:5, :5])

            # ========== 验证计算 ==========
            # 自检：第一个库分子与第一个参考分子的相似性
            test_sim = similarity_matrix[0, 0]
            print(f"[DEBUG] 验证：Lib[0] vs Ref[0] 相似性 = {test_sim:.6f}")

            if similarity_matrix.max() == 0:
                print("[ERROR]  所有相似性都是0！可能指纹格式有问题")
                print(f"[DEBUG] 检查指纹和: Ref[0] sum={ref_array[0].sum()}, Lib[0] sum={lib_array[0].sum()}")
            else:
                print(f"[INFO] 相似性计算成功，最大值={similarity_matrix.max():.6f}")

            return similarity_matrix

        except Exception as e:
            print(f"[ERROR] 指纹相似性计算失败: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((len(lib_fps), len(ref_fps)), dtype=np.float32)

    def _compute_fingerprint_similarity_gpu(self, ref_fps, lib_fps) -> np.ndarray:
        """向后兼容方法"""
        return self._compute_fingerprint_similarity_ultra_fast(ref_fps, lib_fps)

    def _compute_fingerprint_similarity_cpu_vectorized(self, ref_fps, lib_fps) -> np.ndarray:
        """CPU向量化指纹相似性计算"""
        try:
            # 提取有效指纹
            valid_ref = [fp for fp in ref_fps if fp is not None]
            valid_lib = [fp for fp in lib_fps if fp is not None]

            if not valid_ref or not valid_lib:
                return np.zeros((len(lib_fps), len(ref_fps)), dtype=np.float32)

            ref_array = np.array(valid_ref, dtype=np.float32)
            lib_array = np.array(valid_lib, dtype=np.float32)

            # 向量化Tanimoto计算
            intersection = np.dot(lib_array, ref_array.T)
            lib_sum = np.sum(lib_array, axis=1, keepdims=True)
            ref_sum = np.sum(ref_array, axis=1, keepdims=True)
            union = lib_sum + ref_sum.T - intersection

            tanimoto = intersection / (union + 1e-8)
            return tanimoto.astype(np.float32)

        except Exception as e:
            print(f"[WARNING] CPU向量化指纹计算失败: {e}")
            return np.zeros((len(lib_fps), len(ref_fps)), dtype=np.float32)

    def _compute_3d_similarity_fast(self, ref_features: Dict, lib_features: Dict) -> np.ndarray:
        """快速3D形状相似性计算"""
        if not SHAPE3D_AVAILABLE:
            return np.zeros((len(lib_features['molecules']), len(ref_features['molecules'])), dtype=np.float32)

        # 使用采样策略减少计算量
        max_conformers = 3  # 减少构象数量
        sample_ratio = 0.1  # 采样比例

        print("[INFO] 计算3D形状相似性（快速采样版）...")

        ref_mols = ref_features['molecules']
        lib_mols = lib_features['molecules']

        # 采样策略：只计算部分分子对的3D相似性
        n_lib, n_ref = len(lib_mols), len(ref_mols)
        sample_size = max(10, int(min(n_lib, n_ref) * sample_ratio))

        # 随机采样
        import random
        lib_indices = random.sample(range(n_lib), min(sample_size, n_lib))
        ref_indices = random.sample(range(n_ref), min(sample_size, n_ref))

        # 创建相似性矩阵
        similarity = np.zeros((n_lib, n_ref), dtype=np.float32)

        # 只计算采样的分子对
        for i in lib_indices:
            for j in ref_indices:
                try:
                    sim = self._compute_shape_similarity_fast(lib_mols[i], ref_mols[j], max_conformers)
                    similarity[i, j] = sim
                except:
                    similarity[i, j] = 0.0

        return similarity

    def _compute_3d_similarity(self, ref_features: Dict, lib_features: Dict) -> np.ndarray:
        """向后兼容方法"""
        return self._compute_3d_similarity_fast(ref_features, lib_features)

    def _compute_shape_similarity_fast(self, mol1: Chem.Mol, mol2: Chem.Mol, max_conformers: int = 3) -> float:
        """快速形状相似性计算（减少构象数）"""
        if not mol1 or not mol2:
            return 0.0

        try:
            # 快速构象生成
            mol1_h = Chem.AddHs(mol1)
            mol2_h = Chem.AddHs(mol2)

            # 生成少量构象
            ps = AllChem.ETKDGv3()
            ps.numThreads = 1
            ps.randomSeed = 42
            ps.maxAttempts = 10  # 减少尝试次数

            conf_ids_1 = AllChem.EmbedMultipleConfs(mol1_h, numConfs=max_conformers, params=ps)
            conf_ids_2 = AllChem.EmbedMultipleConfs(mol2_h, numConfs=max_conformers, params=ps)

            if not conf_ids_1 or not conf_ids_2:
                return 0.0

            # 计算最大相似性（只取前几个构象）
            max_sim = 0.0
            for conf1 in conf_ids_1[:2]:  # 只取前2个构象
                for conf2 in conf_ids_2[:2]:
                    try:
                        dist = rdShapeHelpers.ShapeTanimotoDist(mol1_h, mol2_h, conf1, conf2)
                        sim = 1.0 - dist
                        max_sim = max(max_sim, sim)
                    except:
                        continue

            return max_sim

        except Exception:
            return 0.0

    # def aggregate_similarities_ultra_fast(self, similarities: Dict) -> Dict:
    #     """超高速相似性聚合"""
    #     print("[INFO] 聚合多维相似性（超高速版）...")
    #
    #     sim_1d = similarities['1d_similarity']
    #     sim_2d = similarities['2d_similarity']
    #     sim_3d = similarities['3d_similarity']
    #
    #     # GPU加速的向量化最大值计算
    #     if TORCH_AVAILABLE and all(torch.is_tensor(s) for s in [sim_1d, sim_2d, sim_3d]):
    #         try:
    #             with torch.no_grad():
    #                 max_sim_1d = torch.max(sim_1d, dim=1)[0]
    #                 max_sim_2d = torch.max(sim_2d, dim=1)[0]
    #                 max_sim_3d = torch.max(sim_3d, dim=1)[0]
    #
    #                 # 归一化
    #                 max_sim_1d = self._minmax_normalize_torch(max_sim_1d)
    #                 max_sim_2d = self._minmax_normalize_torch(max_sim_2d)
    #                 max_sim_3d = self._minmax_normalize_torch(max_sim_3d)
    #
    #                 # 加权组合
    #                 w1 = self.config["similarity"]["w_1d"]
    #                 w2 = self.config["similarity"]["w_2d"]
    #                 w3 = self.config["similarity"]["w_3d"]
    #
    #                 combined_scores = w1 * max_sim_1d + w2 * max_sim_2d + w3 * max_sim_3d
    #
    #                 return {
    #                     'combined_scores': combined_scores.cpu().numpy(),
    #                     'individual_scores': {
    #                         '1d_max': max_sim_1d.cpu().numpy(),
    #                         '2d_max': max_sim_2d.cpu().numpy(),
    #                         '3d_max': max_sim_3d.cpu().numpy(),
    #                         '1d_norm': max_sim_1d.cpu().numpy(),
    #                         '2d_norm': max_sim_2d.cpu().numpy(),
    #                         '3d_norm': max_sim_3d.cpu().numpy()
    #                     }
    #                 }
    #
    #         except Exception as e:
    #             print(f"[WARNING] PyTorch超高速聚合失败: {e}")
    #
    #     # CuPy GPU回退
    #     if GPU_AVAILABLE:
    #         try:
    #             sim_1d_gpu = cp.asarray(sim_1d)
    #             sim_2d_gpu = cp.asarray(sim_2d)
    #             sim_3d_gpu = cp.asarray(sim_3d)
    #
    #             max_sim_1d = cp.max(sim_1d_gpu, axis=1)
    #             max_sim_2d = cp.max(sim_2d_gpu, axis=1)
    #             max_sim_3d = cp.max(sim_3d_gpu, axis=1)
    #
    #             # 归一化
    #             max_sim_1d = self._minmax_normalize_cupy(max_sim_1d)
    #             max_sim_2d = self._minmax_normalize_cupy(max_sim_2d)
    #             max_sim_3d = self._minmax_normalize_cupy(max_sim_3d)
    #
    #             # 加权组合
    #             w1 = self.config["similarity"]["w_1d"]
    #             w2 = self.config["similarity"]["w_2d"]
    #             w3 = self.config["similarity"]["w_3d"]
    #
    #             combined_scores = w1 * max_sim_1d + w2 * max_sim_2d + w3 * max_sim_3d
    #
    #             return {
    #                 'combined_scores': cp.asnumpy(combined_scores),
    #                 'individual_scores': {
    #                     '1d_max': cp.asnumpy(max_sim_1d),
    #                     '2d_max': cp.asnumpy(max_sim_2d),
    #                     '3d_max': cp.asnumpy(max_sim_3d),
    #                     '1d_norm': cp.asnumpy(max_sim_1d),
    #                     '2d_norm': cp.asnumpy(max_sim_2d),
    #                     '3d_norm': cp.asnumpy(max_sim_3d)
    #                 }
    #             }
    #
    #         except Exception as e:
    #             print(f"[WARNING] CuPy超高速聚合失败: {e}")
    #
    #     # CPU向量化回退
    #     max_sim_1d = np.max(sim_1d, axis=1)
    #     max_sim_2d = np.max(sim_2d, axis=1)
    #     max_sim_3d = np.max(sim_3d, axis=1)
    #
    #     # 归一化
    #     max_sim_1d = self._minmax_normalize(max_sim_1d)
    #     max_sim_2d = self._minmax_normalize(max_sim_2d)
    #     max_sim_3d = self._minmax_normalize(max_sim_3d)
    #
    #     # 加权组合
    #     w1 = self.config["similarity"]["w_1d"]
    #     w2 = self.config["similarity"]["w_2d"]
    #     w3 = self.config["similarity"]["w_3d"]
    #
    #     combined_scores = w1 * max_sim_1d + w2 * max_sim_2d + w3 * max_sim_3d
    #
    #     return {
    #         'combined_scores': combined_scores,
    #         'individual_scores': {
    #             '1d_max': max_sim_1d,
    #             '2d_max': max_sim_2d,
    #             '3d_max': max_sim_3d,
    #             '1d_norm': max_sim_1d,
    #             '2d_norm': max_sim_2d,
    #             '3d_norm': max_sim_3d
    #         }
    #     }

    def aggregate_similarities_ultra_fast(self, similarities: Dict) -> Dict:
        """
        相似性聚合 - 完全不归一化版本
        """
        print("[INFO] 聚合多维相似性（无归一化版本）...")

        sim_1d = similarities['1d_similarity']
        sim_2d = similarities['2d_similarity']
        sim_3d = similarities['3d_similarity']

        # 转换为numpy
        if torch.is_tensor(sim_1d):
            sim_1d = sim_1d.cpu().numpy()
            sim_2d = sim_2d.cpu().numpy()
            sim_3d = sim_3d.cpu().numpy()
        elif hasattr(sim_1d, '__array__'):
            sim_1d = np.asarray(sim_1d)
            sim_2d = np.asarray(sim_2d)
            sim_3d = np.asarray(sim_3d)

        print(f"[DEBUG] 输入相似性矩阵形状:")
        print(f"  - 1D: {sim_1d.shape}")
        print(f"  - 2D: {sim_2d.shape}")
        print(f"  - 3D: {sim_3d.shape}")

        print(f"[DEBUG] 输入相似性范围:")
        print(f"  - 1D: {sim_1d.min():.6f} - {sim_1d.max():.6f}")
        print(f"  - 2D: {sim_2d.min():.6f} - {sim_2d.max():.6f}")
        print(f"  - 3D: {sim_3d.min():.6f} - {sim_3d.max():.6f}")

        # 取最大值（每个库分子与所有参考分子的最大相似性）
        max_sim_1d = np.max(sim_1d, axis=1)
        max_sim_2d = np.max(sim_2d, axis=1)
        max_sim_3d = np.max(sim_3d, axis=1)

        print(f"[DEBUG] 最大相似性范围:")
        print(f"  - 1D: {max_sim_1d.min():.6f} - {max_sim_1d.max():.6f}")
        print(f"  - 2D: {max_sim_2d.min():.6f} - {max_sim_2d.max():.6f}")
        print(f"  - 3D: {max_sim_3d.min():.6f} - {max_sim_3d.max():.6f}")

        # ========== 完全不归一化，直接使用原始值 ==========
        w1 = self.config["similarity"]["w_1d"]
        w2 = self.config["similarity"]["w_2d"]
        w3 = self.config["similarity"]["w_3d"]

        print(f"[DEBUG] 权重: w1={w1}, w2={w2}, w3={w3}")

        # 直接加权，不做任何归一化
        combined_scores = w1 * max_sim_1d + w2 * max_sim_2d + w3 * max_sim_3d

        print(f"[DEBUG] 组合得分范围: {combined_scores.min():.6f} - {combined_scores.max():.6f}")
        print(f"[DEBUG] 组合得分平均: {combined_scores.mean():.6f}")
        print(f"[DEBUG] 组合得分中位数: {np.median(combined_scores):.6f}")

        # ========== 详细得分分解（前10个化合物）==========
        print(f"[DEBUG] 前10个化合物得分分解:")
        for i in range(min(10, len(combined_scores))):
            print(f"  [{i}] 1D={max_sim_1d[i]:.4f} * {w1} + "
                  f"2D={max_sim_2d[i]:.4f} * {w2} + "
                  f"3D={max_sim_3d[i]:.4f} * {w3} = "
                  f"{combined_scores[i]:.4f}")

        return {
            'combined_scores': combined_scores,
            'individual_scores': {
                '1d_max': max_sim_1d,
                '2d_max': max_sim_2d,
                '3d_max': max_sim_3d,
                '1d_norm': max_sim_1d,  # 返回原始值，不归一化
                '2d_norm': max_sim_2d,
                '3d_norm': max_sim_3d
            }
        }

    # def aggregate_similarities_ultra_fast(self, similarities: Dict) -> Dict:
    #     """极简版相似性聚合"""
    #     print("[INFO] 聚合相似性（极简版）...")
    #
    #     sim_1d = similarities['1d_similarity']  # Morgan指纹相似性
    #     sim_2d = similarities['2d_similarity']  # 全零
    #     sim_3d = similarities['3d_similarity']  # 全零
    #
    #     # 简单取最大值（只使用Morgan指纹）
    #     max_sim_1d = np.max(sim_1d, axis=1)
    #     max_sim_2d = np.zeros_like(max_sim_1d)
    #     max_sim_3d = np.zeros_like(max_sim_1d)
    #
    #     # 简单归一化
    #     if np.max(max_sim_1d) > 0:
    #         max_sim_1d_norm = max_sim_1d / np.max(max_sim_1d)
    #     else:
    #         max_sim_1d_norm = max_sim_1d
    #
    #     # 只使用1D权重（Morgan指纹）
    #     w1 = 1.0  # 全部权重给Morgan指纹
    #     combined_scores = w1 * max_sim_1d_norm
    #
    #     return {
    #         'combined_scores': combined_scores,
    #         'individual_scores': {
    #             '1d_max': max_sim_1d,
    #             '2d_max': max_sim_2d,
    #             '3d_max': max_sim_3d,
    #             '1d_norm': max_sim_1d_norm,
    #             '2d_norm': max_sim_2d,
    #             '3d_norm': max_sim_3d
    #         }
    #     }


    def aggregate_similarities(self, similarities: Dict) -> Dict:
        """向后兼容方法"""
        return self.aggregate_similarities_ultra_fast(similarities)

    def _minmax_normalize_torch(self, tensor: torch.Tensor) -> torch.Tensor:
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        if max_val - min_val < 1e-12:
            return tensor.clone()  # ✅ 修复
        return (tensor - min_val) / (max_val - min_val)

    def _minmax_normalize_cupy(self, array: cp.ndarray) -> cp.ndarray:
        min_val = cp.min(array)
        max_val = cp.max(array)
        if max_val - min_val < 1e-12:
            return array.copy()  # ✅ 修复
        return (array - min_val) / (max_val - min_val)

    def _minmax_normalize(self, arr: np.ndarray) -> np.ndarray:
        if len(arr) == 0:
            return arr
        min_val, max_val = np.min(arr), np.max(arr)
        if max_val - min_val < 1e-12:
            return arr.copy()  # ✅ 修复
        return (arr - min_val) / (max_val - min_val)

    def apply_ai_enhanced_filtering_ultra_fast(self, lib_df: pd.DataFrame, scores: Dict,
                                               lib_features: Dict) -> pd.DataFrame:
        """超高速AI增强过滤 - 修复配置缺失问题"""
        print("[INFO] 应用AI增强过滤（超高速版）...")
        start_time = time.time()

        # 创建结果数据框（优化内存使用）
        results = lib_df[['id', 'name', 'smiles', 'canonical_smiles']].copy()
        results['combined_score'] = scores['combined_scores']

        # 添加个别相似性得分（向量化操作）
        for key, values in scores['individual_scores'].items():
            results[key] = values

        # 快速描述符处理 - 使用修复后的方法
        try:
            descriptors_data = lib_features.get('descriptors', [])
            if descriptors_data is not None:
                descriptors_df = self._process_descriptors_dataframe(descriptors_data)
                if not descriptors_df.empty:
                    for col in descriptors_df.columns:
                        results[col] = descriptors_df[col]
        except Exception as e:
            print(f"[WARNING] 描述符处理失败: {e}")

        # 并行过滤任务（修复配置访问）
        filtering_results = {}

        with ThreadPoolExecutor(max_workers=min(4, self.cpu_pool.max_workers)) as executor:
            futures = {}

            # PAINS过滤任务
            if self.config.get("filters", {}).get("pains", False):
                futures['pains'] = executor.submit(
                    self._batch_pains_check,
                    lib_df['mol'].tolist()
                )

            # CNS过滤任务（修复：提供默认值）
            if self.config.get("filters", {}).get("cns_filter", False):
                # 提供默认的CNS规则
                default_cns_rules = {
                    "mw_min": 130,
                    "mw_max": 725,
                    "logp_min": -7,
                    "logp_max": 5.5,
                    "hbd_max": 7,
                    "hba_max": 12,
                    "tpsa_max": 200,
                    "rotb_max": 11
                }

                cns_rules = self.config.get("filters", {}).get("cns_rules", default_cns_rules)

                futures['cns'] = executor.submit(
                    self._batch_cns_check,
                    results,
                    cns_rules
                )

            # AI聚类任务（使用MiniBatch）
            if SKLEARN_AVAILABLE and self.config.get("ai_model", {}).get("use_clustering", False):
                futures['cluster'] = executor.submit(
                    self._perform_clustering_fast,
                    descriptors_df if 'descriptors_df' in locals() and not descriptors_df.empty else pd.DataFrame()
                )

            # 异常值检测任务（采样）
            if SKLEARN_AVAILABLE and self.config.get("ai_model", {}).get("use_outlier_filter", False):
                futures['outlier'] = executor.submit(
                    self._detect_outliers_fast,
                    descriptors_df if 'descriptors_df' in locals() and not descriptors_df.empty else pd.DataFrame()
                )

            # 收集结果
            for task_name, future in futures.items():
                try:
                    filtering_results[task_name] = future.result()
                except Exception as e:
                    print(f"[WARNING] {task_name}任务失败: {e}")
                    filtering_results[task_name] = None

        # 应用过滤结果
        results['is_pains'] = filtering_results.get('pains', [False] * len(results))
        results['cns_compliant'] = filtering_results.get('cns', [True] * len(results))

        if filtering_results.get('cluster') is not None:
            results['cluster'] = filtering_results['cluster']
        if filtering_results.get('outlier') is not None:
            results['is_outlier'] = filtering_results['outlier'] == -1

        # 快速排序（使用numpy）
        sort_indices = np.argsort(-results['combined_score'].values)
        results = results.iloc[sort_indices].reset_index(drop=True)

        self.perf_stats['filtering_time'] = time.time() - start_time
        return results

    # def apply_ai_enhanced_filtering_ultra_fast(self, lib_df: pd.DataFrame, scores: Dict,
    #                                            lib_features: Dict) -> pd.DataFrame:
    #     """极简版AI增强过滤"""
    #     print("[INFO] 应用过滤（极简版）...")
    #     start_time = time.time()
    #
    #     # 创建结果数据框
    #     results = lib_df[['id', 'name', 'smiles', 'canonical_smiles']].copy()
    #     results['combined_score'] = scores['combined_scores']
    #
    #     # 添加个别相似性得分
    #     for key, values in scores['individual_scores'].items():
    #         results[key] = values
    #
    #     # 添加基本描述符
    #     descriptors_list = lib_features['descriptors']
    #     for i, desc_dict in enumerate(descriptors_list):
    #         for key, value in desc_dict.items():
    #             if key not in results.columns:
    #                 results[key] = 0.0
    #             results.loc[i, key] = value
    #
    #     # 简化过滤（跳过PAINS和复杂AI功能）
    #     results['is_pains'] = False  # 暂时跳过PAINS检查
    #     results['cns_compliant'] = True  # 暂时跳过CNS检查
    #
    #     # 快速排序
    #     results = results.sort_values('combined_score', ascending=False).reset_index(drop=True)
    #
    #     elapsed_time = time.time() - start_time
    #     print(f"[INFO] 过滤完成 - 耗时: {elapsed_time:.2f}秒")
    #
    #     return results



    # def apply_ai_enhanced_filtering(self, lib_df: pd.DataFrame, scores: Dict, lib_features: Dict) -> pd.DataFrame:
    #     """向后兼容方法"""
    #     return self.apply_ai_enhanced_filtering_ultra_fast(lib_df, scores, lib_features)

    # def _process_descriptors_dataframe(self, descriptors_list: List[Dict]) -> pd.DataFrame:
    #     """快速处理描述符列表为DataFrame"""
    #     if not descriptors_list:
    #         return pd.DataFrame()
    #
    #     # 收集所有键
    #     all_keys = set()
    #     for desc in descriptors_list:
    #         if isinstance(desc, dict):
    #             all_keys.update(desc.keys())
    #
    #     all_keys = sorted(list(all_keys))
    #
    #     # 向量化构建矩阵
    #     data_matrix = np.zeros((len(descriptors_list), len(all_keys)), dtype=np.float32)
    #
    #     for i, desc in enumerate(descriptors_list):
    #         if isinstance(desc, dict):
    #             for j, key in enumerate(all_keys):
    #                 data_matrix[i, j] = float(desc.get(key, 0.0))
    #
    #     return pd.DataFrame(data_matrix, columns=all_keys)

    def _process_descriptors_dataframe(self, descriptors_list) -> pd.DataFrame:
        """快速处理描述符列表为DataFrame - 修复张量判断问题"""

        # 修复：安全的空值检查
        if descriptors_list is None:
            return pd.DataFrame()

        # 检查是否为 PyTorch 张量
        if TORCH_AVAILABLE and torch.is_tensor(descriptors_list):
            # 转换张量为 numpy 数组再转为 DataFrame
            if descriptors_list.numel() == 0:  # 空张量检查
                return pd.DataFrame()

            # 如果是GPU张量，先移到CPU
            if descriptors_list.is_cuda:
                descriptors_array = descriptors_list.cpu().numpy()
            else:
                descriptors_array = descriptors_list.numpy()

            # 生成列名
            n_features = descriptors_array.shape[1] if len(descriptors_array.shape) > 1 else descriptors_array.shape[0]
            columns = [f'desc_{i}' for i in range(n_features)]

            return pd.DataFrame(descriptors_array, columns=columns)

        # 检查是否为 CuPy 数组
        if GPU_AVAILABLE and hasattr(descriptors_list, '__array__'):
            try:
                import cupy as cp
                if isinstance(descriptors_list, cp.ndarray):
                    if descriptors_list.size == 0:
                        return pd.DataFrame()

                    descriptors_array = cp.asnumpy(descriptors_list)
                    n_features = descriptors_array.shape[1] if len(descriptors_array.shape) > 1 else \
                    descriptors_array.shape[0]
                    columns = [f'desc_{i}' for i in range(n_features)]

                    return pd.DataFrame(descriptors_array, columns=columns)
            except:
                pass

        # 检查是否为普通列表
        if isinstance(descriptors_list, list):
            if len(descriptors_list) == 0:
                return pd.DataFrame()

            # 收集所有键
            all_keys = set()
            for desc in descriptors_list:
                if isinstance(desc, dict):
                    all_keys.update(desc.keys())

            if not all_keys:
                return pd.DataFrame()

            all_keys = sorted(list(all_keys))

            # 向量化构建矩阵
            data_matrix = np.zeros((len(descriptors_list), len(all_keys)), dtype=np.float32)

            for i, desc in enumerate(descriptors_list):
                if isinstance(desc, dict):
                    for j, key in enumerate(all_keys):
                        data_matrix[i, j] = float(desc.get(key, 0.0))

            return pd.DataFrame(data_matrix, columns=all_keys)

        # 检查是否为 numpy 数组
        if isinstance(descriptors_list, np.ndarray):
            if descriptors_list.size == 0:
                return pd.DataFrame()

            n_features = descriptors_list.shape[1] if len(descriptors_list.shape) > 1 else descriptors_list.shape[0]
            columns = [f'desc_{i}' for i in range(n_features)]

            return pd.DataFrame(descriptors_list, columns=columns)

        # 其他情况返回空 DataFrame
        print(f"[WARNING] 未知的描述符数据类型: {type(descriptors_list)}")
        return pd.DataFrame()

    def _batch_pains_check(self, molecules: List[Chem.Mol]) -> List[bool]:
        """批量PAINS检查"""
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
            catalog = FilterCatalog(params)

            # 并行检查
            check_func = lambda mol: catalog.HasMatch(mol) if mol else False
            results = self.cpu_pool.map_parallel(
                check_func,
                molecules,
                desc="PAINS批量检查",
                use_processes=False,
                batch_size=1000
            )

            return results

        except Exception as e:
            print(f"[WARNING] 批量PAINS检查失败: {e}")
            return [False] * len(molecules)

    def _check_pains(self, mol: Chem.Mol) -> bool:
        """向后兼容方法"""
        if mol is None:
            return False
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
            catalog = FilterCatalog(params)
            return catalog.HasMatch(mol)
        except:
            return False

    def _batch_cns_check(self, results_df: pd.DataFrame, rules: Dict) -> List[bool]:
        """批量CNS规则检查（向量化）"""
        try:
            # 向量化规则检查
            conditions = []

            if 'mw' in results_df.columns:
                mw_check = (results_df['mw'] >= rules.get("mw_min", 0)) & \
                           (results_df['mw'] <= rules.get("mw_max", 1000))
                conditions.append(mw_check)

            if 'tpsa' in results_df.columns:
                tpsa_check = results_df['tpsa'] <= rules.get("tpsa_max", 200)
                conditions.append(tpsa_check)

            if 'hbd' in results_df.columns:
                hbd_check = results_df['hbd'] <= rules.get("hbd_max", 10)
                conditions.append(hbd_check)

            if 'hba' in results_df.columns:
                hba_check = results_df['hba'] <= rules.get("hba_max", 15)
                conditions.append(hba_check)

            if 'rotb' in results_df.columns:
                rotb_check = results_df['rotb'] <= rules.get("rotb_max", 20)
                conditions.append(rotb_check)

            if 'logp' in results_df.columns:
                logp_check = (results_df['logp'] >= rules.get("logp_min", -10)) & \
                             (results_df['logp'] <= rules.get("logp_max", 10))
                conditions.append(logp_check)

            # 组合所有条件
            if conditions:
                combined_check = conditions[0]
                for condition in conditions[1:]:
                    combined_check = combined_check & condition
                return combined_check.tolist()
            else:
                return [True] * len(results_df)

        except Exception as e:
            print(f"[WARNING] 批量CNS检查失败: {e}")
            return [True] * len(results_df)

    def _check_cns_compliance_row(self, row: pd.Series, rules: Dict) -> bool:
        """向后兼容方法"""
        try:
            return (
                    rules.get("mw_min", 0) <= row.get("mw", 0) <= rules.get("mw_max", 1000) and
                    row.get("tpsa", 0) <= rules.get("tpsa_max", 200) and
                    row.get("hbd", 0) <= rules.get("hbd_max", 10) and
                    row.get("hba", 0) <= rules.get("hba_max", 15) and
                    row.get("rotb", 0) <= rules.get("rotb_max", 20) and
                    rules.get("logp_min", -10) <= row.get("logp", 0) <= rules.get("logp_max", 10)
            )
        except:
            return False

    def _perform_clustering_fast(self, descriptors_df: pd.DataFrame, n_clusters: int = 5) -> np.ndarray:
        """快速聚类分析（使用MiniBatchKMeans）"""
        try:
            # 选择数值列
            numeric_cols = descriptors_df.select_dtypes(include=[np.number]).columns
            X = descriptors_df[numeric_cols].fillna(0).values

            # 数据采样（如果数据量大）
            if len(X) > 10000:
                sample_size = 5000
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[indices]

                # 标准化采样数据
                scaler = StandardScaler()
                X_sample_scaled = scaler.fit_transform(X_sample)

                # MiniBatch K-means（更快）
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    batch_size=1000,
                    n_init=3,
                    max_iter=100
                )
                kmeans.fit(X_sample_scaled)

                # 预测所有数据
                X_scaled = scaler.transform(X)
                clusters = kmeans.predict(X_scaled)
            else:
                # 小数据集直接处理
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    batch_size=min(1000, len(X)),
                    n_init=3
                )
                clusters = kmeans.fit_predict(X_scaled)

            return clusters

        except Exception as e:
            print(f"[WARNING] 快速聚类失败: {e}")
            return np.zeros(len(descriptors_df))

    def _perform_clustering(self, descriptors_df: pd.DataFrame, n_clusters: int = 5) -> np.ndarray:
        """向后兼容方法"""
        return self._perform_clustering_fast(descriptors_df, n_clusters)

    def _detect_outliers_fast(self, descriptors_df: pd.DataFrame, contamination: float = 0.1) -> np.ndarray:
        """快速异常值检测（采样策略）"""
        try:
            # 选择数值列
            numeric_cols = descriptors_df.select_dtypes(include=[np.number]).columns
            X = descriptors_df[numeric_cols].fillna(0).values

            # 数据采样策略
            if len(X) > 5000:
                # 大数据集：训练采样，预测全部
                sample_size = 2000
                train_indices = np.random.choice(len(X), sample_size, replace=False)
                X_train = X[train_indices]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                # 训练异常检测器
                iso_forest = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_jobs=-1,
                    n_estimators=50  # 减少估计器数量
                )
                iso_forest.fit(X_train_scaled)

                # 预测所有数据
                X_scaled = scaler.transform(X)
                outliers = iso_forest.predict(X_scaled)
            else:
                # 小数据集直接处理
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                iso_forest = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_jobs=-1,
                    n_estimators=50
                )
                outliers = iso_forest.fit_predict(X_scaled)

            return outliers

        except Exception as e:
            print(f"[WARNING] 快速异常值检测失败: {e}")
            return np.ones(len(descriptors_df))

    def _detect_outliers(self, descriptors_df: pd.DataFrame, contamination: float = 0.1) -> np.ndarray:
        """向后兼容方法"""
        return self._detect_outliers_fast(descriptors_df, contamination)

    # def generate_comprehensive_report(self, results: pd.DataFrame, ref_df: pd.DataFrame,
    #                                   lib_df: pd.DataFrame, processing_time: float):
    #     """生成综合报告（增强版）"""
    #     print("[INFO] 生成综合报告...")
    #
    #     # 统计信息
    #     total_compounds = len(results)
    #     threshold = self.config.get("ai_model", {}).get("similarity_threshold", 0.75)
    #     hits = results[results['combined_score'] >= threshold]
    #
    #     # 性能统计
    #     cache_hit_rate = (self.perf_stats['cache_hits'] /
    #                       (self.perf_stats['cache_hits'] + self.perf_stats['cache_misses']) * 100) if \
    #         (self.perf_stats['cache_hits'] + self.perf_stats['cache_misses']) > 0 else 0
    #
    #     # 报告内容
    #     report_lines = [
    #         "=" * 90,
    #         " 超高速GPU + 多线程CPU优化版 Mitophagy诱导剂虚拟筛选报告",
    #         "=" * 90,
    #         "",
    #         "=== 系统配置 ===",
    #         f"CPU线程数: {self.cpu_pool.max_workers}",
    #         f"可用内存: {psutil.virtual_memory().available / (1024 ** 3):.1f} GB",
    #         f"GPU加速: {'启用' if (GPU_AVAILABLE or TORCH_AVAILABLE) else '禁用'}",
    #     ]
    #
    #     if GPU_AVAILABLE:
    #         report_lines.extend([
    #             f"CuPy GPU: {self.gpu_info.get('device_name', 'Unknown')}",
    #             f"GPU显存: {self.gpu_info.get('total_memory', 0):.1f} GB",
    #             f"GPU内存限制: {self.gpu_info.get('memory_limit', 0):.1f} GB",
    #         ])
    #
    #     if TORCH_AVAILABLE:
    #         report_lines.append(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
    #
    #     report_lines.extend([
    #         f"总处理时间: {processing_time:.2f} 秒",
    #         f"处理速度: {total_compounds / processing_time:.1f} 化合物/秒",
    #         "",
    #         "=== 性能统计 ===",
    #         f"预处理时间: {self.perf_stats['preprocessing_time']:.2f}s",
    #         f"特征提取时间: {self.perf_stats['feature_extraction_time']:.2f}s",
    #         f"相似性计算时间: {self.perf_stats['similarity_computation_time']:.2f}s",
    #         f"过滤时间: {self.perf_stats['filtering_time']:.2f}s",
    #         f"缓存命中率: {cache_hit_rate:.1f}%",
    #         "",
    #         "=== 数据概览 ===",
    #         f"参考分子数: {len(ref_df)}",
    #         f"库分子数: {len(lib_df)}",
    #         f"有效化合物: {total_compounds}",
    #         "",
    #         "===  相似性权重 ===",
    #         f"1D权重 (Mol2vec/Morgan): {self.config['similarity']['w_1d']:.3f}",
    #         f"2D权重 (药效团): {self.config['similarity']['w_2d']:.3f}",
    #         f"3D权重 (形状): {self.config['similarity']['w_3d']:.3f}",
    #         "",
    #         "=== 筛选结果 ===",
    #         f"相似性阈值: {threshold:.3f}",
    #         f"命中化合物数: {len(hits)}",
    #         f"命中率: {len(hits) / total_compounds * 100:.2f}%",
    #         "",
    #         "=== 得分统计 ===",
    #         f"最高得分: {results['combined_score'].max():.4f}",
    #         f"平均得分: {results['combined_score'].mean():.4f}",
    #         f"中位数得分: {results['combined_score'].median():.4f}",
    #         f"标准差: {results['combined_score'].std():.4f}",
    #         f"75分位数: {results['combined_score'].quantile(0.75):.4f}",
    #         f"90分位数: {results['combined_score'].quantile(0.90):.4f}",
    #         "",
    #     ])
    #
    #     # Top候选化合物
    #     top_k = min(30, len(results))  # 增加显示数量
    #     report_lines.extend([
    #         f"=== Top-{top_k} 候选化合物 ===",
    #     ])
    #
    #     for i, (_, row) in enumerate(results.head(top_k).iterrows(), 1):
    #         pains_status = "PAINS+" if row.get('is_pains', False) else "PAINS-"
    #         cns_status = "CNS+" if row.get('cns_compliant', True) else "CNS-"
    #         cluster_info = f"C{row.get('cluster', 'N/A')}" if 'cluster' in row else ""
    #
    #         score_bar = "█" * int(row['combined_score'] * 20) + "░" * (20 - int(row['combined_score'] * 20))
    #
    #         report_lines.append(
    #             f"{i:2d}. {row['id'][:15]:15s} | {row.get('name', 'Unknown')[:25]:25s} | "
    #             f"得分={row['combined_score']:.4f} [{score_bar}] | {pains_status} | {cns_status} | {cluster_info}"
    #         )
    #
    #     # 性能对比
    #     estimated_speedup = self.cpu_pool.max_workers
    #     if GPU_AVAILABLE or TORCH_AVAILABLE:
    #         estimated_speedup *= 3  # GPU大约3倍加速
    #
    #     report_lines.extend([
    #         "",
    #         "===  性能对比 ===",
    #         f"理论加速比: ~{estimated_speedup:.1f}x",
    #         f"实际处理速度: {total_compounds / processing_time:.1f} 化合物/秒",
    #         f"CPU线程利用率: {self.cpu_pool.max_workers}x",
    #         f"GPU后端: {'PyTorch' if TORCH_AVAILABLE else 'CuPy' if GPU_AVAILABLE else 'None'}",
    #         f"内存优化: {' 启用' if cache_hit_rate > 10 else ' 禁用'}",
    #     ])
    #
    #     # 过滤统计
    #     if 'is_pains' in results.columns:
    #         pains_count = results['is_pains'].sum()
    #         pains_rate = pains_count / total_compounds * 100
    #         report_lines.extend([
    #             "",
    #             "=== 过滤统计 ===",
    #             f"PAINS阳性: {pains_count} ({pains_rate:.1f}%)",
    #         ])
    #
    #     if 'cns_compliant' in results.columns:
    #         cns_pass = results['cns_compliant'].sum()
    #         cns_rate = cns_pass / total_compounds * 100
    #         report_lines.append(f"CNS合规: {cns_pass} ({cns_rate:.1f}%)")
    #
    #     # 聚类分析
    #     if 'cluster' in results.columns:
    #         report_lines.extend([
    #             "",
    #             "=== 🔬 聚类分析 ===",
    #         ])
    #         cluster_counts = results['cluster'].value_counts().sort_index()
    #         for cluster_id, count in cluster_counts.items():
    #             percentage = count / total_compounds * 100
    #             bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
    #             report_lines.append(f"聚类 {cluster_id}: {count:4d} 个化合物 ({percentage:5.1f}%) [{bar}]")
    #
    #     # 内存和资源使用
    #     memory_info = psutil.virtual_memory()
    #     report_lines.extend([
    #         "",
    #         "===资源使用 ===",
    #         f"系统内存: {memory_info.total / (1024 ** 3):.1f} GB",
    #         f"可用内存: {memory_info.available / (1024 ** 3):.1f} GB",
    #         f"内存使用率: {memory_info.percent:.1f}%",
    #         f"CPU核心数: {os.cpu_count()}",
    #     ])
    #
    #     # 保存报告
    #     report_path = os.path.join(self.output_dir, self.config["output"]["report_txt"])
    #     with open(report_path, 'w', encoding='utf-8') as f:
    #         f.write('\n'.join(report_lines))
    #
    #     print(f"[INFO] 报告已保存: {report_path}")
    #
    #     # 保存详细处理信息
    #     processing_info = {
    #         'processing_time_seconds': processing_time,
    #         'performance_stats': self.perf_stats,
    #         'cpu_threads': self.cpu_pool.max_workers,
    #         'gpu_enabled': GPU_AVAILABLE or TORCH_AVAILABLE,
    #         'gpu_backend': 'PyTorch' if TORCH_AVAILABLE else 'CuPy' if GPU_AVAILABLE else 'None',
    #         'gpu_info': getattr(self, 'gpu_info', {}),
    #         'system_info': {
    #             'cpu_count': os.cpu_count(),
    #             'memory_total_gb': psutil.virtual_memory().total / (1024 ** 3),
    #             'memory_available_gb': psutil.virtual_memory().available / (1024 ** 3),
    #         },
    #         'data_stats': {
    #             'total_compounds': total_compounds,
    #             'reference_compounds': len(ref_df),
    #             'library_compounds': len(lib_df),
    #             'hits_count': len(hits),
    #             'hit_rate': len(hits) / total_compounds if total_compounds > 0 else 0,
    #         },
    #         'score_stats': {
    #             'max_score': float(results['combined_score'].max()),
    #             'mean_score': float(results['combined_score'].mean()),
    #             'median_score': float(results['combined_score'].median()),
    #             'std_score': float(results['combined_score'].std()),
    #         },
    #         'config_used': self.config
    #     }
    #
    #     info_path = os.path.join(self.output_dir, "processing_info.json")
    #     import json
    #     with open(info_path, 'w') as f:
    #         json.dump(processing_info, f, indent=2)
    #
    #     # 生成性能图表（如果matplotlib可用）
    #     try:
    #         import matplotlib.pyplot as plt
    #         self._generate_performance_plots(results, processing_info)
    #     except ImportError:
    #         print("[INFO] matplotlib不可用，跳过图表生成")

    def generate_comprehensive_report(self, results: pd.DataFrame, ref_df: pd.DataFrame,
                                      lib_df: pd.DataFrame, processing_time: float):
        """生成综合报告（修复版）"""
        print("[INFO] 生成综合报告...")

        # 统计信息
        total_compounds = len(results)
        threshold = self.config.get("ai_model", {}).get("similarity_threshold", 0.75)
        hits = results[results['combined_score'] >= threshold]

        # 性能统计
        cache_hit_rate = (self.perf_stats.get('cache_hits', 0) /
                          (self.perf_stats.get('cache_hits', 0) + self.perf_stats.get('cache_misses', 1)) * 100)

        # 报告内容
        report_lines = [
            "=" * 90,
            " 超高速GPU + 多线程CPU优化版 Mitophagy诱导剂虚拟筛选报告",
            "=" * 90,
            "",
            "=== 系统配置 ===",
            f"CPU线程数: {self.cpu_pool.max_workers}",
            f"可用内存: {psutil.virtual_memory().available / (1024 ** 3):.1f} GB",
            f"GPU加速: {'启用' if (GPU_AVAILABLE or TORCH_AVAILABLE) else '禁用'}",
            f"总处理时间: {processing_time:.2f} 秒",
            f"处理速度: {total_compounds / processing_time:.1f} 化合物/秒",
            "",
            "===  数据概览 ===",
            f"参考分子数: {len(ref_df)}",
            f"库分子数: {len(lib_df)}",
            f"有效化合物: {total_compounds}",
            "",
            "=== 筛选结果 ===",
            f"相似性阈值: {threshold:.3f}",
            f"命中化合物数: {len(hits)}",
            f"命中率: {len(hits) / total_compounds * 100:.2f}%" if total_compounds > 0 else "命中率: 0%",
            "",
            "=== 得分统计 ===",
            f"最高得分: {results['combined_score'].max():.4f}" if len(results) > 0 else "最高得分: N/A",
            f"平均得分: {results['combined_score'].mean():.4f}" if len(results) > 0 else "平均得分: N/A",
            f"中位数得分: {results['combined_score'].median():.4f}" if len(results) > 0 else "中位数得分: N/A",
            f"标准差: {results['combined_score'].std():.4f}" if len(results) > 0 else "标准差: N/A",
            "",
        ]

        # Top候选化合物 - 修复版本
        if len(results) > 0:
            top_k = min(20, len(results))
            report_lines.extend([
                f"=== Top-{top_k} 候选化合物 ===",
            ])

            for i, (_, row) in enumerate(results.head(top_k).iterrows(), 1):
                try:
                    # 安全的字符串处理
                    row_id = str(row.get('id', 'Unknown'))[:15]
                    row_name = str(row.get('name', 'Unknown'))[:25]
                    combined_score = float(row.get('combined_score', 0.0))

                    pains_status = "PAINS+" if row.get('is_pains', False) else "PAINS-"
                    cns_status = "CNS+" if row.get('cns_compliant', True) else "CNS-"
                    cluster_info = f"C{row.get('cluster', 'N/A')}" if 'cluster' in row else ""

                    score_bar = "█" * int(combined_score * 20) + "░" * (20 - int(combined_score * 20))

                    report_lines.append(
                        f"{i:2d}. {row_id:15s} | {row_name:25s} | "
                        f"得分={combined_score:.4f} [{score_bar}] | {pains_status} | {cns_status} | {cluster_info}"
                    )
                except Exception as e:
                    print(f"[WARNING] 处理第{i}个化合物信息时出错: {e}")
                    report_lines.append(f"{i:2d}. 数据处理错误")

        # 性能对比
        estimated_speedup = self.cpu_pool.max_workers
        if GPU_AVAILABLE or TORCH_AVAILABLE:
            estimated_speedup *= 3

        report_lines.extend([
            "",
            "===  性能统计 ===",
            f"理论加速比: ~{estimated_speedup:.1f}x",
            f"实际处理速度: {total_compounds / processing_time:.1f} 化合物/秒" if processing_time > 0 else "处理速度: N/A",
            f"CPU线程利用率: {self.cpu_pool.max_workers}x",
            f"GPU后端: {'PyTorch' if TORCH_AVAILABLE else 'CuPy' if GPU_AVAILABLE else 'None'}",
            f"缓存命中率: {cache_hit_rate:.1f}%",
        ])

        # 过滤统计
        if len(results) > 0:
            if 'is_pains' in results.columns:
                pains_count = results['is_pains'].sum()
                pains_rate = pains_count / total_compounds * 100
                report_lines.extend([
                    "",
                    "===过滤统计 ===",
                    f"PAINS阳性: {pains_count} ({pains_rate:.1f}%)",
                ])

            if 'cns_compliant' in results.columns:
                cns_pass = results['cns_compliant'].sum()
                cns_rate = cns_pass / total_compounds * 100
                report_lines.append(f"CNS合规: {cns_pass} ({cns_rate:.1f}%)")

        # 系统资源信息
        try:
            memory_info = psutil.virtual_memory()
            report_lines.extend([
                "",
                "=== 资源使用 ===",
                f"系统内存: {memory_info.total / (1024 ** 3):.1f} GB",
                f"可用内存: {memory_info.available / (1024 ** 3):.1f} GB",
                f"内存使用率: {memory_info.percent:.1f}%",
                f"CPU核心数: {os.cpu_count()}",
            ])
        except Exception as e:
            print(f"[WARNING] 获取系统信息失败: {e}")

        # 保存报告
        try:
            report_path = os.path.join(self.output_dir, self.config["output"]["report_txt"])
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            print(f"[INFO] 报告已保存: {report_path}")
        except Exception as e:
            print(f"[WARNING] 报告保存失败: {e}")

        # 保存处理信息
        try:
            processing_info = {
                'processing_time_seconds': processing_time,
                'performance_stats': self.perf_stats,
                'cpu_threads': self.cpu_pool.max_workers,
                'gpu_enabled': GPU_AVAILABLE or TORCH_AVAILABLE,
                'gpu_backend': 'PyTorch' if TORCH_AVAILABLE else 'CuPy' if GPU_AVAILABLE else 'None',
                'system_info': {
                    'cpu_count': os.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / (1024 ** 3),
                    'memory_available_gb': psutil.virtual_memory().available / (1024 ** 3),
                },
                'data_stats': {
                    'total_compounds': total_compounds,
                    'reference_compounds': len(ref_df),
                    'library_compounds': len(lib_df),
                    'hits_count': len(hits),
                    'hit_rate': len(hits) / total_compounds if total_compounds > 0 else 0,
                },
                'config_used': self.config
            }

            info_path = os.path.join(self.output_dir, "processing_info.json")
            import json
            with open(info_path, 'w') as f:
                json.dump(processing_info, f, indent=2)
            print(f"[INFO] 处理信息已保存: {info_path}")
        except Exception as e:
            print(f"[WARNING] 处理信息保存失败: {e}")




    def _generate_performance_plots(self, results: pd.DataFrame, processing_info: Dict):
        """生成性能分析图表"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('虚拟筛选性能分析', fontsize=16, fontweight='bold')

            # 1. 得分分布直方图
            axes[0, 0].hist(results['combined_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('组合相似性得分')
            axes[0, 0].set_ylabel('化合物数量')
            axes[0, 0].set_title('得分分布')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 处理时间饼图
            perf_stats = processing_info['performance_stats']
            labels = ['预处理', '特征提取', '相似性计算', '过滤', '其他']
            sizes = [
                perf_stats['preprocessing_time'],
                perf_stats['feature_extraction_time'],
                perf_stats['similarity_computation_time'],
                perf_stats['filtering_time'],
                max(0, processing_info['processing_time_seconds'] - sum([
                    perf_stats['preprocessing_time'],
                    perf_stats['feature_extraction_time'],
                    perf_stats['similarity_computation_time'],
                    perf_stats['filtering_time']
                ]))
            ]

            axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('处理时间分布')

            # 3. Top化合物得分条形图
            top_20 = results.head(20)
            axes[1, 0].barh(range(len(top_20)), top_20['combined_score'], color='lightcoral')
            axes[1, 0].set_yticks(range(len(top_20)))
            axes[1, 0].set_yticklabels([f"{row['id'][:10]}" for _, row in top_20.iterrows()])
            axes[1, 0].set_xlabel('组合得分')
            axes[1, 0].set_title('Top-20 候选化合物')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. 系统资源使用
            resource_labels = ['CPU线程', 'GPU', '内存(GB)', '处理速度(化合物/秒)']
            resource_values = [
                processing_info['cpu_threads'],
                1 if processing_info['gpu_enabled'] else 0,
                processing_info['system_info']['memory_available_gb'],
                len(results) / processing_info['processing_time_seconds']
            ]

            bars = axes[1, 1].bar(resource_labels, resource_values, color=['gold', 'red', 'green', 'blue'])
            axes[1, 1].set_title('系统资源使用')
            axes[1, 1].set_ylabel('资源量')

            # 添加数值标签
            for bar, value in zip(bars, resource_values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                                f'{value:.1f}', ha='center', va='bottom')

            plt.tight_layout()

            # 保存图表
            plot_path = os.path.join(self.output_dir, "performance_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[INFO] 性能分析图表已保存: {plot_path}")

        except Exception as e:
            print(f"[WARNING] 图表生成失败: {e}")

    def run_virtual_screening_ultra_fast(self):
        """执行超高速虚拟筛选流程"""
        start_time = time.time()

        print("=" * 90)
        print("GPU + 多线程CPU优化版虚拟筛选系统启动")
        print("=" * 90)
        print(f"CPU线程池: {self.cpu_pool.max_workers} 个工作线程")
        print(f"可用内存: {psutil.virtual_memory().available / (1024 ** 3):.1f} GB")

        if TORCH_AVAILABLE:
            print(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
        elif GPU_AVAILABLE:
            print(f"CuPy GPU: {self.gpu_info.get('device_name', 'Unknown')}")
        else:
            print("CPU模式")
        print("=" * 90)

        try:
            # 检查是否使用流式处理
            use_streaming = self.config.get("performance", {}).get("use_streaming", False)

            if use_streaming:
                # 流式处理（处理超大数据集）
                ref_df, lib_stream = self.load_and_preprocess_data_streaming()
                return self._run_streaming_screening(ref_df, lib_stream, start_time)
            else:
                # 标准处理
                return self._run_standard_screening(start_time)

        except Exception as e:
            print(f"[ERROR] 超高速虚拟筛选失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 清理资源
            self._cleanup_resources()

    def run_virtual_screening(self):
        """向后兼容方法"""
        return self.run_virtual_screening_ultra_fast()

    def _run_standard_screening(self, start_time: float):
        """标准筛选流程"""
        # 1. 数据加载和预处理
        ref_df, lib_df = self.load_and_preprocess_data()

        # 2. 特征提取
        ref_features = self.extract_molecular_features_optimized(ref_df, "参考分子")
        lib_features = self.extract_molecular_features_optimized(lib_df, "库分子")

        # 3. 多维相似性计算
        similarities = self.compute_multi_dimensional_similarity_ultra_fast(ref_features, lib_features)

        # 4. 相似性聚合
        scores = self.aggregate_similarities_ultra_fast(similarities)

        # 5. AI增强过滤
        results = self.apply_ai_enhanced_filtering_ultra_fast(lib_df, scores, lib_features)

        # 6. 保存结果
        results_path = os.path.join(self.output_dir, self.config["output"]["hits_csv"])
        results.to_csv(results_path, index=False)
        print(f"[INFO] 结果已保存: {results_path}")

        # 7. 生成报告
        processing_time = time.time() - start_time
        self.generate_comprehensive_report(results, ref_df, lib_df, processing_time)

        # 8. 输出摘要
        self._print_final_summary(results, processing_time)

        return results

    def _run_streaming_screening(self, ref_df: pd.DataFrame, lib_stream: Iterator, start_time: float):
        """流式筛选流程（处理超大数据集）"""
        print("[INFO] 启动流式筛选模式...")

        # 预提取参考分子特征
        ref_features = self.extract_molecular_features_optimized(ref_df, "参考分子")

        all_results = []
        chunk_count = 0

        for lib_chunk in lib_stream:
            chunk_count += 1
            print(f"[INFO] 处理第 {chunk_count} 个数据块 ({len(lib_chunk)} 个化合物)...")

            try:
                # 特征提取
                lib_features = self.extract_molecular_features_optimized(lib_chunk, f"库分子块{chunk_count}")

                # 相似性计算
                similarities = self.compute_multi_dimensional_similarity_ultra_fast(ref_features, lib_features)

                # 聚合和过滤
                scores = self.aggregate_similarities_ultra_fast(similarities)
                chunk_results = self.apply_ai_enhanced_filtering_ultra_fast(lib_chunk, scores, lib_features)

                # 只保留高分化合物（节省内存）
                threshold = self.config.get("ai_model", {}).get("similarity_threshold", 0.5)
                high_score_results = chunk_results[chunk_results['combined_score'] >= threshold]

                if len(high_score_results) > 0:
                    all_results.append(high_score_results)
                    print(f"[INFO] 块 {chunk_count} 发现 {len(high_score_results)} 个候选化合物")
                else:
                    print(f"[INFO] 块 {chunk_count} 未发现符合条件的化合物")

                # 内存清理
                del lib_features, similarities, scores, chunk_results
                gc.collect()

            except Exception as e:
                print(f"[WARNING] 块 {chunk_count} 处理失败: {e}")
                continue

        # 合并所有结果
        if all_results:
            print("[INFO] 合并流式处理结果...")
            final_results = pd.concat(all_results, ignore_index=True)

            # 最终排序
            final_results = final_results.sort_values('combined_score', ascending=False).reset_index(drop=True)

            # 限制结果数量（避免内存溢出）
            max_results = self.config.get("performance", {}).get("max_results", 50000)
            if len(final_results) > max_results:
                print(f"[INFO] 限制结果数量至 {max_results}")
                final_results = final_results.head(max_results)
        else:
            print("[WARNING] 流式处理未发现任何候选化合物")
            final_results = pd.DataFrame()

        # 保存和报告
        if len(final_results) > 0:
            results_path = os.path.join(self.output_dir, self.config["output"]["hits_csv"])
            final_results.to_csv(results_path, index=False)
            print(f"[INFO] 流式处理结果已保存: {results_path}")

            # 生成报告（使用估算的库大小）
            processing_time = time.time() - start_time
            estimated_lib_size = chunk_count * 10000  # 估算
            mock_lib_df = pd.DataFrame({'estimated_size': [estimated_lib_size]})
            self.generate_comprehensive_report(final_results, ref_df, mock_lib_df, processing_time)

            self._print_final_summary(final_results, processing_time)

        return final_results

    def _cleanup_resources(self):
        """清理系统资源"""
        try:
            # 清理GPU内存
            if GPU_AVAILABLE:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                print("[INFO] CuPy GPU内存已清理")

            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
                print("[INFO] PyTorch GPU缓存已清理")

            # 清理Python内存
            gc.collect()

            # 清理缓存
            if hasattr(self, 'feature_cache'):
                self.feature_cache.cache.clear()

        except Exception as e:
            print(f"[WARNING] 资源清理时出错: {e}")

    def _print_final_summary(self, results: pd.DataFrame, processing_time: float):
        """打印最终摘要"""
        threshold = self.config.get("ai_model", {}).get("similarity_threshold", 0.75)
        hits = results[results['combined_score'] >= threshold] if len(results) > 0 else pd.DataFrame()

        print("\n" + "=" * 90)
        print(" 超高速虚拟筛选完成")
        print("=" * 90)
        print(f"️  处理时间: {processing_time:.2f} 秒")
        print(f" CPU线程: {self.cpu_pool.max_workers}")
        print(f" GPU加速: {' 启用' if (GPU_AVAILABLE or TORCH_AVAILABLE) else ' 禁用'}")
        print(f"总化合物: {len(results):,}")
        print(f" 命中数 (≥{threshold}): {len(hits):,}")
        print(f" 最高得分: {results['combined_score'].max():.4f}" if len(results) > 0 else " 最高得分: N/A")
        print(f" 输出目录: {self.output_dir}")

        # 性能指标
        if processing_time > 0:
            compounds_per_sec = len(results) / processing_time
            print(f" 处理速度: {compounds_per_sec:,.1f} 化合物/秒")

        # 缓存性能
        if hasattr(self, 'perf_stats'):
            cache_hits = self.perf_stats.get('cache_hits', 0)
            cache_misses = self.perf_stats.get('cache_misses', 0)
            if cache_hits + cache_misses > 0:
                cache_hit_rate = cache_hits / (cache_hits + cache_misses) * 100
                print(f"缓存命中率: {cache_hit_rate:.1f}%")

        print("=" * 90)


def _safe_tensor_check(self, data):
    """安全的张量/数组检查函数"""
    if data is None:
        return True, "none"

    # PyTorch 张量检查
    if TORCH_AVAILABLE and torch.is_tensor(data):
        return data.numel() == 0, "torch_tensor"

    # CuPy 数组检查
    if GPU_AVAILABLE:
        try:
            import cupy as cp
            if isinstance(data, cp.ndarray):
                return data.size == 0, "cupy_array"
        except:
            pass

    # NumPy 数组检查
    if isinstance(data, np.ndarray):
        return data.size == 0, "numpy_array"

    # 普通列表检查
    if isinstance(data, list):
        return len(data) == 0, "list"

    # 其他类型
    try:
        return not bool(data), "other"
    except:
        return False, "unknown"


# 在需要的地方使用
def _process_descriptors_dataframe(self, descriptors_list) -> pd.DataFrame:
    """快速处理描述符列表为DataFrame - 使用安全检查"""

    is_empty, data_type = self._safe_tensor_check(descriptors_list)

    if is_empty:
        print(f"[INFO] 描述符数据为空 (类型: {data_type})")
        return pd.DataFrame()

    # 根据数据类型进行相应处理
    # [其余处理逻辑...]
def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def optimize_system_settings():
    """优化系统设置"""
    try:
        # 设置CPU亲和性（如果支持）
        if hasattr(os, 'sched_setaffinity'):
            available_cpus = list(range(os.cpu_count()))
            os.sched_setaffinity(0, available_cpus)

        # 设置进程优先级（如果支持）
        if hasattr(os, 'nice'):
            os.nice(-5)  # 提高优先级

        # 设置环境变量优化
        os.environ['OMP_NUM_THREADS'] = str(min(8, os.cpu_count()))
        os.environ['MKL_NUM_THREADS'] = str(min(8, os.cpu_count()))
        os.environ['NUMEXPR_NUM_THREADS'] = str(min(8, os.cpu_count()))

        print("[INFO] 系统优化设置已应用")

    except Exception as e:
        print(f"[WARNING] 系统优化设置失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description=" 超高速GPU + 多线程CPU优化版虚拟筛选")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--threads", type=int, help="CPU线程数（默认自动检测）")
    parser.add_argument("--gpu", action="store_true", help="强制启用GPU（如果可用）")
    parser.add_argument("--streaming", action="store_true", help="启用流式处理（大数据集）")
    parser.add_argument("--optimize", action="store_true", help="应用系统优化设置")

    args = parser.parse_args()

    # 应用系统优化
    if args.optimize:
        optimize_system_settings()

    # 加载配置
    config = load_config(args.config)

    # 设置流式处理
    if args.streaming:
        config.setdefault("performance", {})["use_streaming"] = True

    # 设置线程数
    if args.threads:
        original_cpu_count = os.cpu_count()
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        print(f"[INFO] 设置CPU线程数: {args.threads} (系统: {original_cpu_count})")

    # 创建超高速虚拟筛选系统
    vs_system = UltraFastVirtualScreening(config)

    # 运行虚拟筛选
    results = vs_system.run_virtual_screening_ultra_fast()

    return results


if __name__ == "__main__":
    main()