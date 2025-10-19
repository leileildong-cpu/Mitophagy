#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLRP3抑制剂虚拟筛选系统 - 主程序
完整模块化版本 v3.0
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ✅ 针对 Mitophagy 项目的路径修复
# 获取项目根目录（Mitophagy/）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在可以直接从 src 导入
from src.core.cpu_pool import OptimizedCPUThreadPool
from src.core.cache import FastMolecularFeatureCache
from src.core.gpu_utils import GPUManager
from src.core.logger import ScreeningLogger

from src.features.extractors import FeatureExtractor
from src.features.mol2vec_handler import Mol2VecHandler
from src.features.descriptors import DescriptorCalculator

from src.similarity.fingerprint_sim import FingerprintSimilarity
from src.similarity.shape_3d import Shape3DSimilarity
from src.similarity.aggregator import SimilarityAggregator

# 导入过滤器模块
try:
    from src.filters.nlrp3_filters import NLRP3Filters
    from src.filters.admet_predictor import ADMETPredictor
    from src.filters.toxicity_checker import ToxicityChecker
    from src.filters.pains_brenk import PAINSBrenkFilter
    FILTERS_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] 过滤器模块导入失败: {e}")
    FILTERS_AVAILABLE = False

from src.docking.molecular_docking import MolecularDocking

from src.reporting.text_report import TextReportGenerator
from src.reporting.html_report import HTMLReportGenerator
from src.reporting.visualizer import Visualizer

from src.checkpoint_manager import CheckpointManager

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# 其他
import numpy as np
from tqdm import tqdm
import psutil
import gc


class CompleteNLRP3VirtualScreening:
    """完整的NLRP3虚拟筛选系统"""

    def __init__(self, config: dict):
        """初始化虚拟筛选系统"""
        self.config = config
        self.start_time = time.time()

        # 设置输出目录
        self._setup_output_directory()

        # 初始化日志系统
        self.logger = ScreeningLogger(
            self.output_dir,
            log_level=config.get("logging", {}).get("level", "INFO")
        )

        self.logger.info("=" * 70)
        self.logger.info("NLRP3抑制剂虚拟筛选系统启动")
        self.logger.info("=" * 70)

        # 记录系统信息
        self.logger.log_system_info()

        # 初始化核心组件
        self._initialize_core_components()

        # 初始化功能模块
        self._initialize_modules()

        # 性能统计
        self.perf_stats = {
            'preprocessing_time': 0,
            'feature_extraction_time': 0,
            'similarity_computation_time': 0,
            'filtering_time': 0,
            'docking_time': 0,
            'reporting_time': 0
        }

        self.logger.info("系统初始化完成")

    def _setup_output_directory(self):
        """设置输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = self.config["output"]["dir"]
        self.output_dir = f"{base_dir}_nlrp3_{timestamp}"

        os.makedirs(self.output_dir, exist_ok=True)

        # 保存配置文件副本
        config_backup = os.path.join(self.output_dir, "config_used.yaml")
        with open(config_backup, 'w') as f:
            yaml.dump(self.config, f)

    def _initialize_core_components(self):
        """初始化核心组件"""
        max_workers = self.config.get("performance", {}).get("max_workers")
        self.cpu_pool = OptimizedCPUThreadPool(max_workers=max_workers)

        if self.config.get("performance", {}).get("use_cache", True):
            cache_size = self.config.get("performance", {}).get("cache_size", 20000)
            self.feature_cache = FastMolecularFeatureCache(max_size=cache_size)
        else:
            self.feature_cache = None

        use_gpu = self.config.get("performance", {}).get("use_gpu", True)
        gpu_backend = self.config.get("performance", {}).get("gpu_backend", "pytorch")
        self.gpu_manager = GPUManager(prefer_pytorch=(gpu_backend == "pytorch")) if use_gpu else None

        use_checkpoint = self.config.get("performance", {}).get("use_checkpoint", True)
        self.checkpoint_manager = CheckpointManager(self.output_dir, enabled=use_checkpoint)

    def _initialize_modules(self):
        """初始化功能模块"""
        self.logger.info("初始化功能模块...")

        self.feature_extractor = FeatureExtractor(self.config, cache=self.feature_cache)

        use_gpu = self.config.get("performance", {}).get("use_gpu", True)
        gpu_backend = self.config.get("performance", {}).get("gpu_backend", "pytorch")

        self.fingerprint_sim = FingerprintSimilarity(use_gpu=use_gpu, gpu_backend=gpu_backend)
        self.shape_3d_sim = Shape3DSimilarity()
        self.similarity_aggregator = SimilarityAggregator(self.config)

        # NLRP3过滤器
        if FILTERS_AVAILABLE and self.config.get("filters", {}).get("nlrp3_filter", False):
            try:
                self.nlrp3_filters = NLRP3Filters(self.config)
                self.admet_predictor = ADMETPredictor(self.config)
                self.toxicity_checker = ToxicityChecker(self.config)
                self.pains_brenk_filter = PAINSBrenkFilter()
                self.logger.info("NLRP3特异性过滤器已加载")
            except Exception as e:
                self.logger.warning(f"NLRP3过滤器加载失败: {e}")
                self.nlrp3_filters = None
                self.pains_brenk_filter = None
        else:
            self.nlrp3_filters = None
            self.pains_brenk_filter = None

        self.molecular_docking = MolecularDocking(self.config)
        self.text_reporter = TextReportGenerator(self.config, self.output_dir)
        self.html_reporter = HTMLReportGenerator(self.config, self.output_dir)
        self.visualizer = Visualizer(self.output_dir)

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载和预处理数据"""
        self.logger.info("=" * 70)
        self.logger.info("步骤 1/7: 数据加载和预处理")
        self.logger.info("=" * 70)

        start_time = time.time()

        checkpoint_data = self.checkpoint_manager.load_checkpoint('preprocessing')
        if checkpoint_data:
            self.logger.info("从断点恢复数据...")
            return checkpoint_data['ref_df'], checkpoint_data['lib_df']

        ref_csv = self.config["data"]["references_csv"]
        lib_csv = self.config["data"]["library_csv"]

        self.logger.info(f"读取参考分子: {ref_csv}")
        self.logger.info(f"读取库分子: {lib_csv}")

        try:
            ref_df = pd.read_csv(ref_csv)
            lib_df = pd.read_csv(lib_csv)
        except Exception as e:
            self.logger.error(f"数据读取失败: {e}")
            raise

        self.logger.info(f"原始数据: 参考分子={len(ref_df)}, 库分子={len(lib_df)}")

        required_cols = {"id", "name", "smiles"}
        for df, name in [(ref_df, "参考"), (lib_df, "库")]:
            if not required_cols.issubset(set(df.columns)):
                missing = required_cols - set(df.columns)
                raise ValueError(f"{name}数据缺少必需列: {missing}")

        ref_df = self._preprocess_molecules(ref_df, "参考分子")
        lib_df = self._preprocess_molecules(lib_df, "库分子")

        if len(ref_df) == 0 or len(lib_df) == 0:
            raise ValueError("预处理后没有有效分子")

        self.logger.info(f"有效分子: 参考={len(ref_df)}, 库={len(lib_df)}")

        self.checkpoint_manager.save_checkpoint('preprocessing', {
            'ref_df': ref_df,
            'lib_df': lib_df
        })

        duration = time.time() - start_time
        self.perf_stats['preprocessing_time'] = duration
        self.logger.log_performance('preprocessing', duration, len(ref_df) + len(lib_df))

        return ref_df, lib_df

    def _preprocess_molecules(self, df: pd.DataFrame, desc: str) -> pd.DataFrame:
        """预处理分子"""
        self.logger.info(f"预处理{desc}...")

        valid_mols = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"标准化{desc}"):
            smiles = row['smiles']
            mol = self._standardize_molecule(smiles)

            if mol:
                new_row = row.to_dict()
                new_row['mol'] = mol
                new_row['canonical_smiles'] = Chem.MolToSmiles(mol, isomericSmiles=True)
                valid_mols.append(new_row)

        if not valid_mols:
            return pd.DataFrame()

        result_df = pd.DataFrame(valid_mols)

        before = len(result_df)
        result_df['smiles_hash'] = result_df['canonical_smiles'].apply(hash)
        result_df = result_df.drop_duplicates(subset=['smiles_hash']).reset_index(drop=True)
        result_df = result_df.drop('smiles_hash', axis=1)
        after = len(result_df)

        if before != after:
            self.logger.info(f"{desc}去重: {before} → {after}")

        return result_df

    def _standardize_molecule(self, smiles: str) -> Optional[Chem.Mol]:
        """标准化分子"""
        if not isinstance(smiles, str) or not smiles.strip():
            return None

        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return None

            if mol.GetNumAtoms() == 0 or mol.GetNumAtoms() > 150:
                return None

            Chem.SanitizeMol(mol)

            try:
                from rdkit.Chem import rdMolStandardize
                normalizer = rdMolStandardize.Normalizer()
                mol = normalizer.normalize(mol)
            except:
                pass

            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return Chem.MolFromSmiles(canonical_smiles)

        except:
            return None

    # ✅✅✅ 注意：以下方法不要多缩进！ ✅✅✅
    def extract_features(self, df: pd.DataFrame, desc: str) -> Dict:
        """提取分子特征"""
        self.logger.info("=" * 70)
        self.logger.info(f"提取{desc}特征")
        self.logger.info("=" * 70)

        start_time = time.time()

        checkpoint_key = f'features_{desc.replace(" ", "_")}'
        checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_key)
        if checkpoint_data:
            self.logger.info(f"从断点恢复{desc}特征...")
            return checkpoint_data['features']

        features = self.feature_extractor.extract_features(df['mol'].tolist(), desc)

        self.checkpoint_manager.save_checkpoint(checkpoint_key, {'features': features})

        duration = time.time() - start_time
        self.perf_stats['feature_extraction_time'] += duration
        self.logger.log_performance(f'feature_extraction_{desc}', duration, len(df))

        return features

    def compute_similarity(self, ref_features: Dict, lib_features: Dict) -> Dict:
        """计算多维相似性"""
        self.logger.info("=" * 70)
        self.logger.info("步骤 3/7: 计算多维相似性")
        self.logger.info("=" * 70)

        start_time = time.time()

        checkpoint_data = self.checkpoint_manager.load_checkpoint('similarity')
        if checkpoint_data:
            self.logger.info("从断点恢复相似性数据...")
            return checkpoint_data['similarities']

        n_lib = len(lib_features['molecules'])
        n_ref = len(ref_features['molecules'])

        self.logger.info(f"计算 {n_lib} x {n_ref} = {n_lib * n_ref:,} 对相似性")

        similarities = {}

        # 1D相似性
        self.logger.info("  [1/3] 计算1D相似性（Morgan指纹）...")
        try:
            sim_1d = self.fingerprint_sim.compute_tanimoto_similarity(
                ref_features['morgan_fps'],
                lib_features['morgan_fps']
            )
            similarities['1d_similarity'] = sim_1d
            self.logger.info(f"  1D相似性范围: {sim_1d.min():.4f} - {sim_1d.max():.4f}")
        except Exception as e:
            self.logger.error(f"1D相似性计算失败: {e}")
            similarities['1d_similarity'] = np.zeros((n_lib, n_ref), dtype=np.float32)

        # 2D相似性
        self.logger.info("  [2/3] 计算2D相似性（药效团）...")
        try:
            sim_2d = self.fingerprint_sim.compute_tanimoto_similarity(
                ref_features['pharm2d_fps'],
                lib_features['pharm2d_fps']
            )
            similarities['2d_similarity'] = sim_2d
            self.logger.info(f"  2D相似性范围: {sim_2d.min():.4f} - {sim_2d.max():.4f}")
        except Exception as e:
            self.logger.error(f"2D相似性计算失败: {e}")
            similarities['2d_similarity'] = np.zeros((n_lib, n_ref), dtype=np.float32)

        # 3D相似性
        w_3d = self.config["similarity"]["w_3d"]
        if w_3d > 0:
            self.logger.info("  [3/3] 计算3D相似性（形状）...")
            try:
                sim_3d = self.shape_3d_sim.compute_shape_similarity(
                    ref_features['molecules'],
                    lib_features['molecules']
                )
                similarities['3d_similarity'] = sim_3d
                self.logger.info(f"  3D相似性范围: {sim_3d.min():.4f} - {sim_3d.max():.4f}")
            except Exception as e:
                self.logger.warning(f"3D相似性计算失败: {e}")
                similarities['3d_similarity'] = np.zeros((n_lib, n_ref), dtype=np.float32)
        else:
            self.logger.info("  [3/3] 跳过3D相似性（权重=0）")
            similarities['3d_similarity'] = np.zeros((n_lib, n_ref), dtype=np.float32)

        self.checkpoint_manager.save_checkpoint('similarity', {'similarities': similarities})

        duration = time.time() - start_time
        self.perf_stats['similarity_computation_time'] = duration
        self.logger.log_performance('similarity_computation', duration, n_lib * n_ref)

        return similarities

    def aggregate_similarities(self, similarities: Dict) -> Dict:
        """聚合相似性得分"""
        self.logger.info("=" * 70)
        self.logger.info("步骤 4/7: 聚合相似性得分")
        self.logger.info("=" * 70)

        scores = self.similarity_aggregator.aggregate_similarities(similarities)

        self.logger.info(
            f"综合得分范围: {scores['combined_scores'].min():.4f} - {scores['combined_scores'].max():.4f}")
        self.logger.info(f"综合得分平均: {scores['combined_scores'].mean():.4f}")

        return scores

    def apply_filters(self, lib_df: pd.DataFrame, scores: Dict, lib_features: Dict) -> pd.DataFrame:
        """应用过滤器"""
        self.logger.info("=" * 70)
        self.logger.info("步骤 5/7: 应用NLRP3特异性过滤")
        self.logger.info("=" * 70)

        start_time = time.time()

        checkpoint_data = self.checkpoint_manager.load_checkpoint('filtering')
        if checkpoint_data:
            self.logger.info("从断点恢复过滤结果...")
            return checkpoint_data['results']

        results = lib_df[['id', 'name', 'smiles', 'canonical_smiles']].copy()
        results['combined_score'] = scores['combined_scores']

        for key, values in scores['individual_scores'].items():
            results[key] = values

        descriptors_list = lib_features['descriptors']
        for col in ['mw', 'logp', 'hbd', 'hba', 'tpsa', 'rotb', 'aromatic_rings', 'heavy_atoms', 'qed']:
            results[col] = [desc.get(col, 0) for desc in descriptors_list]

        if self.nlrp3_filters:
            self.logger.info("执行NLRP3特异性过滤...")

            nlrp3_results = []
            admet_results = []
            toxicity_results = []

            for i, (idx, row) in enumerate(tqdm(results.iterrows(), total=len(results), desc="NLRP3过滤")):
                mol = lib_df.loc[idx, 'mol']

                descriptors = {col: row[col] for col in
                               ['mw', 'logp', 'hbd', 'hba', 'tpsa', 'rotb', 'aromatic_rings', 'heavy_atoms']}

                is_valid, fail_reasons = self.nlrp3_filters.check_molecule(mol, descriptors)
                nlrp3_results.append({
                    'compliant': is_valid,
                    'fail_reasons': "; ".join(fail_reasons) if fail_reasons else "Pass"
                })

                admet_scores = self.admet_predictor.predict_admet(mol, descriptors)
                admet_results.append(admet_scores)

                is_safe, alerts = self.toxicity_checker.check_toxicity(mol)
                toxicity_results.append({
                    'is_safe': is_safe,
                    'alerts': "; ".join(alerts) if alerts else "Safe"
                })

            results['nlrp3_compliant'] = [r['compliant'] for r in nlrp3_results]
            results['nlrp3_fail_reasons'] = [r['fail_reasons'] for r in nlrp3_results]

            for key in ['lipinski', 'veber', 'ghose', 'caco2_pass', 'bioavailability_pass']:
                results[key] = [scores.get(key, False) for scores in admet_results]

            results['bioavailability_score'] = [scores.get('bioavailability', 0) for scores in admet_results]
            results['caco2_permeability'] = [scores.get('caco2_permeability', 0) for scores in admet_results]

            results['is_safe'] = [r['is_safe'] for r in toxicity_results]
            results['toxicity_alerts'] = [r['alerts'] for r in toxicity_results]

            nlrp3_pass = sum(results['nlrp3_compliant'])
            admet_pass = sum(s['lipinski'] and s['veber'] for s in admet_results)
            safe_count = sum(results['is_safe'])

            self.logger.info(f"NLRP3过滤结果:")
            self.logger.info(
                f"  - NLRP3规则通过: {nlrp3_pass}/{len(results)} ({nlrp3_pass / len(results) * 100:.1f}%)")
            self.logger.info(f"  - ADMET通过: {admet_pass}/{len(results)} ({admet_pass / len(results) * 100:.1f}%)")
            self.logger.info(
                f"  - 安全性通过: {safe_count}/{len(results)} ({safe_count / len(results) * 100:.1f}%)")
        else:
            results['nlrp3_compliant'] = True
            results['nlrp3_fail_reasons'] = "Not evaluated"
            results['is_safe'] = True
            results['toxicity_alerts'] = "Not evaluated"

        if self.config.get("filters", {}).get("pains", False) and self.pains_brenk_filter:
            self.logger.info("执行PAINS过滤...")
            results['is_pains'] = self.pains_brenk_filter.batch_check_pains(lib_df['mol'].tolist())
            pains_count = sum(results['is_pains'])
            self.logger.info(
                f"  - PAINS阳性: {pains_count}/{len(results)} ({pains_count / len(results) * 100:.1f}%)")
        else:
            results['is_pains'] = False

        if self.config.get("filters", {}).get("brenk", False) and self.pains_brenk_filter:
            self.logger.info("执行Brenk过滤...")
            brenk_results = self.pains_brenk_filter.batch_check_brenk(lib_df['mol'].tolist())
            results['brenk_clean'] = [r['is_clean'] for r in brenk_results]
            results['brenk_alerts'] = [r['alerts'] for r in brenk_results]
            brenk_pass = sum(results['brenk_clean'])
            self.logger.info(f"  - Brenk通过: {brenk_pass}/{len(results)} ({brenk_pass / len(results) * 100:.1f}%)")

        results['final_score'] = self._calculate_comprehensive_score(results)

        results = results.sort_values('final_score', ascending=False).reset_index(drop=True)

        self.checkpoint_manager.save_checkpoint('filtering', {'results': results})

        duration = time.time() - start_time
        self.perf_stats['filtering_time'] = duration
        self.logger.log_performance('filtering', duration, len(results))

        return results

    def _calculate_comprehensive_score(self, results: pd.DataFrame) -> np.ndarray:
        """计算综合评分"""
        n = len(results)
        final_scores = np.zeros(n)

        for i in range(n):
            similarity_score = results.loc[i, 'combined_score']

            admet_score = 0.0
            if 'lipinski' in results.columns and results.loc[i, 'lipinski']:
                admet_score += 0.25
            if 'veber' in results.columns and results.loc[i, 'veber']:
                admet_score += 0.25
            if 'caco2_pass' in results.columns and results.loc[i, 'caco2_pass']:
                admet_score += 0.25
            if 'bioavailability_pass' in results.columns and results.loc[i, 'bioavailability_pass']:
                admet_score += 0.25

            safety_score = 0.0
            if 'is_pains' in results.columns and not results.loc[i, 'is_pains']:
                safety_score += 0.33
            if 'is_safe' in results.columns and results.loc[i, 'is_safe']:
                safety_score += 0.33
            if 'nlrp3_compliant' in results.columns and results.loc[i, 'nlrp3_compliant']:
                safety_score += 0.34

            final_scores[i] = (
                    0.5 * similarity_score +
                    0.3 * admet_score +
                    0.2 * safety_score
            )

        return final_scores

    def perform_docking(self, results: pd.DataFrame, lib_df: pd.DataFrame) -> pd.DataFrame:
        """执行分子对接"""
        if not self.molecular_docking.enabled:
            self.logger.info("分子对接未启用，跳过")
            results['docking_affinity'] = 0.0
            results['docking_success'] = False
            return results

        self.logger.info("=" * 70)
        self.logger.info("步骤 6/7: 分子对接")
        self.logger.info("=" * 70)

        start_time = time.time()

        checkpoint_data = self.checkpoint_manager.load_checkpoint('docking')
        if checkpoint_data:
            self.logger.info("从断点恢复对接结果...")
            return checkpoint_data['results']

        max_dock = self.config["docking"].get("max_molecules", 50)
        top_results = results.head(max_dock)

        self.logger.info(f"对接前 {len(top_results)} 个化合物...")

        top_mols = []
        for idx in top_results.index:
            mol = lib_df.loc[lib_df['id'] == results.loc[idx, 'id'], 'mol'].iloc[0]
            top_mols.append(mol)

        docking_results = self.molecular_docking.batch_dock_molecules(
            top_mols,
            max_molecules=max_dock
        )

        results['docking_affinity'] = 0.0
        results['docking_success'] = False

        for i, dock_result in enumerate(docking_results):
            idx = top_results.index[i]
            results.loc[idx, 'docking_affinity'] = dock_result.get('affinity', 0.0)
            results.loc[idx, 'docking_success'] = dock_result.get('success', False)

        success_count = sum(results['docking_success'])
        self.logger.info(f"对接完成: {success_count}/{len(top_results)} 成功")

        if success_count > 0:
            docked = results[results['docking_success'] == True]
            affinities = docked['docking_affinity'].values
            self.logger.info(f"  - 最佳亲和力: {affinities.min():.2f} kcal/mol")
            self.logger.info(f"  - 平均亲和力: {affinities.mean():.2f} kcal/mol")

        self.checkpoint_manager.save_checkpoint('docking', {'results': results})

        duration = time.time() - start_time
        self.perf_stats['docking_time'] = duration
        self.logger.log_performance('docking', duration, success_count)

        return results

    def generate_reports(self, results: pd.DataFrame, ref_df: pd.DataFrame,
                         lib_df: pd.DataFrame, processing_time: float):
        """生成报告"""
        self.logger.info("=" * 70)
        self.logger.info("步骤 7/7: 生成报告")
        self.logger.info("=" * 70)

        start_time = time.time()

        results_csv = os.path.join(self.output_dir, self.config["output"]["hits_csv"])
        results.to_csv(results_csv, index=False)
        self.logger.info(f"CSV结果已保存: {results_csv}")

        self.text_reporter.generate_report(
            results, ref_df, lib_df, processing_time, self.perf_stats
        )

        if self.config["output"].get("generate_html_report", True):
            self.html_reporter.generate_report(
                results, lib_df, ref_df, processing_time, self.perf_stats
            )

        if self.config["output"].get("generate_plots", True):
            processing_info = {
                'processing_time_seconds': processing_time,
                'performance_stats': self.perf_stats,
                'system_info': {
                    'cpu_threads': self.cpu_pool.max_workers,
                    'gpu_enabled': self.gpu_manager is not None,
                },
                'data_stats': {
                    'total_compounds': len(results),
                    'reference_compounds': len(ref_df),
                    'library_compounds': len(lib_df),
                }
            }

            self.visualizer.generate_all_plots(results, processing_info)

        duration = time.time() - start_time
        self.perf_stats['reporting_time'] = duration
        self.logger.log_performance('reporting', duration)

    def run(self) -> pd.DataFrame:
        """运行完整的虚拟筛选流程"""
        try:
            ref_df, lib_df = self.load_and_preprocess_data()
            ref_features = self.extract_features(ref_df, "参考分子")
            lib_features = self.extract_features(lib_df, "库分子")
            similarities = self.compute_similarity(ref_features, lib_features)
            scores = self.aggregate_similarities(similarities)
            results = self.apply_filters(lib_df, scores, lib_features)
            results = self.perform_docking(results, lib_df)

            total_time = time.time() - self.start_time
            self.generate_reports(results, ref_df, lib_df, total_time)
            self._print_summary(results, total_time)

            return results

        except Exception as e:
            self.logger.error(f"虚拟筛选失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            self._cleanup()

    def _cleanup(self):
        """清理资源"""
        self.logger.info("清理资源...")

        if self.gpu_manager:
            self.gpu_manager.empty_cache()

        if self.feature_cache:
            self.feature_cache.clear()

        gc.collect()

        self.logger.info("资源清理完成")

    def _print_summary(self, results: pd.DataFrame, processing_time: float):
        """打印最终摘要"""
        threshold = self.config.get("ai_model", {}).get("similarity_threshold", 0.30)
        hits = results[results['final_score'] >= threshold] if len(results) > 0 else pd.DataFrame()

        print("\n" + "=" * 70)
        print(" NLRP3虚拟筛选完成")
        print("=" * 70)
        print(f"⏱️  处理时间: {processing_time:.2f} 秒")
        print(f"⚡ 处理速度: {len(results) / processing_time:.1f} 化合物/秒")
        print(f"📊 总化合物: {len(results):,}")
        print(f"🎯 命中数 (≥{threshold}): {len(hits):,}")
        print(f"🏆 最高得分: {results['final_score'].max():.4f}" if len(results) > 0 else "🏆 最高得分: N/A")
        print(f"📁 输出目录: {self.output_dir}")

        if 'nlrp3_compliant' in results.columns:
            nlrp3_pass = sum(results['nlrp3_compliant'])
            print(f"✅ NLRP3规则通过: {nlrp3_pass}/{len(results)} ({nlrp3_pass / len(results) * 100:.1f}%)")

        if 'is_safe' in results.columns:
            safe_count = sum(results['is_safe'])
            print(f"🛡️  安全性通过: {safe_count}/{len(results)} ({safe_count / len(results) * 100:.1f}%)")

        print("=" * 70)


# ============================================================================
# 工具函数（类外部）
# ============================================================================

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="NLRP3抑制剂超高速虚拟筛选系统 v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础运行
  python run_complete_screening.py --config configs/nlrp3_complete_config.yaml

  # 指定CPU线程数
  python run_complete_screening.py --config configs/nlrp3_complete_config.yaml --threads 32

  # 启用流式处理（大数据集）
  python run_complete_screening.py --config configs/nlrp3_complete_config.yaml --streaming

  # 清除断点重新运行
  python run_complete_screening.py --config configs/nlrp3_complete_config.yaml --clear-checkpoints
        """
    )

    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--threads", type=int, help="CPU线程数（默认自动检测）")
    parser.add_argument("--streaming", action="store_true", help="启用流式处理（大数据集）")
    parser.add_argument("--clear-checkpoints", action="store_true", help="清除所有断点")

    args = parser.parse_args()

    print("[INFO] 加载配置文件...")
    config = load_config(args.config)

    if args.streaming:
        config.setdefault("performance", {})["use_streaming"] = True
        print("[INFO] 流式处理已启用")

    if args.threads:
        config.setdefault("performance", {})["max_workers"] = args.threads
        print(f"[INFO] 设置CPU线程数: {args.threads}")

    screening = CompleteNLRP3VirtualScreening(config)

    if args.clear_checkpoints:
        screening.checkpoint_manager.clear_all_checkpoints()
        print("[INFO] 所有断点已清除")

    results = screening.run()

    return results


if __name__ == "__main__":
    main()