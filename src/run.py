#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows兼容版虚拟筛选脚本 - 无警告优化版
"""

import os
import argparse
import math
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import pickle
import warnings
import platform
import threading

# 完全禁用所有警告
warnings.filterwarnings('ignore')
os.environ['RDKIT_LOG_LEVEL'] = 'ERROR'

# RDKit相关导入
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, Lipinski
from rdkit import DataStructs
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem import rdShapeHelpers

# 禁用RDKit日志
RDLogger.DisableLog('rdApp.*')

# 使用线程池代替进程池（Windows兼容）
from concurrent.futures import ThreadPoolExecutor, as_completed

# 机器学习相关导入
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not available, ML features disabled")

# Mol2vec相关导入（可选）
try:
    from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
    from gensim.models import word2vec
    MOL2VEC_AVAILABLE = True
except ImportError:
    MOL2VEC_AVAILABLE = False
    print("[INFO] mol2vec not available, using Morgan fingerprints for 1D similarity")

# 线程本地存储
thread_local = threading.local()

def get_thread_resources():
    """获取线程本地资源"""
    if not hasattr(thread_local, 'pharm_factory'):
        thread_local.pharm_factory = Gobbi_Pharm2D.factory
    if not hasattr(thread_local, 'mol2vec_model'):
        thread_local.mol2vec_model = None
    return thread_local.pharm_factory, thread_local.mol2vec_model

def set_mol2vec_model(model_path):
    """设置mol2vec模型"""
    if model_path and os.path.exists(model_path):
        try:
            thread_local.mol2vec_model = word2vec.Word2Vec.load(model_path)
        except:
            thread_local.mol2vec_model = None

def load_config(path: str) -> dict:
    """加载配置文件"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_smiles_csv(path: str) -> pd.DataFrame:
    """读取SMILES CSV文件"""
    df = pd.read_csv(path)
    needed = {"id", "name", "smiles"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"{path} 必须包含列: {needed}")
    return df

def mol_from_smiles(smi: str):
    """从SMILES创建分子对象"""
    if not isinstance(smi, str) or not smi.strip():
        return None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None

# 尝试导入分子标准化模块
try:
    from rdkit.Chem import rdMolStandardize as rdMS
except ImportError:
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize as rdMS
    except ImportError:
        rdMS = None

def standardize_mol(m: Chem.Mol) -> Chem.Mol:
    """标准化分子"""
    if m is None:
        return None
    
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    
    m2 = m
    if rdMS is not None:
        try:
            normalizer = rdMS.Normalizer()
            reionizer = rdMS.Reionizer()
            uncharger = rdMS.Uncharger()
            m2 = normalizer.normalize(m2)
            m2 = reionizer.reionize(m2)
            m2 = uncharger.uncharge(m2)
        except Exception:
            m2 = m
    
    try:
        smi = Chem.MolToSmiles(m2, isomericSmiles=True)
        return Chem.MolFromSmiles(smi)
    except Exception:
        return m

def compute_morgan_fp(m: Chem.Mol, radius=2, nBits=2048):
    """计算Morgan指纹"""
    if m is None:
        return None
    try:
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
    except Exception:
        return None

def compute_pharm2d_fp(m: Chem.Mol, factory) -> DataStructs.ExplicitBitVect:
    """计算药效团指纹"""
    if m is None:
        from rdkit.DataStructs.cDataStructs import ExplicitBitVect
        return ExplicitBitVect(2048)
    
    try:
        return Generate.Gen2DFingerprint(m, factory)
    except Exception:
        from rdkit.DataStructs.cDataStructs import ExplicitBitVect
        return ExplicitBitVect(2048)

def compute_molecular_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """计算基本分子描述符（无警告版）"""
    if mol is None:
        return {}
    
    descriptors = {}
    
    # 基本描述符列表
    basic_descriptors = [
        ('mw', lambda m: Descriptors.MolWt(m)),
        ('logp', lambda m: Crippen.MolLogP(m)),
        ('tpsa', lambda m: rdMolDescriptors.CalcTPSA(m)),
        ('hbd', lambda m: Lipinski.NumHDonors(m)),
        ('hba', lambda m: Lipinski.NumHAcceptors(m)),
        ('rotb', lambda m: Lipinski.NumRotatableBonds(m)),
        ('aromatic_rings', lambda m: rdMolDescriptors.CalcNumAromaticRings(m)),
        ('saturated_rings', lambda m: rdMolDescriptors.CalcNumSaturatedRings(m)),
        ('hetero_atoms', lambda m: rdMolDescriptors.CalcNumHeteroatoms(m)),
        ('qed', lambda m: Descriptors.qed(m)),
    ]
    
    # 计算基本描述符
    for name, func in basic_descriptors:
        try:
            descriptors[name] = float(func(mol))
        except Exception:
            descriptors[name] = 0.0
    
    # 额外的安全描述符
    try:
        descriptors['num_rings'] = rdMolDescriptors.CalcNumRings(mol)
    except:
        descriptors['num_rings'] = 0
    
    try:
        descriptors['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
    except:
        descriptors['num_heavy_atoms'] = 0
    
    return descriptors

def pains_flag(m: Chem.Mol) -> bool:
    """检查PAINS"""
    if m is None:
        return False
    
    try:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
        catalog = FilterCatalog(params)
        return catalog.HasMatch(m)
    except Exception:
        return False

def physchem_props(m: Chem.Mol) -> Dict[str, float]:
    """计算理化性质"""
    if m is None:
        return dict(mw=0, tpsa=0, hbd=0, hba=0, rotb=0, logp=0)
    
    try:
        mw = Descriptors.MolWt(m)
        tpsa = rdMolDescriptors.CalcTPSA(m)
        hbd = Lipinski.NumHDonors(m)
        hba = Lipinski.NumHAcceptors(m)
        rotb = Lipinski.NumRotatableBonds(m)
        logp = Crippen.MolLogP(m)
        return dict(mw=mw, tpsa=tpsa, hbd=hbd, hba=hba, rotb=rotb, logp=logp)
    except Exception:
        return dict(mw=0, tpsa=0, hbd=0, hba=0, rotb=0, logp=0)

def cns_pass(props: Dict[str, float], rules: Dict[str, float]) -> bool:
    """CNS药物过滤器"""
    try:
        return (rules["mw_min"] <= props["mw"] <= rules["mw_max"] and
                props["tpsa"] <= rules["tpsa_max"] and
                props["hbd"] <= rules["hbd_max"] and
                props["hba"] <= rules["hba_max"] and
                props["rotb"] <= rules["rotb_max"] and
                rules["logp_min"] <= props["logp"] <= rules["logp_max"])
    except Exception:
        return False

def minmax_norm(arr: np.ndarray) -> np.ndarray:
    """最小-最大归一化"""
    if len(arr) == 0:
        return arr
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)

# Windows兼容的处理函数
def process_reference_molecule_thread(args):
    """线程安全的参考分子处理"""
    mol, radius, nbits, mol2vec_model_path = args
    
    # 设置mol2vec模型
    if mol2vec_model_path:
        set_mol2vec_model(mol2vec_model_path)
    
    pharm_factory, mol2vec_model = get_thread_resources()
    
    # Morgan指纹
    morgan_fp = compute_morgan_fp(mol, radius, nbits)
    
    # 药效团指纹  
    pharm2d_fp = compute_pharm2d_fp(mol, pharm_factory)
    
    # Mol2vec特征
    mol2vec_vec = None
    if mol2vec_model is not None:
        try:
            sentence = mol2alt_sentence(mol, 1)
            vec = sentences2vec([sentence], mol2vec_model, unseen='UNK')
            mol2vec_vec = vec[0] if len(vec) > 0 else None
        except:
            pass
    
    return morgan_fp, pharm2d_fp, mol2vec_vec

def process_library_molecule_thread(args):
    """线程安全的库分子处理"""
    mol, radius, nbits, mol2vec_model_path = args
    
    # 设置mol2vec模型
    if mol2vec_model_path:
        set_mol2vec_model(mol2vec_model_path)
    
    pharm_factory, mol2vec_model = get_thread_resources()
    
    # 基本指纹
    morgan_fp = compute_morgan_fp(mol, radius, nbits)
    pharm2d_fp = compute_pharm2d_fp(mol, pharm_factory)
    
    # Mol2vec特征
    mol2vec_vec = None
    if mol2vec_model is not None:
        try:
            sentence = mol2alt_sentence(mol, 1)
            vec = sentences2vec([sentence], mol2vec_model, unseen='UNK')
            mol2vec_vec = vec[0] if len(vec) > 0 else None
        except:
            pass
    
    # 分子描述符
    descriptors = compute_molecular_descriptors(mol)
    
    return morgan_fp, pharm2d_fp, mol2vec_vec, descriptors

def compute_similarity_batch(args):
    """批量计算相似性"""
    lib_morgan, lib_pharm2d, lib_mol2vec, ref_morgan_list, ref_pharm2d_list, ref_mol2vec_list, use_mol2vec = args
    
    # 1D相似性
    oneD_max = 0.0
    if use_mol2vec and lib_mol2vec is not None:
        for ref_vec in ref_mol2vec_list:
            if ref_vec is not None:
                try:
                    sim = cosine_similarity([lib_mol2vec], [ref_vec])[0][0]
                    oneD_max = max(oneD_max, sim)
                except:
                    continue
    else:
        if lib_morgan is not None:
            for ref_morgan in ref_morgan_list:
                if ref_morgan is not None:
                    try:
                        sim = DataStructs.TanimotoSimilarity(lib_morgan, ref_morgan)
                        oneD_max = max(oneD_max, sim)
                    except:
                        continue
    
    # 2D药效团相似性
    twoD_max = 0.0
    for ref_pharm2d in ref_pharm2d_list:
        try:
            sim = DataStructs.TanimotoSimilarity(lib_pharm2d, ref_pharm2d)
            twoD_max = max(twoD_max, sim)
        except:
            continue
    
    return oneD_max, twoD_max

def apply_ai_clustering(features: np.ndarray, n_clusters: int = 5) -> Tuple[np.ndarray, object]:
    """应用AI聚类分析"""
    if not SKLEARN_AVAILABLE:
        return np.zeros(len(features)), None
    
    try:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        return clusters, kmeans
    except Exception as e:
        print(f"[WARNING] Clustering failed: {e}")
        return np.zeros(len(features)), None

def detect_outliers(features: np.ndarray, contamination: float = 0.1) -> np.ndarray:
    """异常值检测"""
    if not SKLEARN_AVAILABLE:
        return np.ones(len(features))
    
    try:
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(features)
        return outliers  # 1: normal, -1: outlier
    except Exception as e:
        print(f"[WARNING] Outlier detection failed: {e}")
        return np.ones(len(features))

def evaluate_predictions(predictions: List[Dict], cfg: dict) -> Dict:
    """评估预测结果"""
    threshold = cfg.get("ai_model", {}).get("similarity_threshold", 0.75)
    
    hits = [p for p in predictions if p['score'] >= threshold]
    hit_rate = len(hits) / len(predictions) if len(predictions) > 0 else 0
    
    scores = [p['score'] for p in predictions]
    score_stats = {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'median': np.median(scores),
        'max': np.max(scores),
        'min': np.min(scores)
    }
    
    return {
        'total_compounds': len(predictions),
        'hits': len(hits),
        'hit_rate': hit_rate,
        'threshold': threshold,
        'score_statistics': score_stats
    }

def main(cfg: dict):
    # 生成唯一的时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_dir = cfg["output"]["dir"]
    unique_dir = f"{original_dir}_{timestamp}"
    cfg["output"]["dir"] = unique_dir
    
    print(f"[INFO] Windows兼容版虚拟筛选启动")
    print(f"[INFO] 输出目录: {unique_dir}")
    
    # 检测系统并设置并发数
    if platform.system() == "Windows":
        max_workers = min(4, os.cpu_count() or 1)  # Windows上限制线程数
        print(f"[INFO] Windows系统检测到，使用 {max_workers} 个线程")
    else:
        max_workers = min(8, os.cpu_count() or 1)
        print(f"[INFO] 使用 {max_workers} 个线程进行并行处理")
    
    # 读取数据
    print("[INFO] 读取数据文件...")
    ref_df = read_smiles_csv(cfg["data"]["references_csv"])
    lib_df = read_smiles_csv(cfg["data"]["library_csv"])
    print(f"[INFO] 参考分子: {len(ref_df)}, 库分子: {len(lib_df)}")

    # 检查Mol2vec模型
    mol2vec_model_path = cfg.get("mol2vec", {}).get("model_path")
    use_mol2vec = MOL2VEC_AVAILABLE and mol2vec_model_path and os.path.exists(mol2vec_model_path)
    if use_mol2vec:
        print(f"[INFO] 使用Mol2vec模型: {mol2vec_model_path}")
    else:
        print("[INFO] 使用Morgan指纹替代Mol2vec")
        mol2vec_model_path = None

    # 禁用3D形状相似性以提高速度
    print("[INFO] 为提高速度，禁用3D形状相似性")
    cfg["similarity"]["w_3d"] = 0.0

    # 分子预处理
    print("[INFO] 处理参考分子...")
    ref_df["mol"] = ref_df["smiles"].apply(mol_from_smiles).apply(standardize_mol)
    ref_df = ref_df[ref_df["mol"].notnull()].reset_index(drop=True)
    
    print("[INFO] 处理库分子...")
    lib_df["mol"] = lib_df["smiles"].apply(mol_from_smiles).apply(standardize_mol)
    lib_df = lib_df[lib_df["mol"].notnull()].reset_index(drop=True)
    
    if len(ref_df) == 0 or len(lib_df) == 0:
        raise ValueError("参考集或库为空或SMILES解析失败。请检查输入数据。")
    
    print(f"[INFO] 有效分子数 - 参考: {len(ref_df)}, 库: {len(lib_df)}")

    # 参数设置
    radius = cfg["fingerprints"]["morgan_radius"]
    nbits = cfg["fingerprints"]["morgan_nbits"]

    # 使用线程池计算参考分子特征
    print("[INFO] 计算参考分子特征...")
    ref_mol_data = [(mol, radius, nbits, mol2vec_model_path) for mol in ref_df["mol"]]
    
    ref_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_reference_molecule_thread, data): idx 
                        for idx, data in enumerate(ref_mol_data)}
        
        for future in tqdm(as_completed(future_to_idx), total=len(ref_mol_data), desc="参考分子特征"):
            idx = future_to_idx[future]
            try:
                result = future.result()
                ref_results.append((idx, result))
            except Exception as e:
                ref_results.append((idx, (None, None, None)))
    
    # 按索引排序结果
    ref_results.sort(key=lambda x: x[0])
    ref_morgan = [r[1][0] for r in ref_results]
    ref_pharm2d = [r[1][1] for r in ref_results]
    ref_mol2vec = [r[1][2] for r in ref_results] if use_mol2vec else []

    # 使用线程池计算库分子特征
    print("[INFO] 计算库分子特征...")
    lib_mol_data = [(mol, radius, nbits, mol2vec_model_path) for mol in lib_df["mol"]]
    
    lib_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_library_molecule_thread, data): idx 
                        for idx, data in enumerate(lib_mol_data)}
        
        for future in tqdm(as_completed(future_to_idx), total=len(lib_mol_data), desc="库分子特征"):
            idx = future_to_idx[future]
            try:
                result = future.result()
                lib_results.append((idx, result))
            except Exception as e:
                lib_results.append((idx, (None, None, None, {})))
    
    # 按索引排序结果
    lib_results.sort(key=lambda x: x[0])
    lib_morgan = [r[1][0] for r in lib_results]
    lib_pharm2d = [r[1][1] for r in lib_results]
    lib_mol2vec = [r[1][2] for r in lib_results] if use_mol2vec else []
    lib_descriptors = [r[1][3] for r in lib_results]

    # AI分析（如果启用）
    clusters = None
    outliers = None
    if SKLEARN_AVAILABLE and cfg.get("ai_model", {}).get("use_clustering", False):
        print("[INFO] 执行AI聚类分析...")
        # 准备特征矩阵
        feature_matrix = []
        for desc in lib_descriptors:
            if desc:
                feature_row = [desc.get(k, 0) for k in ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotb', 'aromatic_rings']]
                feature_matrix.append(feature_row)
            else:
                feature_matrix.append([0] * 7)
        
        feature_matrix = np.array(feature_matrix)
        clusters, kmeans_model = apply_ai_clustering(feature_matrix, n_clusters=5)
        
        if cfg.get("ai_model", {}).get("use_outlier_filter", False):
            print("[INFO] 执行异常值检测...")
            outliers = detect_outliers(feature_matrix, contamination=0.1)

    # 计算相似性（使用线程池）
    print("[INFO] 计算相似性...")
    similarity_args = [
        (lib_morgan[i], lib_pharm2d[i], lib_mol2vec[i] if use_mol2vec else None,
         ref_morgan, ref_pharm2d, ref_mol2vec if use_mol2vec else [],
         use_mol2vec)
        for i in range(len(lib_df))
    ]
    
    similarity_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(compute_similarity_batch, args): idx 
                        for idx, args in enumerate(similarity_args)}
        
        for future in tqdm(as_completed(future_to_idx), total=len(similarity_args), desc="计算相似性"):
            idx = future_to_idx[future]
            try:
                result = future.result()
                similarity_results.append((idx, result))
            except Exception as e:
                similarity_results.append((idx, (0.0, 0.0)))
    
    # 按索引排序结果
    similarity_results.sort(key=lambda x: x[0])
    s1_list = [r[1][0] for r in similarity_results]
    s2_list = [r[1][1] for r in similarity_results]
    s3_list = [0.0] * len(lib_df)  # 3D禁用

    # 计算过滤器和性质
    print("[INFO] 计算分子过滤器...")
    pains_list = []
    cns_list = []
    props_list = []
    
    for i, m in enumerate(tqdm(lib_df["mol"], desc="分子过滤器")):
        pflag = pains_flag(m) if cfg["filters"]["pains"] else False
        props = physchem_props(m)
        cflag = cns_pass(props, cfg["filters"]["cns_rules"]) if cfg["filters"]["cns_filter"] else True
        
        pains_list.append(pflag)
        cns_list.append(cflag)
        props_list.append(props)

    # 归一化和评分
    print("[INFO] 计算最终评分...")
    s1 = np.array(s1_list, dtype=float)
    s2 = np.array(s2_list, dtype=float)
    s3 = np.array(s3_list, dtype=float)

    s1n = minmax_norm(s1)
    s2n = minmax_norm(s2)
    s3n = minmax_norm(s3)

    w1, w2, w3 = cfg["similarity"]["w_1d"], cfg["similarity"]["w_2d"], cfg["similarity"]["w_3d"]
    combined = w1 * s1n + w2 * s2n + w3 * s3n

    # 组装结果
    out = lib_df[["id", "name", "smiles"]].copy()
    out["s1D_similarity"] = s1
    out["s2D_pharm2d"] = s2
    out["s3D_shape"] = s3
    out["s1D_norm"] = s1n
    out["s2D_norm"] = s2n
    out["s3D_norm"] = s3n
    out["score"] = combined
    out["pains"] = pains_list
    out["cns_pass"] = cns_list
    
    # 添加AI分析结果
    if clusters is not None:
        out["cluster"] = clusters
    if outliers is not None:
        out["outlier"] = outliers
    
    # 添加分子描述符
    for i, props in enumerate(props_list):
        for k, v in props.items():
            out.loc[i, k] = v
    
    # 添加额外描述符
    for i, desc in enumerate(lib_descriptors):
        for k, v in desc.items():
            if k not in out.columns:
                out.loc[i, k] = v

    out = out.sort_values("score", ascending=False).reset_index(drop=True)

    # 评估
    predictions = out.to_dict('records')
    evaluation = evaluate_predictions(predictions, cfg)

    # 保存结果
    print("[INFO] 保存结果...")
    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    hits_path = os.path.join(cfg["output"]["dir"], cfg["output"]["hits_csv"])
    out.to_csv(hits_path, index=False)

    # 生成报告
    report_lines = []
    report_lines.append("=== Windows兼容版 Mitophagy VS 报告 ===")
    report_lines.append(f"参考分子数: {len(ref_df)}")
    report_lines.append(f"库分子数: {len(lib_df)}")
    report_lines.append(f"使用线程数: {max_workers}")
    report_lines.append(f"权重设置: w1D={w1}, w2D={w2}, w3D={w3}")
    report_lines.append(f"特征提取: {'Mol2vec' if use_mol2vec else 'Morgan'} + 药效团")
    report_lines.append("")
    
    # 评估结果
    report_lines.append("=== 评估结果 ===")
    report_lines.append(f"总化合物数: {evaluation['total_compounds']}")
    report_lines.append(f"命中数 (>={evaluation['threshold']}): {evaluation['hits']}")
    report_lines.append(f"命中率: {evaluation['hit_rate']:.3f}")
    report_lines.append(f"得分统计: 均值={evaluation['score_statistics']['mean']:.3f}, "
                       f"中位数={evaluation['score_statistics']['median']:.3f}, "
                       f"最大值={evaluation['score_statistics']['max']:.3f}")
    report_lines.append("")
    
    # AI分析结果
    if clusters is not None:
        report_lines.append("=== AI聚类分析 ===")
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        for cluster_id, count in zip(unique_clusters, counts):
            report_lines.append(f"聚类 {cluster_id}: {count} 个化合物")
    
    if outliers is not None:
        n_outliers = np.sum(outliers == -1)
        report_lines.append(f"检测到异常值: {n_outliers} 个 ({n_outliers/len(outliers)*100:.1f}%)")
    
    # Top化合物
    report_lines.append("")
    report_lines.append("=== Top-20 候选化合物 ===")
    topk = out.head(20)
    for idx, (_, row) in enumerate(topk.iterrows(), 1):
        cluster_info = f", 聚类={row.get('cluster', 'N/A')}" if 'cluster' in row else ""
        outlier_info = f", 异常值={'是' if row.get('outlier', 1) == -1 else '否'}" if 'outlier' in row else ""
        report_lines.append(f"{idx:2d}. {row['id']} | {row['name'][:30]} | "
                           f"得分={row['score']:.3f} | PAINS={'是' if row['pains'] else '否'} | "
                           f"CNS={'通过' if row['cns_pass'] else '不通过'}{cluster_info}{outlier_info}")

    report_path = os.path.join(cfg["output"]["dir"], cfg["output"]["report_txt"])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 保存模型和特征（如果有AI分析）
    if clusters is not None and 'kmeans_model' in locals():
        model_path = os.path.join(cfg["output"]["dir"], "clustering_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(kmeans_model, f)

    print(f"[完成] Windows兼容版虚拟筛选完成")
    print(f"  - 详细结果: {hits_path}")
    print(f"  - 分析报告: {report_path}")
    print(f"[统计] 最高得分: {out.iloc[0]['score']:.3f}")
    print(f"[统计] 命中率: {evaluation['hit_rate']:.3f} ({evaluation['hits']}/{evaluation['total_compounds']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Windows兼容版虚拟筛选")
    parser.add_argument("--config", type=str, default="config_paper.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    main(cfg)