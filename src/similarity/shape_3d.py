#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D形状相似性计算
支持多种3D对齐和形状比较方法
"""

import numpy as np
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import warnings

# 3D形状相似性
try:
    from rdkit.Chem import rdShapeHelpers
    from rdkit.Chem.rdMolAlign import AlignMol
    SHAPE3D_AVAILABLE = True
except ImportError:
    SHAPE3D_AVAILABLE = False


class Shape3DSimilarity:
    """3D形状相似性计算器"""

    def __init__(self, max_conformers: int = 1, optimize_conformers: bool = True):
        """
        初始化3D形状相似性计算器

        Args:
            max_conformers: 每个分子生成的最大构象数
            optimize_conformers: 是否优化构象
        """
        self.max_conformers = max_conformers
        self.optimize_conformers = optimize_conformers
        self.available = SHAPE3D_AVAILABLE

        if self.available:
            print(f"[INFO] 3D形状相似性: 启用 (max_conformers={max_conformers})")
        else:
            print("[WARNING] 3D形状相似性: 不可用")

    def calculate_similarity(self, ref_mols: List[Chem.Mol],
                             lib_mols: List[Chem.Mol]) -> np.ndarray:
        """
        计算3D形状相似性矩阵

        Args:
            ref_mols: 参考分子列表
            lib_mols: 库分子列表

        Returns:
            相似性矩阵 (n_lib, n_ref)
        """
        if not self.available:
            print("[WARNING] 3D形状相似性不可用，返回零矩阵")
            return np.zeros((len(lib_mols), len(ref_mols)), dtype=np.float32)

        print("[INFO] 计算3D形状相似性...")

        n_lib, n_ref = len(lib_mols), len(ref_mols)
        similarity = np.zeros((n_lib, n_ref), dtype=np.float32)

        # 策略1: 快速批量计算
        try:
            print("[DEBUG] 批量生成3D构象...")
            ref_mols_3d = self._batch_generate_conformers(ref_mols)
            lib_mols_3d = self._batch_generate_conformers(lib_mols)

            # 计算所有分子对
            computed = 0
            for i in tqdm(range(n_lib), desc="3D相似性计算"):
                lib_mol = lib_mols_3d[i]
                if lib_mol is None:
                    continue

                for j in range(n_ref):
                    ref_mol = ref_mols_3d[j]
                    if ref_mol is None:
                        continue

                    try:
                        sim = self._compute_shape_similarity(lib_mol, ref_mol)
                        similarity[i, j] = sim
                        computed += 1
                    except Exception:
                        continue

            print(f"[INFO] 3D相似性计算完成: {computed}/{n_lib * n_ref} 对")

            # 验证结果
            non_zero = np.count_nonzero(similarity)
            print(f"[DEBUG] 3D相似性非零数: {non_zero} ({non_zero / similarity.size * 100:.2f}%)")
            print(f"[DEBUG] 3D相似性范围: {similarity.min():.4f} - {similarity.max():.4f}")

            if non_zero > 0:
                return similarity

        except Exception as e:
            print(f"[WARNING] 3D批量计算失败: {e}")

        # 策略2: 备用方法
        return self._fallback_calculation(ref_mols, lib_mols)

    # ✅ 添加接口方法
    def compute_shape_similarity(self, ref_mols: List[Chem.Mol],
                                  lib_mols: List[Chem.Mol]) -> np.ndarray:
        """
        计算形状相似性（主接口方法）

        Args:
            ref_mols: 参考分子列表
            lib_mols: 库分子列表

        Returns:
            相似性矩阵 (n_lib, n_ref)
        """
        return self.calculate_similarity(ref_mols, lib_mols)

    def _batch_generate_conformers(self, mols: List[Chem.Mol]) -> List[Optional[Chem.Mol]]:
        """批量生成分子3D构象"""
        mols_3d = []
        success_count = 0

        for mol in mols:
            if mol is None:
                mols_3d.append(None)
                continue

            try:
                mol_h = Chem.AddHs(mol)

                # 检查是否已有构象
                if mol_h.GetNumConformers() > 0:
                    mols_3d.append(mol_h)
                    success_count += 1
                    continue

                # 生成构象
                ps = AllChem.ETKDGv3()
                ps.randomSeed = 42
                ps.numThreads = 1
                ps.maxAttempts = 5

                if self.max_conformers == 1:
                    conf_id = AllChem.EmbedMolecule(mol_h, params=ps)
                    success = (conf_id >= 0)
                else:
                    conf_ids = AllChem.EmbedMultipleConfs(
                        mol_h,
                        numConfs=self.max_conformers,
                        params=ps
                    )
                    success = len(conf_ids) > 0

                if success:
                    # 优化构象
                    if self.optimize_conformers:
                        if self.max_conformers == 1:
                            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=50)
                        else:
                            for conf_id in range(mol_h.GetNumConformers()):
                                AllChem.MMFFOptimizeMolecule(mol_h, confId=conf_id, maxIters=50)

                    mols_3d.append(mol_h)
                    success_count += 1
                else:
                    mols_3d.append(None)

            except Exception:
                mols_3d.append(None)

        print(f"[INFO] 构象生成成功: {success_count}/{len(mols)}")
        return mols_3d

    def _compute_shape_similarity(self, mol1: Chem.Mol, mol2: Chem.Mol) -> float:
        """计算两个分子的形状相似性"""
        if not mol1 or not mol2:
            return 0.0

        if mol1.GetNumConformers() == 0 or mol2.GetNumConformers() == 0:
            return 0.0

        try:
            max_sim = 0.0

            conf_count1 = min(self.max_conformers, mol1.GetNumConformers())
            conf_count2 = min(self.max_conformers, mol2.GetNumConformers())

            for conf1 in range(conf_count1):
                for conf2 in range(conf_count2):
                    try:
                        dist = rdShapeHelpers.ShapeTanimotoDist(mol1, mol2, conf1, conf2)
                        sim = 1.0 - dist
                        max_sim = max(max_sim, sim)
                    except:
                        continue

            return max(0.0, min(1.0, max_sim))

        except Exception:
            return 0.0

    def _fallback_calculation(self, ref_mols: List[Chem.Mol],
                              lib_mols: List[Chem.Mol]) -> np.ndarray:
        """备用计算方法"""
        print("[DEBUG] 使用备用3D计算方法...")

        n_lib, n_ref = len(lib_mols), len(ref_mols)
        similarity = np.zeros((n_lib, n_ref), dtype=np.float32)

        computed = 0
        for i in range(n_lib):
            for j in range(n_ref):
                try:
                    sim = self._compute_shape_similarity_robust(lib_mols[i], ref_mols[j])
                    similarity[i, j] = sim
                    computed += 1
                except:
                    continue

        print(f"[INFO] 备用计算完成: {computed} 对")
        return similarity

    def _compute_shape_similarity_robust(self, mol1: Chem.Mol, mol2: Chem.Mol) -> float:
        """鲁棒的形状相似性计算"""
        if not mol1 or not mol2:
            return 0.0

        try:
            mol1_h = Chem.AddHs(mol1)
            mol2_h = Chem.AddHs(mol2)

            if mol1_h.GetNumConformers() == 0:
                ps = AllChem.ETKDGv3()
                ps.randomSeed = 42
                AllChem.EmbedMolecule(mol1_h, params=ps)

            if mol2_h.GetNumConformers() == 0:
                ps = AllChem.ETKDGv3()
                ps.randomSeed = 42
                AllChem.EmbedMolecule(mol2_h, params=ps)

            if mol1_h.GetNumConformers() == 0 or mol2_h.GetNumConformers() == 0:
                return 0.0

            dist = rdShapeHelpers.ShapeTanimotoDist(mol1_h, mol2_h, 0, 0)
            sim = 1.0 - dist

            return max(0.0, min(1.0, sim))

        except Exception:
            return 0.0

    def __repr__(self):
        status = "enabled" if self.available else "disabled"
        return f"Shape3DSimilarity(status={status}, max_conformers={self.max_conformers})"