#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAINS和Brenk过滤器
检测泛激动剂和不良化学结构
"""

from typing import List, Tuple, Dict  # ✅ 添加 Dict
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from tqdm import tqdm


class PAINSFilter:
    """PAINS（Pan Assay Interference Compounds）过滤器"""

    def __init__(self):
        """初始化PAINS过滤器"""
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
            self.catalog = FilterCatalog(params)
            self.available = True
            print("[INFO] PAINS过滤器已加载")
        except Exception as e:
            print(f"[WARNING] PAINS过滤器初始化失败: {e}")
            self.available = False

    def check_molecule(self, mol: Chem.Mol) -> bool:
        """
        检查分子是否为PAINS

        Args:
            mol: RDKit分子对象

        Returns:
            True表示是PAINS（应过滤掉）
        """
        if not self.available or mol is None:
            return False

        try:
            return self.catalog.HasMatch(mol)
        except:
            return False

    def batch_check(self, molecules: List[Chem.Mol]) -> List[bool]:
        """批量PAINS检查"""
        if not self.available:
            return [False] * len(molecules)

        results = []
        for mol in tqdm(molecules, desc="PAINS检查"):
            results.append(self.check_molecule(mol))

        return results

    def __repr__(self):
        status = "available" if self.available else "unavailable"
        return f"PAINSFilter(status={status})"


class BrenkFilter:
    """Brenk不良结构过滤器"""

    def __init__(self):
        """初始化Brenk过滤器"""
        # Brenk不良结构的SMARTS模式
        self.unwanted_patterns = [
            # 金属离子
            ('[Li,Na,K,Rb,Cs,Fr]', 'Alkali_metal'),
            ('[Be,Mg,Ca,Sr,Ba,Ra]', 'Alkaline_earth_metal'),
            ('[Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn]', 'Transition_metal'),
            ('[Al,Ga,In,Tl]', 'Group_13_metal'),

            # 反应性官能团
            ('[N+](=O)[O-]', 'Nitro_group'),
            ('C(=O)Cl', 'Acyl_chloride'),
            ('S(=O)(=O)Cl', 'Sulfonyl_chloride'),
            ('[C,c]S(=O)(=O)O[C,c]', 'Sulfonic_ester'),
            ('P(=O)([OH])[OH]', 'Phosphonic_acid'),
            ('P(=O)([OH])([OH])[OH]', 'Phosphoric_acid'),

            # 高反应性基团
            ('C=C=O', 'Ketene'),
            ('C=C=C', 'Allene'),
            ('C#C', 'Terminal_alkyne'),
            ('[N,S]=[N+]=[N-]', 'Azide'),
            ('C=N-N', 'Hydrazone'),
            ('N-N=O', 'N_nitroso'),
            ('C(=O)O[N+](=O)[O-]', 'Nitrate_ester'),

            # 不良杂环
            ('c1ccccc1[N+](=O)[O-]', 'Nitrobenzene'),
            ('c1cc([N+](=O)[O-])ccc1', 'para_Nitrobenzene'),
            ('c1ccc([N+](=O)[O-])c([N+](=O)[O-])c1', 'Dinitrobenzene'),

            # 其他不良结构
            ('[SH]', 'Thiol'),
            ('C(=S)', 'Thiocarbonyl'),
            ('[Si]', 'Silicon'),
            ('[As,Sb,Bi]', 'Heavy_metalloid'),
            ('C#N', 'Nitrile'),
            ('N=C=N', 'Carbodiimide'),
        ]

        print(f"[INFO] Brenk过滤器已加载: {len(self.unwanted_patterns)} 个不良结构模式")

    def check_molecule(self, mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """
        检查分子是否包含Brenk不良结构

        Args:
            mol: RDKit分子对象

        Returns:
            (is_clean, matched_patterns): 是否干净，匹配的不良结构列表
        """
        if mol is None:
            return False, ["Invalid molecule"]

        matched = []

        for smarts, description in self.unwanted_patterns:
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    matched.append(description)
            except Exception:
                continue

        is_clean = len(matched) == 0
        return is_clean, matched

    def batch_check(self, molecules: List[Chem.Mol]) -> Dict:
        """批量Brenk检查"""
        results = {
            'brenk_clean': [],
            'brenk_alerts': []
        }

        for mol in tqdm(molecules, desc="Brenk过滤"):
            is_clean, alerts = self.check_molecule(mol)
            results['brenk_clean'].append(is_clean)
            results['brenk_alerts'].append("; ".join(alerts) if alerts else "Pass")

        return results

    def get_patterns(self) -> List[Tuple[str, str]]:
        """获取所有不良结构模式"""
        return self.unwanted_patterns.copy()

    def __repr__(self):
        return f"BrenkFilter({len(self.unwanted_patterns)} patterns)"


# 在 pains_brenk.py 末尾添加
class PAINSBrenkFilter:
    """PAINS和Brenk统一过滤器"""

    def __init__(self):
        """初始化过滤器"""
        self.pains_filter = PAINSFilter()
        self.brenk_filter = BrenkFilter()

    def batch_check_pains(self, molecules: List[Chem.Mol]) -> List[bool]:
        """批量PAINS检查"""
        return self.pains_filter.batch_check(molecules)

    def batch_check_brenk(self, molecules: List[Chem.Mol]) -> Dict:
        """批量Brenk检查"""
        return self.brenk_filter.batch_check(molecules)

    def check_molecule(self, mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """综合检查单个分子"""
        alerts = []

        # PAINS检查
        if self.pains_filter.check_molecule(mol):
            alerts.append("PAINS")

        # Brenk检查
        is_clean, brenk_alerts = self.brenk_filter.check_molecule(mol)
        if not is_clean:
            alerts.extend(brenk_alerts)

        is_valid = len(alerts) == 0
        return is_valid, alerts

    def __repr__(self):
        return "PAINSBrenkFilter(PAINS+Brenk)"