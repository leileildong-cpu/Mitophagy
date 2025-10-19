#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAINS和Brenk过滤器
检测泛激动剂和不良化学结构
"""

from typing import List, Tuple, Dict
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

    def batch_check(self, molecules: List[Chem.Mol]) -> List[Dict[str, any]]:
        """
        批量Brenk检查

        Args:
            molecules: RDKit分子对象列表

        Returns:
            list: 字典列表，每个字典包含 'is_clean' 和 'alerts' 键
        """
        results = []

        for mol in tqdm(molecules, desc="Brenk过滤"):
            try:
                is_clean, alerts = self.check_molecule(mol)
                results.append({
                    'is_clean': is_clean,
                    'alerts': "; ".join(alerts) if alerts else "Pass"
                })
            except Exception as e:
                results.append({
                    'is_clean': False,
                    'alerts': f"Check failed: {str(e)}"
                })

        return results

    def get_patterns(self) -> List[Tuple[str, str]]:
        """获取所有不良结构模式"""
        return self.unwanted_patterns.copy()

    def __repr__(self):
        return f"BrenkFilter({len(self.unwanted_patterns)} patterns)"


class PAINSBrenkFilter:
    """PAINS和Brenk统一过滤器"""

    def __init__(self):
        """初始化过滤器"""
        self.pains_filter = PAINSFilter()
        self.brenk_filter = BrenkFilter()

    def batch_check_pains(self, molecules: List[Chem.Mol]) -> List[bool]:
        """
        批量PAINS检查

        Args:
            molecules: RDKit分子对象列表

        Returns:
            list: 布尔值列表，True表示是PAINS（应过滤）
        """
        return self.pains_filter.batch_check(molecules)

    def batch_check_brenk(self, molecules: List[Chem.Mol]) -> List[Dict[str, any]]:
        """
        批量Brenk检查

        Args:
            molecules: RDKit分子对象列表

        Returns:
            list: 字典列表，每个字典包含：
                - 'is_clean': bool, 是否通过检查
                - 'alerts': str, 警报信息
        """
        return self.brenk_filter.batch_check(molecules)

    def check_molecule(self, mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """
        综合检查单个分子

        Args:
            mol: RDKit分子对象

        Returns:
            (is_valid, alerts): 是否有效，警报列表
        """
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

    def batch_check_comprehensive(self, molecules: List[Chem.Mol]) -> List[Dict[str, any]]:
        """
        综合批量检查（PAINS + Brenk）

        Args:
            molecules: RDKit分子对象列表

        Returns:
            list: 字典列表，包含所有检查结果
        """
        results = []

        for mol in tqdm(molecules, desc="综合过滤检查"):
            try:
                is_valid, alerts = self.check_molecule(mol)

                # PAINS详细结果
                is_pains = self.pains_filter.check_molecule(mol)

                # Brenk详细结果
                is_brenk_clean, brenk_alerts = self.brenk_filter.check_molecule(mol)

                results.append({
                    'overall_valid': is_valid,
                    'all_alerts': "; ".join(alerts) if alerts else "Pass",
                    'is_pains': is_pains,
                    'is_brenk_clean': is_brenk_clean,
                    'brenk_alerts': "; ".join(brenk_alerts) if brenk_alerts else "Pass"
                })
            except Exception as e:
                results.append({
                    'overall_valid': False,
                    'all_alerts': f"Check failed: {str(e)}",
                    'is_pains': True,
                    'is_brenk_clean': False,
                    'brenk_alerts': f"Check failed: {str(e)}"
                })

        return results

    def __repr__(self):
        return "PAINSBrenkFilter(PAINS+Brenk)"


# 便捷函数
def create_filter() -> PAINSBrenkFilter:
    """创建PAINS+Brenk过滤器实例"""
    return PAINSBrenkFilter()


def quick_check_pains(smiles: str) -> bool:
    """快速检查SMILES是否为PAINS"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True  # 无效分子视为PAINS

    filter_obj = PAINSFilter()
    return filter_obj.check_molecule(mol)


def quick_check_brenk(smiles: str) -> Tuple[bool, List[str]]:
    """快速检查SMILES的Brenk结构"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, ["Invalid SMILES"]

    filter_obj = BrenkFilter()
    return filter_obj.check_molecule(mol)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("PAINS和Brenk过滤器测试")
    print("=" * 60)

    # 测试分子
    test_smiles = [
        "CCO",  # 乙醇（应该通过）
        "c1ccccc1[N+](=O)[O-]",  # 硝基苯（应该被Brenk标记）
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # 布洛芬（应该通过）
    ]

    filter_obj = PAINSBrenkFilter()

    print("\n单分子测试:")
    for smiles in test_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            is_valid, alerts = filter_obj.check_molecule(mol)
            print(f"  SMILES: {smiles}")
            print(f"  结果: {'通过' if is_valid else '未通过'}")
            if alerts:
                print(f"  警报: {', '.join(alerts)}")
            print()

    print("\n批量测试:")
    mols = [Chem.MolFromSmiles(s) for s in test_smiles]
    mols = [m for m in mols if m is not None]

    # PAINS批量检查
    pains_results = filter_obj.batch_check_pains(mols)
    print(f"PAINS检查: {sum(pains_results)}/{len(pains_results)} 个分子为PAINS")

    # Brenk批量检查
    brenk_results = filter_obj.batch_check_brenk(mols)
    brenk_clean_count = sum(r['is_clean'] for r in brenk_results)
    print(f"Brenk检查: {brenk_clean_count}/{len(brenk_results)} 个分子通过")

    print("\n测试完成!")