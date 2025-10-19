#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
毒性检查器
检查致突变性、肝毒性、心脏毒性等
"""

from typing import List, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import Descriptors


class ToxicityChecker:
    """毒性检查器"""

    def __init__(self, config: dict):
        """
        初始化毒性检查器

        Args:
            config: 配置字典
        """
        self.config = config
        self.toxicity_rules = config.get("filters", {}).get("toxicity_rules", {})

        print("[INFO] 毒性检查器初始化")

    def check_toxicity(self, mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """
        综合毒性检查

        Args:
            mol: RDKit分子对象

        Returns:
            (is_safe, alerts): 是否安全，毒性警报列表
        """
        if mol is None:
            return False, ["Invalid molecule"]

        alerts = []

        # 1. 致突变性
        if self.toxicity_rules.get('mutagenicity', True):
            if self._check_mutagenicity(mol):
                alerts.append("潜在致突变性")

        # 2. 肝毒性
        if self.toxicity_rules.get('hepatotoxicity', True):
            if self._check_hepatotoxicity(mol):
                alerts.append("潜在肝毒性")

        # 3. 心脏毒性（hERG抑制）
        if self.toxicity_rules.get('cardiotoxicity', True):
            if self._check_herg_liability(mol):
                alerts.append("潜在hERG抑制")

        # 4. 肾毒性
        if self._check_nephrotoxicity(mol):
            alerts.append("潜在肾毒性")

        # 5. 线粒体毒性
        if self._check_mitochondrial_toxicity(mol):
            alerts.append("潜在线粒体毒性")

        is_safe = len(alerts) == 0
        return is_safe, alerts

    def _check_mutagenicity(self, mol: Chem.Mol) -> bool:
        """检查致突变性（Ames阳性结构）"""
        mutagenic_smarts = [
            'c1ccccc1[N+](=O)[O-]',  # 硝基芳烃
            'NN',  # 肼类
            'N=N',  # 偶氮
            'C(=O)N([H])N',  # 酰肼
            'c1ccccc1N',  # 芳香胺（某些）
            'N=C=O',  # 异氰酸酯
            'N=C=S',  # 异硫氰酸酯
        ]

        for smarts in mutagenic_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True

        return False

    def _check_hepatotoxicity(self, mol: Chem.Mol) -> bool:
        """检查肝毒性警报"""
        hepatotoxic_smarts = [
            'c1ccc(N)cc1',  # 苯胺
            'c1ccccc1S',  # 芳香硫醚
            'CC(C)(C)c1ccccc1',  # 叔丁基苯
            'c1ccc(cc1)C(F)(F)F',  # 三氟甲基苯
            'c1ccc(O)cc1',  # 酚类（某些）
        ]

        for smarts in hepatotoxic_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True

        return False

    def _check_herg_liability(self, mol: Chem.Mol) -> bool:
        """
        检查hERG通道抑制风险
        hERG抑制剂特征：碱性氮 + 疏水芳香环 + 合适分子大小

        Args:
            mol: RDKit分子对象

        Returns:
            是否有hERG风险
        """
        # 1. 检查碱性氮
        basic_nitrogen = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
        if not mol.HasSubstructMatch(basic_nitrogen):
            return False

        # 2. 检查芳香环系统
        try:
            aromatic_rings = Chem.GetSSSR(mol)
            aromatic_count = sum(1 for ring in aromatic_rings
                                 if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
        except:
            aromatic_count = 0

        if aromatic_count < 2:
            return False

        # 3. 检查分子大小
        mw = Descriptors.MolWt(mol)
        if mw < 300:
            return False

        # 4. 检查已知hERG毒性基团
        herg_alerts = [
            'c1ccc(cc1)N(C)C',  # N,N-二甲基苯胺
            'c1ccc(cc1)CCN',  # 苯乙胺
        ]

        for smarts in herg_alerts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True

        return True  # 满足所有条件，可能有风险

    def _check_nephrotoxicity(self, mol: Chem.Mol) -> bool:
        """检查肾毒性警报"""
        nephrotoxic_smarts = [
            '[C,c]S(=O)(=O)N',  # 磺胺类
            'c1ccc([N+](=O)[O-])cc1',  # 硝基芳烃
        ]

        for smarts in nephrotoxic_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True

        return False

    def _check_mitochondrial_toxicity(self, mol: Chem.Mol) -> bool:
        """检查线粒体毒性"""
        mito_toxic_smarts = [
            'c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]',  # 二硝基苯
            'ClC=CCl',  # 二氯乙烯
        ]

        for smarts in mito_toxic_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True

        return False

    def batch_check(self, molecules: list, descriptors_list: list) -> Dict:
        """批量毒性检查"""
        results = {
            'toxicity_alerts': [],
            'is_safe': []
        }

        for mol, desc in zip(molecules, descriptors_list):
            is_safe, alerts = self.check_toxicity(mol)
            results['toxicity_alerts'].append("; ".join(alerts) if alerts else "Safe")
            results['is_safe'].append(is_safe)

        return results

    def __repr__(self):
        return "ToxicityChecker(mutagenicity+hepatotoxicity+cardiotoxicity)"