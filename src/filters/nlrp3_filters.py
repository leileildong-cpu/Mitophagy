#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLRP3抑制剂特异性过滤器
基于已知NLRP3抑制剂的理化性质特征
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from tqdm import tqdm


class NLRP3Filters:
    """NLRP3抑制剂特异性过滤器"""

    # NLRP3抑制剂典型特征（基于文献）
    NLRP3_PROFILE = {
        'mw_range': (200, 600),
        'logp_range': (1.5, 5.0),
        'hbd_max': 3,
        'hba_max': 8,
        'tpsa_range': (40, 140),
        'rotb_max': 10,
        'aromatic_rings_range': (2, 4)
    }

    def __init__(self, config: dict):
        """
        初始化NLRP3过滤器

        Args:
            config: 配置字典
        """
        self.config = config
        self.rules = config.get("filters", {}).get("nlrp3_rules", self.NLRP3_PROFILE)

        print("[INFO] NLRP3特异性过滤器初始化")
        print(f"  - 分子量: {self.rules.get('mw_min', 200)}-{self.rules.get('mw_max', 600)}")
        print(f"  - LogP: {self.rules.get('logp_min', 1.5)}-{self.rules.get('logp_max', 5.0)}")
        print(f"  - TPSA: {self.rules.get('tpsa_min', 40)}-{self.rules.get('tpsa_max', 140)}")

    def check_molecule(self, mol: Chem.Mol, descriptors: Dict) -> Tuple[bool, List[str]]:
        """
        检查单个分子是否符合NLRP3规则

        Args:
            mol: RDKit分子对象
            descriptors: 分子描述符字典

        Returns:
            (is_compliant, fail_reasons): 是否合规，不合规原因列表
        """
        if mol is None:
            return False, ["Invalid molecule"]

        fail_reasons = []

        # 1. 分子量
        mw = descriptors.get('mw', 0)
        mw_min = self.rules.get('mw_min', 200)
        mw_max = self.rules.get('mw_max', 600)
        if not (mw_min <= mw <= mw_max):
            fail_reasons.append(f"MW={mw:.1f} (要求: {mw_min}-{mw_max})")

        # 2. LogP（脂溶性）
        logp = descriptors.get('logp', 0)
        logp_min = self.rules.get('logp_min', 1.5)
        logp_max = self.rules.get('logp_max', 5.0)
        if not (logp_min <= logp <= logp_max):
            fail_reasons.append(f"LogP={logp:.2f} (要求: {logp_min}-{logp_max})")

        # 3. 氢键供体
        hbd = descriptors.get('hbd', 0)
        hbd_max = self.rules.get('hbd_max', 3)
        if hbd > hbd_max:
            fail_reasons.append(f"HBD={hbd} (最大: {hbd_max})")

        # 4. 氢键受体
        hba = descriptors.get('hba', 0)
        hba_max = self.rules.get('hba_max', 8)
        if hba > hba_max:
            fail_reasons.append(f"HBA={hba} (最大: {hba_max})")

        # 5. 极性表面积
        tpsa = descriptors.get('tpsa', 0)
        tpsa_min = self.rules.get('tpsa_min', 40)
        tpsa_max = self.rules.get('tpsa_max', 140)
        if not (tpsa_min <= tpsa <= tpsa_max):
            fail_reasons.append(f"TPSA={tpsa:.1f} (要求: {tpsa_min}-{tpsa_max})")

        # 6. 可旋转键
        rotb = descriptors.get('rotb', 0)
        rotb_max = self.rules.get('rotb_max', 10)
        if rotb > rotb_max:
            fail_reasons.append(f"RotB={rotb} (最大: {rotb_max})")

        # 7. 芳香环数量
        aromatic_rings = descriptors.get('aromatic_rings', 0)
        ar_min = self.rules.get('aromatic_rings_min', 2)
        ar_max = self.rules.get('aromatic_rings_max', 4)
        if not (ar_min <= aromatic_rings <= ar_max):
            fail_reasons.append(f"AromaticRings={aromatic_rings} (要求: {ar_min}-{ar_max})")

        # 8. 检查反应性官能团
        if self._has_reactive_groups(mol):
            fail_reasons.append("含有反应性官能团")

        # 9. 检查线粒体毒性相关结构
        if self._check_mitochondrial_toxicity(mol):
            fail_reasons.append("可能具有线粒体毒性")

        is_compliant = len(fail_reasons) == 0
        return is_compliant, fail_reasons

    def _has_reactive_groups(self, mol: Chem.Mol) -> bool:
        """检查反应性官能团"""
        reactive_smarts = [
            '[N+](=O)[O-]',  # 硝基
            'C(=O)Cl',  # 酰氯
            'S(=O)(=O)Cl',  # 磺酰氯
            '[C,S](=[O,S])[F,Cl,Br,I]',  # 卤代羰基
            'C=N-N',  # 重氮
            'N=[N+]=[N-]',  # 叠氮
            '[C,c]S(=O)(=O)O',  # 磺酸
            'C(=O)O[N+](=O)[O-]',  # 硝酸酯
        ]

        for smarts in reactive_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True

        return False

    def _check_mitochondrial_toxicity(self, mol: Chem.Mol) -> bool:
        """检查线粒体毒性警报"""
        toxicity_smarts = [
            'c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]',  # 二硝基苯
            'ClC=CCl',  # 二氯乙烯
            'c1ccc(Cl)c(Cl)c1',  # 二氯苯
            'c1ccc(cc1)N(=O)=O',  # 硝基苯
        ]

        for smarts in toxicity_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True

        return False

    def batch_check(self, molecules: List[Chem.Mol],
                    descriptors_list: List[Dict]) -> Dict:
        """
        批量检查分子

        Args:
            molecules: 分子列表
            descriptors_list: 描述符列表

        Returns:
            检查结果字典
        """
        results = {
            'nlrp3_compliant': [],
            'nlrp3_fail_reasons': []
        }

        for mol, desc in tqdm(zip(molecules, descriptors_list),
                              total=len(molecules),
                              desc="NLRP3规则检查"):
            is_compliant, reasons = self.check_molecule(mol, desc)
            results['nlrp3_compliant'].append(is_compliant)
            results['nlrp3_fail_reasons'].append("; ".join(reasons) if reasons else "Pass")

        return results

    def get_compliance_rate(self, results: Dict) -> float:
        """
        计算合规率

        Args:
            results: 批量检查结果

        Returns:
            合规率 (0-1)
        """
        compliant_count = sum(results['nlrp3_compliant'])
        total_count = len(results['nlrp3_compliant'])

        return compliant_count / total_count if total_count > 0 else 0.0

    def __repr__(self):
        return f"NLRP3Filters(rules={len(self.rules)} criteria)"