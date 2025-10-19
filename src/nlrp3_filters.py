#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLRP3抑制剂特异性过滤器模块
包含：NLRP3规则、ADMET预测、毒性预测
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from tqdm import tqdm


class NLRP3SpecificFilters:
    """NLRP3抑制剂特异性过滤器"""

    def __init__(self, config: dict):
        self.config = config
        self.nlrp3_rules = config.get("filters", {}).get("nlrp3_rules", {})
        self.admet_rules = config.get("filters", {}).get("admet_rules", {})
        self.toxicity_rules = config.get("filters", {}).get("toxicity_rules", {})

        print("[INFO] NLRP3特异性过滤器初始化完成")
        print(f"  - NLRP3规则: MW={self.nlrp3_rules.get('mw_min', 200)}-{self.nlrp3_rules.get('mw_max', 600)}, "
              f"LogP={self.nlrp3_rules.get('logp_min', 1.5)}-{self.nlrp3_rules.get('logp_max', 5.0)}")

    def apply_nlrp3_filters(self, mol: Chem.Mol, descriptors: Dict) -> Tuple[bool, List[str]]:
        """
        应用NLRP3特异性过滤器

        Args:
            mol: RDKit分子对象
            descriptors: 分子描述符字典

        Returns:
            (is_valid, fail_reasons): 是否通过，未通过的原因列表
        """
        if mol is None:
            return False, ["Invalid molecule"]

        fail_reasons = []

        # 1. 分子量检查
        mw = descriptors.get('mw', 0)
        if not (self.nlrp3_rules.get('mw_min', 200) <= mw <= self.nlrp3_rules.get('mw_max', 600)):
            fail_reasons.append(
                f"MW={mw:.1f} (要求: {self.nlrp3_rules.get('mw_min', 200)}-{self.nlrp3_rules.get('mw_max', 600)})")

        # 2. LogP检查（细胞膜通透性关键）
        logp = descriptors.get('logp', 0)
        if not (self.nlrp3_rules.get('logp_min', 1.5) <= logp <= self.nlrp3_rules.get('logp_max', 5.0)):
            fail_reasons.append(
                f"LogP={logp:.2f} (要求: {self.nlrp3_rules.get('logp_min', 1.5)}-{self.nlrp3_rules.get('logp_max', 5.0)})")

        # 3. 氢键供体/受体
        hbd = descriptors.get('hbd', 0)
        hba = descriptors.get('hba', 0)
        if hbd > self.nlrp3_rules.get('hbd_max', 3):
            fail_reasons.append(f"HBD={hbd} (最大: {self.nlrp3_rules.get('hbd_max', 3)})")
        if hba > self.nlrp3_rules.get('hba_max', 8):
            fail_reasons.append(f"HBA={hba} (最大: {self.nlrp3_rules.get('hba_max', 8)})")

        # 4. TPSA检查（膜通透性）
        tpsa = descriptors.get('tpsa', 0)
        if not (self.nlrp3_rules.get('tpsa_min', 40) <= tpsa <= self.nlrp3_rules.get('tpsa_max', 140)):
            fail_reasons.append(
                f"TPSA={tpsa:.1f} (要求: {self.nlrp3_rules.get('tpsa_min', 40)}-{self.nlrp3_rules.get('tpsa_max', 140)})")

        # 5. 可旋转键（柔性）
        rotb = descriptors.get('rotb', 0)
        if rotb > self.nlrp3_rules.get('rotb_max', 10):
            fail_reasons.append(f"RotB={rotb} (最大: {self.nlrp3_rules.get('rotb_max', 10)})")

        # 6. 芳香环数量（结合关键）
        aromatic_rings = descriptors.get('aromatic_rings', 0)
        if not (self.nlrp3_rules.get('aromatic_rings_min', 2) <= aromatic_rings <= self.nlrp3_rules.get(
                'aromatic_rings_max', 4)):
            fail_reasons.append(
                f"AromaticRings={aromatic_rings} (要求: {self.nlrp3_rules.get('aromatic_rings_min', 2)}-{self.nlrp3_rules.get('aromatic_rings_max', 4)})")

        # 7. 检查不良官能团
        if self._has_reactive_groups(mol):
            fail_reasons.append("含有反应性官能团")

        # 8. 检查线粒体毒性相关结构
        if self._check_mitochondrial_toxicity_alerts(mol):
            fail_reasons.append("可能具有线粒体毒性")

        is_valid = len(fail_reasons) == 0
        return is_valid, fail_reasons

    def _has_reactive_groups(self, mol: Chem.Mol) -> bool:
        """检查反应性官能团"""
        reactive_smarts = [
            '[N+](=O)[O-]',  # 硝基
            'C(=O)Cl',  # 酰氯
            'S(=O)(=O)Cl',  # 磺酰氯
            '[C,S](=[O,S])[F,Cl,Br,I]',  # 卤代羰基
            'C=N-N',  # 重氮
            'N=[N+]=[N-]',  # 叠氮
        ]

        for smarts in reactive_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True

        return False

    def _check_mitochondrial_toxicity_alerts(self, mol: Chem.Mol) -> bool:
        """检查线粒体毒性警报"""
        toxicity_smarts = [
            'c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]',  # 二硝基苯
            'ClC=CCl',  # 二氯乙烯
            'c1ccc(Cl)c(Cl)c1',  # 二氯苯
        ]

        for smarts in toxicity_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True

        return False

    def apply_admet_filters(self, mol: Chem.Mol, descriptors: Dict) -> Tuple[bool, Dict]:
        """
        应用ADMET过滤器

        Returns:
            (is_valid, admet_scores): 是否通过，ADMET评分字典
        """
        admet_scores = {}

        # 1. Lipinski五规则
        lipinski_pass = (
                descriptors.get('mw', 0) <= 500 and
                descriptors.get('logp', 0) <= 5 and
                descriptors.get('hbd', 0) <= 5 and
                descriptors.get('hba', 0) <= 10
        )
        admet_scores['lipinski'] = lipinski_pass

        # 2. Veber规则（口服生物利用度）
        veber_pass = (
                descriptors.get('rotb', 0) <= 10 and
                descriptors.get('tpsa', 0) <= 140
        )
        admet_scores['veber'] = veber_pass

        # 3. Ghose规则
        ghose_pass = (
                160 <= descriptors.get('mw', 0) <= 480 and
                -0.4 <= descriptors.get('logp', 0) <= 5.6 and
                40 <= descriptors.get('heavy_atoms', 0) <= 70
        )
        admet_scores['ghose'] = ghose_pass

        # 4. 预测Caco-2通透性
        caco2_logp = self._predict_caco2_permeability(descriptors)
        admet_scores['caco2_permeability'] = caco2_logp
        admet_scores['caco2_pass'] = caco2_logp >= self.admet_rules.get('caco2_permeability', -5.5)

        # 5. 预测口服生物利用度
        bioavailability = self._predict_oral_bioavailability(descriptors)
        admet_scores['bioavailability'] = bioavailability
        admet_scores['bioavailability_pass'] = bioavailability >= 0.3

        # 综合判断
        is_valid = (
                admet_scores['lipinski'] and
                admet_scores['veber'] and
                admet_scores['caco2_pass'] and
                admet_scores['bioavailability_pass']
        )

        return is_valid, admet_scores

    def _predict_caco2_permeability(self, descriptors: Dict) -> float:
        """预测Caco-2通透性（log Papp）"""
        logp = descriptors.get('logp', 0)
        tpsa = descriptors.get('tpsa', 0)
        mw = descriptors.get('mw', 0)

        # 经验公式
        log_papp = (0.8 * logp) - (0.01 * tpsa) - (0.0025 * mw) - 3.5

        return log_papp

    def _predict_oral_bioavailability(self, descriptors: Dict) -> float:
        """预测口服生物利用度（Abbott评分）"""
        mw = descriptors.get('mw', 0)
        logp = descriptors.get('logp', 0)
        tpsa = descriptors.get('tpsa', 0)
        rotb = descriptors.get('rotb', 0)
        aromatic_rings = descriptors.get('aromatic_rings', 0)

        score = 0.0

        # 分子量贡献
        if mw <= 500:
            score += 1.0
        elif mw <= 600:
            score += 0.5

        # LogP贡献
        if -0.5 <= logp <= 5.0:
            score += 1.0
        elif logp <= 6.0:
            score += 0.5

        # TPSA贡献
        if tpsa <= 140:
            score += 1.0
        elif tpsa <= 200:
            score += 0.5

        # 可旋转键贡献
        if rotb <= 10:
            score += 1.0
        elif rotb <= 15:
            score += 0.5

        # 芳香环贡献
        if 1 <= aromatic_rings <= 4:
            score += 1.0

        # 归一化到[0,1]
        bioavailability = score / 5.0

        return bioavailability

    def check_toxicity_alerts(self, mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """
        检查毒性警报

        Returns:
            (is_safe, alert_messages): 是否安全，警报信息列表
        """
        alerts = []

        # 1. 致突变性结构警报
        if self._check_mutagenicity_alerts(mol):
            alerts.append("潜在致突变性")

        # 2. 肝毒性结构警报
        if self._check_hepatotoxicity_alerts(mol):
            alerts.append("潜在肝毒性")

        # 3. 心脏毒性（hERG抑制）
        if self._check_herg_liability(mol):
            alerts.append("潜在hERG抑制")

        is_safe = len(alerts) == 0
        return is_safe, alerts

    def _check_mutagenicity_alerts(self, mol: Chem.Mol) -> bool:
        """检查致突变性警报"""
        mutagenic_smarts = [
            'c1ccccc1[N+](=O)[O-]',  # 硝基芳烃
            'NN',  # 肼类
            'N=N',  # 偶氮
            'C(=O)N([H])N',  # 酰肼
        ]

        for smarts in mutagenic_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True

        return False

    def _check_hepatotoxicity_alerts(self, mol: Chem.Mol) -> bool:
        """检查肝毒性警报"""
        hepatotoxic_smarts = [
            'c1ccc(N)cc1',  # 苯胺
            'c1ccccc1S',  # 芳香硫醚
            'CC(C)(C)c1ccccc1',  # 叔丁基苯
        ]

        for smarts in hepatotoxic_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True

        return False

    def _check_herg_liability(self, mol: Chem.Mol) -> bool:
        """检查hERG通道抑制风险"""
        # 检查碱性氮
        basic_nitrogen_pattern = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
        if not mol.HasSubstructMatch(basic_nitrogen_pattern):
            return False

        # 检查芳香环系统
        try:
            aromatic_rings = Chem.GetSSSR(mol)
            aromatic_count = sum(1 for ring in aromatic_rings
                                 if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
        except:
            aromatic_count = 0

        if aromatic_count < 2:
            return False

        # 检查分子大小
        mw = Descriptors.MolWt(mol)
        if mw < 300:
            return False

        return True

    def batch_filter_molecules(self, molecules: List[Chem.Mol],
                               descriptors_list: List[Dict]) -> Dict:
        """
        批量过滤分子

        Returns:
            results: {
                'nlrp3_compliant': List[bool],
                'nlrp3_fail_reasons': List[str],
                'admet_scores': List[Dict],
                'toxicity_alerts': List[str],
                'is_safe': List[bool]
            }
        """
        results = {
            'nlrp3_compliant': [],
            'nlrp3_fail_reasons': [],
            'admet_scores': [],
            'toxicity_alerts': [],
            'is_safe': []
        }

        for mol, descriptors in tqdm(zip(molecules, descriptors_list),
                                     total=len(molecules),
                                     desc="NLRP3过滤"):
            # NLRP3规则
            is_valid, fail_reasons = self.apply_nlrp3_filters(mol, descriptors)
            results['nlrp3_compliant'].append(is_valid)
            results['nlrp3_fail_reasons'].append("; ".join(fail_reasons) if fail_reasons else "Pass")

            # ADMET
            admet_valid, admet_scores = self.apply_admet_filters(mol, descriptors)
            results['admet_scores'].append(admet_scores)

            # 毒性
            is_safe, alerts = self.check_toxicity_alerts(mol)
            results['toxicity_alerts'].append("; ".join(alerts) if alerts else "Safe")
            results['is_safe'].append(is_safe)

        return results


class Brenk_Filter:
    """Brenk不良结构过滤器"""

    def __init__(self):
        # Brenk不良结构的SMARTS模式
        self.unwanted_patterns = [
            # 金属离子
            ('[Li,Na,K,Rb,Cs,Fr]', 'Alkali metal'),
            ('[Be,Mg,Ca,Sr,Ba,Ra]', 'Alkaline earth metal'),
            ('[Al,Ga,In,Tl]', 'Group 13 metal'),
            ('[Si]', 'Silicon'),
            ('[Ti,Zr,Hf]', 'Group 4 metal'),
            ('[V,Nb,Ta]', 'Group 5 metal'),
            ('[Cr,Mo,W]', 'Group 6 metal'),
            ('[Mn,Tc,Re]', 'Group 7 metal'),
            ('[Fe,Ru,Os]', 'Group 8 metal'),
            ('[Co,Rh,Ir]', 'Group 9 metal'),
            ('[Ni,Pd,Pt]', 'Group 10 metal'),
            ('[Cu,Ag,Au]', 'Group 11 metal'),
            ('[Zn,Cd,Hg]', 'Group 12 metal'),

            # 不良官能团
            ('[N+](=O)[O-]', 'Nitro group'),
            ('C(=O)Cl', 'Acyl chloride'),
            ('S(=O)(=O)Cl', 'Sulfonyl chloride'),
            ('[C,c]S(=O)(=O)O[C,c]', 'Sulfonic ester'),
            ('P(=O)([OH])[OH]', 'Phosphonic acid'),
            ('P(=O)([OH])([OH])[OH]', 'Phosphoric acid'),

            # 反应性基团
            ('C=C=O', 'Ketene'),
            ('C=C=C', 'Allene'),
            ('C#C', 'Alkyne'),
            ('[N,S]=[N+]=[N-]', 'Azide'),
            ('C=N-N', 'Hydrazone'),
            ('N-N=O', 'N-nitroso'),

            # 杂环系统问题
            ('c1ccccc1[N+](=O)[O-]', 'Nitrobenzene'),
            ('c1cc([N+](=O)[O-])ccc1', 'p-Nitrobenzene'),
        ]

    def check_molecule(self, mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """
        检查分子是否包含Brenk不良结构

        Returns:
            (is_clean, matched_patterns): 是否通过，匹配的不良结构列表
        """
        if mol is None:
            return False, ["Invalid molecule"]

        matched = []

        for smarts, description in self.unwanted_patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                matched.append(description)

        is_clean = len(matched) == 0
        return is_clean, matched