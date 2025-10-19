#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADMET预测器
预测吸收、分布、代谢、排泄、毒性
"""

import numpy as np
from typing import Dict, Tuple, List


class ADMETPredictor:
    """ADMET性质预测器"""

    def __init__(self, config: dict):
        """
        初始化ADMET预测器

        Args:
            config: 配置字典
        """
        self.config = config
        self.admet_rules = config.get("filters", {}).get("admet_rules", {})

        print("[INFO] ADMET预测器初始化")

    def predict(self, mol, descriptors: Dict) -> Tuple[bool, Dict]:
        """
        预测分子的ADMET性质

        Args:
            mol: RDKit分子对象
            descriptors: 分子描述符

        Returns:
            (is_valid, admet_scores): 是否通过，ADMET评分字典
        """
        admet_scores = {}

        # 1. Lipinski五规则
        admet_scores['lipinski'] = self._check_lipinski(descriptors)

        # 2. Veber规则
        admet_scores['veber'] = self._check_veber(descriptors)

        # 3. Ghose规则
        admet_scores['ghose'] = self._check_ghose(descriptors)

        # 4. Caco-2通透性预测
        caco2 = self._predict_caco2(descriptors)
        admet_scores['caco2_permeability'] = caco2
        admet_scores['caco2_pass'] = caco2 >= self.admet_rules.get('caco2_permeability', -5.5)

        # 5. 口服生物利用度预测
        bioavail = self._predict_bioavailability(descriptors)
        admet_scores['bioavailability'] = bioavail
        admet_scores['bioavailability_pass'] = bioavail >= 0.3

        # 6. 血脑屏障通透性预测（BBB）
        bbb = self._predict_bbb_permeability(descriptors)
        admet_scores['bbb_permeability'] = bbb
        admet_scores['bbb_pass'] = bbb > 0

        # 7. 人血浆蛋白结合率预测
        ppb = self._predict_plasma_protein_binding(descriptors)
        admet_scores['plasma_protein_binding'] = ppb

        # 综合判断
        is_valid = (
                admet_scores['lipinski'] and
                admet_scores['veber'] and
                admet_scores['caco2_pass'] and
                admet_scores['bioavailability_pass']
        )

        return is_valid, admet_scores

    def _check_lipinski(self, desc: Dict) -> bool:
        """检查Lipinski五规则"""
        return (
                desc.get('mw', 0) <= 500 and
                desc.get('logp', 0) <= 5 and
                desc.get('hbd', 0) <= 5 and
                desc.get('hba', 0) <= 10
        )

    def _check_veber(self, desc: Dict) -> bool:
        """检查Veber规则"""
        return (
                desc.get('rotb', 0) <= 10 and
                desc.get('tpsa', 0) <= 140
        )

    def _check_ghose(self, desc: Dict) -> bool:
        """检查Ghose规则"""
        return (
                160 <= desc.get('mw', 0) <= 480 and
                -0.4 <= desc.get('logp', 0) <= 5.6 and
                40 <= desc.get('heavy_atoms', 0) <= 70
        )

    def _predict_caco2(self, desc: Dict) -> float:
        """
        预测Caco-2通透性
        基于线性模型: log Papp = 0.8*LogP - 0.01*TPSA - 0.0025*MW - 3.5

        Args:
            desc: 描述符字典

        Returns:
            log Papp值（>-5.5为良好通透性）
        """
        logp = desc.get('logp', 0)
        tpsa = desc.get('tpsa', 0)
        mw = desc.get('mw', 0)

        log_papp = (0.8 * logp) - (0.01 * tpsa) - (0.0025 * mw) - 3.5

        return log_papp

    def _predict_bioavailability(self, desc: Dict) -> float:
        """
        预测口服生物利用度
        基于Abbott生物利用度评分

        Args:
            desc: 描述符字典

        Returns:
            生物利用度评分 [0, 1]
        """
        score = 0.0

        # 分子量贡献
        mw = desc.get('mw', 0)
        if mw <= 500:
            score += 1.0
        elif mw <= 600:
            score += 0.5

        # LogP贡献
        logp = desc.get('logp', 0)
        if -0.5 <= logp <= 5.0:
            score += 1.0
        elif logp <= 6.0:
            score += 0.5

        # TPSA贡献
        tpsa = desc.get('tpsa', 0)
        if tpsa <= 140:
            score += 1.0
        elif tpsa <= 200:
            score += 0.5

        # 可旋转键贡献
        rotb = desc.get('rotb', 0)
        if rotb <= 10:
            score += 1.0
        elif rotb <= 15:
            score += 0.5

        # 芳香环贡献
        aromatic_rings = desc.get('aromatic_rings', 0)
        if 1 <= aromatic_rings <= 4:
            score += 1.0

        # 归一化
        bioavailability = score / 5.0

        return bioavailability

    def _predict_bbb_permeability(self, desc: Dict) -> float:
        """
        预测血脑屏障通透性
        基于经验公式: BBB = -0.0148*TPSA + 0.152*LogP + 0.139

        Args:
            desc: 描述符字典

        Returns:
            BBB得分（>0表示可通过BBB）
        """
        tpsa = desc.get('tpsa', 0)
        logp = desc.get('logp', 0)

        bbb_score = -0.0148 * tpsa + 0.152 * logp + 0.139

        return bbb_score

    def _predict_plasma_protein_binding(self, desc: Dict) -> float:
        """
        预测人血浆蛋白结合率
        基于经验模型

        Args:
            desc: 描述符字典

        Returns:
            血浆蛋白结合率 [0, 100]%
        """
        logp = desc.get('logp', 0)
        mw = desc.get('mw', 0)

        # 经验公式（高LogP和高MW倾向于高结合率）
        ppb = 50 + 10 * logp + 0.02 * (mw - 300)

        # 限制在[0, 100]范围
        ppb = max(0, min(100, ppb))

        return ppb

    def batch_predict(self, molecules: list, descriptors_list: list) -> Dict:
        """
        批量预测ADMET性质

        Args:
            molecules: 分子列表
            descriptors_list: 描述符列表

        Returns:
            预测结果字典
        """
        results = {
            'admet_valid': [],
            'admet_scores': []
        }

        for mol, desc in zip(molecules, descriptors_list):
            is_valid, admet_scores = self.predict(mol, desc)
            results['admet_valid'].append(is_valid)
            results['admet_scores'].append(admet_scores)

        return results

    @staticmethod
    def get_default_admet_scores() -> Dict:
        """
        获取默认的ADMET评分（用于失败情况）

        Returns:
            默认ADMET评分字典
        """
        return {
            'lipinski': False,
            'veber': False,
            'ghose': False,
            'caco2_permeability': -999.0,
            'caco2_pass': False,
            'bioavailability': 0.0,
            'bioavailability_pass': False,
            'bbb_permeability': -999.0,
            'bbb_pass': False,
            'plasma_protein_binding': 0.0
        }

    @staticmethod
    def get_admet_keys() -> List[str]:
        """
        获取所有ADMET指标的键名

        Returns:
            ADMET键名列表
        """
        return [
            'lipinski',
            'veber',
            'ghose',
            'caco2_permeability',
            'caco2_pass',
            'bioavailability',
            'bioavailability_pass',
            'bbb_permeability',
            'bbb_pass',
            'plasma_protein_binding'
        ]

    def __repr__(self):
        return "ADMETPredictor(Lipinski+Veber+Caco2+Bioavailability)"