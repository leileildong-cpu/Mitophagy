#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子描述符计算器
计算各种理化性质和拓扑描述符
"""

import numpy as np
from typing import Dict, List
from rdkit import Chem
# ✅ 添加这三个导入
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
import warnings


class DescriptorCalculator:
    """分子描述符计算器"""

    def __init__(self):
        """初始化描述符计算器"""
        # 定义要计算的描述符列表
        self.descriptor_functions = {
            # Lipinski五规则
            'mw': Descriptors.MolWt,
            'logp': Crippen.MolLogP,
            'hbd': Lipinski.NumHDonors,
            'hba': Lipinski.NumHAcceptors,
            'rotb': Lipinski.NumRotatableBonds,

            # 拓扑描述符
            'tpsa': rdMolDescriptors.CalcTPSA,
            'heavy_atoms': lambda mol: mol.GetNumHeavyAtoms(),
            'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings,
            'aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings,
            'rings': rdMolDescriptors.CalcNumRings,

            # 分子复杂度
            'num_atoms': lambda mol: mol.GetNumAtoms(),
            'num_bonds': lambda mol: mol.GetNumBonds(),
            'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms,

            # 电子性质
            'num_valence_electrons': Descriptors.NumValenceElectrons,

            # 药物相似性
            'qed': lambda mol: Descriptors.qed(mol) if hasattr(Descriptors, 'qed') else 0.0,
        }

        # 扩展描述符（可选）
        self.extended_descriptors = {
            'fraction_csp3': rdMolDescriptors.CalcFractionCSP3,
            'num_saturated_rings': rdMolDescriptors.CalcNumSaturatedRings,
            'num_aromatic_carbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles,
            'num_aromatic_heterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles,
            'num_aliphatic_carbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles,
            'num_aliphatic_heterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles,
        }

    # ... 其余方法保持不变

    def calculate_all_descriptors(self, mol: Chem.Mol, include_extended: bool = False) -> Dict[str, float]:
        """
        计算所有分子描述符

        Args:
            mol: RDKit分子对象
            include_extended: 是否包含扩展描述符

        Returns:
            描述符字典
        """
        if mol is None:
            return {}

        descriptors = {}

        # 计算基础描述符
        for name, func in self.descriptor_functions.items():
            try:
                value = float(func(mol))
                descriptors[name] = value
            except Exception as e:
                warnings.warn(f"描述符 {name} 计算失败: {e}")
                descriptors[name] = 0.0

        # 计算扩展描述符
        if include_extended:
            for name, func in self.extended_descriptors.items():
                try:
                    value = float(func(mol))
                    descriptors[name] = value
                except Exception as e:
                    warnings.warn(f"扩展描述符 {name} 计算失败: {e}")
                    descriptors[name] = 0.0

        return descriptors

    def calculate_essential_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        计算关键描述符（用于过滤）

        Args:
            mol: RDKit分子对象

        Returns:
            关键描述符字典
        """
        if mol is None:
            return {}

        essential_names = ['mw', 'logp', 'hbd', 'hba', 'tpsa', 'rotb', 'aromatic_rings', 'heavy_atoms']

        descriptors = {}
        for name in essential_names:
            if name in self.descriptor_functions:
                try:
                    value = float(self.descriptor_functions[name](mol))
                    descriptors[name] = value
                except Exception:
                    descriptors[name] = 0.0

        return descriptors

    def calculate_lipinski_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        计算Lipinski五规则相关描述符

        Args:
            mol: RDKit分子对象

        Returns:
            Lipinski描述符字典
        """
        if mol is None:
            return {}

        lipinski_names = ['mw', 'logp', 'hbd', 'hba', 'rotb']

        descriptors = {}
        for name in lipinski_names:
            if name in self.descriptor_functions:
                try:
                    value = float(self.descriptor_functions[name](mol))
                    descriptors[name] = value
                except Exception:
                    descriptors[name] = 0.0

        return descriptors

    def get_descriptor_names(self, include_extended: bool = False) -> List[str]:
        """
        获取所有描述符名称

        Args:
            include_extended: 是否包含扩展描述符

        Returns:
            描述符名称列表
        """
        names = list(self.descriptor_functions.keys())

        if include_extended:
            names.extend(self.extended_descriptors.keys())

        return names

    def batch_calculate(self, molecules: List[Chem.Mol],
                        include_extended: bool = False) -> List[Dict[str, float]]:
        """
        批量计算描述符

        Args:
            molecules: 分子列表
            include_extended: 是否包含扩展描述符

        Returns:
            描述符字典列表
        """
        descriptors_list = []

        for mol in molecules:
            desc = self.calculate_all_descriptors(mol, include_extended)
            descriptors_list.append(desc)

        return descriptors_list

    def validate_lipinski_rule(self, descriptors: Dict[str, float]) -> bool:
        """
        验证Lipinski五规则

        Args:
            descriptors: 描述符字典

        Returns:
            是否满足Lipinski五规则
        """
        return (
                descriptors.get('mw', 0) <= 500 and
                descriptors.get('logp', 0) <= 5 and
                descriptors.get('hbd', 0) <= 5 and
                descriptors.get('hba', 0) <= 10
        )

    def validate_veber_rule(self, descriptors: Dict[str, float]) -> bool:
        """
        验证Veber规则（口服生物利用度）

        Args:
            descriptors: 描述符字典

        Returns:
            是否满足Veber规则
        """
        return (
                descriptors.get('rotb', 0) <= 10 and
                descriptors.get('tpsa', 0) <= 140
        )

    def __repr__(self):
        return f"DescriptorCalculator({len(self.descriptor_functions)} base descriptors)"