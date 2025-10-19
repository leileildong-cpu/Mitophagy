#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选库准备工程代码
将ChEMBL数据库处理为虚拟筛选系统兼容格式
版本: v1.0
"""
# python src/library_preparation.py --source sdf --sdf-path data/chembl_36.sdf.gz --max-compounds 100000 --output-dir data
# python src/library_preparation.py --source sdf --sdf-path data/chembl_36.sdf.gz --output-dir data

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# ⚠️ 在所有imports之前设置
os.environ['RDK_PICKLE_PROTOCOL'] = '2'
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings('ignore')

# ✅ 添加这个：禁用RDKit日志
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # 只显示严重错误

print("[INFO] RDKit警告已禁用")

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Iterator
import pickle
import gzip
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import yaml
import warnings

warnings.filterwarnings('ignore')

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit import DataStructs
# 可选的标准化模块
try:
    from rdkit.Chem import rdMolStandardize as rdMS

    STANDARDIZE_AVAILABLE = True
except ImportError:
    STANDARDIZE_AVAILABLE = False
    print("[WARNING] RDKit标准化模块不可用，使用基础标准化")

# ChEMBL API客户端（可选）
try:
    from chembl_webresource_client.new_client import new_client

    CHEMBL_API_AVAILABLE = True
except ImportError:
    CHEMBL_API_AVAILABLE = False
    print("[INFO] ChEMBL API客户端不可用，使用本地文件")


class ChEMBLDownloader:
    """ChEMBL数据库下载器"""

    def __init__(self, data_dir='data/chembl_raw'):
        self.data_dir = data_dir
        self.base_url = 'https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/'
        os.makedirs(data_dir, exist_ok=True)

    def get_sample_chembl_data(self, sample_size=100000):
        """获取ChEMBL样本数据（用于测试）"""

        if CHEMBL_API_AVAILABLE:
            print(f"[INFO] 通过API获取 {sample_size} 个ChEMBL化合物样本...")

            try:
                molecule = new_client.molecule

                # 查询条件
                molecules = molecule.filter(
                    molecule_structures__canonical_smiles__isnull=False,
                    molecule_properties__mw_freebase__lte=800,  # 分子量限制
                    molecule_properties__mw_freebase__gte=150
                ).only([
                    'molecule_chembl_id',
                    'molecule_structures',
                    'molecule_properties'
                ])[:sample_size]

                compounds = []

                for i, mol_data in enumerate(molecules):
                    if i >= sample_size:
                        break

                    try:
                        smiles = mol_data['molecule_structures']['canonical_smiles']
                        mol = Chem.MolFromSmiles(smiles)

                        if mol:
                            compound = {
                                'id': mol_data['molecule_chembl_id'],
                                'name': mol_data['molecule_chembl_id'],
                                'smiles': smiles,
                                'mol': mol,
                                'source': 'ChEMBL_API'
                            }

                            # 添加已有的性质
                            if mol_data.get('molecule_properties'):
                                props = mol_data['molecule_properties']
                                compound.update({
                                    'mw': props.get('mw_freebase'),
                                    'logp': props.get('alogp'),
                                    'tpsa': props.get('psa'),
                                    'hbd': props.get('hbd'),
                                    'hba': props.get('hba')
                                })

                            compounds.append(compound)

                    except Exception as e:
                        continue

                    if (i + 1) % 1000 == 0:
                        print(f"[INFO] API获取进度: {i + 1:,}")

                print(f"[INFO] 通过API获取了 {len(compounds):,} 个化合物")
                return compounds

            except Exception as e:
                print(f"[ERROR] API获取失败: {e}")
                return []

        else:
            print("[WARNING] ChEMBL API不可用，生成测试数据")
            return self._generate_test_data(sample_size)

    def _generate_test_data(self, sample_size=1000):
        """生成测试数据"""

        # 一些常见药物的SMILES
        test_smiles = [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'CC(=O)NC1=CC=C(C=C1)O',  # Paracetamol
            'CC1=CC=C(C=C1)N2C=C(N=N2)C3=CC=CC=C3',
            'CN1CCN(CC1)C2=CC=C(C=C2)Cl',
            'COC1=CC=C(C=C1)C2=NC=NC3=C2C=CC=C3',
            'CC1=CC(=C(C=C1)N2C(=O)C=CC2=O)C',
            'C1=CC=C(C=C1)C2=NC3=CC=CC=C3N=C2N',
            'CC1=C(C2=CC=CC=C2N1)CC(=O)O',
        ]

        compounds = []

        for i in range(min(sample_size, len(test_smiles))):
            smiles = test_smiles[i % len(test_smiles)]

            # 为了增加多样性，对SMILES进行一些修改
            if i >= len(test_smiles):
                # 添加一些简单的官能团
                modifications = ['C', 'CC', 'O', 'N', 'F', 'Cl']
                mod = modifications[i % len(modifications)]
                smiles = f"{mod}{smiles}"

            mol = Chem.MolFromSmiles(smiles)

            if mol:
                compound = {
                    'id': f'TEST_{i:06d}',
                    'name': f'Test_Compound_{i:06d}',
                    'smiles': Chem.MolToSmiles(mol),
                    'mol': mol,
                    'source': 'Generated'
                }
                compounds.append(compound)

        print(f"[INFO] 生成了 {len(compounds):,} 个测试化合物")
        return compounds


class MolecularStandardizer:
    """分子标准化器"""

    def __init__(self):
        self.setup_standardizers()

    def setup_standardizers(self):
        """设置标准化工具"""

        if STANDARDIZE_AVAILABLE:
            try:
                self.normalizer = rdMS.Normalizer()
                self.tautomer_enumerator = rdMS.TautomerEnumerator()
                self.uncharger = rdMS.Uncharger()
                print("[INFO] RDKit高级标准化工具已启用")
            except:
                self.normalizer = None
                self.tautomer_enumerator = None
                self.uncharger = None
        else:
            self.normalizer = None
            self.tautomer_enumerator = None
            self.uncharger = None

    def standardize_molecule(self, mol_input):
        """标准化分子"""

        # 处理输入（SMILES字符串或分子对象）
        if isinstance(mol_input, str):
            mol = Chem.MolFromSmiles(mol_input)
        else:
            mol = mol_input

        if mol is None:
            return None

        try:
            # 基础清理
            mol = Chem.RemoveHs(mol)

            if mol.GetNumAtoms() == 0:
                return None

            # 基础合理性检查
            try:
                Chem.SanitizeMol(mol)
            except:
                return None

            # 高级标准化（如果可用）
            if self.normalizer:
                try:
                    mol = self.normalizer.normalize(mol)
                except:
                    pass

            if self.uncharger:
                try:
                    mol = self.uncharger.uncharge(mol)
                except:
                    pass

            if self.tautomer_enumerator:
                try:
                    mol = self.tautomer_enumerator.Canonicalize(mol)
                except:
                    pass

            # 最终标准化
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            final_mol = Chem.MolFromSmiles(canonical_smiles)

            return final_mol

        except Exception as e:
            return None

    def batch_standardize(self, compounds, max_workers=4):
        """批量标准化"""

        print(f"[INFO] 开始批量标准化 {len(compounds):,} 个化合物...")

        def process_compound(compound):
            """处理单个化合物"""
            try:
                standardized_mol = self.standardize_molecule(compound['mol'])

                if standardized_mol:
                    result = compound.copy()
                    result['mol'] = standardized_mol
                    result['canonical_smiles'] = Chem.MolToSmiles(standardized_mol, isomericSmiles=True)
                    return result
                else:
                    return None
            except:
                return None

        # 多线程处理
        standardized_compounds = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_compound, comp): i for i, comp in enumerate(compounds)}

            for future in tqdm(as_completed(futures), total=len(compounds), desc="标准化进度"):
                result = future.result()
                if result:
                    standardized_compounds.append(result)

        print(f"[INFO] 标准化完成: {len(standardized_compounds):,}/{len(compounds):,} 成功")
        return standardized_compounds


class DrugLikenessFilter:
    """药物相似性过滤器"""

    def __init__(self, config=None):
        # 默认过滤条件
        self.filters = {
            'mw_min': 150,
            'mw_max': 500,
            'logp_min': -3,
            'logp_max': 5,
            'hbd_max': 5,
            'hba_max': 10,
            'tpsa_max': 140,
            'rotb_max': 10,
            'heavy_atoms_min': 10,
            'heavy_atoms_max': 50
        }

        if config:
            self.filters.update(config)

        # PAINS过滤器
        self.setup_pains_filter()

    def setup_pains_filter(self):
        """设置PAINS过滤器"""

        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
            self.pains_catalog = FilterCatalog(params)
            print(f"[INFO] PAINS过滤器已设置，包含 {self.pains_catalog.GetNumEntries()} 个模式")
        except Exception as e:
            print(f"[WARNING] PAINS过滤器设置失败: {e}")
            self.pains_catalog = None

    def calculate_properties(self, mol):
        """计算分子性质"""

        if mol is None:
            return None

        try:
            properties = {
                'mw': float(Descriptors.MolWt(mol)),
                'logp': float(Crippen.MolLogP(mol)),
                'tpsa': float(rdMolDescriptors.CalcTPSA(mol)),
                'hbd': float(Lipinski.NumHDonors(mol)),
                'hba': float(Lipinski.NumHAcceptors(mol)),
                'rotb': float(Lipinski.NumRotatableBonds(mol)),
                'heavy_atoms': float(mol.GetNumHeavyAtoms()),
                'aromatic_rings': float(rdMolDescriptors.CalcNumAromaticRings(mol))
            }
            return properties
        except:
            return None

    def is_pains(self, mol):
        """检查是否为PAINS"""

        if self.pains_catalog is None:
            return False

        try:
            return self.pains_catalog.HasMatch(mol)
        except:
            return False

    def passes_filters(self, mol):
        """检查分子是否通过所有过滤器"""

        # 计算性质
        props = self.calculate_properties(mol)
        if props is None:
            return False, {}, "性质计算失败"

        # PAINS检查
        is_pains = self.is_pains(mol)
        if is_pains:
            return False, props, "PAINS阳性"

        # 药物相似性检查
        violations = []

        if not (self.filters['mw_min'] <= props['mw'] <= self.filters['mw_max']):
            violations.append(f"MW({props['mw']:.1f})")

        if not (self.filters['logp_min'] <= props['logp'] <= self.filters['logp_max']):
            violations.append(f"LogP({props['logp']:.1f})")

        if props['hbd'] > self.filters['hbd_max']:
            violations.append(f"HBD({props['hbd']})")

        if props['hba'] > self.filters['hba_max']:
            violations.append(f"HBA({props['hba']})")

        if props['tpsa'] > self.filters['tpsa_max']:
            violations.append(f"TPSA({props['tpsa']:.1f})")

        if props['rotb'] > self.filters['rotb_max']:
            violations.append(f"RotB({props['rotb']})")

        if not (self.filters['heavy_atoms_min'] <= props['heavy_atoms'] <= self.filters['heavy_atoms_max']):
            violations.append(f"Heavy({props['heavy_atoms']})")

        if violations:
            return False, props, f"违反规则: {', '.join(violations)}"

        return True, props, "通过"

    def filter_compounds(self, compounds, max_workers=4):
        """批量过滤化合物"""

        print(f"[INFO] 开始药物相似性过滤 {len(compounds):,} 个化合物...")

        def process_compound(compound):
            """处理单个化合物"""
            passes, props, reason = self.passes_filters(compound['mol'])

            if passes:
                result = compound.copy()
                result.update(props)
                result['filter_reason'] = reason
                return result
            else:
                return None

        # 多线程过滤
        filtered_compounds = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_compound, comp): i for i, comp in enumerate(compounds)}

            for future in tqdm(as_completed(futures), total=len(compounds), desc="过滤进度"):
                result = future.result()
                if result:
                    filtered_compounds.append(result)

        print(f"[INFO] 过滤完成: {len(filtered_compounds):,}/{len(compounds):,} 通过 "
              f"({len(filtered_compounds) / len(compounds) * 100:.1f}%)")

        return filtered_compounds


class NLRP3SpecificFilter(DrugLikenessFilter):
    """NLRP3特异性过滤器 - 继承自通用过滤器"""

    def __init__(self, config=None):
        # 调用父类初始化
        super().__init__(config)

        print("[INFO] 初始化NLRP3特异性过滤器...")

        # NLRP3特异性规则
        self.nlrp3_rules = {
            'aromatic_rings_min': config.get('aromatic_rings_min', 2) if config else 2,
            'aromatic_rings_max': config.get('aromatic_rings_max', 5) if config else 5,
            'tpsa_min': config.get('tpsa_min', 30) if config else 30,
        }

        # 更新通用规则为NLRP3优化的参数
        if config and config.get('nlrp3_mode', False):
            self.filters.update({
                'mw_min': 250,
                'mw_max': 600,
                'logp_min': 1.5,
                'logp_max': 6.0,
                'hbd_max': 3,
                'hba_max': 8,
                'tpsa_max': 120,
            })
            print("[INFO] 已应用NLRP3优化参数")

        # 已知NLRP3抑制剂（用于相似性筛选）
        self.reference_inhibitors = self._load_reference_inhibitors()

        # 优选结构特征的SMARTS模式
        self._setup_preferred_patterns()

    def _load_reference_inhibitors(self):
        """加载已知NLRP3抑制剂"""

        known_inhibitors = {
            'MCC950': 'CC(C)CC1=CC=C(C=C1)C(=O)N[C@@H]2CCN(C2)S(=O)(=O)C3=CC=C(C=C3)C',
            'CY-09': 'CN1CCN(CC1)C2=NC(=NC=C2)NC3=CC=C(C=C3)C(=O)N',
            'OLT1177': 'CC1=C(C=C(C=C1)C2=CC=C(C=C2)Cl)N3C=NC=N3',
            'Tranilast': 'CC1=C(C=CC=C1NC(=O)C2=CC=CC=C2)C(=O)O'
        }

        reference_mols = {}
        for name, smiles in known_inhibitors.items():
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    reference_mols[name] = mol
                    print(f"[INFO] 加载参考抑制剂: {name}")
            except Exception as e:
                print(f"[WARNING] 无法加载 {name}: {e}")

        return reference_mols

    def _setup_preferred_patterns(self):
        """设置优选结构模式"""

        try:
            self.preferred_patterns = {
                'sulfonamide': Chem.MolFromSmarts('S(=O)(=O)N'),
                'pyridine': Chem.MolFromSmarts('c1ccncc1'),
                'pyrimidine': Chem.MolFromSmarts('c1ncncn1'),
                'imidazole': Chem.MolFromSmarts('c1c[nH]cn1'),
                'benzamide': Chem.MolFromSmarts('c1ccccc1C(=O)N'),
                'halogen': Chem.MolFromSmarts('[F,Cl,Br,I]'),
                'urea': Chem.MolFromSmarts('NC(=O)N'),
                'sulfone': Chem.MolFromSmarts('S(=O)(=O)'),
            }
            print(f"[INFO] 已加载 {len(self.preferred_patterns)} 个优选结构模式")
        except Exception as e:
            print(f"[WARNING] 优选模式设置失败: {e}")
            self.preferred_patterns = {}

    def check_nlrp3_specific_rules(self, mol):
        """检查NLRP3特异性规则"""

        if mol is None:
            return False, {}, "无效分子"

        try:
            nlrp3_props = {}

            # 1. 检查芳香环数量
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            nlrp3_props['aromatic_rings'] = aromatic_rings

            if not (self.nlrp3_rules['aromatic_rings_min'] <= aromatic_rings <=
                    self.nlrp3_rules['aromatic_rings_max']):
                return False, nlrp3_props, f"芳香环数量不符合({aromatic_rings}，需要{self.nlrp3_rules['aromatic_rings_min']}-{self.nlrp3_rules['aromatic_rings_max']})"

            # 2. 检查极性表面积最小值
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            if tpsa < self.nlrp3_rules['tpsa_min']:
                return False, nlrp3_props, f"TPSA过低({tpsa:.1f}，需要≥{self.nlrp3_rules['tpsa_min']})"

            # 3. 检查优选的结构特征
            preferred_score = self._score_preferred_features(mol)
            nlrp3_props['preferred_feature_score'] = preferred_score

            if preferred_score < 1:  # 至少要有1个优选特征
                return False, nlrp3_props, "缺少优选结构特征（磺酰胺/氮杂环等）"

            return True, nlrp3_props, "通过NLRP3特异性规则"

        except Exception as e:
            return False, {}, f"NLRP3规则检查失败: {e}"

    def _score_preferred_features(self, mol):
        """评分优选结构特征"""

        score = 0
        matched_features = []

        if not self.preferred_patterns:
            return 0

        try:
            # 检查磺酰胺（高权重）
            if self.preferred_patterns.get('sulfonamide'):
                if mol.HasSubstructMatch(self.preferred_patterns['sulfonamide']):
                    score += 2
                    matched_features.append('sulfonamide')

            # 检查氮杂环
            for pattern_name in ['pyridine', 'pyrimidine', 'imidazole']:
                pattern = self.preferred_patterns.get(pattern_name)
                if pattern and mol.HasSubstructMatch(pattern):
                    score += 1
                    matched_features.append(pattern_name)

            # 检查其他特征
            for pattern_name in ['benzamide', 'urea', 'sulfone']:
                pattern = self.preferred_patterns.get(pattern_name)
                if pattern and mol.HasSubstructMatch(pattern):
                    score += 0.5
                    matched_features.append(pattern_name)

            # 检查卤素
            if self.preferred_patterns.get('halogen'):
                if mol.HasSubstructMatch(self.preferred_patterns['halogen']):
                    score += 0.5
                    matched_features.append('halogen')

        except Exception as e:
            print(f"[WARNING] 特征评分失败: {e}")

        return score

    def calculate_similarity_to_references(self, mol):
        """计算与已知抑制剂的相似性"""

        if not self.reference_inhibitors:
            return 0.0

        try:
            # 计算Morgan指纹
            mol_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=3, nBits=2048
            )

            max_similarity = 0.0
            best_match = None

            for ref_name, ref_mol in self.reference_inhibitors.items():
                ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    ref_mol, radius=3, nBits=2048
                )
                similarity = DataStructs.TanimotoSimilarity(mol_fp, ref_fp)

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = ref_name

            return max_similarity

        except Exception as e:
            print(f"[WARNING] 相似性计算失败: {e}")
            return 0.0

    def passes_filters(self, mol):
        """增强的过滤检查（包含NLRP3特异性）"""

        # 1. 先进行通用药物相似性过滤
        passes_general, props, reason = super().passes_filters(mol)

        if not passes_general:
            return False, props, f"通用过滤失败: {reason}"

        # 2. 再进行NLRP3特异性检查
        passes_nlrp3, nlrp3_props, nlrp3_reason = self.check_nlrp3_specific_rules(mol)

        # 合并属性
        props.update(nlrp3_props)

        if not passes_nlrp3:
            return False, props, f"NLRP3特异性过滤失败: {nlrp3_reason}"

        # 3. 计算相似性得分
        similarity = self.calculate_similarity_to_references(mol)
        props['reference_similarity'] = similarity

        return True, props, f"通过所有过滤（特征分: {nlrp3_props.get('preferred_feature_score', 0):.1f}, 相似性: {similarity:.3f}）"


class LibraryProcessor:
    """筛选库处理器 - 支持通用和特异性过滤"""

    def __init__(self, config):
        self.config = config
        self.downloader = ChEMBLDownloader(config.get('raw_data_dir', 'data/chembl_raw'))
        self.standardizer = MolecularStandardizer()

        # 根据配置选择过滤器类型
        filter_config = config.get('filters', {})

        # 检查是否使用NLRP3特异性过滤
        if filter_config.get('nlrp3_mode', False) or filter_config.get('target_specific', False):
            print("[INFO] 使用NLRP3特异性过滤器")
            self.filter = NLRP3SpecificFilter(filter_config)
        else:
            print("[INFO] 使用通用药物相似性过滤器")
            self.filter = DrugLikenessFilter(filter_config)

        self.output_dir = config.get('output_dir', 'data')
        os.makedirs(self.output_dir, exist_ok=True)

    def load_compounds_from_sdf(self, sdf_path, max_compounds=None):
        """从SDF文件加载化合物"""

        print(f"[INFO] 从SDF文件加载化合物: {sdf_path}")

        if not os.path.exists(sdf_path):
            print(f"[ERROR] SDF文件不存在: {sdf_path}")
            return []

        compounds = []

        try:
            # 处理gzip压缩文件
            if sdf_path.endswith('.gz'):
                with gzip.open(sdf_path, 'rb') as gz_file:
                    # 创建一个临时的二进制流给SDMolSupplier
                    supplier = Chem.ForwardSDMolSupplier(gz_file)
                    compounds = self._process_sdf_supplier(supplier, max_compounds)
            else:
                supplier = Chem.SDMolSupplier(sdf_path)
                compounds = self._process_sdf_supplier(supplier, max_compounds)

        except Exception as e:
            print(f"[ERROR] SDF文件处理失败: {e}")
            return []

        print(f"[INFO] SDF加载完成: {len(compounds):,} 个化合物")
        return compounds

    def _process_sdf_supplier(self, supplier, max_compounds=None):
        """处理SDF供应器"""

        compounds = []

        for i, mol in enumerate(supplier):
            if max_compounds and i >= max_compounds:
                break

            if mol is None:
                continue

            try:
                # 提取ChEMBL ID
                chembl_id = None
                for prop_name in ['chembl_id', 'ID', 'Name', '_Name']:
                    if mol.HasProp(prop_name):
                        chembl_id = mol.GetProp(prop_name)
                        break

                if not chembl_id:
                    chembl_id = f'COMPOUND_{i:06d}'

                # 获取SMILES
                smiles = None
                for prop_name in ['canonical_smiles', 'SMILES', 'smiles']:
                    if mol.HasProp(prop_name):
                        smiles = mol.GetProp(prop_name)
                        break

                if not smiles:
                    smiles = Chem.MolToSmiles(mol)

                compound = {
                    'id': chembl_id,
                    'name': chembl_id,  # 使用ChEMBL ID作为name
                    'smiles': smiles,
                    'mol': mol,
                    'source': 'ChEMBL_SDF'
                }

                compounds.append(compound)

                if (i + 1) % 10000 == 0:
                    print(f"[INFO] 已加载 {i + 1:,} 个化合物")

            except Exception as e:
                continue

        return compounds

    def remove_duplicates(self, compounds):
        """去除重复化合物"""

        print(f"[INFO] 去除重复化合物...")

        seen_smiles = set()
        unique_compounds = []

        for compound in compounds:
            smiles = compound.get('canonical_smiles', compound.get('smiles', ''))

            if smiles and smiles not in seen_smiles:
                seen_smiles.add(smiles)
                unique_compounds.append(compound)

        print(f"[INFO] 去重完成: {len(unique_compounds):,}/{len(compounds):,} 保留")
        return unique_compounds

    def save_library_compatible_format(self, compounds):
        """保存为兼容现有筛选代码的格式"""

        # 准备数据
        data_for_csv = []

        for compound in compounds:
            data_for_csv.append({
                'id': compound['id'],
                'name': compound.get('name', compound['id']),
                'smiles': compound.get('canonical_smiles', compound['smiles']),
                # 理化性质
                'mw': compound.get('mw', 0),
                'logp': compound.get('logp', 0),
                'tpsa': compound.get('tpsa', 0),
                'hbd': compound.get('hbd', 0),
                'hba': compound.get('hba', 0),
                'rotb': compound.get('rotb', 0),
                'heavy_atoms': compound.get('heavy_atoms', 0),
                'aromatic_rings': compound.get('aromatic_rings', 0)
            })

        # 保存为CSV（与现有代码兼容）
        df = pd.DataFrame(data_for_csv)
        csv_path = os.path.join(self.output_dir, 'library.csv')
        df.to_csv(csv_path, index=False)

        print(f"[INFO] 筛选库已保存: {csv_path}")
        print(f"[INFO] 格式兼容现有虚拟筛选代码")

        # 保存统计信息
        stats = {
            'total_compounds': len(compounds),
            'file_path': csv_path,
            'columns': list(df.columns),
            'mw_range': [float(df['mw'].min()), float(df['mw'].max())] if len(df) > 0 else [0, 0],
            'logp_range': [float(df['logp'].min()), float(df['logp'].max())] if len(df) > 0 else [0, 0],
            'creation_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        stats_path = os.path.join(self.output_dir, 'library_stats.yaml')
        with open(stats_path, 'w') as f:
            yaml.dump(stats, f)

        return csv_path, len(compounds)

    def process_library(self, source='sample', max_compounds=None):
        """处理筛选库的完整流程"""

        print("=" * 80)
        print("筛选库处理流程开始")
        print("=" * 80)

        # 步骤1: 获取原始数据
        if source == 'sample':
            print("\n[STEP 1] 获取ChEMBL样本数据...")
            compounds = self.downloader.get_sample_chembl_data(
                sample_size=max_compounds or 100000
            )
        elif source == 'sdf':
            print("\n[STEP 1] 从SDF文件加载数据...")
            sdf_path = self.config.get('sdf_path')
            if not sdf_path or not os.path.exists(sdf_path):
                print(f"[ERROR] SDF文件路径未配置或文件不存在: {sdf_path}")
                return None
            compounds = self.load_compounds_from_sdf(sdf_path, max_compounds)
        else:
            print(f"[ERROR] 未知的数据源: {source}")
            return None

        if not compounds:
            print("[ERROR] 未获取到任何化合物数据")
            return None

        # 步骤2: 分子标准化
        print(f"\n[STEP 2] 分子标准化 ({len(compounds):,} 个化合物)...")
        compounds = self.standardizer.batch_standardize(compounds)

        if not compounds:
            print("[ERROR] 标准化后没有有效化合物")
            return None

        # 步骤3: 去重
        print(f"\n[STEP 3] 去除重复 ({len(compounds):,} 个化合物)...")
        compounds = self.remove_duplicates(compounds)

        # 步骤4: 药物相似性过滤
        print(f"\n[STEP 4] 药物相似性过滤 ({len(compounds):,} 个化合物)...")
        compounds = self.filter.filter_compounds(compounds)

        if not compounds:
            print("[WARNING] 过滤后没有化合物通过，可能过滤条件过严")
            return None

        # 步骤5: 保存结果
        print(f"\n[STEP 5] 保存筛选库 ({len(compounds):,} 个化合物)...")
        library_path, count = self.save_library_compatible_format(compounds)

        # 生成摘要报告
        self.generate_summary_report(compounds, library_path)

        print("\n" + "=" * 80)
        print("筛选库处理完成")
        print(f"最终库文件: {library_path}")
        print(f"化合物数量: {count:,}")
        print("=" * 80)

        return library_path

    def generate_summary_report(self, compounds, library_path):
        """生成摘要报告"""

        df = pd.DataFrame([{k: v for k, v in comp.items() if k != 'mol'} for comp in compounds])

        report = [
            "筛选库处理摘要报告",
            "=" * 50,
            f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"最终化合物数量: {len(compounds):,}",
            f"输出文件: {library_path}",
            "",
            "理化性质统计:",
            f"  分子量: {df['mw'].min():.1f} - {df['mw'].max():.1f} (平均: {df['mw'].mean():.1f})" if 'mw' in df.columns else "  分子量: N/A",
            f"  LogP: {df['logp'].min():.1f} - {df['logp'].max():.1f} (平均: {df['logp'].mean():.1f})" if 'logp' in df.columns else "  LogP: N/A",
            f"  TPSA: {df['tpsa'].min():.1f} - {df['tpsa'].max():.1f} (平均: {df['tpsa'].mean():.1f})" if 'tpsa' in df.columns else "  TPSA: N/A",
            f"  HBD: {df['hbd'].min():.0f} - {df['hbd'].max():.0f} (平均: {df['hbd'].mean():.1f})" if 'hbd' in df.columns else "  HBD: N/A",
            f"  HBA: {df['hba'].min():.0f} - {df['hba'].max():.0f} (平均: {df['hba'].mean():.1f})" if 'hba' in df.columns else "  HBA: N/A",
            "",
            "文件格式: CSV",
            "兼容性: 与现有虚拟筛选代码完全兼容"
        ]

        report_path = os.path.join(self.output_dir, 'processing_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"[INFO] 处理报告已保存: {report_path}")


def create_default_config():
    """创建默认配置"""

    config = {
        'raw_data_dir': 'data/chembl_raw',
        'output_dir': 'data',
        'sdf_path': None,  # 如果使用本地SDF文件

        'filters': {
            'mw_min': 150,
            'mw_max': 500,
            'logp_min': -3,
            'logp_max': 5,
            'hbd_max': 5,
            'hba_max': 10,
            'tpsa_max': 140,
            'rotb_max': 10,
            'heavy_atoms_min': 10,
            'heavy_atoms_max': 50
        },

        'processing': {
            'max_workers': 4,
            'chunk_size': 10000
        }
    }

    return config


def main():
    """主函数"""

    parser = argparse.ArgumentParser(description="筛选库准备工具")
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--source', choices=['sample', 'sdf'], default='sample',
                        help='数据源：sample(ChEMBL API样本) 或 sdf(本地SDF文件)')
    parser.add_argument('--max-compounds', type=int, help='最大化合物数量限制')
    parser.add_argument('--output-dir', type=str, default='data', help='输出目录')
    parser.add_argument('--sdf-path', type=str, help='SDF文件路径（当source=sdf时）')
    parser.add_argument('--nlrp3-mode', action='store_true', help='启用NLRP3特异性过滤')  # 新增

    args = parser.parse_args()

    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_default_config()

    # 命令行参数覆盖配置
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.sdf_path:
        config['sdf_path'] = args.sdf_path
    if args.nlrp3_mode:  # 新增
        config.setdefault('filters', {})['nlrp3_mode'] = True

    # 创建处理器
    processor = LibraryProcessor(config)

    # 处理筛选库
    try:
        library_path = processor.process_library(
            source=args.source,
            max_compounds=args.max_compounds
        )

        if library_path:
            print(f"\n✅ 成功！筛选库已准备完成")
            print(f"📁 库文件位置: {library_path}")
            print(f"🔄 可直接用于虚拟筛选系统")
        else:
            print("\n❌ 筛选库处理失败")
            return 1

    except Exception as e:
        print(f"\n❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())