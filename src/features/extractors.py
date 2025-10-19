#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分子特征提取器"""

import numpy as np
from typing import Dict, List, Optional  # ✅ 移除 Any
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Pharm2D import Generate
from tqdm import tqdm
import warnings

# ✅ 从 descriptors.py 导入（保持不变）
from .mol2vec_handler import Mol2VecHandler
from .descriptors import DescriptorCalculator


class FeatureExtractor:
    # ... 代码保持不变
    """统一的分子特征提取器"""

    def __init__(self, config: dict, cache=None):
        """初始化特征提取器"""
        self.config = config
        self.cache = cache

        # 初始化子模块
        try:
            self.mol2vec_handler = Mol2VecHandler(config)
        except Exception as e:
            print(f"[WARNING] Mol2VecHandler初始化失败: {e}")
            self.mol2vec_handler = None

        try:
            self.descriptor_calc = DescriptorCalculator()
        except Exception as e:
            print(f"[WARNING] DescriptorCalculator初始化失败: {e}")
            self.descriptor_calc = None

        # 预加载药效团工厂
        self.pharm_factory = self._load_pharmacophore_factory()

        # Morgan指纹参数
        fingerprints_config = config.get("fingerprints", {})
        self.morgan_radius = fingerprints_config.get("morgan_radius", 3)
        self.morgan_nbits = fingerprints_config.get("morgan_nbits", 2048)

        # 打印配置摘要
        self._print_initialization_summary()

    def _load_pharmacophore_factory(self):
        """加载药效团工厂"""
        # 方法1：标准导入
        try:
            from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
            factory = Gobbi_Pharm2D.factory
            print("[INFO] 药效团工厂已加载（标准方法）")

            # 验证factory
            try:
                sig_size = factory.GetSigSize()
                print(f"[INFO] 药效团工厂验证成功，签名大小: {sig_size}")
                return factory
            except Exception as e:
                print(f"[WARNING] 药效团工厂验证失败: {e}")
                return None

        except (AttributeError, ImportError) as e:
            print(f"[WARNING] 标准方法加载失败: {e}")

        # 方法2：直接导入factory
        try:
            from rdkit.Chem.Pharm2D.Gobbi_Pharm2D import factory
            print("[INFO] 药效团工厂已加载（直接导入）")
            sig_size = factory.GetSigSize()
            print(f"[INFO] 药效团工厂验证成功，签名大小: {sig_size}")
            return factory
        except (ImportError, AttributeError) as e:
            print(f"[WARNING] 直接导入失败: {e}")

        # 方法3：动态构建factory
        try:
            from rdkit.Chem import ChemicalFeatures
            from rdkit import RDConfig
            import os as os_module

            fdef_path = os_module.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)
            print("[INFO] 药效团工厂已加载（动态构建）")
            return factory
        except Exception as e:
            print(f"[WARNING] 动态构建失败: {e}")

        # 所有方法都失败
        print("[WARNING] 药效团指纹功能将被禁用")
        return None

    def _print_initialization_summary(self):
        """打印初始化摘要"""
        print(f"[INFO] 特征提取器初始化完成")
        print(f"  - Morgan指纹: radius={self.morgan_radius}, nbits={self.morgan_nbits}")
        print(f"  - 药效团: {'✓ 可用' if self.pharm_factory else '✗ 不可用'}")

        if self.mol2vec_handler:
            try:
                is_available = self.mol2vec_handler.is_available()
                print(f"  - Mol2vec: {'✓ 可用' if is_available else '✗ 不可用'}")
            except Exception:
                print(f"  - Mol2vec: ✗ 不可用（检查失败）")
        else:
            print(f"  - Mol2vec: ✗ 不可用（未初始化）")

        print(f"  - 缓存: {'✓ 启用' if self.cache else '✗ 禁用'}")

    def extract_features(self, molecules: List[Chem.Mol], desc: str = "") -> Dict:
        """提取完整的分子特征集合"""
        print(f"[INFO] 提取{desc}特征...")

        features = {
            'morgan_fps': [],
            'pharm2d_fps': [],
            'mol2vec_vecs': [],
            'descriptors': [],
            'molecules': molecules
        }

        for mol in tqdm(molecules, desc=f"{desc}特征提取"):
            if mol is None:
                self._append_empty_features(features)
                continue

            # 尝试从缓存获取
            if self.cache:
                cached_features = self._get_from_cache(mol)
                if cached_features:
                    for key in ['morgan_fps', 'pharm2d_fps', 'mol2vec_vecs', 'descriptors']:
                        features[key].append(cached_features.get(key))
                    continue

            # 提取各类特征
            mol_features = self._extract_single_molecule_features(mol)

            # 添加到结果中
            for key in ['morgan_fps', 'pharm2d_fps', 'mol2vec_vecs', 'descriptors']:
                features[key].append(mol_features.get(key))

            # 缓存特征
            if self.cache:
                self._set_to_cache(mol, mol_features)

        # 验证提取结果
        self._validate_features(features, desc)

        return features

    def _extract_single_molecule_features(self, mol: Chem.Mol) -> Dict:
        """提取单个分子的所有特征"""
        features = {}

        try:
            # 1. Morgan指纹
            features['morgan_fps'] = self._extract_morgan_fingerprint(mol)

            # 2. 药效团指纹
            features['pharm2d_fps'] = self._extract_pharmacophore_fingerprint(mol)

            # 3. Mol2vec向量
            if self.mol2vec_handler:
                features['mol2vec_vecs'] = self.mol2vec_handler.get_mol2vec_vector(mol)
            else:
                features['mol2vec_vecs'] = None

            # 4. 分子描述符
            if self.descriptor_calc:
                features['descriptors'] = self.descriptor_calc.calculate_all_descriptors(mol)
            else:
                features['descriptors'] = {}

        except Exception as e:
            warnings.warn(f"特征提取失败: {e}")
            features = {
                'morgan_fps': None,
                'pharm2d_fps': None,
                'mol2vec_vecs': None,
                'descriptors': {}
            }

        return features

    def _extract_morgan_fingerprint(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        """提取Morgan圆形指纹"""
        if mol is None:
            return None

        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.morgan_radius,
                nBits=self.morgan_nbits
            )
            fp_array = np.array(fp, dtype=np.uint8)
            return fp_array

        except Exception as e:
            warnings.warn(f"Morgan指纹提取失败: {e}")
            return None

    def _extract_pharmacophore_fingerprint(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        """提取药效团指纹"""
        if mol is None:
            return None

        factory = self.pharm_factory
        if factory is None:
            factory = self._load_pharmacophore_factory()
            if factory is None:
                return None

        try:
            fp = Generate.Gen2DFingerprint(mol, factory)
            fp_bits = fp.ToBitString()
            fp_array = np.array([int(bit) for bit in fp_bits], dtype=np.uint8)
            return fp_array

        except Exception as e:
            warnings.warn(f"药效团指纹提取失败: {e}")
            try:
                fp_length = factory.GetSigSize() if factory else 39972
                return np.zeros(fp_length, dtype=np.uint8)
            except:
                return None

    def _get_from_cache(self, mol: Chem.Mol) -> Optional[Dict]:
        """从缓存获取特征"""
        if not self.cache:
            return None

        cached = {}
        feature_types = ['morgan_fps', 'pharm2d_fps', 'mol2vec_vecs', 'descriptors']

        for feature_type in feature_types:
            feature = self.cache.get(mol, feature_type)
            if feature is None:
                return None
            cached[feature_type] = feature

        return cached

    def _set_to_cache(self, mol: Chem.Mol, features: Dict):
        """设置缓存"""
        if not self.cache:
            return

        for feature_type, feature_value in features.items():
            if feature_value is not None:
                self.cache.set(mol, feature_type, feature_value)

    def _append_empty_features(self, features: Dict):
        """追加空特征"""
        features['morgan_fps'].append(None)
        features['pharm2d_fps'].append(None)
        features['mol2vec_vecs'].append(None)
        features['descriptors'].append({})

    def _validate_features(self, features: Dict, desc: str):
        """验证提取的特征"""
        n_mols = len(features['molecules'])

        valid_morgan = sum(1 for fp in features['morgan_fps'] if fp is not None)
        valid_pharm = sum(1 for fp in features['pharm2d_fps'] if fp is not None)
        valid_mol2vec = sum(1 for vec in features['mol2vec_vecs'] if vec is not None)
        valid_desc = sum(1 for d in features['descriptors'] if d)

        print(f"[INFO] {desc}特征提取完成:")
        print(f"  - Morgan指纹: {valid_morgan}/{n_mols} ({valid_morgan / n_mols * 100:.1f}%)")
        print(f"  - 药效团指纹: {valid_pharm}/{n_mols} ({valid_pharm / n_mols * 100:.1f}%)")
        print(f"  - Mol2vec向量: {valid_mol2vec}/{n_mols} ({valid_mol2vec / n_mols * 100:.1f}%)")
        print(f"  - 分子描述符: {valid_desc}/{n_mols} ({valid_desc / n_mols * 100:.1f}%)")

        if valid_pharm > 0:
            non_zero_pharm = sum(1 for fp in features['pharm2d_fps']
                                 if fp is not None and np.sum(fp) > 0)
            print(f"  - 药效团非零: {non_zero_pharm}/{valid_pharm} ({non_zero_pharm / valid_pharm * 100:.1f}%)")

            if non_zero_pharm == 0:
                warnings.warn(f"{desc}所有药效团指纹都是0！")

    def get_feature_dimensions(self) -> Dict[str, int]:
        """获取各类特征的维度"""
        dimensions = {
            'morgan': self.morgan_nbits,
            'pharm2d': 39972,
        }

        if self.mol2vec_handler:
            try:
                dimensions['mol2vec'] = self.mol2vec_handler.get_vector_size()
            except:
                dimensions['mol2vec'] = 300
        else:
            dimensions['mol2vec'] = 300

        if self.descriptor_calc:
            try:
                dimensions['descriptors'] = len(self.descriptor_calc.get_descriptor_names())
            except:
                dimensions['descriptors'] = 13
        else:
            dimensions['descriptors'] = 13

        return dimensions

    def __repr__(self):
        pharm_status = '✓' if self.pharm_factory else '✗'
        mol2vec_status = '✗'
        if self.mol2vec_handler:
            try:
                mol2vec_status = '✓' if self.mol2vec_handler.is_available() else '✗'
            except:
                pass
        cache_status = '✓' if self.cache else '✗'

        return (f"FeatureExtractor("
                f"morgan={self.morgan_nbits}bits, "
                f"pharm={pharm_status}, "
                f"mol2vec={mol2vec_status}, "
                f"cache={cache_status})")

# ✅ 到此为止！不要在下面再添加 DescriptorCalculator 类！
# DescriptorCalculator 在 descriptors.py 中