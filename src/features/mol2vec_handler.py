#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mol2vec特征处理器
处理Mol2vec模型加载和向量生成
"""

import os
import numpy as np
from typing import Optional
from rdkit import Chem
import warnings

# Mol2vec依赖
try:
    from mol2vec.features import mol2alt_sentence, sentences2vec
    from gensim.models import word2vec

    MOL2VEC_AVAILABLE = True
except ImportError:
    MOL2VEC_AVAILABLE = False


class Mol2VecHandler:
    """Mol2vec特征处理器"""

    def __init__(self, config: dict):
        """
        初始化Mol2vec处理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.model_loaded = False
        self.vector_size = 300  # 默认维度

        if MOL2VEC_AVAILABLE:
            self._load_model()
        else:
            print("[WARNING] Mol2vec不可用，将跳过mol2vec特征")

    def _load_model(self):
        """加载Mol2vec模型"""
        model_path = self.config.get("mol2vec", {}).get("model_path")

        if not model_path:
            print("[INFO] 未配置Mol2vec模型路径，将跳过mol2vec特征")
            return

        if not os.path.exists(model_path):
            print(f"[WARNING] Mol2vec模型文件不存在: {model_path}")
            return

        try:
            self.model = word2vec.Word2Vec.load(model_path)
            self.model_loaded = True
            self.vector_size = self.model.wv.vector_size
            print(f"[INFO] Mol2vec模型已加载: {model_path} (维度: {self.vector_size})")
        except Exception as e:
            print(f"[WARNING] Mol2vec模型加载失败: {e}")
            self.model = None

    def is_available(self) -> bool:
        """检查Mol2vec是否可用"""
        return MOL2VEC_AVAILABLE and self.model_loaded

    def get_mol2vec_vector(self, mol: Chem.Mol, radius: int = 1) -> Optional[np.ndarray]:
        """
        获取分子的Mol2vec向量

        Args:
            mol: RDKit分子对象
            radius: 子结构半径

        Returns:
            Mol2vec向量（维度300），失败返回None
        """
        if not self.is_available():
            return None

        if mol is None:
            return None

        try:
            # 生成分子句子
            sentence = mol2alt_sentence(mol, radius=radius)

            if not sentence:
                return None

            # 转换为向量
            try:
                vec = sentences2vec([sentence], self.model.wv, unseen='UNK')
            except AttributeError:
                # 兼容旧版本Gensim
                vec = sentences2vec([sentence], self.model, unseen='UNK')

            if len(vec) > 0:
                return vec[0].astype(np.float32)
            else:
                return None

        except Exception as e:
            warnings.warn(f"Mol2vec向量生成失败: {e}")
            return None

    def get_vector_size(self) -> int:
        """获取向量维度"""
        return self.vector_size

    def batch_get_vectors(self, molecules: list, radius: int = 1) -> list:
        """
        批量获取Mol2vec向量

        Args:
            molecules: 分子列表
            radius: 子结构半径

        Returns:
            向量列表
        """
        if not self.is_available():
            return [None] * len(molecules)

        vectors = []
        for mol in molecules:
            vec = self.get_mol2vec_vector(mol, radius)
            vectors.append(vec)

        return vectors

    def __repr__(self):
        status = "loaded" if self.model_loaded else "not loaded"
        return f"Mol2VecHandler(status={status}, vector_size={self.vector_size})"