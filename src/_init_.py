#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLRP3抑制剂虚拟筛选系统 - 完整模块化版本
Version: 3.0 Complete Modular Edition
"""

__version__ = "3.0.0"
__author__ = "NLRP3 Screening Team"

# 导入策略：尽可能导入，失败则跳过
import warnings

# Core模块（这个应该没问题）
try:
    from .core import (
        OptimizedCPUThreadPool,
        FastMolecularFeatureCache,
        GPUManager,
        ScreeningLogger
    )
    _CORE_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Core模块导入失败: {e}")
    _CORE_AVAILABLE = False

# Features模块
try:
    from .features import (
        FeatureExtractor,
        Mol2VecHandler,
        DescriptorCalculator
    )
    _FEATURES_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Features模块导入失败: {e}")
    _FEATURES_AVAILABLE = False

# Similarity模块
try:
    from .similarity import (
        FingerprintSimilarity,
        Shape3DSimilarity,
        SimilarityAggregator
    )
    _SIMILARITY_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Similarity模块导入失败: {e}")
    _SIMILARITY_AVAILABLE = False

# Filters模块
try:
    from .filters import (
        NLRP3Filters,
        ADMETPredictor,
        ToxicityChecker,
        PAINSBrenkFilter
    )
    _FILTERS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Filters模块导入失败: {e}")
    _FILTERS_AVAILABLE = False

# Docking模块
try:
    from .docking import MolecularDocking
    _DOCKING_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Docking模块导入失败: {e}")
    _DOCKING_AVAILABLE = False

# Reporting模块
try:
    from .reporting import (
        TextReportGenerator,
        HTMLReportGenerator,
        Visualizer
    )
    _REPORTING_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Reporting模块导入失败: {e}")
    _REPORTING_AVAILABLE = False

# 构建 __all__（只包含成功导入的）
__all__ = []

if _CORE_AVAILABLE:
    __all__.extend([
        'OptimizedCPUThreadPool',
        'FastMolecularFeatureCache',
        'GPUManager',
        'ScreeningLogger'
    ])

if _FEATURES_AVAILABLE:
    __all__.extend([
        'FeatureExtractor',
        'Mol2VecHandler',
        'DescriptorCalculator'
    ])

if _SIMILARITY_AVAILABLE:
    __all__.extend([
        'FingerprintSimilarity',
        'Shape3DSimilarity',
        'SimilarityAggregator'
    ])

if _FILTERS_AVAILABLE:
    __all__.extend([
        'NLRP3Filters',
        'ADMETPredictor',
        'ToxicityChecker',
        'PAINSBrenkFilter'
    ])

if _DOCKING_AVAILABLE:
    __all__.append('MolecularDocking')

if _REPORTING_AVAILABLE:
    __all__.extend([
        'TextReportGenerator',
        'HTMLReportGenerator',
        'Visualizer'
    ])

# 打印导入状态
def print_module_status():
    """打印模块可用性状态"""
    print("=" * 60)
    print("NLRP3虚拟筛选系统 - 模块状态")
    print("=" * 60)
    print(f"Core:       {'✓' if _CORE_AVAILABLE else '✗'}")
    print(f"Features:   {'✓' if _FEATURES_AVAILABLE else '✗'}")
    print(f"Similarity: {'✓' if _SIMILARITY_AVAILABLE else '✗'}")
    print(f"Filters:    {'✓' if _FILTERS_AVAILABLE else '✗'}")
    print(f"Docking:    {'✓' if _DOCKING_AVAILABLE else '✗'}")
    print(f"Reporting:  {'✓' if _REPORTING_AVAILABLE else '✗'}")
    print("=" * 60)