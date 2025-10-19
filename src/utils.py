#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工具函数"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    保存配置到YAML文件

    Args:
        config: 配置字典
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置完整性

    Args:
        config: 配置字典

    Returns:
        是否有效
    """
    required_keys = [
        'paths',
        'fingerprints',
        'filters',
        'similarity',
        'screening'
    ]

    for key in required_keys:
        if key not in config:
            print(f"[ERROR] 缺少配置项: {key}")
            return False

    return True