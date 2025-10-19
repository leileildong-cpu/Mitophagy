#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
断点续传管理器
支持保存和恢复筛选进度，避免意外中断导致的重复计算
"""

import os
import pickle
import json
from typing import Dict, Optional, Any
from datetime import datetime
import hashlib


class CheckpointManager:
    """断点续传管理器"""

    def __init__(self, output_dir: str, enabled: bool = True):
        """
        初始化断点管理器

        Args:
            output_dir: 输出目录
            enabled: 是否启用断点续传
        """
        self.output_dir = output_dir
        self.enabled = enabled
        self.checkpoint_dir = os.path.join(output_dir, '.checkpoints')

        if self.enabled:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"[INFO] 断点续传已启用: {self.checkpoint_dir}")
        else:
            print("[INFO] 断点续传已禁用")

    def save_checkpoint(self,
                        stage: str,
                        data: Dict[str, Any],
                        metadata: Optional[Dict] = None) -> bool:
        """
        保存断点

        Args:
            stage: 阶段名称（如'preprocessing', 'features', 'similarity'）
            data: 要保存的数据
            metadata: 元数据（可选）

        Returns:
            是否成功保存
        """
        if not self.enabled:
            return False

        try:
            # 生成断点文件名
            checkpoint_file = os.path.join(self.checkpoint_dir, f'{stage}.pkl')
            metadata_file = os.path.join(self.checkpoint_dir, f'{stage}_meta.json')

            # 保存数据
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 保存元数据
            if metadata is None:
                metadata = {}

            metadata.update({
                'stage': stage,
                'timestamp': datetime.now().isoformat(),
                'file_size': os.path.getsize(checkpoint_file),
                'checksum': self._calculate_checksum(checkpoint_file)
            })

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"[INFO] 断点已保存: {stage}")
            return True

        except Exception as e:
            print(f"[WARNING] 断点保存失败 ({stage}): {e}")
            return False

    def load_checkpoint(self, stage: str) -> Optional[Dict[str, Any]]:
        """
        加载断点

        Args:
            stage: 阶段名称

        Returns:
            保存的数据，失败返回None
        """
        if not self.enabled:
            return None

        checkpoint_file = os.path.join(self.checkpoint_dir, f'{stage}.pkl')
        metadata_file = os.path.join(self.checkpoint_dir, f'{stage}_meta.json')

        # 检查文件是否存在
        if not os.path.exists(checkpoint_file):
            return None

        try:
            # 验证元数据
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # 验证校验和
                current_checksum = self._calculate_checksum(checkpoint_file)
                if current_checksum != metadata.get('checksum'):
                    print(f"[WARNING] 断点文件校验失败 ({stage})，将重新计算")
                    return None

                print(f"[INFO] 发现有效断点: {stage} (保存于 {metadata.get('timestamp')})")

            # 加载数据
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)

            print(f"[INFO] 断点已加载: {stage}")
            return data

        except Exception as e:
            print(f"[WARNING] 断点加载失败 ({stage}): {e}")
            return None

    def has_checkpoint(self, stage: str) -> bool:
        """
        检查是否存在有效的断点

        Args:
            stage: 阶段名称

        Returns:
            是否存在有效断点
        """
        if not self.enabled:
            return False

        checkpoint_file = os.path.join(self.checkpoint_dir, f'{stage}.pkl')
        return os.path.exists(checkpoint_file)

    def remove_checkpoint(self, stage: str) -> bool:
        """
        删除断点

        Args:
            stage: 阶段名称

        Returns:
            是否成功删除
        """
        if not self.enabled:
            return False

        checkpoint_file = os.path.join(self.checkpoint_dir, f'{stage}.pkl')
        metadata_file = os.path.join(self.checkpoint_dir, f'{stage}_meta.json')

        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            if os.path.exists(metadata_file):
                os.remove(metadata_file)

            print(f"[INFO] 断点已删除: {stage}")
            return True

        except Exception as e:
            print(f"[WARNING] 断点删除失败 ({stage}): {e}")
            return False

    def clear_all_checkpoints(self) -> bool:
        """
        清除所有断点

        Returns:
            是否成功清除
        """
        if not self.enabled:
            return False

        try:
            import shutil
            if os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
                os.makedirs(self.checkpoint_dir, exist_ok=True)

            print("[INFO] 所有断点已清除")
            return True

        except Exception as e:
            print(f"[WARNING] 断点清除失败: {e}")
            return False

    def list_checkpoints(self) -> list:
        """
        列出所有可用的断点

        Returns:
            断点列表
        """
        if not self.enabled or not os.path.exists(self.checkpoint_dir):
            return []

        checkpoints = []

        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('_meta.json'):
                stage = filename.replace('_meta.json', '')
                metadata_file = os.path.join(self.checkpoint_dir, filename)

                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    checkpoints.append({
                        'stage': stage,
                        'timestamp': metadata.get('timestamp'),
                        'file_size': metadata.get('file_size', 0)
                    })
                except:
                    continue

        return checkpoints

    def _calculate_checksum(self, file_path: str) -> str:
        """计算文件校验和（MD5）"""
        md5_hash = hashlib.md5()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)

        return md5_hash.hexdigest()

    def get_checkpoint_info(self, stage: str) -> Optional[Dict]:
        """
        获取断点信息

        Args:
            stage: 阶段名称

        Returns:
            断点信息字典
        """
        if not self.enabled:
            return None

        metadata_file = os.path.join(self.checkpoint_dir, f'{stage}_meta.json')

        if not os.path.exists(metadata_file):
            return None

        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except:
            return None

    def __repr__(self):
        status = "enabled" if self.enabled else "disabled"
        return f"CheckpointManager(status={status}, dir='{self.checkpoint_dir}')"