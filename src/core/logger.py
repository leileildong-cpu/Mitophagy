#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志系统
支持文件轮转、多级别日志和性能监控
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional
import psutil


class ScreeningLogger:
    """虚拟筛选日志系统"""

    def __init__(self,
                 log_dir: str,
                 log_level: str = 'INFO',
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        初始化日志系统

        Args:
            log_dir: 日志目录
            log_level: 日志级别
            max_bytes: 单个日志文件最大大小
            backup_count: 保留的日志文件数量
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 创建logger
        self.logger = logging.getLogger('VirtualScreening')
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # 清除现有handlers
        self.logger.handlers.clear()

        # 文件处理器（带轮转）
        log_file = os.path.join(log_dir, 'screening.log')
        fh = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        fh.setLevel(logging.DEBUG)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # 性能监控日志
        perf_log = os.path.join(log_dir, 'performance.log')
        self.perf_logger = logging.getLogger('Performance')
        perf_fh = RotatingFileHandler(perf_log, maxBytes=max_bytes, backupCount=backup_count)
        perf_fh.setFormatter(formatter)
        self.perf_logger.addHandler(perf_fh)
        self.perf_logger.setLevel(logging.DEBUG)

        self.logger.info("=" * 60)
        self.logger.info("虚拟筛选系统日志启动")
        self.logger.info(f"日志目录: {log_dir}")
        self.logger.info("=" * 60)

    def info(self, msg: str):
        """记录INFO级别日志"""
        self.logger.info(msg)

    def debug(self, msg: str):
        """记录DEBUG级别日志"""
        self.logger.debug(msg)

    def warning(self, msg: str):
        """记录WARNING级别日志"""
        self.logger.warning(msg)

    def error(self, msg: str):
        """记录ERROR级别日志"""
        self.logger.error(msg)

    def critical(self, msg: str):
        """记录CRITICAL级别日志"""
        self.logger.critical(msg)

    def log_performance(self, stage: str, duration: float, items_processed: int = 0):
        """
        记录性能指标

        Args:
            stage: 处理阶段
            duration: 持续时间（秒）
            items_processed: 处理的项目数
        """
        speed = items_processed / duration if duration > 0 else 0

        memory_info = psutil.virtual_memory()

        perf_msg = (
            f"[{stage}] "
            f"Duration: {duration:.2f}s, "
            f"Items: {items_processed}, "
            f"Speed: {speed:.1f} items/s, "
            f"Memory: {memory_info.percent:.1f}%"
        )

        self.perf_logger.info(perf_msg)
        self.logger.info(perf_msg)

    def log_system_info(self):
        """记录系统信息"""
        import platform

        sys_info = [
            "系统信息:",
            f"  OS: {platform.system()} {platform.release()}",
            f"  Python: {platform.python_version()}",
            f"  CPU核心数: {os.cpu_count()}",
            f"  总内存: {psutil.virtual_memory().total / (1024 ** 3):.1f} GB",
            f"  可用内存: {psutil.virtual_memory().available / (1024 ** 3):.1f} GB"
        ]

        for line in sys_info:
            self.logger.info(line)

    def __repr__(self):
        return f"ScreeningLogger(log_dir='{self.log_dir}')"