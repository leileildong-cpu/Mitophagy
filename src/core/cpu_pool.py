#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的CPU多线程池管理器
支持智能任务调度、批处理和早停
"""

import os
import psutil
from typing import List, Callable, Optional, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm


class OptimizedCPUThreadPool:
    """优化的CPU多线程管理器"""

    def __init__(self, max_workers: Optional[int] = None):
        """
        初始化CPU线程池

        Args:
            max_workers: 最大工作线程数，None则自动检测
        """
        cpu_count = os.cpu_count() or 1
        available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB

        # 基于CPU和内存动态调整线程数
        if max_workers is None:
            if available_memory > 16:  # 大内存系统
                self.max_workers = min(32, cpu_count * 2)
            elif available_memory > 8:  # 中等内存
                self.max_workers = min(16, cpu_count + 4)
            else:  # 小内存系统
                self.max_workers = min(8, cpu_count)
        else:
            self.max_workers = max_workers

        print(f"[INFO] 优化CPU线程池: {self.max_workers} 个工作线程 (可用内存: {available_memory:.1f}GB)")

        self.process_pool = None
        self.thread_pool = None

    def map_parallel(self,
                     func: Callable,
                     items: List[Any],
                     desc: str = "处理中",
                     use_processes: bool = False,
                     batch_size: Optional[int] = None,
                     early_stop_threshold: Optional[int] = None) -> List[Any]:
        """
        并行映射函数到items列表

        Args:
            func: 要应用的函数
            items: 输入项列表
            desc: 进度条描述
            use_processes: 是否使用进程池（默认线程池）
            batch_size: 批处理大小
            early_stop_threshold: 早停阈值

        Returns:
            处理结果列表
        """
        # 小数据集直接串行处理
        if len(items) < 50:
            return [func(item) for item in tqdm(items, desc=desc)]

        # 动态批处理
        if batch_size is None:
            batch_size = max(1, len(items) // (self.max_workers * 4))

        # 选择执行器
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        results = []
        with executor_class(max_workers=self.max_workers) as executor:
            # 批量提交任务
            futures = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                if len(batch) == 1:
                    future = executor.submit(func, batch[0])
                else:
                    # 批处理函数
                    batch_func = partial(self._batch_process, func)
                    future = executor.submit(batch_func, batch)
                futures.append(future)

            # 收集结果，支持早停
            with tqdm(total=len(items), desc=f"{desc} (批处理)") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        if isinstance(batch_results, list):
                            results.extend(batch_results)
                            pbar.update(len(batch_results))
                        else:
                            results.append(batch_results)
                            pbar.update(1)

                        # 早停检查
                        if early_stop_threshold and len(results) >= early_stop_threshold:
                            print(f"[INFO] 达到早停阈值 {early_stop_threshold}，停止处理")
                            break

                    except Exception as e:
                        print(f"[WARNING] 批处理任务失败: {e}")
                        continue

        return results[:len(items)]  # 确保返回正确数量的结果

    def _batch_process(self, func: Callable, batch: List[Any]) -> List[Any]:
        """批处理函数"""
        return [func(item) for item in batch]

    def map_parallel_advanced(self,
                              func: Callable,
                              items: List[Any],
                              desc: str = "处理中",
                              use_processes: bool = False,
                              chunk_generator: Optional[Callable] = None) -> List[Any]:
        """
        高级并行处理，支持自定义分块策略

        Args:
            func: 处理函数
            items: 输入项
            desc: 描述
            use_processes: 是否使用进程
            chunk_generator: 自定义分块生成器

        Returns:
            结果列表
        """
        if chunk_generator is None:
            # 默认分块策略
            return self.map_parallel(func, items, desc, use_processes)

        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        results = []

        with executor_class(max_workers=self.max_workers) as executor:
            futures = []

            for chunk in chunk_generator(items):
                future = executor.submit(func, chunk)
                futures.append(future)

            with tqdm(total=len(futures), desc=desc) as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        print(f"[WARNING] 任务失败: {e}")

        return results

    def get_optimal_batch_size(self, total_items: int, memory_per_item: float = 1.0) -> int:
        """
        计算最优批处理大小

        Args:
            total_items: 总项目数
            memory_per_item: 每项占用内存（MB）

        Returns:
            最优批处理大小
        """
        available_memory = psutil.virtual_memory().available / (1024 ** 2)  # MB

        # 预留20%内存
        usable_memory = available_memory * 0.8

        # 基于内存的批处理大小
        memory_based = int(usable_memory / (memory_per_item * self.max_workers))

        # 基于工作线程的批处理大小
        worker_based = max(1, total_items // (self.max_workers * 4))

        # 取较小值，确保不会内存溢出
        optimal = min(memory_based, worker_based, 1000)  # 最大1000

        return max(1, optimal)

    def __del__(self):
        """清理资源"""
        if self.process_pool:
            self.process_pool.shutdown(wait=False)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)

    def __repr__(self):
        return f"OptimizedCPUThreadPool(max_workers={self.max_workers})"