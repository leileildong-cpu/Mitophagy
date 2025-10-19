#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具
生成各种数据可视化图表（如果matplotlib可用）
"""

import os
from typing import Dict, Optional, List
import pandas as pd
import numpy as np


class Visualizer:
    """数据可视化工具"""

    def __init__(self, output_dir: str):
        """
        初始化可视化工具

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir

        # 检查matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt
            import seaborn as sns

            self.plt = plt
            self.sns = sns
            self.available = True

            # 设置样式
            sns.set_style("whitegrid")
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False

            print("[INFO] 可视化功能已启用")

        except ImportError:
            self.available = False
            print("[WARNING] matplotlib/seaborn不可用，可视化功能禁用")

    def generate_all_plots(self, results: pd.DataFrame, processing_info: Dict) -> List[str]:
        """
        生成所有可视化图表

        Args:
            results: 筛选结果DataFrame
            processing_info: 处理信息字典

        Returns:
            生成的图表文件路径列表
        """
        if not self.available or len(results) == 0:
            return []

        print("[INFO] 生成可视化图表...")

        plots = []

        # 1. 综合性能分析图
        try:
            path = self.plot_performance_analysis(results, processing_info)
            if path:
                plots.append(path)
        except Exception as e:
            print(f"[WARNING] 性能分析图生成失败: {e}")

        # 2. 得分分布图
        try:
            path = self.plot_score_distribution(results)
            if path:
                plots.append(path)
        except Exception as e:
            print(f"[WARNING] 得分分布图生成失败: {e}")

        # 3. 分子性质分布图
        try:
            path = self.plot_property_distribution(results)
            if path:
                plots.append(path)
        except Exception as e:
            print(f"[WARNING] 性质分布图生成失败: {e}")

        # 4. 过滤器通过率图
        try:
            path = self.plot_filter_pass_rates(results)
            if path:
                plots.append(path)
        except Exception as e:
            print(f"[WARNING] 过滤器图生成失败: {e}")

        # 5. 对接结果图（如果有）
        if 'docking_affinity' in results.columns:
            try:
                path = self.plot_docking_results(results)
                if path:
                    plots.append(path)
            except Exception as e:
                print(f"[WARNING] 对接结果图生成失败: {e}")

        print(f"[INFO] 生成了 {len(plots)} 个图表")
        return plots

    def plot_performance_analysis(self, results: pd.DataFrame, processing_info: Dict) -> Optional[str]:
        """绘制综合性能分析图（2x2布局）"""
        if not self.available:
            return None

        fig, axes = self.plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('NLRP3虚拟筛选性能分析', fontsize=16, fontweight='bold')

        # 1. 得分分布直方图
        if 'final_score' in results.columns:
            axes[0, 0].hist(results['final_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('综合得分')
            axes[0, 0].set_ylabel('化合物数量')
            axes[0, 0].set_title('得分分布')
            axes[0, 0].grid(True, alpha=0.3)

        # 2. 处理时间饼图
        perf_stats = processing_info.get('performance_stats', {})
        labels = ['预处理', '特征提取', '相似性计算', '过滤', '其他']
        sizes = [
            perf_stats.get('preprocessing_time', 0),
            perf_stats.get('feature_extraction_time', 0),
            perf_stats.get('similarity_computation_time', 0),
            perf_stats.get('filtering_time', 0),
            max(0, processing_info.get('processing_time_seconds', 0) - sum([
                perf_stats.get('preprocessing_time', 0),
                perf_stats.get('feature_extraction_time', 0),
                perf_stats.get('similarity_computation_time', 0),
                perf_stats.get('filtering_time', 0)
            ]))
        ]

        # 过滤零值
        labels = [l for l, s in zip(labels, sizes) if s > 0]
        sizes = [s for s in sizes if s > 0]

        if sizes:
            axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('处理时间分布')

        # 3. Top-20化合物得分条形图
        if len(results) > 0:
            top_20 = results.head(20)
            y_pos = np.arange(len(top_20))
            axes[1, 0].barh(y_pos, top_20['final_score'].values, color='lightcoral')
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels([f"{row['id'][:10]}" for _, row in top_20.iterrows()])
            axes[1, 0].set_xlabel('综合得分')
            axes[1, 0].set_title('Top-20 候选化合物')
            axes[1, 0].grid(True, alpha=0.3, axis='x')
            axes[1, 0].invert_yaxis()

        # 4. 系统资源使用
        resource_labels = ['CPU线程', 'GPU', '内存(GB)', '处理速度\n(化合物/秒)']
        resource_values = [
            processing_info.get('system_info', {}).get('cpu_threads', 0),
            1 if processing_info.get('system_info', {}).get('gpu_enabled', False) else 0,
            processing_info.get('system_info', {}).get('memory_available_gb', 0),
            processing_info.get('data_stats', {}).get('total_compounds', 0) /
            processing_info.get('processing_time_seconds', 1)
        ]

        bars = axes[1, 1].bar(resource_labels, resource_values, color=['gold', 'red', 'green', 'blue'])
        axes[1, 1].set_title('系统资源使用')
        axes[1, 1].set_ylabel('资源量')

        # 添加数值标签
        for bar, value in zip(bars, resource_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                            f'{value:.1f}', ha='center', va='bottom')

        self.plt.tight_layout()

        # 保存
        output_path = os.path.join(self.output_dir, 'performance_analysis.png')
        self.plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.plt.close()

        return output_path

    def plot_score_distribution(self, results: pd.DataFrame) -> Optional[str]:
        """绘制得分分布图"""
        if not self.available or len(results) == 0:
            return None

        fig, axes = self.plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('得分分布分析', fontsize=16, fontweight='bold')

        # 1. 综合得分分布
        if 'final_score' in results.columns:
            self.sns.histplot(results['final_score'], kde=True, ax=axes[0, 0], color='skyblue')
            axes[0, 0].set_xlabel('综合得分')
            axes[0, 0].set_ylabel('频数')
            axes[0, 0].set_title('综合得分分布')

        # 2. 相似性得分分布
        if 'combined_score' in results.columns:
            self.sns.histplot(results['combined_score'], kde=True, ax=axes[0, 1], color='salmon')
            axes[0, 1].set_xlabel('相似性得分')
            axes[0, 1].set_ylabel('频数')
            axes[0, 1].set_title('相似性得分分布')

        # 3. 各维度相似性对比
        if all(col in results.columns for col in ['1d_max', '2d_max', '3d_max']):
            similarity_data = {
                '1D (Morgan)': results['1d_max'].values,
                '2D (药效团)': results['2d_max'].values,
                '3D (形状)': results['3d_max'].values
            }

            axes[1, 0].boxplot(similarity_data.values(), labels=similarity_data.keys())
            axes[1, 0].set_ylabel('相似性得分')
            axes[1, 0].set_title('各维度相似性对比')
            axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 4. 得分累积分布
        if 'final_score' in results.columns:
            sorted_scores = np.sort(results['final_score'].values)
            cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100

            axes[1, 1].plot(sorted_scores, cumulative, linewidth=2, color='green')
            axes[1, 1].set_xlabel('综合得分')
            axes[1, 1].set_ylabel('累积百分比 (%)')
            axes[1, 1].set_title('得分累积分布')
            axes[1, 1].grid(True, alpha=0.3)

        self.plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'score_distribution.png')
        self.plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.plt.close()

        return output_path

    def plot_property_distribution(self, results: pd.DataFrame) -> Optional[str]:
        """绘制分子性质分布图"""
        if not self.available or len(results) == 0:
            return None

        # 检查必需的列
        required_cols = ['mw', 'logp', 'tpsa', 'hba']
        if not all(col in results.columns for col in required_cols):
            return None

        fig, axes = self.plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('分子性质分布', fontsize=16, fontweight='bold')

        # 1. 分子量分布
        self.sns.histplot(results['mw'], kde=True, ax=axes[0, 0], color='purple')
        axes[0, 0].axvline(500, color='red', linestyle='--', label='Lipinski限制')
        axes[0, 0].set_xlabel('分子量 (Da)')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('分子量分布')
        axes[0, 0].legend()

        # 2. LogP分布
        self.sns.histplot(results['logp'], kde=True, ax=axes[0, 1], color='orange')
        axes[0, 1].axvline(5, color='red', linestyle='--', label='Lipinski限制')
        axes[0, 1].set_xlabel('LogP')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('LogP分布')
        axes[0, 1].legend()

        # 3. TPSA分布
        self.sns.histplot(results['tpsa'], kde=True, ax=axes[1, 0], color='green')
        axes[1, 0].axvline(140, color='red', linestyle='--', label='理想范围')
        axes[1, 0].set_xlabel('TPSA (Ų)')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('TPSA分布')
        axes[1, 0].legend()

        # 4. MW vs LogP散点图
        scatter = axes[1, 1].scatter(results['mw'], results['logp'],
                                     c=results['final_score'],
                                     cmap='viridis', alpha=0.6)
        axes[1, 1].set_xlabel('分子量 (Da)')
        axes[1, 1].set_ylabel('LogP')
        axes[1, 1].set_title('分子量 vs LogP (颜色=得分)')

        # 添加Lipinski界限
        axes[1, 1].axvline(500, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(5, color='red', linestyle='--', alpha=0.5)

        cbar = self.plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('综合得分')

        self.plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'property_distribution.png')
        self.plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.plt.close()

        return output_path

    def plot_filter_pass_rates(self, results: pd.DataFrame) -> Optional[str]:
        """绘制过滤器通过率图"""
        if not self.available or len(results) == 0:
            return None

        # 收集过滤器数据
        filters = []

        if 'nlrp3_compliant' in results.columns:
            rate = sum(results['nlrp3_compliant']) / len(results) * 100
            filters.append(('NLRP3规则', rate))

        if 'lipinski' in results.columns:
            rate = sum(results['lipinski']) / len(results) * 100
            filters.append(('Lipinski五规则', rate))

        if 'veber' in results.columns:
            rate = sum(results['veber']) / len(results) * 100
            filters.append(('Veber规则', rate))

        if 'ghose' in results.columns:
            rate = sum(results['ghose']) / len(results) * 100
            filters.append(('Ghose规则', rate))

        if 'is_safe' in results.columns:
            rate = sum(results['is_safe']) / len(results) * 100
            filters.append(('安全性', rate))

        if not filters:
            return None

        # 绘图
        fig, ax = self.plt.subplots(figsize=(10, 6))

        filter_names, pass_rates = zip(*filters)

        colors = ['#27ae60' if rate >= 70 else '#f39c12' if rate >= 50 else '#e74c3c'
                  for rate in pass_rates]

        bars = ax.barh(filter_names, pass_rates, color=colors)

        # 添加数值标签
        for bar, rate in zip(bars, pass_rates):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                    f'{rate:.1f}%',
                    ha='left', va='center', fontweight='bold')

        ax.set_xlabel('通过率 (%)')
        ax.set_title('过滤器通过率', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 105)
        ax.grid(True, alpha=0.3, axis='x')

        self.plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'filter_pass_rates.png')
        self.plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.plt.close()

        return output_path

    def plot_docking_results(self, results: pd.DataFrame) -> Optional[str]:
        """绘制对接结果分析图"""
        if not self.available:
            return None

        if 'docking_affinity' not in results.columns:
            return None

        docked = results[results['docking_success'] == True]

        if len(docked) == 0:
            return None

        fig, axes = self.plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('分子对接结果分析', fontsize=16, fontweight='bold')

        # 1. 亲和力分布
        self.sns.histplot(docked['docking_affinity'], kde=True, ax=axes[0], color='teal')
        axes[0].set_xlabel('对接亲和力 (kcal/mol)')
        axes[0].set_ylabel('频数')
        axes[0].set_title('对接亲和力分布')
        axes[0].axvline(docked['docking_affinity'].mean(),
                        color='red', linestyle='--', label='平均值')
        axes[0].legend()

        # 2. 亲和力 vs 综合得分
        scatter = axes[1].scatter(docked['final_score'],
                                  docked['docking_affinity'],
                                  alpha=0.6, c='coral')
        axes[1].set_xlabel('综合得分')
        axes[1].set_ylabel('对接亲和力 (kcal/mol)')
        axes[1].set_title('综合得分 vs 对接亲和力')
        axes[1].grid(True, alpha=0.3)

        self.plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'docking_results.png')
        self.plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.plt.close()

        return output_path

    def __repr__(self):
        status = "available" if self.available else "unavailable"
        return f"Visualizer(status={status}, output_dir='{self.output_dir}')"