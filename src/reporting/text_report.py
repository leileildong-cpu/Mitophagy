#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本报告生成器
生成详细的文本格式筛选报告
"""

import os
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import psutil


class TextReportGenerator:
    """文本报告生成器"""

    def __init__(self, config: dict, output_dir: str):
        """
        初始化文本报告生成器

        Args:
            config: 配置字典
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = output_dir
        self.report_file = os.path.join(output_dir, config["output"]["report_txt"])

    def generate_report(self,
                        results: pd.DataFrame,
                        ref_df: pd.DataFrame,
                        lib_df: pd.DataFrame,
                        processing_time: float,
                        perf_stats: Dict) -> str:
        """
        生成完整文本报告

        Args:
            results: 筛选结果DataFrame
            ref_df: 参考分子DataFrame
            lib_df: 库分子DataFrame
            processing_time: 总处理时间
            perf_stats: 性能统计字典

        Returns:
            报告文件路径
        """
        print("[INFO] 生成文本报告...")

        # 收集报告内容
        report_lines = []

        # 标题
        report_lines.extend(self._generate_header())

        # 系统信息
        report_lines.extend(self._generate_system_info(perf_stats))

        # 数据概览
        report_lines.extend(self._generate_data_overview(ref_df, lib_df, results))

        # 筛选结果
        report_lines.extend(self._generate_screening_results(results))

        # 得分统计
        report_lines.extend(self._generate_score_statistics(results))

        # NLRP3过滤统计
        report_lines.extend(self._generate_nlrp3_statistics(results))

        # Top候选化合物
        report_lines.extend(self._generate_top_compounds(results))

        # 对接结果（如果有）
        if 'docking_affinity' in results.columns:
            report_lines.extend(self._generate_docking_results(results))

        # 性能统计
        report_lines.extend(self._generate_performance_stats(processing_time, perf_stats, results))

        # 过滤统计详情
        report_lines.extend(self._generate_filter_statistics(results))

        # 聚类分析（如果有）
        if 'cluster' in results.columns:
            report_lines.extend(self._generate_cluster_analysis(results))

        # 保存报告
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"[INFO] 文本报告已保存: {self.report_file}")
        return self.report_file

    def _generate_header(self) -> list:
        """生成报告头部"""
        return [
            "=" * 90,
            " NLRP3炎症小体抑制剂虚拟筛选报告",
            " AI辅助的多维相似性筛选系统",
            "=" * 90,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"项目名称: {self.config.get('project_name', 'NLRP3 Screening')}",
            f"靶点: {self.config.get('target_protein', 'NLRP3')}",
            ""
        ]

    def _generate_system_info(self, perf_stats: Dict) -> list:
        """生成系统信息"""
        memory_info = psutil.virtual_memory()

        lines = [
            "=== 系统配置 ===",
            f"CPU核心数: {os.cpu_count()}",
            f"可用内存: {memory_info.available / (1024 ** 3):.1f} GB / {memory_info.total / (1024 ** 3):.1f} GB",
        ]

        # GPU信息
        try:
            import torch
            if torch.cuda.is_available():
                lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
                lines.append("GPU加速: 启用")
            else:
                lines.append("GPU加速: 禁用")
        except:
            lines.append("GPU加速: 禁用")

        lines.append("")
        return lines

    def _generate_data_overview(self, ref_df: pd.DataFrame,
                                lib_df: pd.DataFrame,
                                results: pd.DataFrame) -> list:
        """生成数据概览"""
        return [
            "=== 数据概览 ===",
            f"参考分子数: {len(ref_df)}",
            f"库分子数: {len(lib_df)}",
            f"有效化合物: {len(results)}",
            ""
        ]

    def _generate_screening_results(self, results: pd.DataFrame) -> list:
        """生成筛选结果"""
        threshold = self.config.get("ai_model", {}).get("similarity_threshold", 0.30)

        if len(results) > 0:
            hits = results[results['final_score'] >= threshold]
            hit_rate = len(hits) / len(results) * 100
        else:
            hits = pd.DataFrame()
            hit_rate = 0.0

        return [
            "=== 筛选结果 ===",
            f"评分阈值: {threshold:.3f}",
            f"命中化合物数: {len(hits)}",
            f"命中率: {hit_rate:.2f}%",
            ""
        ]

    def _generate_score_statistics(self, results: pd.DataFrame) -> list:
        """生成得分统计"""
        if len(results) == 0:
            return [
                "=== 得分统计 ===",
                "无有效结果",
                ""
            ]

        lines = [
            "=== 得分统计 ===",
            f"最高得分: {results['final_score'].max():.4f}",
            f"平均得分: {results['final_score'].mean():.4f}",
            f"中位数得分: {results['final_score'].median():.4f}",
            f"标准差: {results['final_score'].std():.4f}",
            f"75分位数: {results['final_score'].quantile(0.75):.4f}",
            f"90分位数: {results['final_score'].quantile(0.90):.4f}",
            ""
        ]

        # 相似性得分统计
        if 'combined_score' in results.columns:
            lines.extend([
                "相似性得分统计:",
                f"  最高相似性: {results['combined_score'].max():.4f}",
                f"  平均相似性: {results['combined_score'].mean():.4f}",
                ""
            ])

        return lines

    def _generate_nlrp3_statistics(self, results: pd.DataFrame) -> list:
        """生成NLRP3过滤统计"""
        if 'nlrp3_compliant' not in results.columns:
            return []

        lines = [
            "=== NLRP3特异性过滤统计 ===",
        ]

        # NLRP3规则
        if 'nlrp3_compliant' in results.columns:
            nlrp3_pass = sum(results['nlrp3_compliant'])
            nlrp3_rate = nlrp3_pass / len(results) * 100 if len(results) > 0 else 0
            lines.append(f"NLRP3规则通过: {nlrp3_pass}/{len(results)} ({nlrp3_rate:.1f}%)")

        # Lipinski
        if 'lipinski' in results.columns:
            lipinski_pass = sum(results['lipinski'])
            lipinski_rate = lipinski_pass / len(results) * 100 if len(results) > 0 else 0
            lines.append(f"Lipinski五规则: {lipinski_pass}/{len(results)} ({lipinski_rate:.1f}%)")

        # Veber
        if 'veber' in results.columns:
            veber_pass = sum(results['veber'])
            veber_rate = veber_pass / len(results) * 100 if len(results) > 0 else 0
            lines.append(f"Veber规则: {veber_pass}/{len(results)} ({veber_rate:.1f}%)")

        # 安全性
        if 'is_safe' in results.columns:
            safe_count = sum(results['is_safe'])
            safe_rate = safe_count / len(results) * 100 if len(results) > 0 else 0
            lines.append(f"安全性通过: {safe_count}/{len(results)} ({safe_rate:.1f}%)")

        # PAINS
        if 'is_pains' in results.columns:
            pains_count = sum(results['is_pains'])
            pains_rate = pains_count / len(results) * 100 if len(results) > 0 else 0
            lines.append(f"PAINS阳性: {pains_count}/{len(results)} ({pains_rate:.1f}%)")

        # Brenk
        if 'brenk_clean' in results.columns:
            brenk_pass = sum(results['brenk_clean'])
            brenk_rate = brenk_pass / len(results) * 100 if len(results) > 0 else 0
            lines.append(f"Brenk通过: {brenk_pass}/{len(results)} ({brenk_rate:.1f}%)")

        lines.append("")
        return lines

    def _generate_top_compounds(self, results: pd.DataFrame, top_n: int = 30) -> list:
        """生成Top化合物列表"""
        if len(results) == 0:
            return []

        top_n = min(top_n, len(results))
        lines = [
            f"=== Top-{top_n} 候选化合物 ===",
            ""
        ]

        for i, (_, row) in enumerate(results.head(top_n).iterrows(), 1):
            row_id = str(row.get('id', 'Unknown'))[:20]
            row_name = str(row.get('name', 'Unknown'))[:30]
            final_score = float(row.get('final_score', 0.0))
            combined_score = float(row.get('combined_score', 0.0))

            # 状态标记
            nlrp3_status = "✓" if row.get('nlrp3_compliant', False) else "✗"
            safe_status = "✓" if row.get('is_safe', False) else "✗"
            pains_status = "✗" if row.get('is_pains', False) else "✓"

            # 生物利用度
            bioavail = float(row.get('bioavailability_score', 0.0)) if 'bioavailability_score' in row else 0.0

            # 得分条
            score_bar = "█" * int(final_score * 20) + "░" * (20 - int(final_score * 20))

            lines.append(
                f"{i:2d}. {row_id:20s} | {row_name:30s}"
            )
            lines.append(
                f"    综合得分: {final_score:.4f} [{score_bar}]"
            )
            lines.append(
                f"    相似性: {combined_score:.4f} | 生物利用度: {bioavail:.2f}"
            )
            lines.append(
                f"    状态: NLRP3={nlrp3_status} 安全={safe_status} PAINS={pains_status}"
            )

            # 对接结果
            if 'docking_affinity' in row and row.get('docking_success', False):
                lines.append(
                    f"    对接亲和力: {row['docking_affinity']:.2f} kcal/mol"
                )

            lines.append("")

        return lines

    def _generate_docking_results(self, results: pd.DataFrame) -> list:
        """生成对接结果统计"""
        docked = results[results['docking_success'] == True]

        if len(docked) == 0:
            return []

        affinities = docked['docking_affinity'].values

        return [
            "=== 分子对接结果 ===",
            f"成功对接: {len(docked)} 个化合物",
            f"最佳亲和力: {affinities.min():.2f} kcal/mol",
            f"平均亲和力: {affinities.mean():.2f} kcal/mol",
            f"中位数亲和力: {pd.Series(affinities).median():.2f} kcal/mol",
            ""
        ]

    def _generate_performance_stats(self, processing_time: float,
                                    perf_stats: Dict,
                                    results: pd.DataFrame) -> list:
        """生成性能统计"""
        speed = len(results) / processing_time if processing_time > 0 else 0

        lines = [
            "=== 性能统计 ===",
            f"总处理时间: {processing_time:.2f} 秒",
            f"处理速度: {speed:.1f} 化合物/秒",
            "",
            "各阶段耗时:",
            f"  - 预处理: {perf_stats.get('preprocessing_time', 0):.2f}s",
            f"  - 特征提取: {perf_stats.get('feature_extraction_time', 0):.2f}s",
            f"  - 相似性计算: {perf_stats.get('similarity_computation_time', 0):.2f}s",
            f"  - 过滤: {perf_stats.get('filtering_time', 0):.2f}s",
        ]

        if 'docking_time' in perf_stats and perf_stats['docking_time'] > 0:
            lines.append(f"  - 对接: {perf_stats['docking_time']:.2f}s")

        # 缓存性能
        if 'cache_hits' in perf_stats and 'cache_misses' in perf_stats:
            cache_hits = perf_stats['cache_hits']
            cache_misses = perf_stats['cache_misses']
            total_accesses = cache_hits + cache_misses
            hit_rate = cache_hits / total_accesses * 100 if total_accesses > 0 else 0

            lines.extend([
                "",
                "缓存性能:",
                f"  - 命中次数: {cache_hits}",
                f"  - 未命中次数: {cache_misses}",
                f"  - 命中率: {hit_rate:.1f}%"
            ])

        lines.append("")
        return lines

    def _generate_filter_statistics(self, results: pd.DataFrame) -> list:
        """生成详细过滤统计"""
        if len(results) == 0:
            return []

        lines = [
            "=== 详细过滤统计 ===",
        ]

        # 分子量分布
        if 'mw' in results.columns:
            lines.extend([
                "分子量分布:",
                f"  - 范围: {results['mw'].min():.1f} - {results['mw'].max():.1f}",
                f"  - 平均: {results['mw'].mean():.1f}",
                f"  - 中位数: {results['mw'].median():.1f}",
            ])

        # LogP分布
        if 'logp' in results.columns:
            lines.extend([
                "LogP分布:",
                f"  - 范围: {results['logp'].min():.2f} - {results['logp'].max():.2f}",
                f"  - 平均: {results['logp'].mean():.2f}",
            ])

        # TPSA分布
        if 'tpsa' in results.columns:
            lines.extend([
                "TPSA分布:",
                f"  - 范围: {results['tpsa'].min():.1f} - {results['tpsa'].max():.1f}",
                f"  - 平均: {results['tpsa'].mean():.1f}",
            ])

        lines.append("")
        return lines

    def _generate_cluster_analysis(self, results: pd.DataFrame) -> list:
        """生成聚类分析"""
        if 'cluster' not in results.columns:
            return []

        lines = [
            "=== 聚类分析 ===",
        ]

        cluster_counts = results['cluster'].value_counts().sort_index()

        for cluster_id, count in cluster_counts.items():
            percentage = count / len(results) * 100
            bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
            lines.append(
                f"聚类 {cluster_id}: {count:4d} 个化合物 ({percentage:5.1f}%) [{bar}]"
            )

        lines.append("")
        return lines

    def __repr__(self):
        return f"TextReportGenerator(output='{self.report_file}')"