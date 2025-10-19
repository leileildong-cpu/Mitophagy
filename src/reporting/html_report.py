#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML报告生成器
生成交互式HTML格式的筛选报告，包含分子结构图像
"""

import os
import base64
from io import BytesIO
from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd
import numpy as np


class HTMLReportGenerator:
    """HTML报告生成器"""

    def __init__(self, config: dict, output_dir: str):
        """
        初始化HTML报告生成器

        Args:
            config: 配置字典
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = output_dir
        self.report_file = os.path.join(output_dir, "nlrp3_screening_report.html")

        # 检查RDKit绘图功能
        try:
            from rdkit.Chem import Draw
            self.draw_available = True
        except ImportError:
            self.draw_available = False
            print("[WARNING] RDKit Draw不可用，HTML报告将不包含分子结构图")

    def generate_report(self,
                        results: pd.DataFrame,
                        lib_df: pd.DataFrame,
                        ref_df: pd.DataFrame,
                        processing_time: float,
                        perf_stats: Dict) -> str:
        """
        生成完整HTML报告

        Args:
            results: 筛选结果DataFrame
            lib_df: 库分子DataFrame（包含mol对象）
            ref_df: 参考分子DataFrame
            processing_time: 总处理时间
            perf_stats: 性能统计字典

        Returns:
            报告文件路径
        """
        print("[INFO] 生成HTML报告...")

        # 构建HTML内容
        html_content = self._build_html_document(
            results, lib_df, ref_df, processing_time, perf_stats
        )

        # 保存HTML文件
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"[INFO] HTML报告已保存: {self.report_file}")
        return self.report_file

    def _build_html_document(self,
                             results: pd.DataFrame,
                             lib_df: pd.DataFrame,
                             ref_df: pd.DataFrame,
                             processing_time: float,
                             perf_stats: Dict) -> str:
        """构建完整的HTML文档"""

        # HTML各部分
        html_head = self._generate_html_head()
        html_header = self._generate_header()
        html_stats = self._generate_statistics_section(results, processing_time)
        html_nlrp3 = self._generate_nlrp3_section(results)
        html_top = self._generate_top_compounds_section(results, lib_df)
        html_docking = self._generate_docking_section(results)
        html_charts = self._generate_charts_section(results)
        html_performance = self._generate_performance_section(processing_time, perf_stats)
        html_footer = self._generate_footer()

        # 组装HTML
        html_document = f"""<!DOCTYPE html>
<html lang="zh-CN">
{html_head}
<body>
    <div class="container">
        {html_header}
        {html_stats}
        {html_nlrp3}
        {html_top}
        {html_docking}
        {html_charts}
        {html_performance}
        {html_footer}
    </div>
</body>
</html>
"""
        return html_document

    def _generate_html_head(self) -> str:
        """生成HTML头部"""
        return """<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLRP3抑制剂虚拟筛选报告</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        .header h1 {
            font-size: 36px;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 16px;
            opacity: 0.9;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }

        .stat-card {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
        }

        .stat-card h3 {
            font-size: 42px;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .stat-card p {
            font-size: 14px;
            opacity: 0.9;
        }

        .stat-card.green {
            background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        }

        .stat-card.red {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        }

        .stat-card.orange {
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        }

        .section {
            padding: 30px;
            margin: 20px 0;
        }

        .section h2 {
            color: #2c3e50;
            font-size: 28px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #3498db;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        th, td {
            padding: 12px;
            border: 1px solid #ddd;
            text-align: left;
        }

        th {
            background: #34495e;
            color: white;
            font-weight: bold;
            position: sticky;
            top: 0;
        }

        tr:nth-child(even) {
            background: #f8f9fa;
        }

        tr:hover {
            background: #e8f4f8;
        }

        .molecule-img {
            width: 200px;
            height: 200px;
            border: 2px solid #ddd;
            border-radius: 8px;
            object-fit: contain;
        }

        .pass {
            color: #27ae60;
            font-weight: bold;
            font-size: 18px;
        }

        .fail {
            color: #e74c3c;
            font-weight: bold;
            font-size: 18px;
        }

        .score-bar-container {
            width: 100%;
            height: 25px;
            background: #ecf0f1;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }

        .score-bar {
            height: 100%;
            background: linear-gradient(90deg, #27ae60 0%, #f39c12 50%, #e74c3c 100%);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 12px;
        }

        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            font-weight: bold;
            margin: 2px;
        }

        .badge-success {
            background: #27ae60;
            color: white;
        }

        .badge-warning {
            background: #f39c12;
            color: white;
        }

        .badge-danger {
            background: #e74c3c;
            color: white;
        }

        .badge-info {
            background: #3498db;
            color: white;
        }

        .progress-container {
            margin: 20px 0;
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 14px;
        }

        .progress-bar-bg {
            width: 100%;
            height: 30px;
            background: #ecf0f1;
            border-radius: 15px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2980b9);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.5s ease;
        }

        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
        }

        .footer {
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 14px;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .info-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }

        .info-box strong {
            color: #2c3e50;
            display: block;
            margin-bottom: 5px;
        }
    </style>
</head>"""

    def _generate_header(self) -> str:
        """生成页面头部"""
        return f"""
        <div class="header">
            <h1>🔬 NLRP3炎症小体抑制剂虚拟筛选报告</h1>
            <p>AI辅助的多维相似性筛选系统</p>
            <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>项目: {self.config.get('project_name', 'NLRP3 Screening')}</p>
        </div>
"""

    def _generate_statistics_section(self, results: pd.DataFrame, processing_time: float) -> str:
        """生成统计卡片部分"""
        threshold = self.config.get("ai_model", {}).get("similarity_threshold", 0.30)

        if len(results) > 0:
            hits = results[results['final_score'] >= threshold]
            max_score = results['final_score'].max()
            speed = len(results) / processing_time
        else:
            hits = pd.DataFrame()
            max_score = 0.0
            speed = 0.0

        return f"""
        <div class="stats-grid">
            <div class="stat-card">
                <h3>{len(results)}</h3>
                <p>总化合物数</p>
            </div>
            <div class="stat-card green">
                <h3>{len(hits)}</h3>
                <p>命中化合物</p>
            </div>
            <div class="stat-card orange">
                <h3>{max_score:.3f}</h3>
                <p>最高得分</p>
            </div>
            <div class="stat-card red">
                <h3>{processing_time:.1f}s</h3>
                <p>处理时间</p>
            </div>
            <div class="stat-card">
                <h3>{speed:.1f}</h3>
                <p>化合物/秒</p>
            </div>
        </div>
"""

    def _generate_nlrp3_section(self, results: pd.DataFrame) -> str:
        """生成NLRP3过滤结果部分"""
        if 'nlrp3_compliant' not in results.columns:
            return ""

        # 计算各项通过率
        filters = []

        if 'nlrp3_compliant' in results.columns:
            count = sum(results['nlrp3_compliant'])
            rate = count / len(results) * 100 if len(results) > 0 else 0
            filters.append(('NLRP3规则', count, len(results), rate))

        if 'lipinski' in results.columns:
            count = sum(results['lipinski'])
            rate = count / len(results) * 100 if len(results) > 0 else 0
            filters.append(('Lipinski五规则', count, len(results), rate))

        if 'veber' in results.columns:
            count = sum(results['veber'])
            rate = count / len(results) * 100 if len(results) > 0 else 0
            filters.append(('Veber规则', count, len(results), rate))

        if 'ghose' in results.columns:
            count = sum(results['ghose'])
            rate = count / len(results) * 100 if len(results) > 0 else 0
            filters.append(('Ghose规则', count, len(results), rate))

        if 'is_safe' in results.columns:
            count = sum(results['is_safe'])
            rate = count / len(results) * 100 if len(results) > 0 else 0
            filters.append(('安全性', count, len(results), rate))

        # 构建HTML
        html = """
        <div class="section">
            <h2>📊 NLRP3特异性过滤结果</h2>
"""

        for filter_name, passed, total, rate in filters:
            html += f"""
            <div class="progress-container">
                <div class="progress-label">
                    <span><strong>{filter_name}</strong></span>
                    <span>{passed}/{total} ({rate:.1f}%)</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: {rate}%;">
                        {rate:.1f}%
                    </div>
                </div>
            </div>
"""

        html += """
        </div>
"""
        return html

    def _generate_top_compounds_section(self, results: pd.DataFrame, lib_df: pd.DataFrame, top_n: int = 20) -> str:
        """生成Top化合物部分"""
        if len(results) == 0:
            return ""

        top_n = min(top_n, len(results))

        html = f"""
        <div class="section">
            <h2>🏆 Top-{top_n} 候选化合物</h2>
            <table>
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>ID</th>
                        <th>结构</th>
                        <th>综合评分</th>
                        <th>相似性</th>
                        <th>生物利用度</th>
                        <th>状态</th>
                    </tr>
                </thead>
                <tbody>
"""

        for i, (idx, row) in enumerate(results.head(top_n).iterrows(), 1):
            # 获取分子图像
            if self.draw_available:
                try:
                    mol = lib_df.loc[lib_df['id'] == row['id'], 'mol'].iloc[0]
                    img_str = self._mol_to_base64(mol)
                except:
                    img_str = ""
            else:
                img_str = ""

            # 状态标记
            nlrp3_status = '<span class="pass">✓</span>' if row.get('nlrp3_compliant',
                                                                    False) else '<span class="fail">✗</span>'
            safe_status = '<span class="pass">✓</span>' if row.get('is_safe', False) else '<span class="fail">✗</span>'

            # 得分条
            final_score = float(row.get('final_score', 0.0))
            combined_score = float(row.get('combined_score', 0.0))
            bioavail = float(row.get('bioavailability_score', 0.0)) if 'bioavailability_score' in row else 0.0

            score_percent = final_score * 100

            html += f"""
                    <tr>
                        <td><strong>{i}</strong></td>
                        <td>{row['id']}</td>
                        <td>
                            {f'<img src="data:image/png;base64,{img_str}" class="molecule-img" alt="Molecule">' if img_str else '结构不可用'}
                        </td>
                        <td>
                            <div class="score-bar-container">
                                <div class="score-bar" style="width: {score_percent}%;">
                                    {final_score:.3f}
                                </div>
                            </div>
                        </td>
                        <td>{combined_score:.3f}</td>
                        <td>{bioavail:.2f}</td>
                        <td>
                            NLRP3: {nlrp3_status}<br>
                            安全: {safe_status}
                        </td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>
"""
        return html

    def _generate_docking_section(self, results: pd.DataFrame) -> str:
        """生成对接结果部分"""
        if 'docking_affinity' not in results.columns:
            return ""

        docked = results[results['docking_success'] == True]

        if len(docked) == 0:
            return ""

        affinities = docked['docking_affinity'].values

        html = f"""
        <div class="section">
            <h2>⚡ 分子对接结果</h2>
            <div class="info-grid">
                <div class="info-box">
                    <strong>成功对接</strong>
                    {len(docked)} 个化合物
                </div>
                <div class="info-box">
                    <strong>最佳亲和力</strong>
                    {affinities.min():.2f} kcal/mol
                </div>
                <div class="info-box">
                    <strong>平均亲和力</strong>
                    {affinities.mean():.2f} kcal/mol
                </div>
                <div class="info-box">
                    <strong>中位数亲和力</strong>
                    {np.median(affinities):.2f} kcal/mol
                </div>
            </div>

            <h3>Top-10 对接结果</h3>
            <table>
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>化合物ID</th>
                        <th>对接亲和力 (kcal/mol)</th>
                        <th>综合评分</th>
                    </tr>
                </thead>
                <tbody>
"""

        top_docked = docked.nsmallest(10, 'docking_affinity')

        for i, (_, row) in enumerate(top_docked.iterrows(), 1):
            html += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{row['id']}</td>
                        <td><strong>{row['docking_affinity']:.2f}</strong></td>
                        <td>{row.get('final_score', 0.0):.3f}</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>
"""
        return html

    def _generate_charts_section(self, results: pd.DataFrame) -> str:
        """生成图表部分（简化版本，使用纯HTML/CSS）"""
        if len(results) == 0:
            return ""

        return """
        <div class="section">
            <h2>📈 数据可视化</h2>
            <p style="color: #7f8c8d; font-style: italic;">
                提示: 完整的交互式图表需要JavaScript库支持。当前显示简化版本。
            </p>
        </div>
"""

    def _generate_performance_section(self, processing_time: float, perf_stats: Dict) -> str:
        """生成性能统计部分"""
        html = """
        <div class="section">
            <h2>⚙️ 性能统计</h2>
            <div class="info-grid">
"""

        stats = [
            ("总处理时间", f"{processing_time:.2f} 秒"),
            ("预处理", f"{perf_stats.get('preprocessing_time', 0):.2f} 秒"),
            ("特征提取", f"{perf_stats.get('feature_extraction_time', 0):.2f} 秒"),
            ("相似性计算", f"{perf_stats.get('similarity_computation_time', 0):.2f} 秒"),
            ("过滤", f"{perf_stats.get('filtering_time', 0):.2f} 秒"),
        ]

        if 'docking_time' in perf_stats and perf_stats['docking_time'] > 0:
            stats.append(("对接", f"{perf_stats['docking_time']:.2f} 秒"))

        for label, value in stats:
            html += f"""
                <div class="info-box">
                    <strong>{label}</strong>
                    {value}
                </div>
"""

        html += """
            </div>
        </div>
"""
        return html

    def _generate_footer(self) -> str:
        """生成页脚"""
        return f"""
        <div class="footer">
            <p>NLRP3抑制剂虚拟筛选系统 v3.0</p>
            <p>AI辅助的多维相似性筛选 | 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
"""

    def _mol_to_base64(self, mol, size=(200, 200)) -> str:
        """
        将RDKit分子对象转换为Base64编码的PNG图像

        Args:
            mol: RDKit分子对象
            size: 图像尺寸

        Returns:
            Base64编码字符串
        """
        try:
            from rdkit.Chem import Draw

            img = Draw.MolToImage(mol, size=size)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return img_str

        except Exception as e:
            return ""

    def __repr__(self):
        return f"HTMLReportGenerator(output='{self.report_file}')"