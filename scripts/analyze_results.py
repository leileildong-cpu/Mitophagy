# 创建 analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取结果
df = pd.read_csv('hits.csv')

print("=== 虚拟筛选结果分析 ===")
print(f"总化合物数: {len(df)}")
print(f"最高得分: {df['combined_score'].max():.4f}")
print(f"平均得分: {df['combined_score'].mean():.4f}")
print(f"命中数 (≥0.75): {len(df[df['combined_score'] >= 0.75])}")

# 绘制得分分布图
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(df['combined_score'], bins=50, alpha=0.7, color='skyblue')
plt.xlabel('相似性得分')
plt.ylabel('化合物数量')
plt.title('得分分布')
plt.axvline(x=0.75, color='red', linestyle='--', label='阈值 (0.75)')
plt.legend()

plt.subplot(1, 2, 2)
top_20 = df.head(20)
plt.barh(range(len(top_20)), top_20['combined_score'])
plt.ylabel('化合物排名')
plt.xlabel('相似性得分')
plt.title('Top-20 化合物得分')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('results_analysis.png', dpi=300, bbox_inches='tight')
print("分析图表已保存为 results_analysis.png")

# 显示命中化合物详情
print("\n=== 命中化合物详情 ===")
hits = df[df['combined_score'] >= 0.75]
if len(hits) > 0:
    for i, (_, row) in enumerate(hits.iterrows(), 1):
        print(f"{i}. ID: {row['id']}")
        print(f"   名称: {row.get('name', 'Unknown')}")
        print(f"   SMILES: {row['smiles']}")
        print(f"   得分: {row['combined_score']:.4f}")
        print(f"   分子量: {row.get('mw', 'N/A'):.2f}" if pd.notna(row.get('mw')) else "   分子量: N/A")
        print(f"   LogP: {row.get('logp', 'N/A'):.2f}" if pd.notna(row.get('logp')) else "   LogP: N/A")
        print()
else:
    print("没有找到命中化合物")