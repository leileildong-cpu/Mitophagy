import os
print("=== 环境检查 ===")

# 1. 检查文件
files_to_check = [
    "models/mol2vec_model.pkl",
    "data/references.csv", 
    "data/library_5k.csv"
]

for file in files_to_check:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"✅ {file} - {size} bytes")
    else:
        print(f"❌ {file} - 文件不存在")

# 2. 测试导入
try:
    from mol2vec.features import mol2alt_sentence
    from gensim.models import word2vec
    from sklearn.cluster import KMeans
    from rdkit import Chem
    print("✅ 所有必要包导入成功")
except ImportError as e:
    print(f"❌ 包导入失败: {e}")

# 3. 测试mol2vec模型
try:
    model = word2vec.Word2Vec.load("models/mol2vec_model.pkl")
    print(f"✅ Mol2vec模型加载成功 - 向量维度: {model.vector_size}")
except Exception as e:
    print(f"❌ Mol2vec模型加载失败: {e}")

print("=== 环境检查完成 ===")