<<<<<<< HEAD
# Mitophagy Virtual Screening (Demo)

本工程提供三视角相似度融合（1D/2D/3D）的最小可复现流程：
- 1D：Morgan 指纹相似度（Tanimoto）
- 2D：Pharm2D 拓扑药效团相似度（Tanimoto）
- 3D：ETKDG 构象 + 形状相似度（Shape Tanimoto）


## 快速开始

1) 安装 Conda（Miniconda/Anaconda 均可）
2) 创建并激活环境
   conda env create -f environment.yml
   conda run -n mitophagy python -c "import rdkit; print('RDKit OK')"
   conda activate mitophagy


3) 获得数据集
make_library_from_coconut(对数据集全部处理)
    python src/convert_coconut.py --input data/coconut_csv_lite-09-2025.csv --output data/library.csv --smiles-col canonical_smiles --id-col identifier --name-col iupac_name --limit 50000  

make_library_small.py(输入多少，处理三倍的数据，输出指定的数据个数)
   python src/make_small_library.py --input data/coconut_csv_lite-09-2025.csv --output data/library_small.csv --topn 200

3) 环境检查
   python src\test_setup.py


4) 做测试

 python src\run.py --config config.yaml



或使用脚本：
   bash run.sh

4) 查看结果
 

## 配置
参见 config.yaml，可调权重、3D开销、阈值和过滤规则。

# Mitophagy AI VS Repro (Mol2vec + Pharm2D + 3D shape)

本工程复现论文的AI虚拟筛选思路：1D(Mol2vec) + 2D(Pharm2D) + 3D(形状Tanimoto) 多通道相似度融合，按阈值（默认0.75）选出命中候选。包含PAINS与CNS启发式过滤、Top-N预筛加速3D、可配置权重。

特性
- 1D：Mol2vec（余弦）或Morgan（Tanimoto，兜底）；Mol2vec需预训练向量
- 2D：RDKit Gobbi Pharm2D（Tanimoto）
- 3D：ETKDGv3多构象 + 形状Tanimoto（只对Top-N做3D，显著加速）
- 过滤：PAINS + CNS启发式（可关）
- 输出：hits.csv（命中）+ full_scores.csv（全量排名）+ report.txt（摘要）


=======
# Mitophagy
>>>>>>> 67bf3d0e50ff62d0c2ec04f5986b1c8f4a358ea8
