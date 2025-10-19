# diagnose_scores.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs

# 你的14个参考化合物SMILES
reference_compounds = {
    'Quercetin': 'O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12',
    'Curcumin': 'COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O',
    'Resveratrol': 'Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1',
    'EGCG': 'Oc1cc(O)c2c(c1)O[C@H](c1cc(O)c(O)c(O)c1)[C@H](O)C2c1c(O)cc(O)c(O)c1O',
    'Urolithin A': 'O=c1oc2cc(O)cc(O)c2c2c1ccc1c2cccc1',
    'Spermidine': 'NCCCCNCCCN',
    'Nicotinamide': 'NC(=O)c1cccnc1',
    'NAD+': 'NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]2O[C@H]([C@H](O)[C@@H]2O)n2cnc3c(N)ncnc23)[C@@H](O)[C@H]1O',
    'Actinonin': 'CCCCCC[C@H](O)C(=O)N[C@@H](CO)C(=O)N[C@@H](C)C(=O)NO',
    'Berberine': 'COc1ccc2cc3[n+](cc2c1OC)CCc1cc2c(cc1-3)OCO2',
    'Tomatidine': 'C[C@H]1[C@H]2[C@H](C[C@@H]3[C@@H]4CC=C5C[C@@H](O)CC[C@]5(C)[C@H]4CC[C@]23C)O[C@@]11CC[C@@H](C)CN1',
    'Metformin': 'CN(C)C(=N)NC(=N)N',
    'Trehalose': 'OC[C@H]1O[C@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O',
    'Rapamycin': 'CO[C@H]1C[C@@H](C[C@@H](C)/C(C)=C/C=C/C=C/[C@@H](C[C@H]2CC[C@H](O)[C@](O)(O2)C(=O)C(=O)N2CCC[C@H]2C(=O)O1)OC)C/C=C(\\C)C(=O)C1=C(O)C(=C(C)C=C1OC)C'
}

print("=" * 80)
print("步骤1：分析参考化合物的描述符分布")
print("=" * 80)

ref_data = []
for name, smiles in reference_compounds.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        ref_data.append({
            'Name': name,
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'Rings': Descriptors.RingCount(mol)
        })

df_ref = pd.DataFrame(ref_data)
print("\n参考化合物描述符统计：")
print(df_ref.describe())

print("\n各化合物详细信息：")
print(df_ref.to_string())

# 分析两两相似性
print("\n" + "=" * 80)
print("步骤2：分析参考化合物之间的相似性")
print("=" * 80)

fps = []
valid_names = []
for name, smiles in reference_compounds.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(fp)
        valid_names.append(name)

similarities = []
for i in range(len(fps)):
    for j in range(i+1, len(fps)):
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        similarities.append({
            'Compound1': valid_names[i],
            'Compound2': valid_names[j],
            'Similarity': sim
        })

df_sim = pd.DataFrame(similarities)
df_sim = df_sim.sort_values('Similarity', ascending=False)

print(f"\n相似性分布：")
print(f"最高相似性: {df_sim['Similarity'].max():.3f}")
print(f"平均相似性: {df_sim['Similarity'].mean():.3f}")
print(f"中位数相似性: {df_sim['Similarity'].median():.3f}")
print(f"最低相似性: {df_sim['Similarity'].min():.3f}")

print(f"\n最相似的10对化合物：")
print(df_sim.head(10).to_string())

print(f"\n最不相似的10对化合物：")
print(df_sim.tail(10).to_string())

print("\n" + "=" * 80)
print("步骤3：建议")
print("=" * 80)
print("""
基于以上分析，你的参考化合物：
1. 分子量范围：{:.0f} - {:.0f}
2. LogP范围：{:.1f} - {:.1f}
3. 结构多样性很高（平均相似性可能较低）

这意味着：
- 如果相似性阈值太高（如0.5），很难找到同时与多个参考化合物相似的分子
- 需要降低相似性阈值，或者改变评分策略
""".format(df_ref['MW'].min(), df_ref['MW'].max(),
           df_ref['LogP'].min(), df_ref['LogP'].max()))