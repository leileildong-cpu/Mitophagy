# test_fix.py
import fix_rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# 测试分子
mol = Chem.MolFromSmiles('c1ccccc1CCO')  # 苯乙醇

# 测试修复的描述符
print(f"CalcFractionCsp3: {rdMolDescriptors.CalcFractionCsp3(mol)}")
print(f"BertzCT: {rdMolDescriptors.BertzCT(mol)}")