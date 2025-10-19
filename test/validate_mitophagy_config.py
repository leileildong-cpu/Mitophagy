#!/usr/bin/env python3
"""验证线粒体自噬配置是否能通过参考化合物"""

import yaml
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors


def validate_config():
    """验证配置"""

    # 加载配置
    with open('src/mitophagy_library_config.yaml') as f:
        config = yaml.safe_load(f)

    # 加载参考化合物
    ref_df = pd.read_csv('data/references.csv')

    print("=" * 70)
    print("验证线粒体自噬配置")
    print("=" * 70)

    filters = config['filters']
    print("\n配置的过滤规则:")
    for key, value in filters.items():
        print(f"  {key}: {value}")

    print("\n检查14个参考化合物:")
    passed = 0

    for idx, row in ref_df.iterrows():
        name = row['name']
        mol = Chem.MolFromSmiles(row['smiles'])

        if mol is None:
            continue

        # 计算性质
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        heavy = mol.GetNumHeavyAtoms()

        # 检查
        checks = [
            filters['mw_min'] <= mw <= filters['mw_max'],
            filters['logp_min'] <= logp <= filters['logp_max'],
            hbd <= filters['hbd_max'],
            hba <= filters['hba_max'],
            tpsa <= filters['tpsa_max'],
            filters['heavy_atoms_min'] <= heavy <= filters['heavy_atoms_max']
        ]

        if all(checks):
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name} (MW={mw:.1f}, LogP={logp:.2f}, HBD={hbd}, HBA={hba}, TPSA={tpsa:.1f})")

    print(f"\n通过率: {passed}/14 ({passed / 14 * 100:.1f}%)")

    if passed >= 12:
        print("✅ 配置优秀！")
    elif passed >= 10:
        print("✅ 配置良好")
    else:
        print("⚠️ 可能需要进一步放宽")


if __name__ == "__main__":
    validate_config()