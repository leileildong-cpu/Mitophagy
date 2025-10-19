#!/usr/bin/env python3
"""检查参考化合物是否能通过筛选库的过滤规则"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors


def check_if_references_pass_filters(references_csv, library_config):
    """检查参考化合物是否能通过库准备的过滤"""

    print("=" * 70)
    print("检查参考化合物是否能通过筛选库过滤规则")
    print("=" * 70)

    # 读取参考化合物
    ref_df = pd.read_csv(references_csv)
    print(f"\n参考化合物数量: {len(ref_df)}")

    # library_preparation.py的默认过滤规则
    default_filters = {
        'mw_min': 150,
        'mw_max': 500,  # ⚠️ MCC950=442.6，接近边界
        'logp_min': -3,
        'logp_max': 5,  # ⚠️ MCC950=4.2, OLT1177=4.5，接近边界
        'hbd_max': 5,
        'hba_max': 10,
        'tpsa_max': 140,
        'rotb_max': 10,
        'heavy_atoms_min': 10,
        'heavy_atoms_max': 50
    }

    print("\n筛选库过滤规则:")
    for key, value in default_filters.items():
        print(f"  {key}: {value}")

    # 检查每个参考化合物
    print("\n检查结果:")
    print("-" * 70)

    passed = 0
    failed = 0
    failed_compounds = []

    for idx, row in ref_df.iterrows():
        name = row.get('name', row.get('id', f'Compound_{idx}'))
        smiles = row['smiles']

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"❌ {name}: SMILES无效")
            failed += 1
            continue

        # 计算性质
        props = {
            'mw': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'hbd': Lipinski.NumHDonors(mol),
            'hba': Lipinski.NumHAcceptors(mol),
            'tpsa': rdMolDescriptors.CalcTPSA(mol),
            'rotb': Lipinski.NumRotatableBonds(mol),
            'heavy_atoms': mol.GetNumHeavyAtoms()
        }

        # 检查是否通过过滤
        violations = []

        if props['mw'] < default_filters['mw_min'] or props['mw'] > default_filters['mw_max']:
            violations.append(f"MW={props['mw']:.1f} (范围:{default_filters['mw_min']}-{default_filters['mw_max']})")

        if props['logp'] < default_filters['logp_min'] or props['logp'] > default_filters['logp_max']:
            violations.append(
                f"LogP={props['logp']:.2f} (范围:{default_filters['logp_min']}-{default_filters['logp_max']})")

        if props['hbd'] > default_filters['hbd_max']:
            violations.append(f"HBD={props['hbd']} (最大:{default_filters['hbd_max']})")

        if props['hba'] > default_filters['hba_max']:
            violations.append(f"HBA={props['hba']} (最大:{default_filters['hba_max']})")

        if props['tpsa'] > default_filters['tpsa_max']:
            violations.append(f"TPSA={props['tpsa']:.1f} (最大:{default_filters['tpsa_max']})")

        if props['rotb'] > default_filters['rotb_max']:
            violations.append(f"RotB={props['rotb']} (最大:{default_filters['rotb_max']})")

        if props['heavy_atoms'] < default_filters['heavy_atoms_min'] or props['heavy_atoms'] > default_filters[
            'heavy_atoms_max']:
            violations.append(
                f"Heavy={props['heavy_atoms']} (范围:{default_filters['heavy_atoms_min']}-{default_filters['heavy_atoms_max']})")

        # 输出结果
        if violations:
            print(f"❌ {name}:")
            print(f"   违反规则: {'; '.join(violations)}")
            failed += 1
            failed_compounds.append(name)
        else:
            print(f"✅ {name}: 通过所有过滤")
            passed += 1

    # 统计
    print("\n" + "=" * 70)
    print("统计结果:")
    print(f"  通过过滤: {passed}/{len(ref_df)} ({passed / len(ref_df) * 100:.1f}%)")
    print(f"  未通过: {failed}/{len(ref_df)} ({failed / len(ref_df) * 100:.1f}%)")

    if failed > 0:
        print(f"\n未通过的化合物: {', '.join(failed_compounds)}")
        print("\n⚠️  警告: 部分参考化合物无法通过筛选库的过滤规则！")
        print("这意味着筛选库中可能没有与这些参考物相似的化合物。")

    print("=" * 70)


if __name__ == "__main__":
    check_if_references_pass_filters('data/references.csv', {})