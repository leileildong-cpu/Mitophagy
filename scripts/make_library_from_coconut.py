#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_library_from_coconut.py
-从COCONUT的.smi/.smi.gz或.csv/.csv.gz 生成筛选用的 library.csv
-输出字段:id,name,smiles
-默认输入:data/coconut_csv_lite-09-2025.csv
-默认输出:data/library.csv

用法示例:
python src/make_library_from_coconut.py --topn 3000
python src/make_library_from_coconut.py --input data/coconut_csv_lite-09-2025.csv --topn 3274
python src/make_library_from_coconut.py --input data/custom.smi --output data/library.csv
"""
import os
import sys
import csv
import gzip
import argparse
from typing import List, Tuple, Optional

try:
    import pandas as pd
except Exception as e:
    pd = None

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except Exception:
        rdMolStandardize = None
except Exception as e:
    Chem = None

def detect_ext(path: str) -> str:
    p = path.lower()
    for ext in ('.smi.gz', '.smi', '.csv.gz', '.csv'):
        if p.endswith(ext):
            return ext
    return os.path.splitext(p)[1]

def load_csv_like(path: str, smiles_col: Optional[str]=None, id_col: Optional[str]=None) -> List[Tuple[str, Optional[str]]]:
    """读取 .csv 或 .csv.gz"""
    if pd is None:
        raise RuntimeError("需要 pandas 才能读取 CSV")

    # 尝试多种读取参数
    tried = []
    df = None
    for kwargs in [
        dict(),
        dict(sep=';'),
        dict(encoding='utf-8-sig'),
        dict(sep=';', encoding='utf-8-sig'),
        dict(sep='\t'),
    ]:
        try:
            df = pd.read_csv(path, low_memory=False, **kwargs)
            print(f"[信息] 成功读取CSV，使用参数: {kwargs}")
            break
        except Exception as e:
            tried.append(str(e))
    
    if df is None:
        raise RuntimeError(f"读取 CSV 失败。尝试错误如下：\n" + "\n---\n".join(tried))
    
    print(f"[信息] CSV形状: {df.shape}")
    print(f"[信息] 列名: {list(df.columns)}")
    
    # 统一小写列名映射
    colmap = {c.lower(): c for c in df.columns}
    
    def pick(cands):
        for k in cands:
            if k in colmap:
                return colmap[k]
        return None
    
    smiles_cand = ['smiles', 'canonical_smiles', 'molecule_smiles', 'can_smiles', 'structure_smiles', 'inchi_smiles']
    id_cand = ['coconut_id', 'id', 'coconutid', 'name', 'compound_id', 'identifier', 'entry_id']

    if smiles_col is None:
        smiles_col = pick(smiles_cand)
    if id_col is None:
        id_col = pick(id_cand)

    if smiles_col is None:
        # 如果第一列看起来是 SMILES，也可兜底
        if len(df.columns) >= 1:
            smiles_col = df.columns[0]
            print(f"[警告] 未找到标准SMILES列，使用第一列: {smiles_col}")
        else:
            raise RuntimeError(f"未识别到 SMILES 列。可用列: {list(df.columns)}")
        
    if smiles_col not in df.columns:
        raise RuntimeError(f"指定的 SMILES 列 {smiles_col} 不存在。可用列: {list(df.columns)}")

    print(f"[信息] 使用SMILES列: {smiles_col}")
    if id_col:
        print(f"[信息] 使用ID列: {id_col}")
    else:
        print(f"[信息] 未找到ID列，将自动生成")

    # 正确的DataFrame操作
    if id_col and id_col in df.columns:
        sub = df[[smiles_col, id_col]].copy()
    else:
        sub = df[[smiles_col]].copy()
        id_col = None
    
    sub = sub.dropna(subset=[smiles_col])
    sub = sub.drop_duplicates(subset=[smiles_col])

    items = []
    if id_col:
        for idx, row in sub.iterrows():
            smi = str(row[smiles_col]).strip()
            cid = None if pd.isna(row[id_col]) else str(row[id_col]).strip()
            items.append((smi, cid))
    else:
        for idx, row in sub.iterrows():
            smi = str(row[smiles_col]).strip()
            items.append((smi, None))
    
    return items

def standardize_smiles(smi: str) -> Optional[str]:
    """基础标准化"""
    if Chem is None:
        return smi

    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            return None

        # 取最大片段
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            frags = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
            mol = frags[0]

        # 标准化
        if rdMolStandardize is not None:
            try:
                normalizer = rdMolStandardize.Normalizer()
                mol = normalizer.normalize(mol)
                uncharger = rdMolStandardize.Uncharger()
                mol = uncharger.uncharge(mol)
            except Exception:
                pass

        Chem.SanitizeMol(mol)
        can = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return can
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert COCONUT CSV to library.csv")
    parser.add_argument('--input', '-i', type=str, default='data/coconut_csv_lite-09-2025.csv', help='输入文件路径')
    parser.add_argument('--output', '-o', type=str, default='data/library.csv', help='输出 csv 路径')
    parser.add_argument('--topn', type=int, default=-1, help='仅保留前 N 条 (-1 表示全部)')
    parser.add_argument('--smiles-col', type=str, default=None, help='CSV 模式下指定 SMILES 列名')
    parser.add_argument('--id-col', type=str, default=None, help='CSV 模式下指定 ID 列名')
    parser.add_argument('--source', type=str, default='COCONUT', help='source 字段填充值')
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output
    topn = args.topn

    if not os.path.exists(in_path):
        sys.exit(f"[错误] 输入文件不存在: {in_path}")

    print(f"[信息] 读取输入: {in_path}")

    # 加载原始条目
    try:
        raw_items = load_csv_like(in_path, smiles_col=args.smiles_col, id_col=args.id_col)
    except Exception as e:
        sys.exit(f"[错误] 加载文件失败: {e}")

    total = len(raw_items)
    print(f"[信息] 原始条目数: {total}")

    if total == 0:
        sys.exit("[错误] 未读取到任何数据")

    # 清洗与标准化
    cleaned = []
    invalid = 0
    print("[信息] 开始标准化SMILES...")
    
    for idx, (smi, cid) in enumerate(raw_items, 1):
        if idx % 1000 == 0:
            print(f"[进度] 处理 {idx}/{total}")
            
        if not smi or smi.strip() in ('', 'nan', 'None'):
            invalid += 1
            continue
        std = standardize_smiles(smi.strip())
        if std is None:
            invalid += 1
            continue
        cleaned.append((std, cid))

    print(f"[信息] 解析失败条目: {invalid}")
    print(f"[信息] 解析成功条目: {len(cleaned)}")

    # 去重
    seen = set()
    uniq = []
    for smi, cid in cleaned:
        if smi in seen:
            continue
        seen.add(smi)
        uniq.append((smi, cid))

    print(f"[信息] 去重后条目: {len(uniq)}")

    # 截断 topn
    if topn > 0:
        uniq = uniq[:topn]
        print(f"[信息] 取前 Top-{topn} 条")

    # 生成 ID
    records = []
    for i, (smi, cid) in enumerate(uniq, 1):
        rid = cid if (cid is not None and str(cid).strip() != '') else f'COC_{i:06d}'
        records.append((rid, smi, args.source))

    # 写出 CSV
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', 'smiles'])  # 注意：改为name而不是source，符合run.py的要求
        for rid, smi, source in records:
            writer.writerow([rid, source, smi])  # id, name, smiles

    print(f"[完成] 已写出: {out_path}; 分子数: {len(records)}")

if __name__ == '__main__':
    main()