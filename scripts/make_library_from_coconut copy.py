#!/usr/bin/env python3
#-*_coding:utf-8 -*
""""
make_library_from_coconut.py
-从COCONUT的.smi/.smi.gz或.csv/.csv.gz 生成筛选用的 library.csv
-输出字段:id,smiles, source
-默认输入:data/coconut.smi.gz
-默认输出:data/library.csv

用法示例:
python src/make_library_from_coconut.py --topn 3000
python src/make_library_from_coconut.py --input data/coconut_csv_lite-09-2025.csv --topn 3274
python src/make_library_from_coconut,py --input data/custom.smi --output data/library.csv
"""
import os
import sys
import io
import csv
import gzip
import argparse
from typing import List, Tuple, Optional

# 依赖:pandas(用于 CSV 读取，带回退策略)
try:
    import pandas as pd
except Exception as e:
    pd = None

# 依赖:RDKit(标准化与校验)
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    # 可选:更好的标准化工具
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except Exception:
        dMolStandardize =None
except Exception as e:
    Chem = None


def detect_ext(path: str) -> str:
    p= path.lower()
    for ext in ('.smi.gz','smi','csv.gz','.csv'):
        if p.endswith(ext):
            return ext
    return os.path.splitext(p)[1]


def load_smi_like(path: str)-> List[Tuple[str, Optional[str]]]:
    """
    读取.smi或.smi.gz
    约定:每行格式为"SMILES ID”(用空格或制表符分隔);若只有SMILES，则自动生成ID
    返回:[(smiles, id or None),..]
    """
    opener = gzip.open if path.lower0.endswith('.gz') else open
    items =[]
    with opener(path, 'rt', encoding='utf-8', errors='ignore') as f:
       for i, line in enumerate(f, 1):
            line =line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.splitO
            if len(parts)== 0:
                continue
            smi = parts[0]
            cid = parts[1] if len(parts) > 1 else None
            items.append((smi, cid))
    return items


def load_csv_like(path: str, smiles_col: Optional[str]=None, id_col: Optional[str]=None) -> List[Tuple[str, Optional[str]]]:
    """
    读取 .csv 或 .csv.gz；自动/手动选择 SMILES 与 ID 列
    返回：[(smiles, id_or_None), ...]
    """
    if pd is None:
        raise RuntimeError("需要 pandas 才能读取 CSV；请安装 pandas 或改用 .smi/.smi.gz 输入")

    # 尝试多种读取参数以兼容分隔符/编码
    tried = []
    for kwargs in [
        dict(),
        dict(sep=';'),
        dict(encoding='utf-8-sig'),
        dict(sep=';', encoding='utf-8-sig'),
    ]:
        try:
            df = pd.read_csv(path, low_memory=False, **kwargs)
            break
        except Exception as e:
            tried.append(str(e))
            df = None
    if df is None:
        raise RuntimeError(f"读取 CSV 失败。尝试错误如下：\n" + "\n---\n".join(tried))
    
    #统一小写列名映射
    colmap ={c.lower():c for c in df.columns}

    def pick(cands):
        for k in cands:
            if k in colmap:
                return colmap[k]
        return None
    
    smiles_cand = ['smiles','canonical smiles','molecule smiles','can smiles','structure smiles''inchi smiles']
    id_cand=['coconut id','id','coconutid','name','compound id','identifier','entry id']

    if smiles_col is None:
        smiles_col = pick(smiles_cand)
        if id_col is None:
            id_col = pick(id_cand)

    if smiles_col is None:
        #如果第一列看起来是 SMILES，也可兜底
        if len(df.columns)>= 1:
            smiles_col = df.columns[0]
        else:
            raise RuntimeError(f"未识别到 SMILES 列。可用列:{list(df.columns)}")
        
    if smiles_col not in df.columns:
        raise RuntimeError(f"指定的 SMILES 列 {smiles_col} 不存在。可用列: {list(df.columns)}")

    if id_col is not None and id_col not in df.columns:
        print(f"[警告] 指定的 ID 列 {id_col} 不存在, 将自动生成 ID。可用列{list(df.columns)}", file=sys.stderr)
        id_col = None

    sub = df[smiles_col] + ([id_col] if id_col else []).copy()
    sub = sub.dropna(subset=[smiles_col])
    sub = sub.drop_duplicates(subset=[smiles_col])

    items = []
    if id_col:
        for smi, cid in zip(sub[smiles_col], sub[id_col]):
            items.append((str(smi).strip(), None if pd.isna(cid) else str(cid).strip()))
    else:
        for smi in sub[smiles_col]:
            items.append((str(smi).strip(), None))
    return items


def standardize_smiles(smi: str) -> Optional[str]:
    """
    基础标准化:
    - 解析 SMILES
    - 取最大片段 (去盐)
    - 可选去电中和
    - 规范化并导出 canonical isomeric SMILES
    返回 None 表示失败
    """
    if Chem is None:
        # 无 RDKit 时, 直接回传原 SMILES (不推荐, 但可运行)
        return smi

    mol = Chem.MolFromSmiles(smi, sanitize=True)
    if mol is None:
        return None

    # 取最大片段
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            frags = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
            mol = frags[0]
    except Exception: # 此处省略异常处理逻辑（若有）
        pass

    # RDKit 标准化 (可选)
    if 'rdMolStandardize' in globals() and rdMolStandardize is not None:
        try:
            # 规范化 (reionize/normalize)
            normalizer = rdMolStandardize.Normalize()
            mol = normalizer.normalize(mol)
            # 去电中和
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
            # 金属断裂 (如有)
            lfc = rdMolStandardize.LargestFragmentChooser()
            mol = lfc.choose(mol)
        except Exception:
            pass

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    try:
        can = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None
    return can


def main():
    parser = argparse.ArgumentParser(description="Convert COCONUT SMI/CSV to library.csv")
    parser.add_argument('--input', '-i', type=str, default='data/coconut.smi.gz', help='输入文件路径 (.smi/.smi.gz/.csv/.csv.gz)')
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

    ext = detect_ext(in_path)
    print(f"[信息] 读取输入: {in_path} (detected: {ext})")

    # 加载原始条目
    if ext in ('.smi', '.smi.gz'):
        raw_items = load_smi_like(in_path)
    elif ext in ('.csv', '.csv.gz'):
        raw_items = load_csv_like(in_path, smiles_col=args.smiles_col, id_col=args.id_col)
    else:
        sys.exit(f"[错误] 不支持的输入格式: {ext}")

    total = len(raw_items)
    print(f"[信息] 原始条目数: {total}")

    # 清洗与标准化
    cleaned = []
    invalid = 0
    for idx, (smi, cid) in enumerate(raw_items, 1):
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

    # 去重（按标准化 SMILES 去重）
    seen = set()
    uniq = []
    for smi, cid in cleaned:
        if smi in seen:
            continue
        seen.add(smi)
        uniq.append((smi, cid))

    print(f"[信息] 去重后条目: {len(uniq)}")

    # 截断 topn
    if topn is not None and topn > 0:
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
        writer.writerow(['id','smiles','source'])
        writer.writerows(records)
    
    print(f"[完成] 已写出: {out_path};分子数: {len(records)}")
    print("[提示]后续可运行：python src/run.py 进行相似度筛选")


if __name__=='_main_':
    main()