import os, sys, csv, argparse
import pandas as pd
from typing import Optional, List, Tuple
from rdkit import Chem

def detect_sep(path: str) -> str:
    # 自动探测分隔符，默认逗号
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(20000)
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return ","

def choose_first_available(columns: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in columns}
    for k in candidates:
        if k.lower() in low:
            return low[k.lower()]
    return None

def to_canonical_smiles(smi: str) -> Optional[str]:
    if not isinstance(smi, str) or not smi.strip():
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        # 生成标准SMILES，便于去重
        can = Chem.MolToSmiles(m, isomericSmiles=True)
        return can
    except Exception:
        return None

def convert(input_csv: str, output_csv: str,
            id_col: Optional[str] = None,
            name_col: Optional[str] = None,
            smiles_col: Optional[str] = None,
            limit: Optional[int] = None,
            chunksize: int = 100000,
            encoding: str = "utf-8") -> Tuple[int, int]:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"找不到输入文件: {input_csv}")

    sep = detect_sep(input_csv)
    # 读取表头，识别列名
    hdr = pd.read_csv(input_csv, sep=sep, nrows=0, encoding=encoding)
    cols = list(hdr.columns)

    # 若用户未指定，自动匹配常见列名
    smiles_col = smiles_col or choose_first_available(
        cols, ["smiles", "canonical_smiles", "isomeric_smiles", "clean_smiles"]
    )
    id_col = id_col or choose_first_available(
        cols, ["coconut_id", "id", "identifier", "record_id"]
    )
    name_col = name_col or choose_first_available(
        cols, ["name", "compound_name", "trivial_name", "best_name", "iupac_name", "standard_name"]
    )

    if smiles_col is None:
        raise ValueError(f"未找到可用的SMILES列。表头为: {cols[:12]} ...")
    if id_col is None:
        # 若没有ID列，后面用递增编号代替
        print("[WARN] 未找到ID列，将使用自增ID。")
    if name_col is None:
        print("[WARN] 未找到名称列，将用ID或空字符串代替。")

    usecols = [c for c in [id_col, name_col, smiles_col] if c is not None]
    total_in, total_out = 0, 0

    # 输出目录
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 逐块处理，减少内存占用
    writer = None
    seen = set()  # 用 canonical smiles 去重
    for chunk in pd.read_csv(input_csv, sep=sep, encoding=encoding,
                             usecols=usecols, chunksize=chunksize):
        if limit is not None and total_in >= limit:
            break
        # 只保留非空SMILES
        chunk = chunk[chunk[smiles_col].notna()]
        # 迭代转换
        rows = []
        for _, r in chunk.iterrows():
            if limit is not None and total_in >= limit:
                break
            smi = str(r[smiles_col])
            can = to_canonical_smiles(smi)
            total_in += 1
            if can is None:
                continue
            if can in seen:
                continue
            seen.add(can)

            _id = str(r[id_col]) if id_col in r and pd.notna(r[id_col]) else f"coco_{total_in}"
            _name = str(r[name_col]) if (name_col in r and pd.notna(r[name_col])) else _id

            rows.append((_id, _name, can))

        if not rows:
            continue

        out_df = pd.DataFrame(rows, columns=["id", "name", "smiles"])
        mode = "a" if writer is not None else "w"
        header = writer is None
        out_df.to_csv(output_csv, index=False, mode=mode, header=header)
        writer = True
        total_out += len(out_df)

        print(f"[chunk] 累计读取: {total_in}, 写出: {total_out}, 当前去重后SMILES数: {len(seen)}")

    print(f"[DONE] 输入读取: {total_in} 条（可能含无效/重复），输出有效: {total_out} 条 -> {output_csv}")
    return total_in, total_out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert COCONUT CSV to library.csv (id,name,smiles)")
    ap.add_argument("--input", required=True, help="coconut_csv_lite-09-2025.csv")
    ap.add_argument("--output", required=True, help="data/library.csv")
    ap.add_argument("--id-col", default=None, help="ID列名（可选，自动识别）")
    ap.add_argument("--name-col", default=None, help="名称列名（可选，自动识别）")
    ap.add_argument("--smiles-col", default=None, help="SMILES列名（可选，自动识别）")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前N条（可选）")
    ap.add_argument("--chunksize", type=int, default=100000, help="分块大小")
    args = ap.parse_args()

    try:
        convert(args.input, args.output, args.id_col, args.name_col, args.smiles_col,
                limit=args.limit, chunksize=args.chunksize)
    except UnicodeDecodeError:
        # 如果编码不是UTF-8，尝试latin-1
        print("[WARN] UTF-8读取失败，尝试latin-1编码...")
        convert(args.input, args.output, args.id_col, args.name_col, args.smiles_col,
                limit=args.limit, chunksize=args.chunksize, encoding="latin-1")