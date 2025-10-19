import csv
import argparse
import os
from typing import List, Tuple
import pubchempy as pcp
from tqdm import tqdm

def fetch_smiles(name: str) -> str:
    """
    根据化合物名称获取其Canonical SMILES
    :param name: 化合物名称
    :return: Canonical SMILES 字符串或 None
    """
    try:
        compounds = pcp.get_compounds(name, 'name')
        if not compounds:
            return None
        return compounds[0].canonical_smiles
    except Exception as e:
        print(f"[ERROR] 获取 {name} 的SMILES失败: {e}")
        return None

def read_names(file_path: str) -> List[str]:
    """读取化合物名称列表"""
    names = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names

def generate_reference_csv(names_file: str, output_csv: str, prefix: str = "ref"):
    """
    从化合物名称文件生成参考CSV文件
    :param names_file: 包含化合物名称的txt文件路径
    :param output_csv: 输出CSV文件路径
    :param prefix: ID前缀，默认为"ref"
    """
    # 读取化合物名称
    names = read_names(names_file)
    if not names:
        raise SystemExit("[ERROR] 输入文件为空，请检查内容。")

    # 准备数据行
    rows = []
    
    print("[INFO] 正在从PubChem获取化合物SMILES...")
    for name in tqdm(names, desc="Processing"):
        smi = fetch_smiles(name)
        if smi:
            # 生成ID: ref_化合物名称(小写，空格替换为下划线)
            cid = f"{prefix}_{name.lower().replace(' ', '_')}"
            rows.append((cid, name, smi))
            print(f"[OK] {name} -> {smi}")
        else:
            print(f"[WARN] 无法获取SMILES: {name}")

    if not rows:
        raise SystemExit("[ERROR] 未获取到任何SMILES，请检查名称或网络。")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 写入CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', 'smiles'])  # 写入表头
        writer.writerows(rows)
    
    print(f"[DONE] 成功写出 {len(rows)} 条记录 -> {output_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="从化合物名称生成参考CSV文件",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--names', required=True, help='包含化合物名称的txt文件，每行一个名称')
    parser.add_argument('--output', default='D:\\Mitophagy\\data\\references.csv', 
                       help='输出CSV路径，默认为 D:\\Mitophagy\\data\\references.csv')
    parser.add_argument('--prefix', default='ref', help='ID前缀，默认为ref')
    
    args = parser.parse_args()
    
    generate_reference_csv(args.names, args.output, args.prefix)

if __name__ == "__main__":
    main()