#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子对接模块
支持AutoDock Vina和Smina
"""

import os
import subprocess
import tempfile
from typing import Dict, List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


class MolecularDocking:
    """分子对接模块"""

    def __init__(self, config: dict):
        self.config = config
        self.docking_config = config.get("docking", {})
        self.enabled = self.docking_config.get("enabled", False)

        if self.enabled:
            self.vina_available = self._check_vina()
            if self.vina_available:
                print("[INFO] 分子对接模块已启用（AutoDock Vina）")
            else:
                print("[WARNING] AutoDock Vina未找到，对接功能不可用")
                self.enabled = False

    def _check_vina(self) -> bool:
        """检查Vina是否可用"""
        try:
            result = subprocess.run(['vina', '--version'],
                                    capture_output=True,
                                    text=True,
                                    timeout=5)
            return result.returncode == 0
        except:
            return False

    def prepare_ligand(self, mol: Chem.Mol) -> Optional[str]:
        """
        准备配体（生成3D结构）

        Returns:
            pdbqt_file: PDBQT文件路径，失败返回None
        """
        try:
            # 添加氢原子
            mol_h = Chem.AddHs(mol)

            # 生成3D构象
            if AllChem.EmbedMolecule(mol_h, randomSeed=42) != 0:
                return None

            # 优化构象
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)

            # 保存为PDB
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as tmp:
                pdb_block = Chem.MolToPDBBlock(mol_h)
                tmp.write(pdb_block)
                pdb_file = tmp.name

            # 转换为PDBQT（需要obabel或prepare_ligand4.py）
            pdbqt_file = pdb_file.replace('.pdb', '.pdbqt')

            # 使用obabel转换
            cmd = ['obabel', pdb_file, '-O', pdbqt_file, '-h']
            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode == 0:
                os.remove(pdb_file)
                return pdbqt_file
            else:
                # 备用：简单复制（仅用于测试）
                os.rename(pdb_file, pdbqt_file)
                return pdbqt_file

        except Exception as e:
            print(f"[WARNING] 配体准备失败: {e}")
            return None

    def dock_molecule(self, ligand_pdbqt: str, protein_pdbqt: str) -> Dict:
        """
        执行分子对接

        Returns:
            docking_results: {
                'affinity': float,
                'rmsd_lb': float,
                'rmsd_ub': float,
                'poses': List[Dict],
                'success': bool
            }
        """
        if not self.enabled or not ligand_pdbqt or not os.path.exists(ligand_pdbqt):
            return {'affinity': 0.0, 'success': False, 'error': 'Docking disabled or invalid ligand'}

        # 创建配置文件
        config_content = f"""receptor = {protein_pdbqt}
ligand = {ligand_pdbqt}

center_x = {self.docking_config['binding_site']['center'][0]}
center_y = {self.docking_config['binding_site']['center'][1]}
center_z = {self.docking_config['binding_site']['center'][2]}

size_x = {self.docking_config['binding_site']['box_size'][0]}
size_y = {self.docking_config['binding_site']['box_size'][1]}
size_z = {self.docking_config['binding_site']['box_size'][2]}

exhaustiveness = {self.docking_config.get('exhaustiveness', 8)}
num_modes = {self.docking_config.get('num_modes', 10)}
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as config_file:
            config_file.write(config_content)
            config_path = config_file.name

        # 输出文件
        out_pdbqt = ligand_pdbqt.replace('.pdbqt', '_out.pdbqt')
        log_file = ligand_pdbqt.replace('.pdbqt', '_log.txt')

        # 运行Vina
        try:
            cmd = [
                'vina',
                '--config', config_path,
                '--out', out_pdbqt,
                '--log', log_file
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # 解析结果
                docking_results = self._parse_vina_output(log_file)
                docking_results['success'] = True
                return docking_results
            else:
                return {'affinity': 0.0, 'success': False, 'error': result.stderr}

        except subprocess.TimeoutExpired:
            return {'affinity': 0.0, 'success': False, 'error': 'Timeout'}

        except Exception as e:
            return {'affinity': 0.0, 'success': False, 'error': str(e)}

        finally:
            # 清理临时文件
            for f in [config_path, ligand_pdbqt, out_pdbqt, log_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass

    def _parse_vina_output(self, log_file: str) -> Dict:
        """解析Vina输出日志"""
        results = {
            'affinity': 0.0,
            'rmsd_lb': 0.0,
            'rmsd_ub': 0.0,
            'poses': []
        }

        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()

            # 查找结果表格
            for i, line in enumerate(lines):
                if 'mode |   affinity' in line:
                    # 解析每个pose
                    for j in range(i + 2, len(lines)):
                        parts = lines[j].split()
                        if len(parts) >= 4:
                            try:
                                pose = {
                                    'mode': int(parts[0]),
                                    'affinity': float(parts[1]),
                                    'rmsd_lb': float(parts[2]),
                                    'rmsd_ub': float(parts[3])
                                }
                                results['poses'].append(pose)
                            except ValueError:
                                break

            # 最佳亲和力
            if results['poses']:
                best_pose = results['poses'][0]
                results['affinity'] = best_pose['affinity']
                results['rmsd_lb'] = best_pose['rmsd_lb']
                results['rmsd_ub'] = best_pose['rmsd_ub']

        except Exception as e:
            print(f"[WARNING] 解析Vina输出失败: {e}")

        return results

    def batch_dock_molecules(self, molecules: List[Chem.Mol],
                             protein_pdbqt: str,
                             max_molecules: int = 100) -> List[Dict]:
        """
        批量分子对接

        Args:
            molecules: 分子列表
            protein_pdbqt: 蛋白PDBQT文件路径
            max_molecules: 最大对接分子数（节省时间）

        Returns:
            results: 对接结果列表
        """
        if not self.enabled:
            print("[WARNING] 分子对接未启用")
            return [{'affinity': 0.0, 'success': False}] * len(molecules)

        # 限制对接数量
        molecules_to_dock = molecules[:max_molecules]
        print(f"[INFO] 开始批量对接 {len(molecules_to_dock)} 个分子...")

        results = []

        for i, mol in enumerate(tqdm(molecules_to_dock, desc="分子对接")):
            try:
                ligand_pdbqt = self.prepare_ligand(mol)

                if ligand_pdbqt:
                    docking_result = self.dock_molecule(ligand_pdbqt, protein_pdbqt)
                    results.append(docking_result)
                else:
                    results.append({'affinity': 0.0, 'success': False, 'error': 'Ligand preparation failed'})

            except Exception as e:
                print(f"[WARNING] 分子{i}对接失败: {e}")
                results.append({'affinity': 0.0, 'success': False, 'error': str(e)})

        # 填充未对接的分子结果
        for i in range(len(molecules_to_dock), len(molecules)):
            results.append({'affinity': 0.0, 'success': False, 'error': 'Not docked (exceeds max_molecules)'})

        success_count = sum(1 for r in results if r.get('success', False))
        print(f"[INFO] 对接完成: {success_count}/{len(molecules_to_dock)} 成功")

        return results