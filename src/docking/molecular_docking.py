# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子对接模块
支持AutoDock Vina和Smina
提供批量对接和结果解析功能
"""

import os
import subprocess
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import warnings


class MolecularDocking:
    """分子对接管理器"""

    def __init__(self, config: dict):
        """
        初始化分子对接模块

        Args:
            config: 配置字典
        """
        self.config = config
        self.docking_config = config.get("docking", {})
        self.enabled = self.docking_config.get("enabled", False)

        # 对接参数
        self.protein_pdb = self.docking_config.get("protein_pdb")
        self.binding_site = self.docking_config.get("binding_site", {})
        self.exhaustiveness = self.docking_config.get("exhaustiveness", 8)
        self.num_modes = self.docking_config.get("num_modes", 10)
        self.max_molecules = self.docking_config.get("max_molecules", 50)

        # 检查对接程序
        self.vina_available = False
        self.smina_available = False

        if self.enabled:
            self._check_docking_software()
            self._validate_protein_structure()
        else:
            print("[INFO] 分子对接未启用")

    def _check_docking_software(self):
        """检查对接软件是否可用"""
        # 检查Vina
        try:
            result = subprocess.run(
                ['vina', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.vina_available = True
                print("[INFO] AutoDock Vina已找到")
        except:
            pass

        # 检查Smina
        try:
            result = subprocess.run(
                ['smina', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.smina_available = True
                print("[INFO] Smina已找到")
        except:
            pass

        if not self.vina_available and not self.smina_available:
            print("[WARNING] 未找到对接软件（Vina或Smina），对接功能不可用")
            self.enabled = False
        else:
            # 优先使用Smina（更快）
            self.docking_program = 'smina' if self.smina_available else 'vina'
            print(f"[INFO] 使用对接程序: {self.docking_program}")

    def _validate_protein_structure(self):
        """验证蛋白质结构文件"""
        if not self.protein_pdb:
            print("[WARNING] 未配置蛋白质结构文件，对接功能不可用")
            self.enabled = False
            return

        if not os.path.exists(self.protein_pdb):
            print(f"[WARNING] 蛋白质结构文件不存在: {self.protein_pdb}")
            self.enabled = False
            return

        print(f"[INFO] 蛋白质结构: {self.protein_pdb}")

        # 验证结合位点配置
        if not self.binding_site.get('center') or not self.binding_site.get('box_size'):
            print("[WARNING] 结合位点配置不完整，对接功能不可用")
            self.enabled = False

    def prepare_protein(self, pdb_file: str, output_pdbqt: Optional[str] = None) -> Optional[str]:
        """
        准备蛋白质结构（转换为PDBQT格式）

        Args:
            pdb_file: 输入PDB文件
            output_pdbqt: 输出PDBQT文件路径（可选）

        Returns:
            PDBQT文件路径，失败返回None
        """
        if not output_pdbqt:
            output_pdbqt = pdb_file.replace('.pdb', '_prepared.pdbqt')

        try:
            # 使用obabel转换（需要安装Open Babel）
            cmd = [
                'obabel',
                pdb_file,
                '-O', output_pdbqt,
                '-xr',  # 保留残基
                '-h'  # 添加氢原子
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0 and os.path.exists(output_pdbqt):
                print(f"[INFO] 蛋白质已准备: {output_pdbqt}")
                return output_pdbqt
            else:
                print(f"[WARNING] 蛋白质准备失败: {result.stderr}")
                return None

        except Exception as e:
            print(f"[WARNING] 蛋白质准备失败: {e}")
            return None

    def prepare_ligand(self, mol: Chem.Mol, output_dir: Optional[str] = None) -> Optional[str]:
        """
        准备配体（生成3D结构并转换为PDBQT）

        Args:
            mol: RDKit分子对象
            output_dir: 输出目录

        Returns:
            PDBQT文件路径，失败返回None
        """
        if mol is None:
            return None

        try:
            # 添加氢原子
            mol_h = Chem.AddHs(mol)

            # 生成3D构象
            if AllChem.EmbedMolecule(mol_h, randomSeed=42) != 0:
                # 回退：使用更宽松的参数
                ps = AllChem.ETKDGv3()
                ps.randomSeed = 42
                ps.useRandomCoords = True
                if AllChem.EmbedMolecule(mol_h, params=ps) != 0:
                    return None

            # 优化构象
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            except:
                pass

            # 保存为PDB
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                pdb_file = os.path.join(output_dir, 'ligand.pdb')
            else:
                with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as tmp:
                    pdb_file = tmp.name

            pdb_block = Chem.MolToPDBBlock(mol_h)
            with open(pdb_file, 'w') as f:
                f.write(pdb_block)

            # 转换为PDBQT
            pdbqt_file = pdb_file.replace('.pdb', '.pdbqt')

            cmd = ['obabel', pdb_file, '-O', pdbqt_file, '-h']
            result = subprocess.run(cmd, capture_output=True, timeout=30)

            # 清理PDB文件
            try:
                os.remove(pdb_file)
            except:
                pass

            if result.returncode == 0 and os.path.exists(pdbqt_file):
                return pdbqt_file
            else:
                return None

        except Exception as e:
            warnings.warn(f"配体准备失败: {e}")
            return None

    def dock_single_molecule(self,
                             ligand_pdbqt: str,
                             protein_pdbqt: Optional[str] = None) -> Dict:
        """
        对接单个分子

        Args:
            ligand_pdbqt: 配体PDBQT文件
            protein_pdbqt: 蛋白PDBQT文件（可选，默认使用配置中的）

        Returns:
            对接结果字典 {
                'affinity': float,
                'rmsd_lb': float,
                'rmsd_ub': float,
                'poses': List[Dict],
                'success': bool
            }
        """
        if not self.enabled or not ligand_pdbqt:
            return {'affinity': 0.0, 'success': False, 'error': 'Docking disabled or invalid ligand'}

        if not protein_pdbqt:
            protein_pdbqt = self.protein_pdb

        # 创建配置文件
        config_content = self._generate_vina_config(ligand_pdbqt, protein_pdbqt)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as config_file:
            config_file.write(config_content)
            config_path = config_file.name

        # 输出文件
        out_pdbqt = ligand_pdbqt.replace('.pdbqt', '_out.pdbqt')
        log_file = ligand_pdbqt.replace('.pdbqt', '_log.txt')

        # 运行对接
        try:
            cmd = [
                self.docking_program,
                '--config', config_path,
                '--out', out_pdbqt,
                '--log', log_file
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # 解析结果
                docking_results = self._parse_docking_output(log_file)
                docking_results['success'] = True
                return docking_results
            else:
                return {
                    'affinity': 0.0,
                    'success': False,
                    'error': result.stderr
                }

        except subprocess.TimeoutExpired:
            return {'affinity': 0.0, 'success': False, 'error': 'Timeout'}

        except Exception as e:
            return {'affinity': 0.0, 'success': False, 'error': str(e)}

        finally:
            # 清理临时文件
            for f in [config_path, out_pdbqt, log_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass

    def _generate_vina_config(self, ligand_pdbqt: str, protein_pdbqt: str) -> str:
        """生成Vina/Smina配置文件内容"""
        center = self.binding_site['center']
        box_size = self.binding_site['box_size']

        config = f"""receptor = {protein_pdbqt}
ligand = {ligand_pdbqt}

center_x = {center[0]}
center_y = {center[1]}
center_z = {center[2]}

size_x = {box_size[0]}
size_y = {box_size[1]}
size_z = {box_size[2]}

exhaustiveness = {self.exhaustiveness}
num_modes = {self.num_modes}
"""
        return config

    def _parse_docking_output(self, log_file: str) -> Dict:
        """
        解析对接输出日志

        Args:
            log_file: 日志文件路径

        Returns:
            解析的结果字典
        """
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
                if 'mode |   affinity' in line.lower():
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
                            except (ValueError, IndexError):
                                break
                        else:
                            break

            # 提取最佳结果
            if results['poses']:
                best_pose = results['poses'][0]
                results['affinity'] = best_pose['affinity']
                results['rmsd_lb'] = best_pose['rmsd_lb']
                results['rmsd_ub'] = best_pose['rmsd_ub']

        except Exception as e:
            warnings.warn(f"解析对接结果失败: {e}")

        return results

    def batch_dock_molecules(self,
                             molecules: List[Chem.Mol],
                             output_dir: Optional[str] = None,
                             max_molecules: Optional[int] = None) -> List[Dict]:
        """
        批量对接分子

        Args:
            molecules: 分子列表
            output_dir: 输出目录
            max_molecules: 最大对接分子数

        Returns:
            对接结果列表
        """
        if not self.enabled:
            print("[INFO] 分子对接未启用")
            return [{'affinity': 0.0, 'success': False}] * len(molecules)

        # 限制对接数量
        if max_molecules is None:
            max_molecules = self.max_molecules

        molecules_to_dock = molecules[:max_molecules]

        print(f"[INFO] 批量对接 {len(molecules_to_dock)} 个分子...")
        print(f"  - 对接程序: {self.docking_program}")
        print(f"  - 蛋白质: {self.protein_pdb}")
        print(f"  - 结合位点中心: {self.binding_site['center']}")
        print(f"  - 盒子大小: {self.binding_site['box_size']}")

        # 创建临时工作目录
        if output_dir:
            work_dir = output_dir
            os.makedirs(work_dir, exist_ok=True)
        else:
            work_dir = tempfile.mkdtemp(prefix='docking_')

        results = []

        for i, mol in enumerate(tqdm(molecules_to_dock, desc="分子对接")):
            try:
                # 准备配体
                ligand_dir = os.path.join(work_dir, f'ligand_{i}')
                os.makedirs(ligand_dir, exist_ok=True)

                ligand_pdbqt = self.prepare_ligand(mol, ligand_dir)

                if ligand_pdbqt:
                    # 对接
                    docking_result = self.dock_single_molecule(ligand_pdbqt)
                    results.append(docking_result)

                    # 清理配体文件
                    try:
                        shutil.rmtree(ligand_dir)
                    except:
                        pass
                else:
                    results.append({
                        'affinity': 0.0,
                        'success': False,
                        'error': 'Ligand preparation failed'
                    })

            except Exception as e:
                warnings.warn(f"分子{i}对接失败: {e}")
                results.append({
                    'affinity': 0.0,
                    'success': False,
                    'error': str(e)
                })

        # 填充未对接的分子
        for i in range(len(molecules_to_dock), len(molecules)):
            results.append({
                'affinity': 0.0,
                'success': False,
                'error': 'Not docked (exceeds max_molecules)'
            })

        # 统计
        success_count = sum(1 for r in results if r.get('success', False))
        print(f"[INFO] 对接完成: {success_count}/{len(molecules_to_dock)} 成功")

        if success_count > 0:
            affinities = [r['affinity'] for r in results if r.get('success', False)]
            print(f"  - 最佳亲和力: {min(affinities):.2f} kcal/mol")
            print(f"  - 平均亲和力: {sum(affinities) / len(affinities):.2f} kcal/mol")

        # 清理工作目录
        if not output_dir:
            try:
                shutil.rmtree(work_dir)
            except:
                pass

        return results

    def get_docking_summary(self, results: List[Dict]) -> Dict:
        """
        获取对接结果摘要

        Args:
            results: 对接结果列表

        Returns:
            摘要字典
        """
        successful = [r for r in results if r.get('success', False)]

        if not successful:
            return {
                'total': len(results),
                'successful': 0,
                'success_rate': 0.0,
                'best_affinity': None,
                'mean_affinity': None,
                'median_affinity': None
            }

        affinities = [r['affinity'] for r in successful]

        import statistics

        return {
            'total': len(results),
            'successful': len(successful),
            'success_rate': len(successful) / len(results) * 100,
            'best_affinity': min(affinities),
            'mean_affinity': statistics.mean(affinities),
            'median_affinity': statistics.median(affinities),
            'worst_affinity': max(affinities)
        }

    def __repr__(self):
        status = "enabled" if self.enabled else "disabled"
        program = self.docking_program if self.enabled else "None"
        return f"MolecularDocking(status={status}, program={program})"