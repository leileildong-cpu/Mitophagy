import os
import subprocess
import pandas as pd
import yaml

def test_morgan_weights():
    """只测试Morgan指纹的不同权重"""
    
    base_config = {
        'data': {
            'references_csv': 'data/references.csv',
            'library_csv': 'data/library_50.csv'
        },
        'output': {
            'dir': 'outputs/morgan_test',
            'hits_csv': 'hits.csv',
            'report_txt': 'report.txt'
        },
        'similarity': {'w_1d': 1.0, 'w_2d': 0.0, 'w_3d': 0.0},  # 只用Morgan
        'mol2vec': {'model_path': None},
        'ai_model': {
            'similarity_threshold': 0.6,
            'use_clustering': False,
            'use_outlier_filter': False
        },
        'thresholds': {'combined_min': 0.5, 'top_k': 20},
        'fingerprints': {'morgan_radius': 2, 'morgan_nbits': 2048},
        'pharm2d': {'use_gobbi': True},
        'shape3d': {'num_confs': 0},
        'filters': {
            'pains': False,  # 跳过PAINS检查加速
            'cns_filter': False,  # 跳过CNS过滤加速
            'cns_rules': {}
        }
    }
    
    print("快速Morgan指纹测试（预计1分钟）...")
    
    config_file = 'config_morgan_quick.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(base_config, f)
    
    os.makedirs('outputs', exist_ok=True)
    
    try:
        result = subprocess.run(['python', 'src/run.py', '--config', config_file], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Morgan指纹测试成功")
            
            # 查找结果目录
            output_dirs = [d for d in os.listdir('outputs') if d.startswith('morgan_test_')]
            if output_dirs:
                latest_dir = os.path.join('outputs', sorted(output_dirs)[-1])
                hits_file = os.path.join(latest_dir, 'hits.csv')
                
                if os.path.exists(hits_file):
                    df = pd.read_csv(hits_file)
                    print(f"结果: 最高得分={df['score'].max():.3f}, "
                          f"命中数(≥0.7)={len(df[df['score'] >= 0.7])}")
        else:
            print("❌ 测试失败")
            print("错误:", result.stderr[-200:])
    
    except Exception as e:
        print(f"❌ 出错: {e}")

if __name__ == "__main__":
    test_morgan_weights()