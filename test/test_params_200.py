import os
import yaml
import subprocess
import pandas as pd

def test_different_weights():
    """测试不同权重组合"""
    
    base_config = {
        'data': {
            'references_csv': 'data/references.csv',
            'library_csv': 'data/library_200.csv'
        },
        'output': {
            'dir': 'outputs/param_200',
            'hits_csv': 'hits.csv',
            'report_txt': 'report.txt'
        },
        'mol2vec': {'model_path': 'models/mol2vec_model.pkl'},
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
            'pains': True, 'cns_filter': True,
            'cns_rules': {
                'mw_min': 100, 'mw_max': 700, 'tpsa_max': 150,
                'hbd_max': 8, 'hba_max': 12, 'rotb_max': 15,
                'logp_min': -2.0, 'logp_max': 8.0
            }
        }
    }
    
    # 权重组合测试
    weight_tests = [
        (1.0, 0.0, 0.0, "morgan_only"),
        (0.0, 1.0, 0.0, "pharm_only"),
        (0.8, 0.2, 0.0, "morgan_dominant"),
        (0.6, 0.4, 0.0, "morgan_heavy"),
        (0.5, 0.5, 0.0, "balanced"),
        (0.4, 0.6, 0.0, "pharm_heavy"),
        (0.2, 0.8, 0.0, "pharm_dominant"),
    ]
    
    results = []
    os.makedirs('outputs', exist_ok=True)
    
    for w1, w2, w3, name in weight_tests:
        print(f"\n测试权重: {name} (w1={w1}, w2={w2})")
        
        config = base_config.copy()
        config['similarity'] = {'w_1d': w1, 'w_2d': w2, 'w_3d': w3}
        config['output']['dir'] = f'outputs/param_200_{name}'
        
        config_file = f'config_200_{name}.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        try:
            result = subprocess.run(['python', 'src/run.py', '--config', config_file], 
                                  capture_output=True, text=True, timeout=3000)
            
            if result.returncode == 0:
                # 分析结果
                output_dirs = [d for d in os.listdir('outputs') if d.startswith(f'param_200_{name}_')]
                if output_dirs:
                    latest_dir = os.path.join('outputs', sorted(output_dirs)[-1])
                    hits_file = os.path.join(latest_dir, 'hits.csv')
                    
                    if os.path.exists(hits_file):
                        df = pd.read_csv(hits_file)
                        results.append({
                            'name': name,
                            'w1': w1, 'w2': w2,
                            'max_score': df['score'].max(),
                            'mean_score': df['score'].mean(),
                            'hits_06': len(df[df['score'] >= 0.6]),
                            'hits_07': len(df[df['score'] >= 0.7]),
                            'pains_rate': df['pains'].mean(),
                            'cns_pass_rate': df['cns_pass'].mean(),
                            'status': 'success'
                        })
                        print(f"✅ 最高得分: {df['score'].max():.3f}, 命中数(≥0.7): {len(df[df['score'] >= 0.7])}")
            else:
                print(f"❌ 运行失败")
                results.append({'name': name, 'status': 'failed'})
                
        except Exception as e:
            print(f"❌ 出错: {e}")
            results.append({'name': name, 'status': 'error'})
    
    # 保存和分析结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/weight_test_200_results.csv', index=False)
    
    print(f"\n{'='*60}")
    print("权重测试结果汇总:")
    success_results = results_df[results_df['status'] == 'success']
    if len(success_results) > 0:
        for _, row in success_results.iterrows():
            print(f"{row['name']:15s} | w1={row['w1']:.1f} w2={row['w2']:.1f} | "
                  f"最高={row['max_score']:.3f} | 命中={row['hits_07']:2d} | "
                  f"PAINS={row['pains_rate']:.2f} | CNS={row['cns_pass_rate']:.2f}")
        
        best = success_results.loc[success_results['max_score'].idxmax()]
        print(f"\n🏆 最佳权重: {best['name']} (w1={best['w1']}, w2={best['w2']})")
        print(f"   最高得分: {best['max_score']:.3f}")
        print(f"   命中数: {best['hits_07']}")
        
        return best['w1'], best['w2']
    
    return 0.5, 0.5  # 默认返回平衡权重

if __name__ == "__main__":
    test_different_weights()