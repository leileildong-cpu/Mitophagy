import sys


def check_gpu_environment():
    print("=== GPU环境检查 ===")

    # 检查NVIDIA驱动
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU驱动正常")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    print(f"   GPU: {line.strip()}")
                    break
        else:
            print("❌ 没有NVIDIA GPU或驱动未安装")
            return False
    except:
        print("❌ nvidia-smi 命令不可用")
        return False

    # 检查CUDA
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   显存: {memory:.1f} GB")
            return True
        else:
            print("❌ PyTorch检测不到CUDA")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False

    # 检查CuPy
    try:
        import cupy as cp
        print(f"✅ CuPy可用: {cp.__version__}")
        return True
    except ImportError:
        print("❌ CuPy未安装")
        return False


if __name__ == "__main__":
    has_gpu = check_gpu_environment()

    if not has_gpu:
        print("\n=== 建议 ===")
        print("1. 如果你有NVIDIA GPU，请安装CUDA驱动")
        print("2. 如果没有GPU，使用CPU优化版本")
        print("3. 运行: python run_cpu_optimized.py --size 10000")