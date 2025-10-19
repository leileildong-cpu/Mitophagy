import urllib.request
import os
import ssl

def download_model():
    # 创建SSL上下文（处理SSL证书问题）
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    model_url = "https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl"
    model_path = "models/mol2vec_model.pkl"
    
    print("重新下载Mol2vec模型...")
    print(f"URL: {model_url}")
    
    try:
        # 删除旧文件
        if os.path.exists(model_path):
            os.remove(model_path)
            print("已删除损坏的文件")
        
        # 确保目录存在
        os.makedirs("models", exist_ok=True)
        
        # 下载文件
        with urllib.request.urlopen(model_url, context=ssl_context) as response:
            total_size = int(response.headers.get('content-length', 0))
            print(f"文件大小: {total_size / (1024*1024):.1f} MB")
            
            with open(model_path, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r下载进度: {percent:.1f}%", end="")
        
        print(f"\n✅ 下载完成!")
        
        # 验证文件
        actual_size = os.path.getsize(model_path)
        print(f"实际文件大小: {actual_size / (1024*1024):.1f} MB")
        
        if actual_size > 1024:  # 至少1KB
            print("✅ 文件下载成功!")
            return True
        else:
            print("❌ 文件可能损坏")
            return False
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

if __name__ == "__main__":
    download_model()