import torch

# 检查 PyTorch 和 CUDA 版本 和 GPU 信息

def check_versions():
    print("=== PyTorch 和 CUDA 版本信息 ===")
    
    # PyTorch 版本
    print(f"PyTorch version: {torch.__version__}")
    
    # CUDA 版本
    print(f"CUDA version: {torch.version.cuda}")
    
    # 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # GPU 信息
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # cuDNN 版本
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        
        # 当前设备能力
        if torch.cuda.device_count() > 0:
            print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

if __name__ == "__main__":
    check_versions()