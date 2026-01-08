import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version used to compile PyTorch: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    # 测试 GPU 计算
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print("✅ GPU computation successful!")
else:
    print("❌ GPU not available — check installation")