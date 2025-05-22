import torch

print("==== PyTorch 環境チェック ====")
print(f"PyTorch version        : {torch.__version__}")
print(f"CUDA available         : {torch.cuda.is_available()}")
print(f"CUDA version (from PyTorch) : {torch.version.cuda}")
print(f"cuDNN version          : {torch.backends.cudnn.version()}")
print(f"cuDNN enabled          : {torch.backends.cudnn.enabled}")
print(
    f"使用デバイス名         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}"
)
