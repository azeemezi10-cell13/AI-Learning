import torch
import sys
def health_check():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB")
        print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
if __name__ == "__main__":
    health_check()