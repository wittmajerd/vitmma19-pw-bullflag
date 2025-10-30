import torch
import sys
import os
import subprocess

print("=== System Information ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA compiled version: {torch.version.cuda}")

print("\n=== CUDA Environment ===")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")

print(f"\n=== PyTorch CUDA Status ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
else:
    print("CUDA not available - checking possible causes...")
    
    # Check if nvidia-smi works
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ nvidia-smi works - GPU hardware detected")
        else:
            print("✗ nvidia-smi failed - no GPU or driver issues")
    except FileNotFoundError:
        print("✗ nvidia-smi not found - NVIDIA drivers not installed")
    
    # Check CUDA toolkit
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ CUDA toolkit installed")
        else:
            print("✗ CUDA toolkit not working")
    except FileNotFoundError:
        print("✗ CUDA toolkit not found")