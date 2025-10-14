import torch
import tensorflow as tf
import sys
import platform
import numpy as np

def print_system_info():
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print()

def test_pytorch():
    print("=" * 60)
    print("PYTORCH TEST")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        # Test GPU computation
        print("\nTesting GPU computation...")
        device = torch.device('cuda')
        x = torch.randn(10000, 10000).to(device)
        y = torch.randn(10000, 10000).to(device)
        
        # Matrix multiplication on GPU
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = torch.matmul(x, y)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"GPU matrix multiplication time: {elapsed_time:.2f} ms")
        print(f"Result shape: {z.shape}")
        print(f"Result device: {z.device}")
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
    else:
        print("❌ CUDA not available for PyTorch")
    print()

def test_tensorflow():
    print("=" * 60)
    print("TENSORFLOW TEST")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU devices available: {len(gpus)}")
    
    for gpu in gpus:
        print(f"GPU: {gpu}")
    
    if gpus:
        # Test GPU computation
        print("\nTesting GPU computation...")
        with tf.device('/GPU:0'):
            # Create large tensors
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            
            # Time the matrix multiplication
            import time
            start_time = time.time()
            c = tf.matmul(a, b)
            end_time = time.time()
            
            print(f"GPU matrix multiplication time: {(end_time - start_time) * 1000:.2f} ms")
            print(f"Result shape: {c.shape}")
            print(f"Result device: {c.device}")
    else:
        print("❌ No GPU available for TensorFlow")
    print()

def test_memory_usage():
    print("=" * 60)
    print("MEMORY USAGE")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        # Test memory allocation
        large_tensor = torch.randn(1000, 1000, 100).cuda()  # ~400MB
        print(f"After allocation - Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        del large_tensor
        torch.cuda.empty_cache()
        print(f"After cleanup - Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print()

if __name__ == "__main__":
    print_system_info()
    test_pytorch()
    test_tensorflow()
    test_memory_usage()
    
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
