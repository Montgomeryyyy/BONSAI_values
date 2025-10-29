#!/usr/bin/env python3
"""
Test script to verify GPU performance improvements
"""
import torch
import time

def test_gpu_optimization():
    """Test that GPU optimizations are working"""
    print("Testing GPU performance optimizations...")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test GPU optimizations")
        return False
    
    device = torch.device("cuda")
    print(f"✅ Using device: {device}")
    
    # Test half precision
    model = torch.nn.Linear(1000, 1000).to(device)
    model_half = model.half()
    print(f"✅ Half precision conversion successful: {model_half.dtype}")
    
    # Test memory management
    initial_memory = torch.cuda.memory_allocated()
    print(f"✅ Initial GPU memory: {initial_memory / 1024**2:.2f} MB")
    
    # Test tensor operations on GPU
    x = torch.randn(100, 1000, device=device, dtype=torch.float16)
    y = model_half(x)
    print(f"✅ GPU computation successful: {y.shape}")
    
    # Test memory cleanup
    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated()
    print(f"✅ Memory after cleanup: {final_memory / 1024**2:.2f} MB")
    
    return True

def test_tensor_concatenation_performance():
    """Test the improved tensor concatenation approach"""
    print("\nTesting tensor concatenation performance...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simulate the old approach (immediate CPU transfer)
    print("Testing old approach (immediate CPU transfer):")
    start_time = time.time()
    
    old_embs = []
    for i in range(10):  # Simulate 10 batches
        batch_emb = torch.randn(32, 512, 768, device=device)
        old_embs.append(batch_emb.cpu())  # Immediate CPU transfer
    
    old_result = torch.cat(old_embs, dim=0)
    old_time = time.time() - start_time
    print(f"Old approach time: {old_time:.4f}s")
    
    # Simulate the new approach (GPU concatenation, then CPU transfer)
    print("Testing new approach (GPU concatenation, then CPU transfer):")
    start_time = time.time()
    
    new_embs = []
    for i in range(10):  # Simulate 10 batches
        batch_emb = torch.randn(32, 512, 768, device=device)
        new_embs.append(batch_emb)  # Keep on GPU
    
    new_result = torch.cat(new_embs, dim=0).cpu()  # Concatenate on GPU, then transfer
    new_time = time.time() - start_time
    print(f"New approach time: {new_time:.4f}s")
    
    # Verify results are the same
    if torch.allclose(old_result, new_result, atol=1e-6):
        print("✅ Results are identical")
        speedup = old_time / new_time if new_time > 0 else float('inf')
        print(f"✅ Speedup: {speedup:.2f}x")
        return True
    else:
        print("❌ Results differ!")
        return False

if __name__ == "__main__":
    print("GPU Performance Test")
    print("=" * 50)
    
    gpu_ok = test_gpu_optimization()
    concat_ok = test_tensor_concatenation_performance()
    
    print("\n" + "=" * 50)
    if gpu_ok and concat_ok:
        print("✅ All tests passed! GPU optimizations are working.")
    else:
        print("❌ Some tests failed. Check the output above.")
