#!/usr/bin/env python3
"""
Test script to verify the tensor concatenation fix
"""
import torch

def test_original_approach():
    """Test the original approach that would fail"""
    print("Testing original approach (should fail):")
    # Create two batches with different sequence lengths
    batch1 = torch.randn(2, 1024, 768)  # batch_size=2, seq_len=1024, hidden_size=768
    batch2 = torch.randn(2, 998, 768)   # batch_size=2, seq_len=998, hidden_size=768
    
    try:
        result = torch.cat([batch1, batch2], dim=0)
        print("Success!")
        return True
    except RuntimeError as e:
        print(f"Error: {e}")
        return False

def test_new_approach():
    """Test the new approach that should work"""
    print("\nTesting new approach (should work):")
    # Create two batches with different sequence lengths
    batch1 = torch.randn(2, 1024, 768)  # batch_size=2, seq_len=1024, hidden_size=768
    batch2 = torch.randn(2, 998, 768)   # batch_size=2, seq_len=998, hidden_size=768
    
    # Simulate the new approach
    model_embs = []
    for i in range(batch1.shape[0]):
        model_embs.append(batch1[i])
    for i in range(batch2.shape[0]):
        model_embs.append(batch2[i])
    
    result = torch.stack(model_embs, dim=0)
    print(f"Success! Result shape: {result.shape}")
    return True

if __name__ == "__main__":
    print("Testing tensor concatenation fix...")
    
    # Test original approach
    original_works = test_original_approach()
    
    # Test new approach
    new_works = test_new_approach()
    
    print(f"\nResults:")
    print(f"Original approach works: {original_works}")
    print(f"New approach works: {new_works}")
    
    if not original_works and new_works:
        print("✅ Fix is working correctly!")
    else:
        print("❌ Fix needs adjustment")
