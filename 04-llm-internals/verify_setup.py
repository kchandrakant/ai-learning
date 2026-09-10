#!/usr/bin/env python3
"""Verify environment setup for LLM Internals course."""

import sys

def check_import(module_name: str, package_name: str = None) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 60)
    print("LLM Internals - Environment Verification")
    print("=" * 60)
    
    checks = [
        ("torch", "torch", "PyTorch for deep learning"),
        ("transformers", "transformers", "Hugging Face Transformers"),
        ("tokenizers", "tokenizers", "Fast tokenizers"),
        ("sentencepiece", "sentencepiece", "SentencePiece tokenization"),
        ("datasets", "datasets", "Hugging Face Datasets"),
        ("accelerate", "accelerate", "Training acceleration"),
        ("peft", "peft", "Parameter-efficient fine-tuning"),
        ("trl", "trl", "Transformer Reinforcement Learning"),
        ("matplotlib", "matplotlib", "Plotting"),
        ("numpy", "numpy", "Numerical computing"),
        ("pandas", "pandas", "Data manipulation"),
    ]
    
    all_passed = True
    
    for module, package, description in checks:
        status = check_import(module)
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {description}: {package}")
        if not status:
            all_passed = False
    
    # Check PyTorch GPU
    print("\n" + "-" * 60)
    print("GPU Check:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠ CUDA not available (CPU-only mode)")
            print("    Note: Many modules require GPU for practical experiments")
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All required packages installed!")
        print("\nYou're ready to start the LLM Internals course.")
    else:
        print("✗ Some packages are missing.")
        print("\nRun: pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
