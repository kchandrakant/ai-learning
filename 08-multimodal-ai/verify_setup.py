#!/usr/bin/env python3
"""Verify environment setup for Multimodal AI course."""

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
    print("Multimodal AI - Environment Verification")
    print("=" * 60)
    
    checks = [
        ("torch", "torch", "PyTorch"),
        ("torchvision", "torchvision", "TorchVision"),
        ("torchaudio", "torchaudio", "TorchAudio"),
        ("transformers", "transformers", "Hugging Face Transformers"),
        ("timm", "timm", "PyTorch Image Models"),
        ("PIL", "pillow", "Image processing"),
        ("diffusers", "diffusers", "Diffusion models"),
        ("open_clip", "open_clip_torch", "OpenCLIP"),
        ("sentence_transformers", "sentence-transformers", "Sentence embeddings"),
        ("librosa", "librosa", "Audio processing"),
        ("matplotlib", "matplotlib", "Plotting"),
        ("numpy", "numpy", "Numerical computing"),
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
            print("    Note: Diffusion models benefit from >8GB VRAM")
        else:
            print("  ⚠ CUDA not available (CPU-only mode)")
            print("    Note: Image generation will be slow without GPU")
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All required packages installed!")
        print("\nYou're ready to start the Multimodal AI course.")
    else:
        print("✗ Some packages are missing.")
        print("\nRun: pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
