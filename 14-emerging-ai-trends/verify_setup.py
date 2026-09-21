"""
Verify setup for Emerging AI Trends course.
Run this script to check if all dependencies are installed correctly.
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError as e:
        print(f"  ✗ {package_name}: {e}")
        return False

def check_optional(module_name, package_name=None):
    """Check optional module."""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError:
        print(f"  ○ {package_name} (optional)")
        return True  # Don't fail on optional

def main():
    print("=" * 50)
    print("Emerging AI Trends - Setup Verification")
    print("=" * 50)
    
    all_passed = True
    
    # Core packages
    print("\n1. Core packages:")
    all_passed &= check_import("numpy")
    all_passed &= check_import("pandas")
    all_passed &= check_import("matplotlib")
    all_passed &= check_import("torch")
    
    # Transformers ecosystem
    print("\n2. Transformers ecosystem:")
    all_passed &= check_import("transformers")
    all_passed &= check_import("datasets")
    all_passed &= check_import("accelerate")
    
    # State Space Models
    print("\n3. State Space Models:")
    check_optional("mamba_ssm", "mamba-ssm (requires CUDA)")
    
    # Tabular
    print("\n4. Tabular packages:")
    check_optional("pytorch_tabular", "pytorch-tabular")
    
    # Time Series
    print("\n5. Time Series packages:")
    check_optional("pytorch_forecasting", "pytorch-forecasting")
    
    # Graph
    print("\n6. Graph packages:")
    check_optional("torch_geometric", "torch-geometric")
    
    # Efficiency
    print("\n7. Efficiency packages:")
    check_optional("bitsandbytes", "bitsandbytes (quantization)")
    check_optional("optimum", "optimum")
    
    # Utilities
    print("\n8. Utilities:")
    all_passed &= check_import("einops")
    all_passed &= check_import("tqdm")
    
    # Check CUDA
    print("\n9. CUDA availability:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ○ CUDA not available (CPU only - some modules limited)")
    except Exception as e:
        print(f"  ○ Could not check CUDA: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ Core packages installed!")
        print("Optional packages may need separate installation.")
        print("You're ready to start the course.")
    else:
        print("✗ Some core packages are missing.")
        print("Run: pip install -r requirements.txt")
    print("=" * 50)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
