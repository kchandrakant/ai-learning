"""
Verify setup for AI Ethics & Responsible AI course.
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

def main():
    print("=" * 50)
    print("AI Ethics & Responsible AI - Setup Verification")
    print("=" * 50)
    
    all_passed = True
    
    # Core packages
    print("\n1. Core packages:")
    all_passed &= check_import("numpy")
    all_passed &= check_import("pandas")
    all_passed &= check_import("matplotlib")
    
    # ML/AI packages
    print("\n2. ML/AI packages:")
    all_passed &= check_import("torch")
    all_passed &= check_import("transformers")
    all_passed &= check_import("datasets")
    
    # Fairness packages
    print("\n3. Fairness & Bias packages:")
    all_passed &= check_import("fairlearn")
    # aif360 is optional - complex dependencies
    try:
        import aif360
        print(f"  ✓ aif360")
    except ImportError:
        print(f"  ○ aif360 (optional - complex dependencies)")
    
    # Interpretability packages
    print("\n4. Interpretability packages:")
    all_passed &= check_import("shap")
    all_passed &= check_import("lime")
    try:
        import captum
        print(f"  ✓ captum")
    except ImportError:
        print(f"  ○ captum (optional)")
    
    # Privacy packages
    print("\n5. Privacy packages:")
    try:
        import opacus
        print(f"  ✓ opacus")
    except ImportError:
        print(f"  ○ opacus (optional - for differential privacy)")
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All required packages installed!")
        print("You're ready to start the course.")
    else:
        print("✗ Some packages are missing.")
        print("Run: pip install -r requirements.txt")
    print("=" * 50)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
