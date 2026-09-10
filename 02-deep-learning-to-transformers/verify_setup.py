"""
Verify environment setup for Deep Learning to Transformers course.
"""

import sys

def check_python_version():
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (need 3.10+)")
        return False

def check_package(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name} ({version})")
        return True
    except ImportError:
        print(f"❌ {package_name} not installed")
        return False

def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available ({torch.cuda.get_device_name(0)})")
        else:
            print("⚠️ CUDA not available (CPU only - training will be slower)")
        return True
    except:
        return False

def main():
    print("=" * 60)
    print("DEEP LEARNING TO TRANSFORMERS - SETUP VERIFICATION")
    print("=" * 60)
    
    all_good = True
    
    print("\n📋 Python Version:")
    all_good &= check_python_version()
    
    print("\n📋 Core Libraries:")
    all_good &= check_package('numpy')
    all_good &= check_package('scipy')
    
    print("\n📋 Deep Learning:")
    all_good &= check_package('torch')
    check_cuda()
    
    print("\n📋 Visualization:")
    all_good &= check_package('matplotlib')
    
    print("\n" + "=" * 60)
    if all_good:
        print("✅ All checks passed!")
    else:
        print("⚠️ Some checks failed. Run: pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
