"""
Verify environment setup for ML Foundations course.
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

def main():
    print("=" * 60)
    print("ML FOUNDATIONS COURSE - SETUP VERIFICATION")
    print("=" * 60)
    
    all_good = True
    
    print("\n📋 Python Version:")
    all_good &= check_python_version()
    
    print("\n📋 Scientific Computing:")
    all_good &= check_package('numpy')
    all_good &= check_package('scipy')
    all_good &= check_package('pandas')
    
    print("\n📋 Visualization:")
    all_good &= check_package('matplotlib')
    all_good &= check_package('seaborn')
    
    print("\n📋 Machine Learning:")
    all_good &= check_package('scikit-learn', 'sklearn')
    
    print("\n" + "=" * 60)
    if all_good:
        print("✅ All checks passed!")
    else:
        print("⚠️ Some checks failed. Run: pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
