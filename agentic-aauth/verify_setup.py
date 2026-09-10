#!/usr/bin/env python3
"""
Verify that all dependencies for the AAuth Learning Path are installed correctly.
Run this after installing requirements.txt to confirm your environment is ready.
"""

import sys
from importlib import import_module


def check_module(name: str, import_name: str = None) -> bool:
    """Check if a module is importable."""
    import_name = import_name or name
    try:
        import_module(import_name)
        return True
    except ImportError:
        return False


def main():
    print("=" * 60)
    print("AAuth Learning Path - Environment Verification")
    print("=" * 60)
    print()

    # Required packages with their import names
    packages = [
        ("cryptography", "cryptography"),
        ("PyJWT", "jwt"),
        ("jwcrypto", "jwcrypto"),
        ("requests", "requests"),
        ("httpx", "httpx"),
        ("flask", "flask"),
        ("pydantic", "pydantic"),
        ("rich", "rich"),
        ("click", "click"),
        ("pytest", "pytest"),
    ]

    # Optional packages
    optional = [
        ("http-message-signatures", "http_message_signatures"),
        ("uvicorn", "uvicorn"),
    ]

    all_good = True
    
    print("Required Packages:")
    print("-" * 40)
    for pkg_name, import_name in packages:
        ok = check_module(pkg_name, import_name)
        status = "✓" if ok else "✗"
        print(f"  {status} {pkg_name}")
        if not ok:
            all_good = False

    print()
    print("Optional Packages:")
    print("-" * 40)
    for pkg_name, import_name in optional:
        ok = check_module(pkg_name, import_name)
        status = "✓" if ok else "○"
        print(f"  {status} {pkg_name}")

    print()
    print("-" * 40)
    
    # Check Python version
    py_version = sys.version_info
    py_ok = py_version >= (3, 10)
    print(f"Python Version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if not py_ok:
        print("  ⚠ Python 3.10+ recommended")
        all_good = False
    else:
        print("  ✓ Version OK")

    print()
    print("=" * 60)
    
    if all_good:
        print("✓ All required packages installed. You're ready to begin!")
        print()
        print("Start with Module 00:")
        print("  cd 00_oauth_limitations")
        print("  # Read README.md, then say 'ready' to start building")
    else:
        print("✗ Some packages are missing. Run:")
        print("  pip install -r requirements.txt")
    
    print("=" * 60)
    
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
