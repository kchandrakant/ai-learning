#!/usr/bin/env python3
"""
Verify Setup for Course 00: Foundations & Pre-Skills

This script checks that your environment is properly configured
for the course. Run it after installing dependencies:

    python verify_setup.py

All checks should pass before proceeding with the course.
"""

import sys
from typing import Tuple, List

# Minimum Python version required
MIN_PYTHON_VERSION = (3, 9)


def check_python_version() -> Tuple[bool, str]:
    """Check Python version is sufficient."""
    current = sys.version_info[:2]
    if current >= MIN_PYTHON_VERSION:
        return True, f"Python {current[0]}.{current[1]}"
    else:
        return False, f"Python {current[0]}.{current[1]} (need {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+)"


def check_numpy() -> Tuple[bool, str]:
    """Check NumPy installation and basic functionality."""
    try:
        import numpy as np
        
        # Basic functionality test
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.dot(a, b)
        
        if c == 32:  # 1*4 + 2*5 + 3*6 = 32
            return True, f"numpy {np.__version__}"
        else:
            return False, "numpy installed but basic operations failing"
    except ImportError:
        return False, "numpy not installed"
    except Exception as e:
        return False, f"numpy error: {e}"


def check_matplotlib() -> Tuple[bool, str]:
    """Check Matplotlib installation."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        
        # Check that we can create a figure (non-interactive)
        matplotlib.use('Agg')  # Non-interactive backend
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close(fig)
        
        return True, f"matplotlib {matplotlib.__version__}"
    except ImportError:
        return False, "matplotlib not installed"
    except Exception as e:
        return False, f"matplotlib error: {e}"


def check_jupyter() -> Tuple[bool, str]:
    """Check Jupyter installation."""
    try:
        import jupyter
        import notebook
        return True, f"jupyter (notebook {notebook.__version__})"
    except ImportError:
        return False, "jupyter not installed"
    except Exception as e:
        return False, f"jupyter error: {e}"


def check_ipython() -> Tuple[bool, str]:
    """Check IPython installation."""
    try:
        import IPython
        return True, f"IPython {IPython.__version__}"
    except ImportError:
        return False, "IPython not installed"
    except Exception as e:
        return False, f"IPython error: {e}"


def check_pandas() -> Tuple[bool, str]:
    """Check Pandas installation (optional but recommended)."""
    try:
        import pandas as pd
        
        # Basic functionality test
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        if len(df) == 3:
            return True, f"pandas {pd.__version__}"
        else:
            return False, "pandas installed but basic operations failing"
    except ImportError:
        return False, "pandas not installed (optional)"
    except Exception as e:
        return False, f"pandas error: {e}"


def check_seaborn() -> Tuple[bool, str]:
    """Check Seaborn installation (optional but recommended)."""
    try:
        import seaborn as sns
        return True, f"seaborn {sns.__version__}"
    except ImportError:
        return False, "seaborn not installed (optional)"
    except Exception as e:
        return False, f"seaborn error: {e}"


def run_numpy_exercises() -> Tuple[bool, str]:
    """Run a few NumPy exercises to verify understanding."""
    try:
        import numpy as np
        
        # Test 1: Broadcasting
        a = np.ones((3, 4))
        b = np.array([1, 2, 3, 4])
        c = a + b  # Should broadcast
        assert c.shape == (3, 4), "Broadcasting test failed"
        
        # Test 2: Vectorization
        x = np.random.randn(1000)
        y = np.exp(x)  # Vectorized operation
        assert y.shape == (1000,), "Vectorization test failed"
        
        # Test 3: Matrix operations
        A = np.random.randn(5, 3)
        B = np.random.randn(3, 4)
        C = np.dot(A, B)
        assert C.shape == (5, 4), "Matrix multiplication test failed"
        
        return True, "NumPy exercises passed"
    except AssertionError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Exercise error: {e}"


def print_result(name: str, passed: bool, message: str, required: bool = True):
    """Print a formatted result line."""
    status = "✓" if passed else ("✗" if required else "○")
    req_label = "" if required else " (optional)"
    print(f"  {status} {name}: {message}{req_label}")


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("  Course 00: Foundations & Pre-Skills - Setup Verification")
    print("=" * 60 + "\n")
    
    all_required_passed = True
    
    # Required checks
    print("Required Dependencies:")
    print("-" * 40)
    
    required_checks = [
        ("Python", check_python_version),
        ("NumPy", check_numpy),
        ("Matplotlib", check_matplotlib),
        ("Jupyter", check_jupyter),
        ("IPython", check_ipython),
    ]
    
    for name, check_fn in required_checks:
        passed, message = check_fn()
        print_result(name, passed, message, required=True)
        if not passed:
            all_required_passed = False
    
    # Optional checks
    print("\nOptional Dependencies:")
    print("-" * 40)
    
    optional_checks = [
        ("Pandas", check_pandas),
        ("Seaborn", check_seaborn),
    ]
    
    for name, check_fn in optional_checks:
        passed, message = check_fn()
        print_result(name, passed, message, required=False)
    
    # Functionality tests
    print("\nFunctionality Tests:")
    print("-" * 40)
    
    passed, message = run_numpy_exercises()
    print_result("NumPy Exercises", passed, message, required=True)
    if not passed:
        all_required_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_required_passed:
        print("  ✓ All required checks passed!")
        print("  You're ready to start Course 00.")
        print("=" * 60 + "\n")
        return 0
    else:
        print("  ✗ Some required checks failed.")
        print("  Please install missing dependencies:")
        print("    pip install -r requirements.txt")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
