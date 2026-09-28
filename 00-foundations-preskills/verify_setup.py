#!/usr/bin/env python3
"""
Verify Setup for Course 00: Foundations & Pre-Skills

This script checks that your environment is properly configured
for the course. Run it after installing dependencies:

    python verify_setup.py
    python verify_setup.py --verbose    # Show detailed version info
    python verify_setup.py --test       # Run extended functionality tests

All checks should pass before proceeding with the course.
"""

import sys
import os
from typing import Tuple, List, Dict, Any
from pathlib import Path

# Minimum Python version required
MIN_PYTHON_VERSION = (3, 9)


# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================

def check_python_version() -> Tuple[bool, str]:
    """Check Python version is sufficient."""
    current = sys.version_info[:2]
    if current >= MIN_PYTHON_VERSION:
        return True, f"Python {current[0]}.{current[1]}.{sys.version_info[2]}"
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
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Check that we can create a figure (non-interactive)
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
        # Try jupyterlab first (preferred)
        try:
            import jupyterlab
            return True, f"jupyterlab {jupyterlab.__version__}"
        except ImportError:
            pass
        
        # Fall back to notebook
        import notebook
        return True, f"notebook {notebook.__version__}"
    except ImportError:
        return False, "jupyter not installed (install jupyterlab or notebook)"
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


def check_sklearn() -> Tuple[bool, str]:
    """Check scikit-learn installation (optional, used in some examples)."""
    try:
        import sklearn
        return True, f"scikit-learn {sklearn.__version__}"
    except ImportError:
        return False, "scikit-learn not installed (optional)"
    except Exception as e:
        return False, f"scikit-learn error: {e}"


# =============================================================================
# FUNCTIONALITY TESTS
# =============================================================================

def run_numpy_exercises() -> Tuple[bool, str]:
    """Run NumPy exercises to verify functionality."""
    try:
        import numpy as np
        np.random.seed(42)
        
        tests_passed = 0
        total_tests = 5
        
        # Test 1: Broadcasting
        a = np.ones((3, 4))
        b = np.array([1, 2, 3, 4])
        c = a + b
        assert c.shape == (3, 4), "Broadcasting shape failed"
        assert np.allclose(c[0], [2, 3, 4, 5]), "Broadcasting values failed"
        tests_passed += 1
        
        # Test 2: Vectorization
        x = np.random.randn(1000)
        y = np.exp(x)
        assert y.shape == (1000,), "Vectorization shape failed"
        tests_passed += 1
        
        # Test 3: Matrix operations
        A = np.random.randn(5, 3)
        B = np.random.randn(3, 4)
        C = A @ B  # Modern syntax
        assert C.shape == (5, 4), "Matrix multiplication shape failed"
        tests_passed += 1
        
        # Test 4: Indexing and slicing
        arr = np.arange(12).reshape(3, 4)
        assert arr[1, 2] == 6, "Indexing failed"
        assert arr[:, 0].tolist() == [0, 4, 8], "Slicing failed"
        tests_passed += 1
        
        # Test 5: Linear algebra
        M = np.random.randn(3, 3)
        M = M @ M.T  # Make symmetric positive definite
        eigenvalues = np.linalg.eigvalsh(M)
        assert len(eigenvalues) == 3, "Eigenvalue computation failed"
        tests_passed += 1
        
        return True, f"All {total_tests} NumPy tests passed"
    except AssertionError as e:
        return False, f"Test failed: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def run_matplotlib_exercises() -> Tuple[bool, str]:
    """Run Matplotlib exercises to verify functionality."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        tests_passed = 0
        total_tests = 3
        
        # Test 1: Basic line plot
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x))
        ax.set_xlabel('x')
        ax.set_ylabel('sin(x)')
        plt.close(fig)
        tests_passed += 1
        
        # Test 2: Subplots
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            ax.set_title(f'Subplot {i+1}')
        plt.tight_layout()
        plt.close(fig)
        tests_passed += 1
        
        # Test 3: Histogram
        fig, ax = plt.subplots()
        data = np.random.randn(1000)
        ax.hist(data, bins=30)
        plt.close(fig)
        tests_passed += 1
        
        return True, f"All {total_tests} Matplotlib tests passed"
    except Exception as e:
        return False, f"Error: {e}"


def run_python_features_test() -> Tuple[bool, str]:
    """Test Python features used in the course."""
    try:
        tests_passed = 0
        total_tests = 5
        
        # Test 1: List comprehensions
        squares = [x**2 for x in range(5)]
        assert squares == [0, 1, 4, 9, 16], "List comprehension failed"
        tests_passed += 1
        
        # Test 2: Dictionary comprehensions
        d = {x: x**2 for x in range(3)}
        assert d == {0: 0, 1: 1, 2: 4}, "Dict comprehension failed"
        tests_passed += 1
        
        # Test 3: Generator expressions
        gen = (x**2 for x in range(5))
        assert list(gen) == [0, 1, 4, 9, 16], "Generator failed"
        tests_passed += 1
        
        # Test 4: Type hints (syntax check)
        def typed_func(x: int, y: str = "default") -> bool:
            return True
        assert typed_func(1), "Type hints syntax failed"
        tests_passed += 1
        
        # Test 5: Context managers
        class TestContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        
        with TestContext() as ctx:
            pass
        tests_passed += 1
        
        return True, f"All {total_tests} Python feature tests passed"
    except Exception as e:
        return False, f"Error: {e}"


# =============================================================================
# COURSE FILES CHECK
# =============================================================================

def check_course_files() -> Tuple[bool, str]:
    """Check that course files are present."""
    try:
        script_dir = Path(__file__).parent
        
        required_files = [
            'LEARNING_PATH.md',
            'requirements.txt',
            '01_python_essentials/README.md',
            '02_numpy_fundamentals/README.md',
            '03_matplotlib_visualization/README.md',
            '04_jupyter_workflow/README.md',
            '05_reading_ml_papers/README.md',
            '06_staying_current/README.md',
            'self_assessment/README.md',
        ]
        
        exercise_files = [
            '01_python_essentials/exercises.py',
            '02_numpy_fundamentals/exercises.py',
            '03_matplotlib_visualization/exercises.py',
            '04_jupyter_workflow/notebook_template.ipynb',
            '05_reading_ml_papers/paper_template.md',
            '06_staying_current/resources.md',
            'self_assessment/assessment.py',
        ]
        
        missing = []
        for f in required_files:
            if not (script_dir / f).exists():
                missing.append(f)
        
        exercise_count = sum(1 for f in exercise_files if (script_dir / f).exists())
        
        if missing:
            return False, f"Missing {len(missing)} files: {', '.join(missing[:3])}..."
        
        return True, f"All required files present, {exercise_count}/{len(exercise_files)} exercises available"
    except Exception as e:
        return False, f"Error checking files: {e}"


# =============================================================================
# MAIN
# =============================================================================

def print_result(name: str, passed: bool, message: str, required: bool = True):
    """Print a formatted result line."""
    status = "✓" if passed else ("✗" if required else "○")
    req_label = "" if required else " (optional)"
    print(f"  {status} {name}: {message}{req_label}")


def print_verbose_info():
    """Print detailed system information."""
    print("\n" + "=" * 60)
    print("  Detailed System Information")
    print("=" * 60 + "\n")
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    try:
        import numpy as np
        print(f"\nNumPy config:")
        print(f"  BLAS: {np.__config__.blas_info if hasattr(np.__config__, 'blas_info') else 'N/A'}")
    except:
        pass
    
    print(f"\nWorking directory: {os.getcwd()}")
    print(f"Script directory: {Path(__file__).parent}")


def main():
    """Run all verification checks."""
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    extended = "--test" in sys.argv or "-t" in sys.argv
    
    print("\n" + "=" * 60)
    print("  Course 00: Foundations & Pre-Skills - Setup Verification")
    print("=" * 60 + "\n")
    
    if verbose:
        print_verbose_info()
        print()
    
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
        ("Scikit-learn", check_sklearn),
    ]
    
    for name, check_fn in optional_checks:
        passed, message = check_fn()
        print_result(name, passed, message, required=False)
    
    # Functionality tests
    print("\nFunctionality Tests:")
    print("-" * 40)
    
    functionality_checks = [
        ("NumPy Operations", run_numpy_exercises),
        ("Matplotlib Plotting", run_matplotlib_exercises),
        ("Python Features", run_python_features_test),
    ]
    
    for name, check_fn in functionality_checks:
        passed, message = check_fn()
        print_result(name, passed, message, required=True)
        if not passed:
            all_required_passed = False
    
    # Course files check
    print("\nCourse Files:")
    print("-" * 40)
    
    passed, message = check_course_files()
    print_result("Course Content", passed, message, required=True)
    if not passed:
        all_required_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_required_passed:
        print("  ✓ All required checks passed!")
        print("  You're ready to start Course 00.")
        print("\n  Quick start:")
        print("    1. Read LEARNING_PATH.md for course overview")
        print("    2. Start with 01_python_essentials/")
        print("    3. Run exercises: python 01_python_essentials/exercises.py")
        print("=" * 60 + "\n")
        return 0
    else:
        print("  ✗ Some required checks failed.")
        print("\n  To fix:")
        print("    1. Install dependencies: pip install -r requirements.txt")
        print("    2. Re-run this script: python verify_setup.py")
        print("\n  For help:")
        print("    - Check Python version: python --version")
        print("    - Verbose mode: python verify_setup.py --verbose")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
