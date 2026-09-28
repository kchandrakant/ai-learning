"""
Self-Assessment: Are You Ready for Course 01?
==============================================

This script tests your skills from Course 00: Foundations & Pre-Skills.
Complete all sections to verify you're ready for Course 01: ML Foundations.

Usage:
    python assessment.py           # Run assessment interactively
    python assessment.py --auto    # Run with auto-generated answers (demo mode)
    python assessment.py --answers # Show solutions without running

Time Limits (suggested):
    - Python: 15 minutes
    - NumPy: 20 minutes
    - Matplotlib: 15 minutes
    - Concepts: 10 minutes

Passing Criteria:
    - Python: 3/3 exercises
    - NumPy: 2.5/3 exercises (partial credit allowed)
    - Matplotlib: Both plots render correctly
    - Concepts: 5/8 questions
"""

import sys
import time
import traceback
from typing import List, Dict, Any, Optional, Callable, Tuple

# Check for numpy and matplotlib
try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: Matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

# Set seed for reproducibility
np.random.seed(42)


# =============================================================================
# ASSESSMENT FRAMEWORK
# =============================================================================

class AssessmentResult:
    """Store result of a single exercise."""
    def __init__(self, name: str, passed: bool, score: float, 
                 max_score: float, message: str = ""):
        self.name = name
        self.passed = passed
        self.score = score
        self.max_score = max_score
        self.message = message
    
    def __repr__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} {self.name}: {self.score}/{self.max_score} - {self.message}"


class Assessment:
    """Run and track assessment exercises."""
    
    def __init__(self):
        self.results: List[AssessmentResult] = []
        self.section_scores: Dict[str, Tuple[float, float]] = {}
    
    def run_exercise(self, name: str, test_func: Callable, max_score: float = 1.0) -> AssessmentResult:
        """Run a single exercise and record result."""
        try:
            passed, score, message = test_func()
            result = AssessmentResult(name, passed, score * max_score, max_score, message)
        except Exception as e:
            result = AssessmentResult(name, False, 0, max_score, f"Error: {str(e)}")
            if "--debug" in sys.argv:
                traceback.print_exc()
        
        self.results.append(result)
        print(f"  {result}")
        return result
    
    def add_section_score(self, section: str, score: float, max_score: float):
        """Add section total."""
        self.section_scores[section] = (score, max_score)
    
    def print_summary(self):
        """Print final assessment summary."""
        print("\n" + "="*60)
        print("ASSESSMENT SUMMARY")
        print("="*60)
        
        total_score = 0
        total_max = 0
        all_passed = True
        
        for section, (score, max_score) in self.section_scores.items():
            pct = (score / max_score * 100) if max_score > 0 else 0
            status = "✓" if pct >= 70 else "✗"
            print(f"{status} {section}: {score:.1f}/{max_score:.1f} ({pct:.0f}%)")
            total_score += score
            total_max += max_score
            if pct < 70:
                all_passed = False
        
        print("-"*40)
        total_pct = (total_score / total_max * 100) if total_max > 0 else 0
        print(f"TOTAL: {total_score:.1f}/{total_max:.1f} ({total_pct:.0f}%)")
        
        print("\n" + "="*60)
        if all_passed:
            print("🎉 CONGRATULATIONS! You're ready for Course 01!")
            print("="*60)
            print("\nNext step: Navigate to ../01-ml-foundations/")
        else:
            print("📚 Some areas need more practice.")
            print("="*60)
            print("\nReview the modules for sections where you scored < 70%:")
            for section, (score, max_score) in self.section_scores.items():
                if score / max_score < 0.7:
                    print(f"  - {section}")


# =============================================================================
# SECTION 1: PYTHON ESSENTIALS
# =============================================================================

def run_python_assessment(assessment: Assessment, auto_mode: bool = False):
    """Run Python assessment exercises."""
    
    print("\n" + "="*60)
    print("SECTION 1: PYTHON ESSENTIALS")
    print("="*60)
    print("Time limit: 15 minutes\n")
    
    section_score = 0.0
    section_max = 3.0
    
    # Test data
    records = [
        {'name': 'Alice', 'score': 92, 'subject': 'math'},
        {'name': 'Bob', 'score': 78, 'subject': 'math'},
        {'name': 'Charlie', 'score': 85, 'subject': 'physics'},
        {'name': 'Diana', 'score': 91, 'subject': 'physics'},
        {'name': 'Eve', 'score': 67, 'subject': 'math'},
    ]
    
    # Exercise 1.1: Comprehensions
    print("--- Exercise 1.1: Comprehensions ---")
    print(f"Given records: {records}\n")
    
    if auto_mode:
        names = [r['name'] for r in records]
        math_scores = [r['score'] for r in records if r['subject'] == 'math']
        name_to_score = {r['name']: r['score'] for r in records}
        avg_score = sum(r['score'] for r in records) / len(records)
    else:
        print("Write ONE LINE for each (press Enter for empty):")
        print("1. List of all names:")
        user_code = input("   names = ").strip()
        try:
            names = eval(user_code) if user_code else []
        except:
            names = []
        
        print("2. List of scores for math students only:")
        user_code = input("   math_scores = ").strip()
        try:
            math_scores = eval(user_code) if user_code else []
        except:
            math_scores = []
        
        print("3. Dictionary mapping name -> score:")
        user_code = input("   name_to_score = ").strip()
        try:
            name_to_score = eval(user_code) if user_code else {}
        except:
            name_to_score = {}
        
        print("4. Average score (use sum() and len()):")
        user_code = input("   avg_score = ").strip()
        try:
            avg_score = eval(user_code) if user_code else 0
        except:
            avg_score = 0
    
    def test_comprehensions():
        correct = 0
        total = 4
        msgs = []
        
        expected_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
        if names == expected_names:
            correct += 1
        else:
            msgs.append(f"names: got {names}, expected {expected_names}")
        
        expected_math = [92, 78, 67]
        if math_scores == expected_math:
            correct += 1
        else:
            msgs.append(f"math_scores: got {math_scores}, expected {expected_math}")
        
        expected_dict = {'Alice': 92, 'Bob': 78, 'Charlie': 85, 'Diana': 91, 'Eve': 67}
        if name_to_score == expected_dict:
            correct += 1
        else:
            msgs.append(f"name_to_score incorrect")
        
        expected_avg = 82.6
        if abs(avg_score - expected_avg) < 0.1:
            correct += 1
        else:
            msgs.append(f"avg_score: got {avg_score}, expected {expected_avg}")
        
        passed = correct == total
        score = correct / total
        message = f"{correct}/{total} correct" + (f" ({'; '.join(msgs)})" if msgs else "")
        return passed, score, message
    
    result = assessment.run_exercise("1.1 Comprehensions", test_comprehensions)
    section_score += result.score
    
    # Exercise 1.2: Generator Function
    print("\n--- Exercise 1.2: Generator Function ---")
    print("Write a generator that yields all pairs (i, j) where i < j")
    print("Example: pairs([1,2,3]) yields (1,2), (1,3), (2,3)\n")
    
    if auto_mode:
        def pairs(items):
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    yield (items[i], items[j])
    else:
        print("Enter your function (type 'done' on new line when finished):")
        lines = []
        while True:
            line = input()
            if line.strip().lower() == 'done':
                break
            lines.append(line)
        
        code = '\n'.join(lines)
        try:
            exec(code, globals())
        except Exception as e:
            def pairs(items):
                return []
    
    def test_generator():
        try:
            result = list(pairs([1, 2, 3, 4]))
            expected = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
            if result == expected:
                return True, 1.0, "Correct!"
            else:
                return False, 0.5, f"Got {result}, expected {expected}"
        except Exception as e:
            return False, 0.0, f"Error: {e}"
    
    result = assessment.run_exercise("1.2 Generator", test_generator)
    section_score += result.score
    
    # Exercise 1.3: Class
    print("\n--- Exercise 1.3: Class Implementation ---")
    print("Implement a Counter class with:")
    print("  - __init__ starts count at 0")
    print("  - increment() adds 1")
    print("  - decrement() subtracts 1 (but not below 0)")
    print("  - value property returns current count")
    print("  - __repr__ returns 'Counter(value=N)'\n")
    
    if auto_mode:
        class Counter:
            def __init__(self):
                self._count = 0
            
            def increment(self):
                self._count += 1
            
            def decrement(self):
                self._count = max(0, self._count - 1)
            
            @property
            def value(self):
                return self._count
            
            def __repr__(self):
                return f"Counter(value={self._count})"
    else:
        print("Enter your class (type 'done' on new line when finished):")
        lines = []
        while True:
            line = input()
            if line.strip().lower() == 'done':
                break
            lines.append(line)
        
        code = '\n'.join(lines)
        try:
            exec(code, globals())
        except Exception as e:
            class Counter:
                pass
    
    def test_class():
        try:
            c = Counter()
            checks = 0
            total = 4
            
            if c.value == 0:
                checks += 1
            
            c.increment()
            c.increment()
            if c.value == 2:
                checks += 1
            
            c.decrement()
            c.decrement()
            c.decrement()  # Should not go below 0
            if c.value == 0:
                checks += 1
            
            c.increment()
            if repr(c) == "Counter(value=1)":
                checks += 1
            
            passed = checks == total
            return passed, checks / total, f"{checks}/{total} checks passed"
        except Exception as e:
            return False, 0.0, f"Error: {e}"
    
    result = assessment.run_exercise("1.3 Class", test_class)
    section_score += result.score
    
    assessment.add_section_score("Python", section_score, section_max)


# =============================================================================
# SECTION 2: NUMPY FUNDAMENTALS
# =============================================================================

def run_numpy_assessment(assessment: Assessment, auto_mode: bool = False):
    """Run NumPy assessment exercises."""
    
    print("\n" + "="*60)
    print("SECTION 2: NUMPY FUNDAMENTALS")
    print("="*60)
    print("Time limit: 20 minutes\n")
    
    section_score = 0.0
    section_max = 3.0
    
    # Exercise 2.1: Broadcasting Predictions
    print("--- Exercise 2.1: Broadcasting Predictions ---")
    print("Predict the OUTPUT SHAPE (or 'Error') for each operation:\n")
    
    test_cases = [
        ("(4, 3) + (3,)", "(4, 3)"),
        ("(4, 3) + (4,)", "Error"),
        ("(2, 3, 4) + (3, 4)", "(2, 3, 4)"),
        ("(5, 1, 3) + (1, 4, 3)", "(5, 4, 3)"),
        ("(3, 4) * (4, 3)", "Error"),
    ]
    
    if auto_mode:
        answers = [case[1] for case in test_cases]
    else:
        answers = []
        for expr, _ in test_cases:
            answer = input(f"  {expr} -> ").strip()
            answers.append(answer)
    
    def test_broadcasting():
        correct = 0
        for i, (expr, expected) in enumerate(test_cases):
            if answers[i].lower().replace(" ", "") == expected.lower().replace(" ", ""):
                correct += 1
        
        passed = correct == len(test_cases)
        return passed, correct / len(test_cases), f"{correct}/{len(test_cases)} correct"
    
    result = assessment.run_exercise("2.1 Broadcasting", test_broadcasting)
    section_score += result.score
    
    # Exercise 2.2: Vectorized Operations
    print("\n--- Exercise 2.2: Vectorized Operations ---")
    print("Given: data = np.random.randn(100, 5)  # 100 samples, 5 features\n")
    
    data = np.random.randn(100, 5)
    
    if auto_mode:
        # Standardize
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        standardized = (data - mean) / std
        
        # Max row index
        max_row_idx = data.sum(axis=1).argmax()
        
        # Correlation
        X = (data - data.mean(axis=0)) / data.std(axis=0)
        correlation = (X.T @ X) / (len(X) - 1)
    else:
        print("1. Standardize each column (zero mean, unit variance):")
        print("   standardized = ?")
        user_code = input("   > ").strip()
        try:
            standardized = eval(user_code)
        except:
            standardized = np.zeros_like(data)
        
        print("\n2. Find the row index with the maximum sum:")
        print("   max_row_idx = ?")
        user_code = input("   > ").strip()
        try:
            max_row_idx = eval(user_code)
        except:
            max_row_idx = -1
        
        print("\n3. Compute the correlation matrix (5x5) between features:")
        print("   correlation = ?")
        user_code = input("   > ").strip()
        try:
            correlation = eval(user_code)
        except:
            correlation = np.zeros((5, 5))
    
    def test_vectorized():
        checks = 0
        total = 3
        msgs = []
        
        # Check standardization
        if standardized.shape == data.shape:
            col_means = standardized.mean(axis=0)
            col_stds = standardized.std(axis=0)
            if np.allclose(col_means, 0, atol=1e-10) and np.allclose(col_stds, 1, atol=0.1):
                checks += 1
            else:
                msgs.append("standardized: mean or std incorrect")
        else:
            msgs.append("standardized: wrong shape")
        
        # Check max row
        expected_max = data.sum(axis=1).argmax()
        if max_row_idx == expected_max:
            checks += 1
        else:
            msgs.append(f"max_row_idx: got {max_row_idx}, expected {expected_max}")
        
        # Check correlation
        if correlation.shape == (5, 5):
            if np.allclose(np.diag(correlation), 1, atol=0.1):
                checks += 1
            else:
                msgs.append("correlation: diagonal should be ~1")
        else:
            msgs.append("correlation: wrong shape")
        
        passed = checks == total
        return passed, checks / total, f"{checks}/{total}" + (f" ({'; '.join(msgs)})" if msgs else "")
    
    result = assessment.run_exercise("2.2 Vectorized", test_vectorized)
    section_score += result.score
    
    # Exercise 2.3: Cosine Similarity
    print("\n--- Exercise 2.3: Cosine Similarity Matrix ---")
    print("Given two sets of vectors:")
    print("  A = np.random.randn(10, 64)  # 10 vectors")
    print("  B = np.random.randn(20, 64)  # 20 vectors")
    print("\nCompute cosine similarity matrix (10x20) WITHOUT loops.\n")
    
    A = np.random.randn(10, 64)
    B = np.random.randn(20, 64)
    
    if auto_mode:
        A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
        B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
        cosine_sim = A_norm @ B_norm.T
    else:
        print("cosine_sim = ?")
        user_code = input("> ").strip()
        try:
            cosine_sim = eval(user_code)
        except:
            cosine_sim = np.zeros((10, 20))
    
    def test_cosine():
        # Expected result
        A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
        B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
        expected = A_norm @ B_norm.T
        
        if cosine_sim.shape != (10, 20):
            return False, 0.0, f"Wrong shape: got {cosine_sim.shape}, expected (10, 20)"
        
        if np.allclose(cosine_sim, expected, atol=1e-6):
            return True, 1.0, "Correct!"
        else:
            # Check if values are in valid range
            if np.all(cosine_sim >= -1.1) and np.all(cosine_sim <= 1.1):
                return False, 0.5, "Shape correct, but values differ"
            return False, 0.25, "Shape correct, values out of range"
    
    result = assessment.run_exercise("2.3 Cosine Similarity", test_cosine)
    section_score += result.score
    
    assessment.add_section_score("NumPy", section_score, section_max)


# =============================================================================
# SECTION 3: MATPLOTLIB
# =============================================================================

def run_matplotlib_assessment(assessment: Assessment, auto_mode: bool = False):
    """Run Matplotlib assessment exercises."""
    
    print("\n" + "="*60)
    print("SECTION 3: MATPLOTLIB")
    print("="*60)
    print("Time limit: 15 minutes\n")
    
    section_score = 0.0
    section_max = 2.0
    
    # Exercise 3.1: Training Curves
    print("--- Exercise 3.1: Training Curves ---")
    print("Create a figure showing training and validation loss curves.\n")
    
    epochs = np.arange(1, 51)
    train_loss = 2.5 * np.exp(-0.08 * epochs) + 0.1 + np.random.randn(50) * 0.02
    val_loss = 2.7 * np.exp(-0.06 * epochs) + 0.15 + np.random.randn(50) * 0.03
    
    def test_training_curves():
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs, train_loss, label='Train', color='blue')
            ax.plot(epochs, val_loss, label='Validation', color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.savefig('assessment_training_curves.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            return True, 1.0, "Plot saved to assessment_training_curves.png"
        except Exception as e:
            return False, 0.0, f"Error: {e}"
    
    if auto_mode:
        result = assessment.run_exercise("3.1 Training Curves", test_training_curves)
    else:
        print("Press Enter to generate the plot (or type 'skip'):")
        response = input("> ").strip().lower()
        if response == 'skip':
            result = AssessmentResult("3.1 Training Curves", False, 0, 1.0, "Skipped")
            assessment.results.append(result)
            print(f"  {result}")
        else:
            result = assessment.run_exercise("3.1 Training Curves", test_training_curves)
    
    section_score += result.score
    
    # Exercise 3.2: Multi-panel Figure
    print("\n--- Exercise 3.2: Multi-panel Figure ---")
    print("Create a 2x2 figure with different plot types.\n")
    
    def test_multipanel():
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Top-left: Histogram
            axes[0, 0].hist(np.random.randn(1000), bins=30, edgecolor='black')
            axes[0, 0].set_title('Normal Distribution')
            
            # Top-right: Scatter
            class_0 = np.random.randn(50, 2)
            class_1 = np.random.randn(50, 2) + 2
            axes[0, 1].scatter(class_0[:, 0], class_0[:, 1], label='Class 0')
            axes[0, 1].scatter(class_1[:, 0], class_1[:, 1], label='Class 1')
            axes[0, 1].set_title('2D Classification')
            axes[0, 1].legend()
            
            # Bottom-left: Line
            x = np.linspace(0, 2*np.pi, 100)
            axes[1, 0].plot(x, np.sin(x), label='sin')
            axes[1, 0].plot(x, np.cos(x), label='cos')
            axes[1, 0].set_title('Trigonometric Functions')
            axes[1, 0].legend()
            
            # Bottom-right: Bar
            axes[1, 1].bar(['A', 'B', 'C', 'D'], [3, 7, 2, 8])
            axes[1, 1].set_title('Bar Chart')
            
            plt.tight_layout()
            fig.savefig('assessment_multipanel.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            return True, 1.0, "Plot saved to assessment_multipanel.png"
        except Exception as e:
            return False, 0.0, f"Error: {e}"
    
    if auto_mode:
        result = assessment.run_exercise("3.2 Multi-panel", test_multipanel)
    else:
        print("Press Enter to generate the plot (or type 'skip'):")
        response = input("> ").strip().lower()
        if response == 'skip':
            result = AssessmentResult("3.2 Multi-panel", False, 0, 1.0, "Skipped")
            assessment.results.append(result)
            print(f"  {result}")
        else:
            result = assessment.run_exercise("3.2 Multi-panel", test_multipanel)
    
    section_score += result.score
    
    assessment.add_section_score("Matplotlib", section_score, section_max)


# =============================================================================
# SECTION 4: CONCEPTS
# =============================================================================

def run_concepts_assessment(assessment: Assessment, auto_mode: bool = False):
    """Run conceptual assessment."""
    
    print("\n" + "="*60)
    print("SECTION 4: CONCEPTS (Jupyter & Paper Reading)")
    print("="*60)
    print("Time limit: 10 minutes\n")
    
    section_score = 0.0
    section_max = 8.0
    
    questions = [
        {
            "question": "What magic command times a single line of code?",
            "answer": "%timeit",
            "alternatives": ["timeit", "%%timeit"]
        },
        {
            "question": "What magic command enters the debugger after an error?",
            "answer": "%debug",
            "alternatives": ["debug", "pdb"]
        },
        {
            "question": "How do you run a shell command in Jupyter?",
            "answer": "!command",
            "alternatives": ["!", "!ls", "shell"]
        },
        {
            "question": "What does 'x ∈ ℝ^d' mean in paper notation?",
            "answer": "x is a d-dimensional real vector",
            "alternatives": ["d-dimensional", "real vector", "vector of dimension d"]
        },
        {
            "question": "What does 'argmax' mean?",
            "answer": "The input that maximizes a function",
            "alternatives": ["input that maximizes", "value that maximizes", "argument of maximum"]
        },
        {
            "question": "What section of a paper should you read FIRST?",
            "answer": "Abstract",
            "alternatives": ["abstract", "the abstract"]
        },
        {
            "question": "In the three-pass reading strategy, how long is Pass 1?",
            "answer": "5-10 minutes",
            "alternatives": ["10 minutes", "5 minutes", "10", "5-10"]
        },
        {
            "question": "What does ∇_θ L(θ) represent?",
            "answer": "The gradient of loss L with respect to parameters θ",
            "alternatives": ["gradient", "gradient of loss", "derivative"]
        },
    ]
    
    print("--- Conceptual Questions ---\n")
    
    correct = 0
    for i, q in enumerate(questions):
        print(f"{i+1}. {q['question']}")
        
        if auto_mode:
            user_answer = q['answer']
            print(f"   > {user_answer}")
        else:
            user_answer = input("   > ").strip()
        
        # Check answer
        is_correct = False
        if user_answer.lower() == q['answer'].lower():
            is_correct = True
        else:
            for alt in q['alternatives']:
                if alt.lower() in user_answer.lower():
                    is_correct = True
                    break
        
        if is_correct:
            correct += 1
            print("   ✓ Correct!")
        else:
            print(f"   ✗ Expected: {q['answer']}")
        print()
    
    section_score = correct
    assessment.add_section_score("Concepts", section_score, section_max)


# =============================================================================
# MAIN
# =============================================================================

def show_answers():
    """Show all answers without running assessment."""
    print("\n" + "="*60)
    print("ASSESSMENT SOLUTIONS")
    print("="*60)
    
    print("""
SECTION 1: PYTHON
-----------------

1.1 Comprehensions:
    names = [r['name'] for r in records]
    math_scores = [r['score'] for r in records if r['subject'] == 'math']
    name_to_score = {r['name']: r['score'] for r in records}
    avg_score = sum(r['score'] for r in records) / len(records)

1.2 Generator:
    def pairs(items):
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                yield (items[i], items[j])

1.3 Class:
    class Counter:
        def __init__(self):
            self._count = 0
        
        def increment(self):
            self._count += 1
        
        def decrement(self):
            self._count = max(0, self._count - 1)
        
        @property
        def value(self):
            return self._count
        
        def __repr__(self):
            return f"Counter(value={self._count})"


SECTION 2: NUMPY
----------------

2.1 Broadcasting:
    (4, 3) + (3,)         → (4, 3)
    (4, 3) + (4,)         → Error
    (2, 3, 4) + (3, 4)    → (2, 3, 4)
    (5, 1, 3) + (1, 4, 3) → (5, 4, 3)
    (3, 4) * (4, 3)       → Error

2.2 Vectorized Operations:
    # Standardize
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    standardized = (data - mean) / std
    
    # Max row index
    max_row_idx = data.sum(axis=1).argmax()
    
    # Correlation
    X = (data - data.mean(axis=0)) / data.std(axis=0)
    correlation = (X.T @ X) / (len(X) - 1)

2.3 Cosine Similarity:
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    cosine_sim = A_norm @ B_norm.T


SECTION 4: CONCEPTS
-------------------

1. %timeit
2. %debug
3. !command (e.g., !ls)
4. x is a d-dimensional real vector
5. The input that maximizes a function
6. Abstract
7. 5-10 minutes
8. The gradient of loss L with respect to parameters θ
""")


def main():
    """Run the full assessment."""
    
    if "--answers" in sys.argv:
        show_answers()
        return
    
    auto_mode = "--auto" in sys.argv
    
    print("\n" + "="*60)
    print("COURSE 00 SELF-ASSESSMENT")
    print("="*60)
    
    if auto_mode:
        print("\nRunning in AUTO mode (demonstrating with correct answers)\n")
    else:
        print("""
This assessment covers the skills from Course 00: Foundations & Pre-Skills.

Instructions:
- Answer each question to the best of your ability
- Time yourself for each section
- It's okay to not know everything - identify areas for review

Passing criteria:
- Python: 3/3 exercises
- NumPy: 2.5/3 exercises
- Matplotlib: Both plots render
- Concepts: 5/8 questions

Press Enter to begin...
""")
        input()
    
    assessment = Assessment()
    
    # Run all sections
    run_python_assessment(assessment, auto_mode)
    run_numpy_assessment(assessment, auto_mode)
    run_matplotlib_assessment(assessment, auto_mode)
    run_concepts_assessment(assessment, auto_mode)
    
    # Print summary
    assessment.print_summary()


if __name__ == "__main__":
    main()
