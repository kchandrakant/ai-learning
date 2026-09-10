# ML Foundations: A Step-by-Step Learning Journey

This guide builds the mathematical and conceptual foundations needed for modern machine learning and deep learning. No prior ML knowledge assumed — just basic programming.

---

## 🎯 Prerequisites

- Python basics (variables, functions, loops)
- High school math (algebra, basic functions)
- Willingness to work through equations

---

## 📚 Part 1: Mathematical Foundations

The math that powers machine learning.

### Step 1: Linear Algebra Essentials
**Why it matters:** Neural networks are matrix operations. Understanding linear algebra is non-negotiable.

**What we'll cover:**
- Vectors and vector operations
- Matrices and matrix multiplication
- Dot products and their geometric meaning
- Transpose, inverse, identity
- Eigenvalues and eigenvectors (intuition)
- NumPy for linear algebra

**Key insight:**
```
Neural network layer: y = Wx + b
This is just matrix multiplication + vector addition!
```

---

### Step 2: Calculus for ML
**Why it matters:** Training = finding parameters that minimize loss. Calculus tells us which direction to move.

**What we'll cover:**
- Derivatives and their meaning
- Partial derivatives
- The chain rule (critical for backprop!)
- Gradients and gradient descent
- Local vs global minima

**Key equation:**
```
Gradient descent: θ_new = θ_old - α × ∇L(θ)
"Move parameters in the direction that reduces loss"
```

---

### Step 3: Probability & Statistics
**Why it matters:** ML is about learning patterns from data. Probability quantifies uncertainty.

**What we'll cover:**
- Probability basics (events, distributions)
- Conditional probability and Bayes' theorem
- Common distributions (Gaussian, Bernoulli)
- Expected value and variance
- Maximum likelihood estimation
- Basic statistical inference

**Key concepts:**
```
P(A|B) = P(B|A) × P(A) / P(B)  ← Bayes' theorem
argmax P(data|θ)               ← Maximum likelihood
```

---

## 📚 Part 2: Core ML Concepts

Understanding learning from data.

### Step 4: The ML Problem Setup
**Why it matters:** Before algorithms, understand what we're trying to do.

**What we'll cover:**
- Supervised vs unsupervised vs reinforcement learning
- Features, labels, and examples
- Training, validation, and test sets
- The bias-variance tradeoff
- Underfitting and overfitting
- Cross-validation

**The fundamental equation:**
```
Prediction error = Bias² + Variance + Irreducible noise
```

---

### Step 5: Linear Regression
**Why it matters:** The simplest supervised learning algorithm. Foundation for everything else.

**What we'll build:**
- Hypothesis function: y = wx + b
- Mean squared error loss
- Gradient descent for linear regression
- Closed-form solution (normal equation)
- Polynomial features for non-linearity
- Regularization (Ridge, Lasso)

**Key equations:**
```
Loss: L = (1/n) Σ (y_pred - y_true)²
Gradient: ∂L/∂w = (2/n) Σ (y_pred - y_true) × x
```

---

### Step 6: Logistic Regression & Classification
**Why it matters:** Classification is everywhere. Logistic regression introduces key concepts like sigmoid and cross-entropy.

**What we'll build:**
- Binary classification setup
- Sigmoid function: σ(z) = 1 / (1 + e^(-z))
- Cross-entropy loss
- Decision boundaries
- Multi-class classification (softmax)
- Evaluation metrics: accuracy, precision, recall, F1

**Key insight:**
```
Logistic regression outputs probabilities, not just classes.
P(y=1|x) = σ(wx + b)
```

---

### Step 7: Gradient Descent Deep Dive
**Why it matters:** Gradient descent trains almost all ML models. Understanding variants is essential.

**What we'll cover:**
- Batch gradient descent
- Stochastic gradient descent (SGD)
- Mini-batch gradient descent
- Learning rate selection
- Momentum
- Adam optimizer
- Learning rate schedules

**Optimizer progression:**
```
SGD → SGD + Momentum → RMSprop → Adam
```

---

### Step 8: Regularization & Generalization
**Why it matters:** Models that memorize training data are useless. Regularization helps generalize.

**What we'll cover:**
- L1 regularization (Lasso) — sparsity
- L2 regularization (Ridge) — weight decay
- Elastic Net
- Early stopping
- Dropout (preview for deep learning)
- Data augmentation concepts

**Key tradeoff:**
```
Training loss ↓ but validation loss ↑ = Overfitting!
Regularization adds penalty for complex models.
```

---

## 📚 Part 3: Classical ML Algorithms

Algorithms every ML practitioner should know.

### Step 9: Decision Trees & Ensembles
**Why it matters:** Tree-based methods are still top performers on tabular data.

**What we'll cover:**
- Decision tree construction
- Information gain and Gini impurity
- Pruning
- Random Forests
- Gradient Boosting (XGBoost, LightGBM)
- When to use trees vs neural networks

---

### Step 10: Support Vector Machines
**Why it matters:** SVMs introduce important concepts like margins and kernels.

**What we'll cover:**
- Maximum margin classifiers
- Support vectors
- Soft margin (handling non-separable data)
- The kernel trick
- RBF and polynomial kernels
- SVMs vs other methods

---

### Step 11: Unsupervised Learning
**Why it matters:** Not all data has labels. Unsupervised learning finds structure.

**What we'll cover:**
- K-means clustering
- Hierarchical clustering
- Principal Component Analysis (PCA)
- Dimensionality reduction
- t-SNE and UMAP (visualization)
- Anomaly detection basics

---

### Step 12: Model Evaluation & Selection
**Why it matters:** Choosing and comparing models correctly is a skill.

**What we'll cover:**
- Train/validation/test methodology
- K-fold cross-validation
- Stratified splits
- Hyperparameter tuning (grid search, random search)
- ROC curves and AUC
- Confusion matrices
- When to use which metric

---

## 📚 Part 4: Practical Skills

Real-world ML workflows.

### Step 13: Feature Engineering
**Why it matters:** Features often matter more than algorithms.

**What we'll cover:**
- Handling missing values
- Encoding categorical variables
- Feature scaling (normalization, standardization)
- Feature selection
- Creating interaction features
- Domain-specific features

---

### Step 14: ML Project Workflow
**Why it matters:** End-to-end skills for real projects.

**What we'll cover:**
- Problem framing
- Data exploration (EDA)
- Baseline models
- Iterative improvement
- Model interpretation
- Documentation and reproducibility

---

## 🗂️ Project Structure

```
ml-foundations/
├── LEARNING_PATH.md
├── requirements.txt
├── verify_setup.py
│
├── 01_linear_algebra/
├── 02_calculus/
├── 03_probability_statistics/
├── 04_ml_problem_setup/
├── 05_linear_regression/
├── 06_logistic_regression/
├── 07_gradient_descent/
├── 08_regularization/
├── 09_trees_ensembles/
├── 10_svm/
├── 11_unsupervised/
├── 12_evaluation/
├── 13_feature_engineering/
├── 14_project_workflow/
│
├── demo/
└── evolutions/
```

---

## 🚀 Let's Begin!

Start with **Step 1: Linear Algebra Essentials** — the foundation for everything.

---

## 📖 References

### Books
- "Mathematics for Machine Learning" — Deisenroth, Faisal, Ong
- "Pattern Recognition and Machine Learning" — Bishop
- "Hands-On Machine Learning" — Géron

### Courses
- Andrew Ng's Machine Learning (Coursera)
- 3Blue1Brown Linear Algebra series
- StatQuest (YouTube)
