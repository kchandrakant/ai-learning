"""
Probability & Statistics for Machine Learning
==============================================

This module demonstrates probability and statistics concepts 
essential for understanding ML:
- Probability basics and Bayes' theorem
- Common distributions
- Maximum likelihood estimation
- Sampling and the Central Limit Theorem

Run this file to see visualizations and examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Create figures directory
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# PART 1: PROBABILITY BASICS
# =============================================================================

def demonstrate_probability_basics():
    """Basic probability concepts."""
    print("\n" + "=" * 60)
    print("PART 1: PROBABILITY BASICS")
    print("=" * 60)
    
    print("""
Probability P(A) measures how likely event A is to occur.
    
Rules:
    - 0 <= P(A) <= 1
    - P(certain event) = 1
    - P(impossible event) = 0
    - P(not A) = 1 - P(A)
""")
    
    # Simulate coin flips
    n_flips = 10000
    flips = np.random.choice([0, 1], size=n_flips)  # 0 = tails, 1 = heads
    
    # Running probability estimate
    running_prob = np.cumsum(flips) / np.arange(1, n_flips + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Law of Large Numbers
    ax = axes[0]
    ax.plot(running_prob, 'b-', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', label='True probability (0.5)')
    ax.set_xlabel('Number of flips')
    ax.set_ylabel('Estimated P(Heads)')
    ax.set_title('Law of Large Numbers\n(Estimate converges to true probability)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 2: Joint and conditional probability
    ax = axes[1]
    
    # Venn diagram visualization
    circle1 = plt.Circle((0.35, 0.5), 0.3, fill=True, alpha=0.5, color='blue', label='P(A)')
    circle2 = plt.Circle((0.65, 0.5), 0.3, fill=True, alpha=0.5, color='red', label='P(B)')
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.text(0.25, 0.5, 'A only', ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.5, 'A&B', ha='center', va='center', fontsize=12)
    ax.text(0.75, 0.5, 'B only', ha='center', va='center', fontsize=12)
    ax.set_title('Joint Probability: P(A and B)\nConditional: P(A|B) = P(A and B) / P(B)')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'probability_basics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'probability_basics.png'}")
    
    # Concrete example
    print("\n--- Example: Disease Testing ---")
    print("""
Suppose:
    - P(Disease) = 0.01          (1% of population has disease)
    - P(Positive | Disease) = 0.95    (test catches 95% of cases)
    - P(Positive | No Disease) = 0.05 (5% false positive rate)

Question: If you test positive, what's P(Disease | Positive)?
""")


# =============================================================================
# PART 2: BAYES' THEOREM
# =============================================================================

def demonstrate_bayes():
    """Bayes' theorem - updating beliefs with evidence."""
    print("\n" + "=" * 60)
    print("PART 2: BAYES' THEOREM")
    print("=" * 60)
    
    print("""
Bayes' Theorem:

    P(A|B) = P(B|A) * P(A) / P(B)
    
    posterior = likelihood * prior / evidence

In words: 
    "Probability of A given B" depends on:
    - How likely B is when A is true (likelihood)
    - How likely A was before seeing B (prior)
    - How likely B is overall (evidence)
""")
    
    # Disease testing example
    p_disease = 0.01        # Prior: 1% have disease
    p_positive_given_disease = 0.95   # Sensitivity
    p_positive_given_healthy = 0.05   # False positive rate
    
    # P(Positive) via total probability
    p_healthy = 1 - p_disease
    p_positive = p_positive_given_disease * p_disease + p_positive_given_healthy * p_healthy
    
    # Bayes' theorem: P(Disease | Positive)
    p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
    
    print("\n--- Disease Testing Example ---")
    print(f"Prior P(Disease) = {p_disease}")
    print(f"P(Positive | Disease) = {p_positive_given_disease}")
    print(f"P(Positive | Healthy) = {p_positive_given_healthy}")
    print(f"\nP(Positive) = {p_positive:.4f}")
    print(f"\nBayes' Theorem:")
    print(f"P(Disease | Positive) = {p_positive_given_disease} * {p_disease} / {p_positive:.4f}")
    print(f"                      = {p_disease_given_positive:.4f}")
    print(f"\n-> Even with a positive test, there's only {p_disease_given_positive*100:.1f}% chance of disease!")
    print("   (Because disease is rare and false positives add up)")
    
    # Visualize how posterior changes with prior
    priors = np.linspace(0.001, 0.5, 100)
    posteriors = []
    
    for prior in priors:
        p_pos = p_positive_given_disease * prior + p_positive_given_healthy * (1 - prior)
        posterior = (p_positive_given_disease * prior) / p_pos
        posteriors.append(posterior)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Posterior vs Prior
    ax = axes[0]
    ax.plot(priors, posteriors, 'b-', linewidth=2)
    ax.axvline(x=0.01, color='red', linestyle='--', alpha=0.7, label='Our example (prior=0.01)')
    ax.axhline(y=p_disease_given_positive, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Prior P(Disease)')
    ax.set_ylabel('Posterior P(Disease | Positive)')
    ax.set_title('How Prior Affects Posterior\n(When you test positive)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Bayesian updating with multiple tests
    ax = axes[1]
    
    # Sequential updating: what if you take multiple tests?
    prior = 0.01
    n_tests = 10
    posteriors_seq = [prior]
    
    for _ in range(n_tests):
        # Assume positive result
        p_pos = p_positive_given_disease * prior + p_positive_given_healthy * (1 - prior)
        prior = (p_positive_given_disease * prior) / p_pos
        posteriors_seq.append(prior)
    
    ax.plot(range(n_tests + 1), posteriors_seq, 'go-', markersize=8, linewidth=2)
    ax.set_xlabel('Number of positive tests')
    ax.set_ylabel('P(Disease)')
    ax.set_title('Bayesian Updating\n(Each positive test increases belief)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bayes_theorem.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {FIGURES_DIR / 'bayes_theorem.png'}")
    
    print("""
ML Applications of Bayes:
    - Naive Bayes classifier
    - Bayesian optimization (hyperparameter tuning)
    - Bayesian neural networks
    - Prior regularization
""")


# =============================================================================
# PART 3: COMMON DISTRIBUTIONS
# =============================================================================

def demonstrate_distributions():
    """Visualize common probability distributions."""
    print("\n" + "=" * 60)
    print("PART 3: COMMON DISTRIBUTIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Bernoulli
    ax = axes[0, 0]
    p = 0.7
    x = [0, 1]
    probs = [1-p, p]
    ax.bar(x, probs, color=['red', 'green'], alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Failure (0)', 'Success (1)'])
    ax.set_ylabel('Probability')
    ax.set_title(f'Bernoulli (p={p})\nSingle coin flip')
    ax.set_ylim(0, 1)
    
    # 2. Binomial
    ax = axes[0, 1]
    n, p = 20, 0.5
    x = np.arange(0, n+1)
    probs = stats.binom.pmf(x, n, p)
    ax.bar(x, probs, color='blue', alpha=0.7)
    ax.set_xlabel('Number of successes')
    ax.set_ylabel('Probability')
    ax.set_title(f'Binomial (n={n}, p={p})\n# heads in {n} coin flips')
    
    # 3. Gaussian (Normal)
    ax = axes[0, 2]
    x = np.linspace(-4, 4, 100)
    for mu, sigma, color in [(0, 1, 'blue'), (0, 2, 'red'), (1, 0.5, 'green')]:
        y = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, y, color=color, linewidth=2, label=f'mu={mu}, sig={sigma}')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability density')
    ax.set_title('Gaussian (Normal) Distribution\nThe most important distribution!')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Uniform
    ax = axes[1, 0]
    a, b = -2, 3
    x = np.linspace(-4, 5, 100)
    y = stats.uniform.pdf(x, a, b-a)
    ax.plot(x, y, 'b-', linewidth=2)
    ax.fill_between(x, y, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('Probability density')
    ax.set_title(f'Uniform [{a}, {b}]\nEqual probability everywhere')
    ax.grid(True, alpha=0.3)
    
    # 5. Exponential
    ax = axes[1, 1]
    x = np.linspace(0, 5, 100)
    for lam in [0.5, 1, 2]:
        y = stats.expon.pdf(x, scale=1/lam)
        ax.plot(x, y, linewidth=2, label=f'lambda={lam}')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability density')
    ax.set_title('Exponential Distribution\nTime until event')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Categorical / Softmax output
    ax = axes[1, 2]
    categories = ['Cat', 'Dog', 'Bird', 'Fish']
    probs = [0.45, 0.35, 0.15, 0.05]  # Softmax output
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    ax.bar(categories, probs, color=colors, alpha=0.7)
    ax.set_ylabel('Probability')
    ax.set_title('Categorical Distribution\n(Softmax output in classification)')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'distributions.png'}")
    
    print("""
Key Distributions in ML:

1. Bernoulli: Binary classification output
2. Binomial: Count of successes in n trials
3. Gaussian: Continuous features, noise, weight initialization
4. Categorical: Multi-class classification (softmax)
5. Exponential: Time-to-event modeling
6. Uniform: Random initialization, dropout

The Gaussian is EVERYWHERE in ML:
    - Weight initialization: N(0, 1/sqrt(n))
    - Batch normalization
    - VAE latent space
    - Noise in diffusion models
""")


# =============================================================================
# PART 4: EXPECTED VALUE AND VARIANCE
# =============================================================================

def demonstrate_statistics():
    """Expected value, variance, and their importance."""
    print("\n" + "=" * 60)
    print("PART 4: EXPECTED VALUE AND VARIANCE")
    print("=" * 60)
    
    print("""
Expected Value E[X]: The "average" outcome
    E[X] = sum( x * P(x) )    (discrete)
    E[X] = integral( x * p(x) dx ) (continuous)

Variance Var[X]: How spread out the distribution is
    Var[X] = E[(X - E[X])^2] = E[X^2] - E[X]^2

Standard Deviation: sigma = sqrt(Var[X])
""")
    
    # Generate samples from different distributions
    n_samples = 10000
    
    # Same mean, different variance
    dist1 = np.random.normal(5, 1, n_samples)   # Low variance
    dist2 = np.random.normal(5, 3, n_samples)   # High variance
    
    print(f"\nDistribution 1: mu=5, sigma=1")
    print(f"  Sample mean: {np.mean(dist1):.3f}")
    print(f"  Sample std:  {np.std(dist1):.3f}")
    
    print(f"\nDistribution 2: mu=5, sigma=3")
    print(f"  Sample mean: {np.mean(dist2):.3f}")
    print(f"  Sample std:  {np.std(dist2):.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Same mean, different variance
    ax = axes[0]
    bins = np.linspace(-5, 15, 50)
    ax.hist(dist1, bins=bins, alpha=0.7, density=True, label='sigma=1 (precise)')
    ax.hist(dist2, bins=bins, alpha=0.7, density=True, label='sigma=3 (uncertain)')
    ax.axvline(x=5, color='black', linestyle='--', label='Mean = 5')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Same Mean, Different Variance\n(Variance = uncertainty)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Bias-Variance tradeoff visualization
    ax = axes[1]
    
    # Simulate predictions from models with different bias-variance
    true_value = 5
    
    # Low bias, low variance (ideal)
    preds1 = np.random.normal(true_value, 0.5, 100)
    # Low bias, high variance
    preds2 = np.random.normal(true_value, 2, 100)
    # High bias, low variance
    preds3 = np.random.normal(true_value + 2, 0.5, 100)
    # High bias, high variance
    preds4 = np.random.normal(true_value + 2, 2, 100)
    
    ax.scatter([1]*100, preds1, alpha=0.5, s=10, label='Low bias, Low var')
    ax.scatter([2]*100, preds2, alpha=0.5, s=10, label='Low bias, High var')
    ax.scatter([3]*100, preds3, alpha=0.5, s=10, label='High bias, Low var')
    ax.scatter([4]*100, preds4, alpha=0.5, s=10, label='High bias, High var')
    ax.axhline(y=true_value, color='red', linestyle='--', linewidth=2, label='True value')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Low B\nLow V', 'Low B\nHigh V', 'High B\nLow V', 'High B\nHigh V'])
    ax.set_ylabel('Predictions')
    ax.set_title('Bias-Variance Tradeoff\nTotal Error = Bias^2 + Variance + Noise')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {FIGURES_DIR / 'statistics.png'}")


# =============================================================================
# PART 5: MAXIMUM LIKELIHOOD ESTIMATION
# =============================================================================

def demonstrate_mle():
    """Maximum Likelihood Estimation - finding best parameters."""
    print("\n" + "=" * 60)
    print("PART 5: MAXIMUM LIKELIHOOD ESTIMATION (MLE)")
    print("=" * 60)
    
    print("""
MLE: Find parameters that make observed data most likely.

    theta_MLE = argmax P(data | theta)
              = argmax product( P(x_i | theta) )  [for independent samples]
              = argmax sum( log P(x_i | theta) )  [log-likelihood, easier]
          
In practice, we MINIMIZE negative log-likelihood (same thing):
    theta_MLE = argmin -sum( log P(x_i | theta) )
""")
    
    # Example: Estimating Gaussian parameters
    true_mu, true_sigma = 3, 1.5
    data = np.random.normal(true_mu, true_sigma, 100)
    
    print(f"\nTrue parameters: mu={true_mu}, sigma={true_sigma}")
    print(f"Data: 100 samples from N({true_mu}, {true_sigma}^2)")
    
    # MLE estimates
    mle_mu = np.mean(data)
    mle_sigma = np.std(data, ddof=0)  # MLE uses n, not n-1
    
    print(f"\nMLE estimates:")
    print(f"  mu_MLE = sample mean = {mle_mu:.3f}")
    print(f"  sigma_MLE = sample std = {mle_sigma:.3f}")
    
    # Visualize likelihood as function of mu
    mus = np.linspace(0, 6, 100)
    
    def log_likelihood(mu, data, sigma):
        """Log-likelihood of data given mu and fixed sigma."""
        return -len(data)/2 * np.log(2*np.pi*sigma**2) - np.sum((data - mu)**2) / (2*sigma**2)
    
    log_likes = [log_likelihood(mu, data, true_sigma) for mu in mus]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Log-likelihood vs mu
    ax = axes[0]
    ax.plot(mus, log_likes, 'b-', linewidth=2)
    ax.axvline(x=mle_mu, color='red', linestyle='--', label=f'MLE mu = {mle_mu:.2f}')
    ax.axvline(x=true_mu, color='green', linestyle=':', label=f'True mu = {true_mu}')
    ax.set_xlabel('mu')
    ax.set_ylabel('Log-likelihood')
    ax.set_title('MLE: Maximize Log-Likelihood\n(Peak = best parameter)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Show the fitted distribution
    ax = axes[1]
    ax.hist(data, bins=20, density=True, alpha=0.7, label='Data histogram')
    
    x = np.linspace(-2, 8, 100)
    ax.plot(x, stats.norm.pdf(x, true_mu, true_sigma), 'g--', 
            linewidth=2, label=f'True: N({true_mu}, {true_sigma}^2)')
    ax.plot(x, stats.norm.pdf(x, mle_mu, mle_sigma), 'r-', 
            linewidth=2, label=f'MLE: N({mle_mu:.2f}, {mle_sigma:.2f}^2)')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('MLE Fits Distribution to Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mle.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {FIGURES_DIR / 'mle.png'}")
    
    print("""
Connection to ML:
    
    Cross-entropy loss = negative log-likelihood!
    
    For classification with softmax:
        Loss = -sum( y_i * log(y_hat_i) ) = -log P(y | x, theta)
    
    So minimizing cross-entropy = maximizing likelihood
    = finding parameters that make training data most probable.
""")


# =============================================================================
# PART 6: CENTRAL LIMIT THEOREM
# =============================================================================

def demonstrate_clt():
    """The Central Limit Theorem - why Gaussians are everywhere."""
    print("\n" + "=" * 60)
    print("PART 6: CENTRAL LIMIT THEOREM")
    print("=" * 60)
    
    print("""
Central Limit Theorem (CLT):

    The average of many independent random variables tends toward
    a Gaussian distribution, REGARDLESS of the original distribution.
    
    If X_1, X_2, ..., X_n are i.i.d. with mean mu and variance sigma^2:
        X_bar = (X_1 + X_2 + ... + X_n) / n
        
        As n -> infinity:  X_bar ~ N(mu, sigma^2/n)

This is why Gaussians appear everywhere!
""")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original distributions (non-Gaussian)
    def bimodal_sampler(n):
        n1 = n // 2
        n2 = n - n1
        return np.concatenate([np.random.normal(-2, 0.5, n1), 
                               np.random.normal(2, 0.5, n2)])
    
    distributions = [
        ('Uniform', lambda n: np.random.uniform(0, 1, n)),
        ('Exponential', lambda n: np.random.exponential(1, n)),
        ('Bimodal', bimodal_sampler)
    ]
    
    sample_sizes = [1, 2, 10, 30]
    n_experiments = 5000
    
    for col, (name, sampler) in enumerate(distributions):
        ax = axes[0, col]
        
        # Show original distribution
        samples = sampler(10000)
        ax.hist(samples, bins=50, density=True, alpha=0.7)
        ax.set_title(f'Original: {name}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Show distribution of means
        ax = axes[1, col]
        
        for n in sample_sizes:
            means = [np.mean(sampler(n)) for _ in range(n_experiments)]
            ax.hist(means, bins=50, density=True, alpha=0.5, label=f'n={n}')
        
        ax.set_title(f'Distribution of Sample Means\n(becomes Gaussian!)')
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'clt.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {FIGURES_DIR / 'clt.png'}")
    
    print("""
Why CLT Matters in ML:

1. Weight initialization: Random weights -> activations become Gaussian
2. Batch statistics: Mean/variance of a batch approximate population
3. Gradient noise: Sum of many small updates -> Gaussian-like
4. Feature engineering: Sum of many features -> Gaussian
5. Confidence intervals: Sample statistics are approximately Gaussian

The Gaussian assumption is often justified because of CLT!
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all probability and statistics demonstrations."""
    print("=" * 60)
    print("PROBABILITY & STATISTICS FOR MACHINE LEARNING")
    print("=" * 60)
    print("\nML deals with uncertainty. Probability quantifies it.")
    
    demonstrate_probability_basics()
    demonstrate_bayes()
    demonstrate_distributions()
    demonstrate_statistics()
    demonstrate_mle()
    demonstrate_clt()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Takeaways:

1. PROBABILITY quantifies uncertainty
   - P(A|B) = conditional probability
   - Joint, marginal, conditional relationships

2. BAYES' THEOREM updates beliefs with evidence
   P(A|B) = P(B|A) * P(A) / P(B)
   posterior ~ likelihood * prior

3. DISTRIBUTIONS model different types of data
   - Gaussian: continuous features, errors
   - Bernoulli/Categorical: classification outputs
   - Binomial: count data

4. EXPECTED VALUE and VARIANCE
   - E[X] = average outcome
   - Var[X] = spread/uncertainty
   - Bias-Variance tradeoff in ML

5. MLE finds best parameters
   theta_MLE = argmax P(data | theta)
   Cross-entropy loss = negative log-likelihood!

6. CLT explains why Gaussians are everywhere
   Average of many things -> Gaussian

Statistical Thinking in ML:
    - Training data is a SAMPLE from true distribution
    - Model learns to estimate distribution P(y|x)
    - Generalization depends on how representative sample is
    - Uncertainty quantification tells us what we don't know
""")
    
    print(f"\n[All figures saved to: {FIGURES_DIR}]")


if __name__ == "__main__":
    main()
