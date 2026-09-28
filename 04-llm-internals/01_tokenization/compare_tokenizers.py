"""
Compare Different Tokenizers

Shows how GPT-2, Llama, and other tokenizers handle the same text differently.
"""

from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


def analyze_tokenization(tokenizer, text: str, name: str):
    """Analyze how a tokenizer breaks down text."""
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text, add_special_tokens=False)
    
    print(f"\n{name}:")
    print(f"  Text:    '{text}'")
    print(f"  Tokens:  {tokens}")
    print(f"  IDs:     {ids}")
    print(f"  Count:   {len(tokens)} tokens")
    return len(tokens)


def main():
    print("=" * 70)
    print("Tokenizer Comparison")
    print("=" * 70)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    # Llama tokenizer - using a small open model's tokenizer
    try:
        llama = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        has_llama = True
    except:
        print("  (Llama tokenizer requires authentication, using fallback)")
        llama = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        has_llama = True
    
    tokenizers = [
        (gpt2, "GPT-2 (50257 vocab)"),
        (llama, "Llama (32000 vocab)"),
    ]
    
    # Test cases
    test_cases = [
        # Basic English
        ("Hello, world!", "Basic greeting"),
        ("The quick brown fox jumps over the lazy dog.", "Pangram"),
        
        # Technical text
        ("def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)", "Python code"),
        ("https://www.example.com/path?query=value", "URL"),
        
        # Numbers
        ("The year is 2024 and pi is 3.14159265359", "Numbers"),
        ("12345678901234567890", "Long number"),
        
        # Rare words
        ("Pneumonoultramicroscopicsilicovolcanoconiosis", "Longest English word"),
        ("tokenization", "Our topic!"),
        
        # Non-English (if you want to see the problem)
        ("こんにちは世界", "Japanese: Hello World"),
        ("Привет мир", "Russian: Hello World"),
        ("مرحبا بالعالم", "Arabic: Hello World"),
    ]
    
    print("\n" + "=" * 70)
    print("Tokenization Results")
    print("=" * 70)
    
    results = []
    
    for text, description in test_cases:
        print(f"\n{'─' * 70}")
        print(f"Test: {description}")
        
        counts = []
        for tokenizer, name in tokenizers:
            try:
                count = analyze_tokenization(tokenizer, text, name)
                counts.append(count)
            except Exception as e:
                print(f"\n{name}: Error - {e}")
                counts.append(-1)
        
        results.append((description, text, counts))
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("Summary: Token Counts per Tokenizer")
    print("=" * 70)
    print(f"\n{'Test Case':<40} {'GPT-2':>10} {'Llama':>10}")
    print("-" * 62)
    
    for description, text, counts in results:
        display = description[:38] + ".." if len(description) > 40 else description
        print(f"{display:<40} {counts[0]:>10} {counts[1]:>10}")
    
    # Key insights
    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    print("""
1. VOCABULARY SIZE MATTERS
   - GPT-2: 50,257 tokens → more tokens per word, but handles rare words
   - Llama: 32,000 tokens → fewer tokens (more efficient for common text)

2. CODE AND URLS
   - Both struggle with code (many small tokens)
   - Larger vocabulary helps with common patterns

3. NON-ENGLISH TEXT (THE BIG PROBLEM!)
   - English: ~1 token per 4 characters
   - Japanese/Chinese: ~1 token per 1-2 characters (2-4x more tokens!)
   - Arabic/Hindi: Often even worse
   
   This means:
   - Non-English users pay MORE for API calls (more tokens)
   - Models have LESS context for non-English text
   - Training is LESS efficient on non-English data

4. NUMBERS
   - Often tokenized digit-by-digit
   - "2024" might be ['20', '24'] or ['2', '0', '2', '4']
   - This is why models struggle with arithmetic!
""")


if __name__ == "__main__":
    main()
