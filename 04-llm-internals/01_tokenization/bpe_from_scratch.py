"""
Byte-Pair Encoding (BPE) Tokenizer from Scratch

This implements the core BPE algorithm to show how modern tokenizers work.
"""

from collections import Counter
from typing import List, Dict, Tuple


def get_pair_counts(tokens: List[List[str]]) -> Counter:
    """Count frequency of adjacent pairs across all tokenized words."""
    pairs = Counter()
    for word_tokens in tokens:
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i + 1])
            pairs[pair] += 1
    return pairs


def merge_pair(tokens: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
    """Merge all occurrences of a pair into a single token."""
    new_token = pair[0] + pair[1]
    result = []
    
    for word_tokens in tokens:
        new_word_tokens = []
        i = 0
        while i < len(word_tokens):
            # Check if current position matches the pair
            if (i < len(word_tokens) - 1 and 
                word_tokens[i] == pair[0] and 
                word_tokens[i + 1] == pair[1]):
                new_word_tokens.append(new_token)
                i += 2  # Skip both tokens
            else:
                new_word_tokens.append(word_tokens[i])
                i += 1
        result.append(new_word_tokens)
    
    return result


class SimpleBPE:
    """A simple BPE tokenizer for educational purposes."""
    
    def __init__(self):
        self.vocab = set()
        self.merges = []  # List of (pair, merged_token) in order
        
    def train(self, text: str, num_merges: int = 100, verbose: bool = True):
        """
        Train BPE on a text corpus.
        
        Args:
            text: Training text
            num_merges: Number of merge operations (controls vocab size)
            verbose: Print progress
        """
        # Step 1: Split into words and characters
        # Add special end-of-word token to preserve word boundaries
        words = text.split()
        word_counts = Counter(words)
        
        # Initialize: each word as a list of characters + end token
        # The </w> token helps us reconstruct word boundaries later
        tokens = []
        token_counts = []  # How many times each word appears
        
        for word, count in word_counts.items():
            word_tokens = list(word) + ['</w>']
            tokens.append(word_tokens)
            token_counts.append(count)
            self.vocab.update(word_tokens)
        
        if verbose:
            print("=" * 60)
            print("BPE Training")
            print("=" * 60)
            print(f"Initial vocabulary size: {len(self.vocab)}")
            print(f"Initial vocab: {sorted(self.vocab)}")
            print(f"\nUnique words: {len(tokens)}")
            print()
        
        # Step 2: Iteratively merge most frequent pairs
        for i in range(num_merges):
            # Count pairs (weighted by word frequency)
            pairs = Counter()
            for word_tokens, count in zip(tokens, token_counts):
                for j in range(len(word_tokens) - 1):
                    pair = (word_tokens[j], word_tokens[j + 1])
                    pairs[pair] += count
            
            if not pairs:
                if verbose:
                    print(f"No more pairs to merge after {i} merges")
                break
            
            # Find most frequent pair
            best_pair = pairs.most_common(1)[0]
            pair, freq = best_pair
            
            if freq < 2:
                if verbose:
                    print(f"Stopping: no pair appears more than once")
                break
            
            # Merge the pair
            new_token = pair[0] + pair[1]
            tokens = merge_pair(tokens, pair)
            self.vocab.add(new_token)
            self.merges.append((pair, new_token))
            
            if verbose:
                print(f"Merge {i+1:3d}: {str(pair):20s} → '{new_token}' (freq: {freq})")
        
        if verbose:
            print()
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"Merge rules learned: {len(self.merges)}")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using learned merges."""
        words = text.split()
        result = []
        
        for word in words:
            # Start with characters
            tokens = list(word) + ['</w>']
            
            # Apply merges in order
            for pair, merged in self.merges:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1 and 
                        tokens[i] == pair[0] and 
                        tokens[i + 1] == pair[1]):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            
            result.extend(tokens)
        
        return result
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        tokens = self.tokenize(text)
        # Create vocab to ID mapping
        vocab_list = sorted(self.vocab)
        token_to_id = {t: i for i, t in enumerate(vocab_list)}
        return [token_to_id.get(t, 0) for t in tokens]  # 0 for unknown
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        vocab_list = sorted(self.vocab)
        id_to_token = {i: t for i, t in enumerate(vocab_list)}
        tokens = [id_to_token.get(i, '') for i in ids]
        text = ''.join(tokens)
        return text.replace('</w>', ' ').strip()


def main():
    """Demonstrate BPE training and tokenization."""
    
    # Training corpus - simple example
    corpus = """
    low lower lowest
    new newer newest
    show showed showing shown
    know knowing known
    the the the the
    a a a
    """
    
    print("Training Corpus:")
    print(corpus)
    print()
    
    # Train BPE
    tokenizer = SimpleBPE()
    tokenizer.train(corpus, num_merges=20, verbose=True)
    
    print("\n" + "=" * 60)
    print("Tokenization Examples")
    print("=" * 60)
    
    test_strings = [
        "low",
        "lower",
        "lowest",
        "showing",
        "unknown",  # Word not in training
        "newer showing",
    ]
    
    for s in test_strings:
        tokens = tokenizer.tokenize(s)
        ids = tokenizer.encode(s)
        print(f"'{s}'")
        print(f"  Tokens: {tokens}")
        print(f"  IDs:    {ids}")
        print()
    
    # Demonstrate the key insight
    print("=" * 60)
    print("Key Insight: Compositional Representations")
    print("=" * 60)
    print()
    print("Notice how related words share tokens:")
    print()
    
    related_words = [("low", "lower", "lowest"), ("new", "newer", "newest")]
    for group in related_words:
        print(f"  {group}:")
        for word in group:
            tokens = tokenizer.tokenize(word)
            print(f"    {word:10s} → {tokens}")
        print()
    
    print("The model can learn that '-er' means 'more' and '-est' means 'most'")
    print("because these suffixes share the same token IDs!")


if __name__ == "__main__":
    main()
