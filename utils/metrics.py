import math
from collections import Counter
from typing import List


def compute_bleu(reference: List[str], hypothesis: List[str], max_n=4, weights=None):
    """
    Compute BLEU score for a single reference-hypothesis pair.
    
    Args:
        reference: List of reference tokens
        hypothesis: List of hypothesis tokens
        max_n: Maximum n-gram order (default: 4)
        weights: Weights for each n-gram order (default: uniform)
        
    Returns:
        BLEU score (float)
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    # Compute precision for each n-gram order
    precisions = []
    
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(reference, n)
        hyp_ngrams = get_ngrams(hypothesis, n)
        
        if len(hyp_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Count matches (clipped)
        matches = 0
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        precision = matches / len(hyp_ngrams)
        precisions.append(precision)
    
    # Compute geometric mean of precisions
    if min(precisions) > 0:
        log_precision_sum = sum(w * math.log(p) for w, p in zip(weights, precisions))
        geo_mean = math.exp(log_precision_sum)
    else:
        geo_mean = 0.0
    
    # Compute brevity penalty
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    
    bleu = bp * geo_mean
    return bleu


def get_ngrams(tokens: List[str], n: int):
    """
    Extract n-grams from a list of tokens.
    
    Args:
        tokens: List of tokens
        n: n-gram order
        
    Returns:
        Counter of n-grams
    """
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] += 1
    return ngrams


def compute_corpus_bleu(references: List[List[str]], hypotheses: List[List[str]], max_n=4):
    """
    Compute corpus-level BLEU score.
    
    Args:
        references: List of reference token lists
        hypotheses: List of hypothesis token lists
        max_n: Maximum n-gram order
        
    Returns:
        BLEU score (float)
    """
    assert len(references) == len(hypotheses), "Number of references and hypotheses must match"
    
    # Accumulate statistics
    total_matches = [0] * max_n
    total_possible = [0] * max_n
    total_ref_len = 0
    total_hyp_len = 0
    
    for ref, hyp in zip(references, hypotheses):
        total_ref_len += len(ref)
        total_hyp_len += len(hyp)
        
        for n in range(1, max_n + 1):
            ref_ngrams = get_ngrams(ref, n)
            hyp_ngrams = get_ngrams(hyp, n)
            
            # Count matches
            matches = 0
            for ngram, count in hyp_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            
            total_matches[n-1] += matches
            total_possible[n-1] += max(len(hyp) - n + 1, 0)
    
    # Compute precisions
    precisions = []
    for matches, possible in zip(total_matches, total_possible):
        if possible > 0:
            precisions.append(matches / possible)
        else:
            precisions.append(0.0)
    
    # Compute geometric mean
    if min(precisions) > 0:
        weights = [1.0 / max_n] * max_n
        log_precision_sum = sum(w * math.log(p) for w, p in zip(weights, precisions))
        geo_mean = math.exp(log_precision_sum)
    else:
        geo_mean = 0.0
    
    # Brevity penalty
    if total_hyp_len > total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - total_ref_len / total_hyp_len) if total_hyp_len > 0 else 0.0
    
    bleu = bp * geo_mean
    return bleu


def format_bleu_score(bleu: float) -> str:
    """Format BLEU score as percentage."""
    return f"{bleu * 100:.2f}"

