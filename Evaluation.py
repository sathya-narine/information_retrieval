#This scipt contains all the methods for calculating the metrics of the ir system
import numpy as np

def mean_reciprocal_rank(bool_results, k=10):
    """Score is reciprocal of the rank of the first relevant item.
    First element is 'rank 1'. Relevance is binary (nonzero is relevant).

    Args:
        bool_results: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider

    Returns:
        Mean reciprocal rank
    """
    bool_results = (np.atleast_1d(r[:k]).nonzero()[0] for r in bool_results)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in bool_results])

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Args:
        r (list): Relevance scores in rank order (first element is the first item)
        k (int): Number of results to consider
    Returns:
        Precision @ k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        return 0.
    return np.mean(r)

def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Args:
        r (list): Relevance scores in rank order (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Args:
        rs (list): Iterator of relevance scores (list or numpy) in rank order (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])

def recall_at_k(r, k, all_rel):
    """Score is recall @ k
    Relevance is binary (nonzero is relevant).
    Args:
        r (list): Relevance scores in rank order (first element is the first item)
        k (int): Number of results to consider
        all_rel (int): Total number of relevant documents
    Returns:
        Recall @ k
    """
    assert k >= 1
    if all_rel == 0:  # Check if there are no relevant documents
        return 0.0  # Return 0 recall if no relevant documents
    r = np.asarray(r)[:k] != 0
    return np.sum(r) / all_rel

def f1_score_at_k(r, k, all_rel):
    """Score is F1 @ k
    Relevance is binary (nonzero is relevant).
    Args:
        r (list): Relevance scores in rank order (first element is the first item)
        k (int): Number of results to consider
        all_rel (int): Total number of relevant documents
    Returns:
        F1 @ k
    """
    if all_rel == 0:  # Check if there are no relevant documents
        return 0.0  # Return 0 F1 score if no relevant documents
    p = precision_at_k(r, k)
    r = recall_at_k(r, k, all_rel)
    if p + r == 0:
        return 0.
    return 2 * (p * r) / (p + r)