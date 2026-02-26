# src/evaluation.py

import numpy as np

def recall_at_k(recommended, ground_truth):
    return int(ground_truth in recommended)

def ndcg_at_k(recommended, ground_truth):
    if ground_truth in recommended:
        idx = list(recommended).index(ground_truth)
        return 1 / np.log2(idx + 2)
    return 0
