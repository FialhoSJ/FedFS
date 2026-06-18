import numpy as np
from sklearn.feature_selection import mutual_info_classif

# ALGORITMO 1 - CE BASED FEATURE SELECTION

def gen_random_samples(S, p):
    return np.random.binomial(n=1, p=p, size=(S, len(p)))

def get_subset(X, z):
    selected = np.where(z == 1)[0]
    if len(selected) == 0:
        return None
    return X[:, selected]


def conditional_entropy_proxy(X_sub, y):
    if X_sub is None:
        return np.inf
    mi = mutual_info_classif(X_sub, y, random_state=42)
    total_mi = np.sum(mi)
    return -total_mi


def compute_gamma(scores, beta):
    percentile = (1 - beta) * 100
    return np.percentile(scores, percentile)


def compute_new_prob(Z, scores, gamma):
    elite_mask = scores <= gamma
    elite_samples = Z[elite_mask]
    return elite_samples.mean(axis=0)


def CE_feature_selection(X,  y,  p_init, T=20, S=100, alpha=0.7, beta=0.9):
    p = p_init.copy()
    for t in range(T):

        Z = gen_random_samples(S, p)
        scores = []
        for z in Z:
            U = get_subset(X, z)
            score = conditional_entropy_proxy(U, y)
            scores.append(score)
        scores = np.array(scores)
        gamma = compute_gamma(scores, beta)
        p_new = compute_new_prob(Z, scores, gamma)
        p = ((1 - alpha) * p + alpha * p_new)
    return p