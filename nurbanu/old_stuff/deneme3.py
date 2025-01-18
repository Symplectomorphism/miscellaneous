import numpy as np
from scipy.stats import rankdata

def iota(X, Y):
    if len(X) != len(Y):
        raise ValueError("X and Y must be of the same length")
    
    ranks_Y = rankdata(Y, method='ordinal')
    X_ordered = np.array(X)[np.argsort(ranks_Y)]
    
    monotonic_pairs = 0
    total_pairs = 0
    n = len(X_ordered)
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            if X_ordered[i] <= X_ordered[j]:
                monotonic_pairs += 1
            total_pairs += 1
    
    iota_value = monotonic_pairs / total_pairs
    return iota_value

def iota_permutation_test(X, Y, num_permutations=1000, alpha=0.05):
    observed_iota = iota(X, Y)
    permuted_iota = np.zeros(num_permutations)
    
    for i in range(num_permutations):
        permuted_Y = np.random.permutation(Y)
        permuted_iota[i] = iota(X, permuted_Y)
    
    p_value = np.mean(permuted_iota >= observed_iota)
    significant = p_value < alpha
    
    return {
        'observed_iota': observed_iota,
        'p_value': p_value,
        'significant': significant,
        'permuted_iota': permuted_iota
    }

# Given datasets
X = [17.33333, 9.66667, 27.91667, 25.66667, 21.0, 26.0, 27.83333, 28.41667, 29.41667, 29.91667]
Y = [40.6, 38.6, 44.2, 46.8, 45.9, 43.0, 43.8, 45.6, 47.7, 48.6]

# Calculate IOTA(X, Y) and IOTA(Y, X)
result1 = iota_permutation_test(X, Y)
print("Observed IOTA XY:", result1['observed_iota'])
print("P-value:", result1['p_value'])
print("Significant:", result1['significant'])

result2 = iota_permutation_test(Y, X)
print("Observed IOTA YX:", result2['observed_iota'])
print("P-value:", result2['p_value'])
print("Significant:", result2['significant'])

iota_XY = iota(X, Y)
iota_YX = iota(Y, X)
print("IOTA(X, Y):", iota_XY)
print("IOTA(Y, X):", iota_YX)
print("Are they equal?", iota_XY == iota_YX)