# Define the IOTA function
IOTA <- function(X, Y) {
  # Ensure X and Y have the same length
  if (length(X) != length(Y)) stop("X and Y must be of the same length")
  
  # Order X according to the ranks of Y
  ranks_Y <- rank(Y, ties.method = "first")
  X_ordered <- X[order(ranks_Y)]
  
  # Calculate the number of monotonic pairs
  monotonic_pairs <- 0
  total_pairs <- 0
  n <- length(X_ordered)
  
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      if (X_ordered[i] <= X_ordered[j]) {
        monotonic_pairs <- monotonic_pairs + 1
      }
      total_pairs <- total_pairs + 1
    }
  }
  
  # IOTA measure
  IOTA_value <- monotonic_pairs / total_pairs
  return(IOTA_value)
}

# Function to perform permutation test
IOTA_permutation_test <- function(X, Y, num_permutations = 1000, alpha = 0.05) {
  # Calculate the observed IOTA
  observed_IOTA <- IOTA(X, Y)
  
  # Initialize a vector to store IOTA values from permutations
  permuted_IOTA <- numeric(num_permutations)
  
  # Perform permutations
  for (i in 1:num_permutations) {
    permuted_Y <- sample(Y)  # Permute Y
    permuted_IOTA[i] <- IOTA(X, permuted_Y)
  }
  
  # Calculate p-value
  p_value <- mean(permuted_IOTA >= observed_IOTA)
  
  # Determine significance
  significant <- p_value < alpha
  
  # Return results
  list(
    observed_IOTA = observed_IOTA,
    p_value = p_value,
    significant = significant,
    permuted_IOTA = permuted_IOTA
  )
}
# Given datasets
X <- c(17.33333, 9.66667, 27.91667, 25.66667, 21.00000, 26.00000, 27.83333, 28.41667, 29.41667, 29.91667)
Y <- c(40.6, 38.6, 44.2, 46.8, 45.9, 43.0, 43.8, 45.6, 47.7, 48.6)

# Calculate IOTA(X, Y) and IOTA(Y, X)
result1 <- IOTA_permutation_test(X, Y)
# Print the results
cat("Observed IOTA XY:", result1$observed_IOTA, "\n")
cat("P-value:", result1$p_value, "\n")
cat("Significant:", result1$significant, "\n")

result2 <- IOTA_permutation_test(Y, X)

# Print the results
cat("Observed IOTA YX:", result2$observed_IOTA, "\n")
cat("P-value:", result2$p_value, "\n")
cat("Significant:", result2$significant, "\n")

iota_XY <- IOTA(X, Y)
iota_YX <- IOTA(Y, X)
# Print the results
cat("IOTA(X, Y):", iota_XY, "\n")
cat("IOTA(Y, X):", iota_YX, "\n")
cat("Are they equal?", iota_XY == iota_YX, "\n")