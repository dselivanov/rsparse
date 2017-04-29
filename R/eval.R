# @param predicted ordered list of predictions
# @param actual relevant values
# @param k precision level
average_precision_at_k = function(predicted, actual, k = 10) {
  k = min(k, length(predicted), length(actual))
  pk_seq = predicted[seq_len(k)] %in% actual
  xx = cumsum(pk_seq) / seq_along(pk_seq)
  mean(xx)
}


# reference naive implementation
average_precision_at_k_naive = function(predicted, actual, k = 10) {
  k = min(k, length(predicted), length(actual))
  precision_at_k = numeric(k)
  for(j in 1:k) {
    precision_at_k[[j]] = length(intersect(predicted[seq_len(j)], actual)) / j
  }
  mean(precision_at_k)
}




# k = 10
#
# predicted = c(3:50)
# actual = c(4:6, 7, 50)
#
# average_precision_at_k(predicted, actual, k)
# average_precision_at_k_naive(predicted, actual, k)
#
# predicted = c(1, 4, 6)
# actual = 4:6
#
# average_precision_at_k(predicted, actual, k)
# average_precision_at_k_naive(predicted, actual, k)
