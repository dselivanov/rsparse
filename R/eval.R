# @param predicted ordered list of predictions
# @param actual relevant values
# @param k precision level
ap_at_k = function(predicted, actual, k = 10) {
  k = min(k, length(predicted), length(actual))
  pk_seq = predicted[seq_len(k)] %in% actual
  xx = cumsum(pk_seq) / seq_along(pk_seq)
  mean(xx)
}


# reference naive implementation
# average_precision_at_k_naive = function(predicted, actual, k = 10) {
#   k = min(k, length(predicted), length(actual))
#   precision_at_k = numeric(k)
#   for(j in 1:k) {
#     precision_at_k[[j]] = length(intersect(predicted[seq_len(j)], actual)) / j
#   }
#   mean(precision_at_k)
# }

dcg_at_k = function(predicted_indices, actual_indices, actual_relevances, k = length(predicted_indices)) {
  k = min(k, length(predicted_indices), length(actual_indices))
  x_match = match(predicted_indices, actual_indices)
  dcg = 0
  for(i in 1:k) {
    j = x_match[[i]]
    if(!is.na(j))
      dcg = dcg + actual_relevances[[j]] / log2(i + 1)
  }
  dcg
}

idcg_at_k = function(actual_relevances, k = length(actual_relevances)) {

  k = min(k, length(actual_relevances))

  if(length(actual_relevances) == 0) return(1)

  res = sort(actual_relevances, decreasing = T)[1:k]
  sum(res / log2(seq_along(res) + 1))
}

ndcg_at_k = function(predicted_indices, actual_indices, actual_relevances, k = length(predicted_indices)) {
  k = min(k, length(predicted_indices), length(actual_indices))
  dcg_at_k(predicted_indices, actual_indices, actual_relevances, k) / idcg_at_k(actual_relevances, k)
}

top_n = function(x, n) {
  order(x, decreasing = TRUE)[seq_len(n)]
  # nx = length(x)
  # p = nx - n
  # xp = sort(x, partial = p)[p]
  # which(x > xp, useNames = FALSE)
}
