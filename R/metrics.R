#' @name metrics
#' @title Ranking Metrics for Top-K Items
#' @param predictions matrix of predictions. Predctions can be defined 2 ways:
#' \enumerate{
#'   \item \code{predictions} = \code{integer} matrix with item indices (correspond to column numbers in \code{actual})
#'   \item \code{predictions} = \code{character} matrix with item identifiers (characters which correspond to \code{colnames(actual)})
#'   which has attribute "indices" (\code{integer} matrix with item indices which correspond to column numbers in \code{actual}).
#' }
#' @param actual sparse Matrix of relevant items. Each non-zero entry considered as relevant item.
#'   Value of the each non-zero entry considered as relevance for calculation of \code{ndcg@@k}.
#'   It should inherit from \code{Matrix::sparseMatrix}. Internally \code{Matrix::RsparseMatrix} is used.
#' @param ... other arguments (not used at the moment)
#' @rdname metrics
NULL

#' @description \code{ap_k} calculates \bold{Average Precision at K (\code{ap@@k})}.
#' Please refer to \href{Information retrieval wikipedia article}{https://en.wikipedia.org/wiki/Information_retrieval#Average_precision}
#' @rdname metrics
#' @export
ap_k = function(predictions, actual, ...) {
  stopifnot(is.matrix(predictions))
  stopifnot(inherits(actual, "sparseMatrix"))

  k = ncol(predictions)
  n_u = nrow(predictions)
  stopifnot(n_u == nrow(actual))

  if(!is.integer(predictions)) {
    predictions = attr(predictions, "indices", TRUE)
  }
  y_csr = as(actual, "RsparseMatrix")
  res = numeric(n_u)
  for(u in seq_len(n_u)) {
    p1 = y_csr@p[[u]]
    p2 = y_csr@p[[u + 1]]
    ind = p1 + seq_len(p2 - p1)
    # adjust from 0-based indices to 1-based
    u_ind = y_csr@j[ind] + 1L
    u_x = y_csr@x[ind]
    ord = order(u_x, decreasing = TRUE)
    res[[u]] = ap_at_k(predictions[u, ], u_ind[ord], k = k)
  }
  res
}

#' @description \code{ndcg_k()} calculates \bold{Normalized Discounted Cumulative Gain at K (\code{ndcg@@k})}.
#' Please refer to \href{Discounted cumulative gain}{https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG}
#' @rdname metrics
#' @export
ndcg_k = function(predictions, actual, ...) {
  stopifnot(is.matrix(predictions))
  stopifnot(inherits(actual, "sparseMatrix"))

  k = ncol(predictions)
  n_u = nrow(predictions)
  stopifnot(n_u == nrow(actual))
  if(!is.integer(predictions)) {
    predictions = attr(predictions, "indices", TRUE)
  }
  y_csr = as(actual, "RsparseMatrix")
  res = numeric(n_u)
  for(u in seq_len(n_u)) {
    p1 = y_csr@p[[u]]
    p2 = y_csr@p[[u + 1]]
    ind = p1 + seq_len(p2 - p1)
    # adjust from 0-based indices to 1-based
    u_ind = y_csr@j[ind] + 1L
    u_x = y_csr@x[ind]
    ord = order(u_x, decreasing = TRUE)
    res[[u]] = ndcg_at_k(predictions[u, ], u_ind[ord], u_x[ord], k)
  }
  res
}

# @param predicted ordered list of predictions
# @param actual relevant values
# @param k precision level
ap_at_k = function(predicted, actual, k = 10) {
  k = min(k, length(predicted), length(actual))
  pk_seq = predicted[seq_len(k)] %in% actual
  xx = cumsum(pk_seq) / seq_along(pk_seq)
  mean(xx)
}

# DCG
dcg_at_k = function(predicted_indices, actual_indices, actual_relevances, k = length(predicted_indices)) {
  k = min(k, length(predicted_indices), length(actual_indices))
  x_match = match(predicted_indices, actual_indices)
  dcg = 0
  for(i in seq_len(k)) {
    j = x_match[[i]]
    if(!is.na(j))
      dcg = dcg + actual_relevances[[j]] / log2(i + 1)
  }
  dcg
}

# ideal DCG
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
