kmeans = function(x, k, n_iter = 10L, init = NULL,
                  seed_mode = c("static_spread", "keep_existing", "static_subset", "random_subset", "random_spread"),
                  is_transposed = FALSE,
                  verbose = FALSE) {

  seed_mode = match.arg(seed_mode)
  seed_mode_codes = c("keep_existing", "static_subset", "static_spread", "random_subset", "random_spread")
  seed_mode = match(seed_mode, seed_mode_codes)

  if(!is_transposed) x = t(x)
  if(!is.null(init)) {
    stopifnot(is.matrix(init) && is.numeric(init) && identical(dim(init), c(nrow(x), k)))
    # trigger a copy
    result = init + 0.0
  } else {
    result = matrix(0, nrow = nrow(x), ncol = k)
  }

  status = arma_kmeans(x, k, seed_mode, n_iter, verbose, result)

  if(is_transposed) result = t(result)
  result
}
