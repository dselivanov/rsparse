nnls = function(X, Y, max_iter = 5000L, tol = 1e-6) {
  c_nnlm_double(X, Y, max_iter, tol)
}
