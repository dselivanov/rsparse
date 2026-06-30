library(rsparse)
library(Matrix)

# Generate large random sparse matrix
set.seed(42)
n_users = 20000
n_items = 20000
nnz = 1000000
I = sample(n_users, nnz, replace = TRUE)
J = sample(n_items, nnz, replace = TRUE)
X = sparseMatrix(i = I, j = J, x = runif(nnz), dims = c(n_users, n_items))

# Benchmark function
run_bench = function(use_omp_threads, use_blas_control) {
  options(rsparse_omp_threads = use_omp_threads)
  options(rsparse_use_rhpcblasctl = use_blas_control)
  
  # Setup initial BLAS threads to simulate system default (e.g. 2 or 4)
  # only if we have RhpcBLASctl available in the script to set it.
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    # We set it to 2 to ensure it has multi-threading enabled
    RhpcBLASctl::blas_set_num_threads(2)
  }
  
  message(sprintf("OMP Threads: %d, BLAS Control: %s", use_omp_threads, use_blas_control))
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
     message(sprintf("  BLAS threads initial: %d", RhpcBLASctl::blas_get_num_procs()))
  }
  
  model = WRMF$new(rank = 120, feedback = "implicit", solver = "cholesky")
  
  start_time = Sys.time()
  # Use small number of iterations for quick benchmark
  model$fit_transform(X, n_iter = 5L)
  end_time = Sys.time()
  
  diff_time = as.numeric(end_time - start_time, units = "secs")
  message(sprintf("  Time: %.2f seconds", diff_time))
  
  # Restore BLAS threads to 1
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    RhpcBLASctl::blas_set_num_threads(1)
  }
  
  return(diff_time)
}

# Run combinations
grid = expand.grid(
  omp_threads = c(1, 2),
  blas_control = c(TRUE, FALSE)
)

results = list()
for (i in seq_len(nrow(grid))) {
  omp = grid$omp_threads[i]
  ctrl = grid$blas_control[i]
  res = run_bench(omp, ctrl)
  results[[i]] = data.frame(omp_threads = omp, blas_control = ctrl, time = res)
}

print(do.call(rbind, results))
