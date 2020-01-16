make_csr_identity_matrix = function(x) {
  res = Matrix::.sparseDiagonal(n = x, shape = 'g', kind = 'd')
  as(res, "RsparseMatrix")
}


RankMF = R6::R6Class(
  inherit = MatrixFactorizationRecommender,
  classname = "RankMF",
  public = list(
    rank = NULL,
    lambda_user = NULL,
    lambda_item_positive = NULL,
    lambda_item_negative = NULL,
    learning_rate = NULL,
    margin = NULL,
    optimizer = NULL,
    kernel = NULL,
    gamma = NULL,
    precision = NULL,
    loss = NULL,
    max_negative_samples = NULL,
    item_features_embeddings = NULL,
    user_features_embeddings = NULL,
    initialize = function(rank = 8L,
                          learning_rate = 0.01,
                          optimizer = c("adagrad", "rmsprop"),
                          lambda = 0,
                          init = NULL,
                          gamma = 0,
                          precision = c("double", "float"),
                          loss = c("bpr", "warp"),
                          kernel = c("identity", "sigmoid"),
                          margin = 0.1,
                          max_negative_samples = 50L,
                          ...) {
      self$rank = rank
      self$learning_rate = learning_rate

      optimizer = match.arg(optimizer)
      allowed_optimizers = c("adagrad" = 0L, "rmsprop" = 1L)
      self$optimizer = allowed_optimizers[[optimizer]]

      stopifnot(is.numeric(lambda))
      if (length(lambda) == 1) {
        lambda = c(
          lambda_user = lambda,
          lambda_item_positive = lambda,
          lambda_item_negative = lambda
        )
      }
      self$lambda_user = lambda[["lambda_user"]]
      self$lambda_item_positive = lambda[["lambda_item_positive"]]
      self$lambda_item_negative = lambda[["lambda_item_negative"]]

      self$gamma = gamma

      self$precision = match.arg(precision)

      allowed_loss = c("bpr" = 0L, "warp" = 1L)
      loss = match.arg(loss)
      self$loss = allowed_loss[[loss]]

      kernel = match.arg(kernel)
      allowed_kernel = c("identity" = 0L, "sigmoid" = 1L)
      self$kernel = allowed_kernel[[kernel]]

      self$margin = margin
      self$max_negative_samples = max_negative_samples
    },
    transform = function(x, user_features = NULL, n_iter = 100, n_threads = getOption("rsparse_omp_threads", 1L)) {
      stop("not implemented yet")
      # private$partial_fit_transform_(x, private$item_features, user_features, n_iter, n_threads, update_items = FALSE)
    },
    partial_fit_transform = function(x, item_features = NULL, user_features = NULL, n_iter = 100, n_threads = getOption("rsparse_omp_threads", 1L)) {
      private$partial_fit_transform_(x, item_features, user_features, n_iter, n_threads, update_items = TRUE)
    }
  ),
  private = list(
    item_features = NULL,
    user_features_squared_grad = NULL,
    item_features_squared_grad = NULL,
    partial_fit_transform_ = function(x, item_features = NULL, user_features = NULL, n_iter = 100, n_threads = getOption("rsparse_omp_threads", 1L), update_items = TRUE) {
      if (is.null(item_features)) item_features = make_csr_identity_matrix(ncol(x))
      if (is.null(user_features)) user_features = make_csr_identity_matrix(nrow(x))

      stopifnot(
        inherits(x, "RsparseMatrix"),
        inherits(item_features, "RsparseMatrix"),
        inherits(user_features, "RsparseMatrix"),
        nrow(x) == nrow(user_features),
        ncol(x) == nrow(item_features)
      )
      private$item_features = item_features
      n_user = nrow(x)
      # n_item = ncol(x)
      n_item_features = ncol(private$item_features)
      n_user_features = ncol(user_features)

      if (is.null(self$user_features_embeddings)) self$user_features_embeddings = matrix(rnorm(self$rank * n_user_features, 0, 1e-3), nrow = self$rank)
      if (is.null(private$user_features_squared_grad)) private$user_features_squared_grad = rep(1.0, n_user_features)

      if (is.null(self$item_features_embeddings)) self$item_features_embeddings = matrix(rnorm(self$rank * n_item_features, 0, 1e-3), nrow = self$rank)
      if (is.null(private$item_features_squared_grad)) private$item_features_squared_grad = rep(1.0, n_item_features)

      # temporary disable BLAS threading to prevent thread contention with OpenMP
      n_threads_blas = RhpcBLASctl::blas_get_num_procs()
      RhpcBLASctl::blas_set_num_threads(1L)
      on.exit(RhpcBLASctl::blas_set_num_threads(n_threads_blas))

      rankmf_solver_double(
        x,
        self$user_features_embeddings,
        self$item_features_embeddings,
        private$user_features_squared_grad,
        private$item_features_squared_grad,
        user_features,
        private$item_features,
        rank = self$rank,
        n_updates = n_iter * n_user,
        self$learning_rate,
        gamma = self$gamma,
        lambda_user = self$lambda_user,
        lambda_item_positive = self$lambda_item_positive,
        lambda_item_negative = self$lambda_item_negative,
        n_threads = n_threads,
        update_items = update_items,
        loss = self$loss,
        kernel = self$kernel,
        max_negative_samples = self$max_negative_samples,
        margin = self$margin,
        self$optimizer
      )
      item_embeddings = tcrossprod(private$item_features, self$item_features_embeddings)
      # transpose to have shape rank * n_items
      private$components_ = t(item_embeddings)
      # transpose to have shape n_users * rank
      user_embeddings = t(self$user_features_embeddings %*% t(user_features))
      invisible(as.matrix(user_embeddings))
    }
  )
)
