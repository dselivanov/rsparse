#' @title Global Vectors
#' @description Creates Global Vectors matrix factorization model
#' @references \url{http://nlp.stanford.edu/projects/glove/}
#' @rdname GloVe
#' @export
#' @examples
#' data('movielens100k')
#' co_occurence = crossprod(movielens100k)
#' glove_model = GloVe$new(rank = 4, x_max = 10, learning_rate = .25)
#' embeddings = glove_model$fit_transform(co_occurence, n_iter = 2, n_threads = 1)
#' embeddings = embeddings + t(glove_model$components) # embeddings + context embedings
#' identical(dim(embeddings), c(ncol(movielens100k), 10L))
GloVe = R6::R6Class(
  classname = c("GloVe"),
  public = list(
    #' @field components represents context embeddings
    components = NULL,
    #' @field bias_i bias term i as per paper
    bias_i = NULL,
    #' @field bias_j bias term j as per paper
    bias_j = NULL,
    #' @field shuffle \code{logical = FALSE} by default. Whether to perform shuffling before
    #' each SGD iteration. Generally shuffling is a good practice for SGD.
    shuffle = FALSE,
    #' @description
    #' Creates GloVe model object
    #' @param rank desired dimension for the latent vectors
    #' @param x_max \code{integer} maximum number of co-occurrences to use in the weighting function
    #' @param learning_rate \code{numeric} learning rate for SGD. I do not recommend that you
    #' modify this parameter, since AdaGrad will quickly adjust it to optimal
    #' @param alpha \code{numeric = 0.75} the alpha in weighting function formula :
    #' \eqn{f(x) = 1 if x > x_max; else (x/x_max)^alpha}
    #' @param lambda \code{numeric = 0.0} regularization parameter
    #' @param shuffle see \code{shuffle} field
    #' @param init \code{list(w_i = NULL, b_i = NULL, w_j = NULL, b_j = NULL)}
    #'  initialization for embeddings (w_i, w_j) and biases (b_i, b_j).
    #'  \code{w_i, w_j} - numeric matrices, should have #rows = rank, #columns =
    #'  expected number of rows (w_i) / columns(w_j) in the input matrix.
    #'  \code{b_i, b_j} = numeric vectors, should have length of
    #'  #expected number of rows(b_i) / columns(b_j) in input matrix
    initialize = function(rank,
                          x_max,
                          learning_rate = 0.15,
                          alpha = 0.75,
                          lambda = 0.0,
                          shuffle = FALSE,
                          init = list(w_i = NULL, b_i = NULL, w_j = NULL, b_j = NULL)
    ) {
      self$shuffle = shuffle

      private$rank = as.integer(rank)
      private$learning_rate = learning_rate
      private$x_max = x_max
      private$alpha = alpha
      private$lambda = lambda
      private$fitted = FALSE
      private$w_i = init$w_i
      private$b_i = init$b_i
      private$w_j = init$w_j
      private$b_j = init$b_j
    },
    #' @description fits model and returns embeddings
    #' @param x An input term co-occurence matrix. Preferably in \code{dgTMatrix} format
    #' @param n_iter \code{integer} number of SGD iterations
    #' @param convergence_tol \code{numeric = -1} defines early stopping strategy. Stop fitting
    #' when one of two following conditions will be satisfied: (a) passed
    #' all iterations (b) \code{cost_previous_iter / cost_current_iter - 1 <
    #' convergence_tol}.
    #' @param n_threads number of threads to use
    #' @param ... not used at the moment
    fit_transform = function(x, n_iter = 10L, convergence_tol = -1,
                             n_threads = getOption("rsparse_omp_threads", 1L), ...) {
      x = as(x, "TsparseMatrix")
      stopifnot(ncol(x) == nrow(x))
      embedding_names = colnames(x)
      if (is.null(embedding_names)) embedding_names = rownames(x)

      stopifnot(all(x@x > 0))

      IS_TRIANGULAR = isTriangular(x)

      n = ncol(x)
      target_dim = c(private$rank, n)
      if (is.null(private$w_i)) private$w_i = matrix(runif(private$rank * n, -0.5, 0.5), private$rank, n)
      if (is.null(private$b_i)) private$b_i = runif(n, -0.5, 0.5)
      if (is.null(private$w_j)) private$w_j = matrix(runif(private$rank * n, -0.5, 0.5), private$rank, n)
      if (is.null(private$b_j)) private$b_j = runif(n, -0.5, 0.5)

      if (!identical(dim(private$w_i), target_dim) ||
         !identical(dim(private$w_j), target_dim) ||
         length(private$b_j) != n ||
         length(private$b_i) != n) {
        stop(sprintf("init values provided in the constructor don't match expected dimensions from the input matrix"))
      }

      # params in a specific format to pass to C++ backend
      initial = list(w_i = private$w_i, w_j = private$w_j,
                     b_i = private$b_i, b_j = private$b_j)

      # initial = list(w_i = private$w_i@Data, w_j = private$w_j@Data,
      #                b_i = private$b_i@Data, b_j = private$b_j@Data)

      glove_params =
        list(vocab_size = n,
             word_vec_size = private$rank,
             learning_rate = private$learning_rate,
             x_max = private$x_max,
             alpha = private$alpha,
             lambda = private$lambda,
             initial = initial)
      #--------------------------------------------------------
      # init C++ class which actually perform fitting
      private$glove_fitter = cpp_glove_create(glove_params)
      private$cost_history = numeric(0)
      # number of non-zero elements in co-occurence matrix
      n_nnz = length(x@i)

      # sometimes it is useful to perform shuffle between SGD iterations
      # by default we will not perfrom shuffling:
      # length(iter_order)==0 will be checked at C++ level
      iter_order = integer(0)
      # perform iterations over input co-occurence matrix
      i = 1
      while (i <= n_iter) {
        # if shuffling is required, perform reordering at each iteration
        if ( self$shuffle ) {
          iter_order = sample.int( n_nnz, replace = FALSE)
        }

        cost = cpp_glove_partial_fit(private$glove_fitter, x@i, x@j, x@x, iter_order, n_threads)
        if (is.nan(cost)) stop("Cost becomes NaN, try to use smaller learning_rate.")

        if (IS_TRIANGULAR) {
          #if matrix is triangular fit another iterations on transposed one
          cost = cost + cpp_glove_partial_fit(private$glove_fitter, x@j, x@i, x@x, iter_order, n_threads)
        }
        if (is.nan(cost)) stop("Cost becomes NaN, try to use smaller learning_rate.")
        if (cost / n_nnz > 1) stop("Cost is too big, probably something goes wrong... try smaller learning rate")

        # save cost history
        private$cost_history = c(private$cost_history, cost / n_nnz)
        logger$info("epoch %d, loss %.4f", i, private$cost_history[[i]])

        # check convergence
        if ( i > 1 && (private$cost_history[[i - 1]] / private$cost_history[[i]] - 1) < convergence_tol) {
          logger$info("Success: early stopping. Improvement at iterartion %d is less then convergence_tol", i)
          break;
        }
        i = i + 1
      }

      private$fitted = TRUE
      colnames(private$w_i) = embedding_names
      colnames(private$w_j) = embedding_names

      self$components = private$w_j

      self$bias_i = private$b_i
      self$bias_j = private$b_j

      t(private$w_i)
    },
    #' @description returns value of the loss function for each epoch
    get_history = function() {
      list(cost_history = private$cost_history)
    }
  ),
  private = list(
    w_i = NULL,
    w_j = NULL,
    b_i = NULL,
    b_j = NULL,
    rank = NULL,
    initial = NULL,
    alpha = NULL,
    x_max = NULL,
    learning_rate = NULL,
    lambda = NULL,
    cost_history = numeric(0),
    glove_fitter = NULL,
    fitted = FALSE
  )
)
