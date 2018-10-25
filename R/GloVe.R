#' @export
GloVe = R6::R6Class(
  classname = c("GloVe"),
  inherit = mlapi::mlapiDecomposition,
  public = list(
    n_dump_every = 0L,
    shuffle = FALSE,
    initialize = function(word_vectors_size,
                          vocabulary,
                          x_max,
                          learning_rate = 0.15,
                          alpha = 0.75,
                          lambda = 0.0,
                          shuffle = FALSE,
                          initial = NULL
    ) {
      self$shuffle = shuffle
      super$set_internal_matrix_formats(sparse = "TsparseMatrix")
      stopifnot(inherits(vocabulary, "character") || inherits(vocabulary, "text2vec_vocabulary"))
      private$vocab_terms =
        if (inherits(vocabulary, "character")) vocabulary
      else vocabulary$term

      private$word_vectors_size = word_vectors_size
      private$learning_rate = learning_rate
      private$x_max = x_max
      private$alpha = alpha
      private$lambda = lambda

      private$fitted = FALSE
      # user didn't provide , so initialize word vectors and corresponding biases
      # randomly as it done in GloVe paper
      m = word_vectors_size
      n = length(private$vocab_terms)

      if (is.null(initial)) {

        private$w_i = matrix(runif(m * n, -0.5, 0.5), m, n)
        private$b_i = runif(n, -0.5, 0.5)
        private$w_j = matrix(runif(m * n, -0.5, 0.5), m, n)
        private$b_j = runif(n, -0.5, 0.5)

        # private$w_i = float::flrunif(m, n, -0.5, 0.5)
        # private$b_i = float::flrunif(n, -0.5, 0.5)
        # private$w_j = float::flrunif(m, n, -0.5, 0.5)
        # private$b_j = float::flrunif(n, -0.5, 0.5)

      } else {
        stopifnot(is.list(initial))
        stopifnot(all(c('w_i', 'w_j', 'b_i', 'b_j') %in% names(initial) ))
        stopifnot(all(dim(initial$w_i) == c(m, n)))
        stopifnot(all(dim(initial$w_j) == c(m, n)))
        stopifnot(length(initial$b_i) == n && length(initial$b_j) == n)

        private$w_i = initial$w_i
        private$w_j = initial$w_j
        private$b_i = initial$b_i
        private$b_j = initial$b_j
      }
    },
    fit_transform = function(x, n_iter = 10L, convergence_tol = -1,
                             n_check_convergence = 1L,
                             n_threads = parallel::detectCores(), ...) {
      # convert to internal native format
      x = super$check_convert_input(x)

      IS_TRIANGULAR = isTriangular(x)
      # params in a specific format to pass to C++ backend
      initial = list(w_i = private$w_i, w_j = private$w_j,
                     b_i = private$b_i, b_j = private$b_j)

      # initial = list(w_i = private$w_i@Data, w_j = private$w_j@Data,
      #                b_i = private$b_i@Data, b_j = private$b_j@Data)

      glove_params =
        list(vocab_size = length(private$vocab_terms),
             word_vec_size = private$word_vectors_size,
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
        if ( self$shuffle )
          iter_order = sample.int( n_nnz, replace = FALSE)
        cost = cpp_glove_partial_fit(private$glove_fitter, x@i, x@j, x@x, iter_order, n_threads)
        if(IS_TRIANGULAR)
          #if matrix is triangular fit another iterations on transposed one
          cost = cost + cpp_glove_partial_fit(private$glove_fitter, x@j, x@i, x@x, iter_order, n_threads)

        # check whether SGD is numerically correct - no NaN at C++ level
        if (is.nan(cost))
          stop("Cost becomes NaN, try to use smaller learning_rate.")
        if (cost / n_nnz / 2 > 0.5)
          warning("Cost is too big, probably something goes wrong... try smaller learning rate", immediate. = TRUE)

        # save cost history
        private$cost_history = c(private$cost_history, cost / n_nnz / 2)
        msg = sprintf("%s - epoch %d, expected cost %.4f", as.character(Sys.time()),
                      i, private$cost_history[[i]])
        flog.info(msg)

        # check convergence
        if ( i > 1 && (private$cost_history[[i - 1]] / private$cost_history[[i]] - 1) < convergence_tol) {
          flog.info("Success: early stopping. Improvement at iterartion %d is less then convergence_tol", i)
          break;
        }
        i = i + 1
      }
      private$fitted = TRUE
      colnames(private$w_i) = private$vocab_terms
      colnames(private$w_j) = private$vocab_terms

      private$components_ = private$w_j

      t(private$w_i)
    },
    transform = function(x, y = NULL, ...) {
      flog.error("transform() method doesn't make sense for GloVe model")
      stop("transform() method doesn't make sense for GloVe model")
    },
    get_history = function() {
      list(cost_history = private$cost_history,
           word_vectors_history = private$word_vectors_history)
    }
  ),
  private = list(
    w_i = NULL,
    w_j = NULL,
    b_i = NULL,
    b_j = NULL,
    vocab_terms = NULL,
    word_vectors_size = NULL,
    initial = NULL,
    alpha = NULL,
    x_max = NULL,
    learning_rate = NULL,
    lambda = NULL,
    cost_history = numeric(0),
    word_vectors_history = NULL,
    glove_fitter = NULL,
    fitted = FALSE
  )
)
