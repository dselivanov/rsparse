#' @name GloVe
#' @title Creates Global Vectors matrix factorization model.
#' @description GloVe matrix factorization model.
#' Model can be trained via fully can asynchronous and parallel
#' AdaGrad with \code{$fit_transform()} method.
#' @format \code{R6Class} object.
#' @section Usage:
#' For usage details see \bold{Methods, Arguments and Examples} sections.
#' \preformatted{
#' glove = GloVe$new(rank, x_max, learning_rate = 0.15,
#'                           alpha = 0.75, lambda = 0.0, shuffle = FALSE)
#' glove$fit_transform(x, n_iter = 10L, convergence_tol = -1,
#'               n_threads = getOption("rsparse_omp_threads", 1L), ...)
#' glove$components
#' }
#' @section Methods:
#' \describe{
#'   \item{\code{$new(rank, x_max, learning_rate = 0.15,
#'                     alpha = 0.75, lambda = 0, shuffle = FALSE)}}{Constructor for Global vectors model.
#'                     For description of arguments see \bold{Arguments} section.}
#'   \item{\code{$fit_transform(x, n_iter = 10L, convergence_tol = -1,
#'               n_threads = getOption("rsparse_omp_threads", 1L), ...)}}{
#'               fit Glove model given input matrix \code{x}}
#'}
#' @field components represents context embeddings
#' @field bias_i bias term i as per paper
#' @field bias_j bias term j as per paper
#' @field shuffle \code{logical = FALSE} by default. Defines shuffling before each SGD iteration.
#'   Generally shuffling is a good idea for stochastic-gradient descent, but
#'   from my experience in this particular case it does not improve convergence.
#' @section Arguments:
#' \describe{
#'  \item{glove}{A \code{GloVe} object}
#'  \item{x}{An input term co-occurence matrix. Preferably in \code{dgTMatrix} format}
#'  \item{n_iter}{\code{integer} number of SGD iterations}
#'  \item{rank}{desired dimension for the latent vectors}
#'  \item{x_max}{\code{integer} maximum number of co-occurrences to use in the weighting function.
#'    see the GloVe paper for details: \url{http://nlp.stanford.edu/pubs/glove.pdf}}
#'  \item{learning_rate}{\code{numeric} learning rate for SGD. I do not recommend that you
#'     modify this parameter, since AdaGrad will quickly adjust it to optimal}
#'  \item{convergence_tol}{\code{numeric = -1} defines early stopping strategy. We stop fitting
#'     when one of two following conditions will be satisfied: (a) we have used
#'     all iterations, or (b) \code{cost_previous_iter / cost_current_iter - 1 <
#'     convergence_tol}. By default perform all iterations.}
#'  \item{alpha}{\code{numeric = 0.75} the alpha in weighting function formula : \eqn{f(x) = 1 if x >
#'   x_max; else (x/x_max)^alpha}}
#'  \item{lambda}{\code{numeric = 0.0} regularization parameter}
#'  \item{init}{\code{list(w_i = NULL, b_i = NULL, w_j = NULL, b_j = NULL)}
#'  initialization for embeddings (w_i, w_j) and biases (b_i, b_j).
#'  \code{w_i, w_j} - numeric matrices, should number of #rows = rank, #columns - expected number of rows/columns in
#'  input matrix. \code{b_i, b_j} = numeric vectors, should have length of
#'  # expected number of rows/columns in input matrix}
#' }
#' @seealso \url{http://nlp.stanford.edu/projects/glove/}
#' @rdname GloVe
#' @export
#' @examples
#' \donttest{
#' temp = tempfile()
#' download.file('http://mattmahoney.net/dc/text8.zip', temp)
#' text8 = readLines(unz(temp, "text8"))
#' it = itoken(text8)
#' vocabulary = create_vocabulary(it)
#' vocabulary = prune_vocabulary(vocabulary, term_count_min = 5)
#' v_vect = vocab_vectorizer(vocabulary)
#' tcm = create_tcm(it, v_vect, skip_grams_window = 5L)
#' glove_model = GloVe$new(rank = 50, x_max = 10, learning_rate = .25)
#' # fit model and get word vectors
#' word_vectors_main = glove_model$fit_transform(tcm, n_iter = 10)
#' word_vectors_context = glove_model$components
#' word_vectors = word_vectors_main + t(word_vectors_context)
#' }
GloVe = R6::R6Class(
  classname = c("GloVe"),
  inherit = mlapi::mlapiDecomposition,
  public = list(
    bias_i = NULL,
    bias_j = NULL,
    n_dump_every = 0L,
    shuffle = FALSE,
    initialize = function(rank,
                          x_max,
                          learning_rate = 0.15,
                          alpha = 0.75,
                          lambda = 0.0,
                          shuffle = FALSE,
                          init = list(w_i = NULL, b_i = NULL, w_j = NULL, b_j = NULL)
    ) {
      self$shuffle = shuffle
      super$set_internal_matrix_formats(sparse = "TsparseMatrix")

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
    fit_transform = function(x, n_iter = 10L, convergence_tol = -1,
                             n_threads = getOption("rsparse_omp_threads", 1L), ...) {
      # convert to internal native format
      x = super$check_convert_input(x)
      stopifnot(ncol(x) == nrow(x))
      embedding_names = colnames(x)
      if(is.null(embedding_names)) embedding_names = rownames(x)

      stopifnot(all(x@x > 0))

      IS_TRIANGULAR = isTriangular(x)

      n = ncol(x)
      target_dim = c(private$rank, n)
      if(is.null(private$w_i)) private$w_i = matrix(runif(private$rank * n, -0.5, 0.5), private$rank, n)
      if(is.null(private$b_i)) private$b_i = runif(n, -0.5, 0.5)
      if(is.null(private$w_j)) private$w_j = matrix(runif(private$rank * n, -0.5, 0.5), private$rank, n)
      if(is.null(private$b_j)) private$b_j = runif(n, -0.5, 0.5)

      if(!identical(dim(private$w_i), target_dim) ||
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

        if(IS_TRIANGULAR) {
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

      private$components_ = private$w_j

      self$bias_i = private$b_i
      self$bias_j = private$b_j

      t(private$w_i)
    },
    transform = function(x, y = NULL, ...) {
      msg = "transform() is not implemented for GloVe model (unclear what it should to)"
      logger$error(msg)
      stop(msg)
    },
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
