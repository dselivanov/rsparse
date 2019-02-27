#' @name GloVe
#' @title Creates Global Vectors word-embeddings model.
#' @description Class for GloVe word-embeddings model.
#' It can be trained via fully can asynchronous and parallel
#' AdaGrad with \code{$fit_transform()} method.
#' @format \code{R6Class} object.
#' @section Usage:
#' For usage details see \bold{Methods, Arguments and Examples} sections.
#' \preformatted{
#' glove = GloVe$new(word_vectors_size, x_max, learning_rate = 0.15,
#'                           alpha = 0.75, lambda = 0.0, shuffle = FALSE)
#' glove$fit_transform(x, n_iter = 10L, convergence_tol = -1,
#'               n_threads = getOption("rsparse_omp_threads", 1L), ...)
#' glove$components
#' }
#' @section Methods:
#' \describe{
#'   \item{\code{$new(word_vectors_size, x_max, learning_rate = 0.15,
#'                     alpha = 0.75, lambda = 0, shuffle = FALSE)}}{Constructor for Global vectors model.
#'                     For description of arguments see \bold{Arguments} section.}
#'   \item{\code{$fit_transform(x, n_iter = 10L, convergence_tol = -1,
#'               n_threads = parallel::detectCores(), ...)}}{fit Glove model to input matrix \code{x}}
#'}
#' @field components represents context word vectors
#' @field shuffle \code{logical = FALSE} by default. Defines shuffling before each SGD iteration.
#'   Generally shuffling is a good idea for stochastic-gradient descent, but
#'   from my experience in this particular case it does not improve convergence.
#' @section Arguments:
#' \describe{
#'  \item{glove}{A \code{GloVe} object}
#'  \item{x}{An input term co-occurence matrix. Preferably in \code{dgTMatrix} format}
#'  \item{n_iter}{\code{integer} number of SGD iterations}
#'  \item{word_vectors_size}{desired dimension for word vectors}
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
#' }
#' @seealso \url{http://nlp.stanford.edu/projects/glove/}
#' @rdname GloVe
#' @export
#' @examples
#' \dontrun{
#' temp = tempfile()
#' download.file('http://mattmahoney.net/dc/text8.zip', temp)
#' text8 = readLines(unz(temp, "text8"))
#' it = itoken(text8)
#' vocabulary = create_vocabulary(it)
#' vocabulary = prune_vocabulary(vocabulary, term_count_min = 5)
#' v_vect = vocab_vectorizer(vocabulary)
#' tcm = create_tcm(it, v_vect, skip_grams_window = 5L)
#' glove_model = GloVe$new(word_vectors_size = 50, x_max = 10, learning_rate = .25)
#' # fit model and get word vectors
#' word_vectors_main = glove_model$fit_transform(tcm, n_iter = 10)
#' word_vectors_context = glove_model$components
#' word_vectors = word_vectors_main + t(word_vectors_context)
#' }
GloVe = R6::R6Class(
  classname = c("GloVe"),
  inherit = mlapi::mlapiDecomposition,
  public = list(
    n_dump_every = 0L,
    shuffle = FALSE,
    initialize = function(word_vectors_size,
                          x_max,
                          learning_rate = 0.15,
                          alpha = 0.75,
                          lambda = 0.0,
                          shuffle = FALSE
    ) {
      self$shuffle = shuffle
      super$set_internal_matrix_formats(sparse = "TsparseMatrix")

      private$word_vectors_size = word_vectors_size
      private$learning_rate = learning_rate
      private$x_max = x_max
      private$alpha = alpha
      private$lambda = lambda
      private$fitted = FALSE
    },
    fit_transform = function(x, n_iter = 10L, convergence_tol = -1,
                             n_threads = getOption("rsparse_omp_threads", 1L), ...) {
      # convert to internal native format
      flog.trace("checking input format")
      x = super$check_convert_input(x)
      stopifnot(ncol(x) == nrow(x))
      embedding_names = colnames(x)
      if(is.null(embedding_names)) embedding_names = rownames(x)

      flog.trace("checking all entries > 0")
      stopifnot(all(x@x > 0))

      flog.trace("checking if input is triangular")
      IS_TRIANGULAR = isTriangular(x)

      n = ncol(x)
      m = private$word_vectors_size

      flog.trace("initializing matrices with random numbers")
      private$w_i = matrix(runif(m * n, -0.5, 0.5), m, n)
      private$b_i = runif(n, -0.5, 0.5)
      private$w_j = matrix(runif(m * n, -0.5, 0.5), m, n)
      private$b_j = runif(n, -0.5, 0.5)

      # params in a specific format to pass to C++ backend
      initial = list(w_i = private$w_i, w_j = private$w_j,
                     b_i = private$b_i, b_j = private$b_j)

      # initial = list(w_i = private$w_i@Data, w_j = private$w_j@Data,
      #                b_i = private$b_i@Data, b_j = private$b_j@Data)

      glove_params =
        list(vocab_size = n,
             word_vec_size = m,
             learning_rate = private$learning_rate,
             x_max = private$x_max,
             alpha = private$alpha,
             lambda = private$lambda,
             initial = initial)
      #--------------------------------------------------------
      # init C++ class which actually perform fitting
      flog.trace("initializing c++ class")
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
          flog.trace("generationg random traverse order")
          iter_order = sample.int( n_nnz, replace = FALSE)
        }

        flog.trace("fitting model")
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
        flog.info(sprintf("epoch %d, expected cost %.4f", i, private$cost_history[[i]]))

        # check convergence
        if ( i > 1 && (private$cost_history[[i - 1]] / private$cost_history[[i]] - 1) < convergence_tol) {
          flog.info("Success: early stopping. Improvement at iterartion %d is less then convergence_tol", i)
          break;
        }
        i = i + 1
      }

      private$fitted = TRUE
      colnames(private$w_i) = embedding_names
      colnames(private$w_j) = embedding_names

      private$components_ = private$w_j

      t(private$w_i)
    },
    transform = function(x, y = NULL, ...) {
      flog.error("transform() method doesn't make sense for GloVe model")
      stop("transform() method doesn't make sense for GloVe model")
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
    word_vectors_size = NULL,
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
