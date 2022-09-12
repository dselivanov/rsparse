#' @name FTRL
#' @title Logistic regression model with FTRL proximal SGD solver.
#' @description Creates 'Follow the Regularized Leader' model.
#' Only logistic regression implemented at the moment.
#' @examples
#' library(rsparse)
#' library(Matrix)
#' i = sample(1000, 1000 * 100, TRUE)
#' j = sample(1000, 1000 * 100, TRUE)
#' y = sample(c(0, 1), 1000, TRUE)
#' x = sample(c(-1, 1), 1000 * 100, TRUE)
#' odd = seq(1, 99, 2)
#' x[i %in% which(y == 1) & j %in% odd] = 1
#' x = sparseMatrix(i = i, j = j, x = x, dims = c(1000, 1000), repr="R")
#'
#' ftrl = FTRL$new(learning_rate = 0.01, learning_rate_decay = 0.1,
#' lambda = 10, l1_ratio = 1, dropout = 0)
#' ftrl$partial_fit(x, y)
#'
#' w = ftrl$coef()
#' head(w)
#' sum(w != 0)
#' p = ftrl$predict(x)
#' @export
FTRL = R6::R6Class(
  classname = "FTRL",
  public = list(
    #-----------------------------------------------------------------
    #' @description creates a model
    #' @param learning_rate learning rate
    #' @param learning_rate_decay learning rate which controls decay. Please refer to FTRL proximal
    #'  paper for details. Usually convergense does not heavily depend on this parameter,
    #'  so default value 0.5 is safe.
    #' @param lambda regularization parameter
    #' @param l1_ratio controls L1 vs L2 penalty mixing.
    #'  1 = Lasso regression, 0 = Ridge regression. Elastic net is in between
    #' @param dropout dropout - percentage of random features to
    #'  exclude from each sample. Acts as regularization.
    #' @param family a description of the error distribution and link function to be used in
    #' the model. Only \code{binomial} (logistic regression) is implemented at the moment.
    initialize = function(learning_rate = 0.1,
                          learning_rate_decay = 0.5,
                          lambda = 0,
                          l1_ratio = 1,
                          dropout = 0,
                          family = c("binomial")) {

      stopifnot(abs(dropout) < 1)
      stopifnot(l1_ratio <= 1 && l1_ratio >= 0)
      stopifnot(lambda >= 0 && learning_rate > 0 && learning_rate_decay > 0)
      family = match.arg(family);
      private$init_model_param(learning_rate = learning_rate,
                               learning_rate_decay = learning_rate_decay,
                               lambda = lambda, l1_ratio = l1_ratio,
                               dropout = dropout, family = family)
    },
    #-----------------------------------------------------------------
    #' @description fits model to the data
    #' @param x input sparse matrix. Native format is \code{Matrix::RsparseMatrix}.
    #' If \code{x} is in different format, model will try to convert it to \code{RsparseMatrix}
    #' with \code{as(x, "RsparseMatrix")}. Dimensions should be (n_samples, n_features)
    #' @param y vector of targets
    #' @param weights numeric vector of length `n_samples`. Defines how to amplify SGD updates
    #' for each sample. May be useful for highly unbalanced problems.
    #' @param ... not used at the moment
    partial_fit = function(x, y, weights = rep(1.0, length(y)), ...) {
      # we can enforce to work only with sparse matrices:
      # stopifnot(inherits(x, "sparseMatrix"))
      if (!inherits(class(x), private$internal_matrix_format)) {
        # message(Sys.time(), " casting input matrix (class ", class(x), ") to ", private$internal_matrix_format)
        x = as(x, private$internal_matrix_format)
      }
      x_ncol = ncol(x)
      # init model during first first fit
      # if (is.null(private$is_initialized)) {
      if (!private$is_initialized) {
        private$init_model_state(n_features = x_ncol,
                                 z = numeric(x_ncol),
                                 n = numeric(x_ncol))
      }
      # on consequent updates check that we are wotking with input matrix with same numner of features
      stopifnot(x_ncol == private$model$n_features)
      # check number of samples = number of outcomes
      stopifnot(nrow(x) == length(y))
      # check no NA - anyNA() is by far fastest solution
      if (anyNA(x@x))
        stop("NA's in input matrix are not allowed")

      # NOTE THAT private$z and private$n will be updated in place during the call !!!
      p = ftrl_partial_fit(m = x, y = y, R_model = private$model, weights = weights,
                           do_update = TRUE, n_threads = getOption("rsparse_omp_threads", 1L))
      invisible(p)
    },
    #' @description
    #' shorthand for applying `partial_fit` `n_iter` times
    #' @param x input sparse matrix. Native format is \code{Matrix::RsparseMatrix}.
    #' If \code{x} is in different format, model will try to convert it to \code{RsparseMatrix}
    #' with \code{as(x, "RsparseMatrix")}. Dimensions should be (n_samples, n_features)
    #' @param y vector of targets
    #' @param weights numeric vector of length `n_samples`. Defines how to amplify SGD updates
    #' for each sample. May be useful for highly unbalanced problems.
    #' @param n_iter number of SGD epochs
    #' @param ... not used at the moment
    fit = function(x, y, weights = rep(1.0, length(y)), n_iter = 1L, ...) {
      for (i in seq_len(n_iter)) {
        logger$trace("iter %03d", i)
        self$partial_fit(x, y, weights, ...)
      }
    },
    #-----------------------------------------------------------------
    #' @description makes predictions based on fitted model
    #' @param x input matrix
    #' @param ... not used at the moment
    predict = function(x, ...) {
      stopifnot(private$is_initialized)
      # stopifnot(inherits(x, "sparseMatrix"))
      if (!inherits(class(x), private$internal_matrix_format)) {
        # message(Sys.time(), " casting input matrix (class ", class(x), ") to ", private$internal_matrix_format)
        x = as(x, private$internal_matrix_format)
      }
      stopifnot(ncol(x) == private$model$n_features)

      if (any(is.na(x)))
        stop("NA's in input matrix are not allowed")

      p = ftrl_partial_fit(m = x, y = numeric(0), R_model = private$model,
                           weights = rep(1, nrow(x)),
                           do_update = FALSE,
                           n_threads = getOption("rsparse_omp_threads", 1L))
      return(p);
    },
    #-----------------------------------------------------------------
    #' @description returns coefficients of the regression model
    coef = function() {
      get_ftrl_weights(private$model)
    }
    #-----------------------------------------------------------------
  ),
  private = list(
    internal_matrix_format = "RsparseMatrix",
    #-----------------------------------------------------------------
    dump = function() {
      # copy because we modify model in place
      model_dump = data.table::copy(private$model)
      class(model_dump) = "ftrl_model_dump"
      model_dump
    },
    #-----------------------------------------------------------------
    load = function(x) {
      if (!inherits(x, "ftrl_model_dump"))
        stop("input should be class of 'ftrl_model_dump' -  list of model parameters")
      private$init_model_param(learning_rate = x$learning_rate, learning_rate_decay = x$learning_rate_decay,
                               lambda = x$lambda, l1_ratio = x$l1_ratio,
                               dropout = x$dropout, family = x$family)
      private$init_model_state(n_features = x$n_features,
                               z = data.table::copy(x$z),
                               n = data.table::copy(x$n))
    },
    # model parameters object
    model = list(
      learning_rate = NULL,
      learning_rate_decay = NULL,
      lambda = NULL,
      l1_ratio = NULL,
      dropout = NULL,
      n_features = NULL,
      family = NULL,
      family_code = NULL,
      z = NULL,
      n = NULL
    ),
    # whether we already called `partial_fit`
    # in this case we fix `n_features`
    is_initialized = FALSE,
    # function to init model
    init_model_param = function(learning_rate = 0.1, learning_rate_decay = 0.5,
                                lambda = 0, l1_ratio = 1,
                                dropout = 0, family = c("binomial")) {
      family = match.arg(family)

      private$model$learning_rate = learning_rate
      private$model$learning_rate_decay = learning_rate_decay
      private$model$lambda = lambda
      private$model$l1_ratio = l1_ratio
      private$model$dropout = dropout
      private$model$family = family
      private$model$family_code =
        switch(family,
               "binomial" = 1,
               "gaussian" = 2,
               "poisson" = 3,
               stop(sprintf("don't know how to work with family = '%s'", family))
        )
    },

    init_model_state = function(n_features = NULL, z = NULL, n = NULL) {
      if (private$is_initialized)
        stop("model already initialized!")

      private$is_initialized = TRUE

      if (!is.null(n_features)) private$model$n_features = n_features
      if (!is.null(z)) private$model$z = z
      if (!is.null(n)) private$model$n = n
    }
  )
)
