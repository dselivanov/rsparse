#' @name FactorizationMachine
#' @title Creates FactorizationMachine model.
#' @description Creates second order Factorization Machines model
#' @section Usage:
#' For usage details see \bold{Methods, Arguments and Examples} sections.
#' \preformatted{
#' fm = FM$new(learning_rate_w = 0.2, rank = 8, lambda_w = 0, lambda_v = 0, family = c("binomial", "gaussian")
#'   intercept = TRUE, learning_rate_v = learning_rate_w)
#' fm$partial_fit(x, y, ...)
#' fm$predict(x, ...)
#' }
#' @format \code{R6Class} object.
#' @section Methods:
#' \describe{
#'   \item{\code{FM$new(learning_rate_w = 0.2, rank = 8, lambda_w = 1e-6, lambda_v = 1e-6,
#'   family = c("binomial", "gaussian"), intercept = TRUE, learning_rate_v = learning_rate_w)}}{Constructor
#'   for FactorizationMachines model. For description of arguments see \bold{Arguments} section.}
#'   \item{\code{$partial_fit(x, y, ...)}}{fits/updates model given input matrix \code{x} and target vector \code{y}.
#'   \code{x} shape = (n_samples, n_features)}
#'   \item{\code{$predict(x, ...)}}{predicts output \code{x}}
#'}
#' @section Arguments:
#' \describe{
#'  \item{fm}{\code{FM} object}
#'  \item{x}{Input sparse matrix - native format is \code{Matrix::RsparseMatrix}.
#'  If \code{x} is in different format, model will try to convert it to \code{RsparseMatrix}
#'  with \code{as(x, "RsparseMatrix")} call}
#'  \item{learning_rate_w}{learning rate for linear weights in AdaGrad SGD}
#'  \item{learning_rate_v}{learning rate for interactions in AdaGrad SGD}
#'  \item{rank}{rank of the latent dimension in factorization}
#'  \item{lambda_w}{regularization parameter for linear terms}
#'  \item{lambda_v}{regularization parameter for interactions terms}
#'  \item{n_features}{number of features in model (number of columns in expected model matrix) }
#'  \item{family}{ \code{"gaussian"} or \code{"binomial"}}
#' }
#' @export
FactorizationMachine = R6::R6Class(
  classname = "FactorizationMachine",
  inherit = mlapi::mlapiEstimationOnline,
  public = list(
    #-----------------------------------------------------------------
    initialize = function(learning_rate_w = 0.2,
                          rank = 4,
                          lambda_w = 0,
                          lambda_v = 0,
                          family = c("binomial", "gaussian"),
                          intercept = TRUE,
                          learning_rate_v = learning_rate_w) {
      stopifnot(lambda_w >= 0 && lambda_v >= 0 && learning_rate_w > 0 && rank >= 1 && learning_rate_v > 0)
      family = match.arg(family);
      private$learning_rate_w = learning_rate_w
      private$learning_rate_v = learning_rate_v
      private$rank = rank
      private$lambda_w = lambda_w
      private$lambda_v = lambda_v
      private$family = family
      private$intercept = intercept
    },
    partial_fit = function(x, y, weights = rep(1.0, length(y)), ...) {
      if(!inherits(class(x), private$internal_matrix_format)) {
        x = as(x, private$internal_matrix_format)
      }
      x_ncol = ncol(x)
      # init model during first first fit
      if(!private$is_initialized) {
        private$n_features = x_ncol
        #---------------------------------------------
        private$w0 = 0L
        fill_float_vector(private$w0, 0.0)
        #---------------------------------------------
        private$w = integer(private$n_features)
        fill_float_vector_randn(private$w, 0.001)
        #---------------------------------------------
        private$v = matrix(0L, nrow = private$rank, ncol = private$n_features)
        fill_float_matrix_randn(private$v, 0.001)
        #---------------------------------------------
        private$grad_w2 = integer(private$n_features)
        fill_float_vector(private$grad_w2, 1.0)
        #---------------------------------------------
        private$grad_v2 = matrix(0L, nrow = private$rank, ncol = private$n_features)
        fill_float_matrix(private$grad_v2, 1.0)
        #---------------------------------------------
        private$ptr_param = fm_create_param(private$learning_rate_w, private$learning_rate_v,
                                            private$rank, private$lambda_w, private$lambda_v,
                                            private$w0,
                                            private$w, private$v,
                                            private$grad_w2, private$grad_v2,
                                            private$family,
                                            private$intercept)
        private$ptr_model = fm_create_model(private$ptr_param)
        private$is_initialized = TRUE
      }
      # on consequent updates check that we are wotking with input matrix with same numner of features
      stopifnot(x_ncol == private$n_features)
      # check number of samples = number of outcomes
      stopifnot(nrow(x) == length(y))
      stopifnot(is.numeric(weights) && length(weights) == length(y))
      stopifnot(!anyNA(y))
      # convert to (1, -1) as it required by loss function in FM
      if(private$family == 'binomial')
        y = ifelse(y == 1, 1, -1)

      # check no NA - anyNA() is by far fastest solution
      if(anyNA(x@x))
        stop("NA's in input matrix are not allowed")

      p = fm_partial_fit(private$ptr_model, x, y, weights, do_update = TRUE, n_threads = getOption("rsparse_omp_threads"))
      invisible(p)
    },
    fit = function(x, y, weights = rep(1.0, length(y)), n_iter = 1L, ...) {
      for(i in seq_len(n_iter)) {
        futile.logger::flog.debug("FactorizationMachine iter %03d", i)
        self$partial_fit(x, y, weights, ...)
      }
    },
    predict =  function(x, ...) {
      if(is.null(private$ptr_param) || is_invalid_ptr(private$ptr_param)) {
        print("is.null(private$ptr_param) || is_invalid_ptr(private$ptr_param)")
        if(private$is_initialized) {
          print("init private$ptr_param")
          private$ptr_param = fm_create_param(private$learning_rate_w, private$learning_rate_v,
                                              private$rank, private$lambda_w, private$lambda_v,
                                              private$w0,
                                              private$w, private$v,
                                              private$grad_w2, private$grad_v2,
                                              private$family,
                                              private$intercept)
          private$ptr_model = fm_create_model(private$ptr_param)
        }
      }
      stopifnot(private$is_initialized)
      if(!inherits(class(x), private$internal_matrix_format)) {
        x = as(x, private$internal_matrix_format)
      }
      stopifnot(ncol(x) == private$model$n_features)

      if(any(is.na(x)))
        stop("NA's in input matrix are not allowed")
      # dummy numeric(0) - don't have y and don't need weights
      p = fm_partial_fit(private$ptr_model, x, numeric(0), numeric(0), do_update = FALSE, n_threads = getOption("rsparse_omp_threads"))
      return(p);
    }
  ),
  private = list(
    #--------------------------------------------------------------
    is_initialized = FALSE,
    internal_matrix_format = "RsparseMatrix",
    #--------------------------------------------------------------
    ptr_param = NULL,
    ptr_model = NULL,
    #--------------------------------------------------------------
    n_features = NULL,
    learning_rate_w = NULL,
    learning_rate_v = NULL,
    rank = NULL,
    lambda_w = NULL,
    lambda_v = NULL,
    family = NULL,
    intercept = NULL,
    #--------------------------------------------------------------
    # these 5 will be modified in place in C++ code
    #--------------------------------------------------------------
    v = NULL,
    w = NULL,
    w0 = NULL,
    grad_v2 = NULL,
    grad_w2 = NULL
  )
)
