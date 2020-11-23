#' @title Weighted Regularized Matrix Factorization for collaborative filtering
#' @description Creates a matrix factorization model which is solved through Alternating Least Squares (Weighted ALS for implicit feedback).
#' For implicit feedback see "Collaborative Filtering for Implicit Feedback Datasets" (Hu, Koren, Volinsky).
#' For explicit feedback it corresponds to the classic model for rating matrix decomposition with MSE error.
#' These two algorithms are proven to work well in recommender systems.
#' @references
#' \itemize{
#'   \item{Hu, Yifan, Yehuda Koren, and Chris Volinsky.
#'         "Collaborative filtering for implicit feedback datasets."
#'         2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.}
#'   \item{\url{https://math.stackexchange.com/questions/1072451/analytic-solution-for-matrix-factorization-using-alternating-least-squares/1073170#1073170}}
#'   \item{\url{http://activisiongamescience.github.io/2016/01/11/Implicit-Recommender-Systems-Biased-Matrix-Factorization/}}
#'   \item{\url{http://datamusing.info/blog/2015/01/07/implicit-feedback-and-collaborative-filtering/}}
#'   \item{\url{https://jessesw.com/Rec-System/}}
#'   \item{\url{http://danielnee.com/2016/09/collaborative-filtering-using-alternating-least-squares/}}
#'   \item{\url{http://www.benfrederickson.com/matrix-factorization/}}
#'   \item{\url{http://www.benfrederickson.com/fast-implicit-matrix-factorization/}}
#'   \item{Franc, Vojtech, Vaclav Hlavac, and Mirko Navara.
#'         "Sequential coordinate-wise algorithm for the
#'         non-negative least squares problem."
#'         International Conference on Computer Analysis of Images
#'         and Patterns. Springer, Berlin, Heidelberg, 2005.}
#' }
#' @export
#' @examples
#' data('movielens100k')
#' train = movielens100k[1:900, ]
#' cv = movielens100k[901:nrow(movielens100k), ]
#' model = WRMF$new(rank = 5,  lambda = 0, feedback = 'implicit')
#' user_emb = model$fit_transform(train, n_iter = 5, convergence_tol = -1)
#' item_emb = model$components
#' preds = model$predict(cv, k = 10, not_recommend = cv)
WRMF = R6::R6Class(
  inherit = MatrixFactorizationRecommender,
  classname = "WRMF",

  public = list(
    #' @description creates WRMF model
    #' @param rank size of the latent dimension
    #' @param lambda regularization parameter
    #' @param init initialization of item embeddings
    #' @param preprocess \code{identity()} by default. User spectified function which will
    #' be applied to user-item interaction matrix before running matrix factorization
    #' (also applied during inference time before making predictions).
    #' For example we may want to normalize each row of user-item matrix to have 1 norm.
    #' Or apply \code{log1p()} to discount large counts.
    #' This corresponds to the "confidence" function from
    #' "Collaborative Filtering for Implicit Feedback Datasets" paper.
    #' Note that it will not automatically add +1 to the weights of the positive entries.
    #' @param feedback \code{character} - feedback type - one of \code{c("implicit", "explicit")}
    #' @param add_biases \code{logical} - whether to add user and item biases in the explicit feedback
    #' models. These will be treated as additional components, so the actual rank will be reduced
    #' by 2. The biases will be taken as the first and last components.
    #' @param non_negative \code{logical}, whether to perform non-negative factorization
    #' @param solver \code{character} - solver for "implicit feedback" problem.
    #' One of \code{c("conjugate_gradient", "cholesky")}.
    #' Usually approximate \code{"conjugate_gradient"} is significantly faster and solution is
    #' on par with \code{"cholesky"}
    #' @param cg_steps \code{integer > 0} - max number of internal steps in conjugate gradient
    #' (if "conjugate_gradient" solver used). \code{cg_steps = 3} by default.
    #' Controls precision of linear equation solution at the each ALS step. Usually no need to tune this parameter
    #' @param precision one of \code{c("double", "float")}. Should embeeding matrices be
    #' numeric or float (from \code{float} package). The latter is usually 2x faster and
    #' consumes less RAM. BUT \code{float} matrices are not "base" objects. Use carefully.
    #' @param ... not used at the moment
    initialize = function(rank = 10L,
                          lambda = 0,
                          init = NULL,
                          preprocess = identity,
                          feedback = c("implicit", "explicit"),
                          add_biases = ifelse(feedback[1] == "explicit", TRUE, FALSE),
                          non_negative = FALSE,
                          solver = c("conjugate_gradient", "cholesky"),
                          cg_steps = 3L,
                          precision = c("double", "float"),
                          ...) {
      stopifnot(is.null(init) || is.matrix(init))
      self$components = init
      solver = match.arg(solver)
      private$precision = match.arg(precision)

      private$als_implicit_fun = if (private$precision == "float") als_implicit_float else als_implicit_double
      private$als_explicit_fun = if (private$precision == "float") als_explicit_float else als_explicit_double

      private$feedback = match.arg(feedback)

      if (add_biases && private$feedback == "implicit") {
        warning("Biases are only supported for explicit feedback")
        add_biases = FALSE
      }
      if (add_biases && rank < 2L)
        stop("Rank must be >= 2 to accommodate biases.")

      if (solver == "cholesky") private$solver_code = 0L
      if (solver == "conjugate_gradient") private$solver_code = 1L

      stopifnot(is.integer(cg_steps) && length(cg_steps) == 1)
      private$cg_steps = cg_steps

      private$lambda = as.numeric(lambda)
      private$rank = as.integer(rank)
      stopifnot(is.function(preprocess))
      private$preprocess = preprocess

      private$scorers = new.env(hash = TRUE, parent = emptyenv())
      private$add_biases = as.logical(add_biases)[1L]
      private$non_negative = non_negative
    },
    #' @description fits the model
    #' @param x input matrix (preferably matrix  in CSC format -`CsparseMatrix`
    #' @param n_iter max number of ALS iterations
    #' @param convergence_tol convergence tolerance checked between iterations
    #' @param ... not used at the moment
    fit_transform = function(x, n_iter = 10L, convergence_tol = ifelse(private$feedback == "implicit", 0.005, 0.001), ...) {
      if (private$feedback == "implicit" ) {
        logger$trace("WRMF$fit_transform(): calling `RhpcBLASctl::blas_set_num_threads(1)` (to avoid thread contention)")
        RhpcBLASctl::blas_set_num_threads(1)
        on.exit({
          n_physical_cores = RhpcBLASctl::get_num_cores()
          logger$trace("WRMF$fit_transform(): on exit `RhpcBLASctl::blas_set_num_threads(%d)` (=number of physical cores)", n_physical_cores)
          RhpcBLASctl::blas_set_num_threads(n_physical_cores)
        })
      }

      logger$trace("convert input to %s if needed", private$internal_matrix_formats$sparse)
      c_ui = as(x, "CsparseMatrix")
      c_ui = private$preprocess(c_ui)
      # strore item_ids in order to use them in predict method
      private$item_ids = colnames(c_ui)

      if ((private$feedback != "explicit") || private$non_negative) {
        logger$trace("check items in input are not negative")
        stopifnot(all(c_ui@x >= 0))
      }
      if (private$feedback == "explicit" && !private$non_negative) {
        self$glob_mean = mean(c_ui@x)
        c_ui@x = c_ui@x - self$glob_mean
      }
      if (private$add_biases) {
        c_ui@x = deep_copy(c_ui@x)
        c_ui_orig = deep_copy(c_ui@x)
      }
      else {
        c_ui_orig = numeric(0L)
      }

      logger$trace("making another matrix for convenient traverse by users - transposing input matrix")
      c_iu = t(c_ui)
      if (private$add_biases) {
        c_iu_orig = deep_copy(c_iu@x)
      } else {
        c_iu_orig = numeric(0L)
      }

      # init
      n_user = nrow(c_ui)
      n_item = ncol(c_ui)

      logger$trace("initializing U")
      if (private$precision == "double")
        private$U = matrix(0.0, ncol = n_user, nrow = private$rank)
      else
        private$U = flrunif(private$rank, n_user, 0, 0)

      if (is.null(self$components)) {
        if (private$precision == "double")
          self$components = matrix(
            rnorm(n_item * private$rank, 0, 0.01),
            ncol = n_item,
            nrow = private$rank
          )
        else
          self$components = flrnorm(private$rank, n_item)
        if (private$non_negative)
          self$components = abs(self$components)
      } else {
        stopifnot(is.matrix(self$components) || is.float(self$components))
        stopifnot(ncol(self$components) == n_item)
        stopifnot(nrow(self$components) == private$rank)
      }


      private$XtX = tcrossprod(self$components) +
        # make float diagonal matrix - if first component is double - result will be automatically casted to double
        fl(diag(x = private$lambda, nrow = private$rank, ncol = private$rank))

      logger$info("starting factorization with %d threads", getOption("rsparse_omp_threads", 1L))
      trace_lst = vector("list", n_iter)
      loss_prev_iter = Inf
      # iterate
      for (i in seq_len(n_iter)) {

        logger$trace("iter %d by item", i)
        stopifnot(ncol(private$U) == ncol(c_iu))
        if (private$feedback == "implicit") {
          # private$U will be modified in place
          loss = private$als_implicit_fun(c_iu, self$components, private$U, private$XtX,
                                          n_threads = getOption("rsparse_omp_threads", 1L),
                                          lambda = private$lambda,
                                          solver = private$solver_code,
                                          cg_steps = private$cg_steps,
                                          non_negative = private$non_negative)
        } else if (private$feedback == "explicit") {
          loss = private$als_explicit_fun(c_iu, c_iu_orig, self$components, private$U,
                                          n_threads = getOption("rsparse_omp_threads", 1L),
                                          lambda = private$lambda,
                                          solver = private$solver_code,
                                          cg_steps = private$cg_steps,
                                          calc_item_bias = private$add_biases,
                                          calc_user_bias = FALSE,
                                          non_negative = private$non_negative)
        }

        logger$trace("iter %d by user", i)
        stopifnot(ncol(self$components) == ncol(c_ui))

        YtY = tcrossprod(private$U) +
          # make float diagonal matrix - if first component is double - result will be automatically casted to double
          fl(diag(x = private$lambda, nrow = private$rank, ncol = private$rank))

        if (private$feedback == "implicit") {
          # self$components will be modified in place
          loss = private$als_implicit_fun(c_ui, private$U,
                                          self$components,
                                          YtY,
                                          n_threads = getOption("rsparse_omp_threads", 1L),
                                          lambda = private$lambda,
                                          private$solver_code,
                                          private$cg_steps,
                                          private$non_negative)
        } else if (private$feedback == "explicit") {
          loss = private$als_explicit_fun(c_ui, c_ui_orig, private$U,
                                          self$components,
                                          n_threads = getOption("rsparse_omp_threads", 1L),
                                          lambda = private$lambda,
                                          private$solver_code,
                                          cg_steps = private$cg_steps,
                                          calc_item_bias = FALSE,
                                          calc_user_bias = private$add_biases,
                                          non_negative = private$non_negative)
        }

        #update XtX
        if (private$feedback == "implicit")
          private$XtX = tcrossprod(self$components) +
            # make float diagonal matrix - if first component is double - result will be automatically casted to double
            fl(diag(x = private$lambda, nrow = private$rank, ncol = private$rank))

        j = 1L
        trace_scors_string = ""
        trace_iter = NULL
        # check if we have scorers
        if (length(private$scorers) > 0) {
          trace_iter = vector("list", length(names(private$scorers)))
          max_k = max(vapply(private$scorers, function(x) as.integer(x[["k"]]), -1L))
          preds = do.call(function(...) self$predict(x = private$cv_data$train, k = max_k, ...),  private$scorers_ellipsis)
          for (sc in names(private$scorers)) {
            scorer = private$scorers[[sc]]
            # preds = do.call(function(...) self$predict(x = private$cv_data$train, k = scorer[["k"]], ...),  private$scorers_ellipsis)
            score = scorer$scorer_function(preds, ...)
            trace_scors_string = sprintf("%s score %s = %f", trace_scors_string, sc, score)
            trace_iter[[j]] = list(iter = i, scorer = sc, value = score)
            j = j + 1L
          }
          trace_iter = data.table::rbindlist(trace_iter)
        }

        trace_lst[[i]] = data.table::rbindlist(list(trace_iter, list(iter = i, scorer = "loss", value = loss)))
        logger$info("iter %d loss = %.4f %s", i, loss, trace_scors_string)
        if (loss_prev_iter / loss - 1 < convergence_tol) {
          logger$info("Converged after %d iterations", i)
          break
        }
        loss_prev_iter = loss
        #------------------------------------------------------------------------
      }

      if (private$precision == "double")
        data.table::setattr(self$components, "dimnames", list(NULL, colnames(x)))
      else
        data.table::setattr(self$components@Data, "dimnames", list(NULL, colnames(x)))

      res = t(private$U)
      private$U = NULL
      setattr(res, "trace", rbindlist(trace_lst))
      if (private$precision == "double")
        setattr(res, "dimnames", list(rownames(x), NULL))
      else
        setattr(res@Data, "dimnames", list(rownames(x), NULL))
      res
    },
    # project new users into latent user space - just make ALS step given fixed items matrix
    #' @description create user embeddings for new input
    #' @param x user-item iteraction matrix
    #' @param ... not used at the moment
    transform = function(x, ...) {
      stopifnot(ncol(x) == ncol(self$components))
      if (private$feedback == "implicit" ) {
        logger$trace("WRMF$transform(): calling `RhpcBLASctl::blas_set_num_threads(1)` (to avoid thread contention)")
        RhpcBLASctl::blas_set_num_threads(1)
        on.exit({
          n_physical_cores = RhpcBLASctl::get_num_cores()
          logger$trace("WRMF$transform(): on exit `RhpcBLASctl::blas_set_num_threads(%d)` (=number of physical cores)", n_physical_cores)
          RhpcBLASctl::blas_set_num_threads(n_physical_cores)
        })
      }
      x = as(x, "CsparseMatrix")
      x = private$preprocess(x)

      if (private$feedback == "implicit") {
        if (private$precision == "double") {
          res = matrix(0, nrow = private$rank, ncol = nrow(x))
        } else {
          res = float(0, nrow = private$rank, ncol = nrow(x))
        }
        private$als_implicit_fun(t(x),
                                 self$components,
                                 res,
                                 private$XtX,
                                 n_threads = getOption("rsparse_omp_threads", 1L),
                                 lambda = private$lambda,
                                 private$solver_code,
                                 private$cg_steps,
                                 private$non_negative)
      } else if (private$feedback == "explicit") {
        if (!private$non_negative)
          x@x = x@x - self$glob_mean
        if (private$precision == "double") {
          res = matrix(0, nrow = private$rank, ncol = nrow(x))
        } else {
          res = float(0, nrow = private$rank, ncol = nrow(x))
        }
        x = t(x)
        if (private$add_biases) {
          x_orig = deep_copy(x@x)
          x@x = x_orig
        } else {
          x_orig = numeric(0L)
        }
        private$als_explicit_fun(x, x_orig,
                                 self$components,
                                 res,
                                 n_threads = getOption("rsparse_omp_threads", 1L),
                                 lambda = private$lambda,
                                 solver = private$solver_code,
                                 cg_steps = private$cg_steps,
                                 calc_user_bias = private$add_biases,
                                 calc_item_bias = FALSE,
                                 non_negative = private$non_negative)
      } else
        stop(sprintf("don't know how to work with feedback = '%s'", private$feedback))
      res = t(res)

      if (private$precision == "double")
        setattr(res, "dimnames", list(rownames(x), NULL))
      else
        setattr(res@Data, "dimnames", list(rownames(x), NULL))
      res
    }
  ),
  #### private -----
  private = list(
    # FIXME - not used anymore - consider to remove
    add_scorers = function(x_train, x_cv, specs = list("map10" = "map@10"), ...) {
      stopifnot(data.table::uniqueN(names(specs)) == length(specs))
      private$cv_data = list(train = x_train, cv = x_cv)
      private$scorers_ellipsis = list(...)
      for (scorer_name in names(specs)) {
        # check scorer exists
        if (exists(scorer_name, where = private$scorers, inherits = FALSE))
          stop(sprintf("scorer with name '%s' already exists", scorer_name))

        metric = specs[[scorer_name]]
        scorer_placeholder = list("scorer_function" = NULL, "k" = NULL)

        if (length(grep(pattern = "(ndcg|map)\\@[[:digit:]]+", x = metric)) != 1 )
          stop(sprintf("don't know how add '%s' metric. Only 'loss', 'map@k', 'ndcg@k' are supported", metric))

        scorer_conf = strsplit(metric, "@", T)[[1]]
        scorer_placeholder[["k"]] = as.integer(tail(scorer_conf, 1))

        scorer_fun = scorer_conf[[1]]
        if (scorer_fun == "map")
          scorer_placeholder[["scorer_function"]] =
          function(predictions, ...) mean(ap_k(predictions, private$cv_data$cv, ...), na.rm = T)
        if (scorer_fun == "ndcg")
          scorer_placeholder[["scorer_function"]] =
          function(predictions, ...) mean(ndcg_k(predictions, private$cv_data$cv, ...), na.rm = T)

        private$scorers[[scorer_name]] = scorer_placeholder
      }
    },
    remove_scorer = function(scorer_name) {
      if (!exists(scorer_name, where = private$scorers))
        stop(sprintf("can't find scorer '%s'", scorer_name))
      rm(list = scorer_name, envir = private$scorers)
    },
    solver_code = NULL,
    cg_steps = NULL,
    scorers = NULL,
    lambda = NULL,
    rank = NULL,
    non_negative = NULL,
    # user factor matrix = rank * n_users
    U = NULL,
    # item factor matrix = rank * n_items
    I = NULL,
    # preprocess - transformation of input matrix before passing it to ALS
    # for example we can scale each row or apply log() to values
    # this is essentially "confidence" transformation from WRMF article
    preprocess = NULL,
    feedback = NULL,
    cv_data = NULL,
    scorers_ellipsis = NULL,
    precision = NULL,
    XtX = NULL,
    als_implicit_fun = NULL,
    als_explicit_fun = NULL
  )
)
