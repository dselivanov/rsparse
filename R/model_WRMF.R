#' @name WRMF
#'
#' @title (Weighted) Regularized Matrix Facrtorization for collaborative filtering
#' @description Creates matrix factorization model which could be solved with Alternating Least Squares (Weighted ALS for implicit feedback).
#' For implicit feedback see (Hu, Koren, Volinsky)'2008 paper \url{http://yifanhu.net/PUB/cf.pdf}.
#' For explicit feedback model is classic model for rating matrix decomposition with MSE error (without biases at the moment).
#' These two algorithms are proven to work well in recommender systems.
#' @seealso
#' \itemize{
#'   \item{\url{https://math.stackexchange.com/questions/1072451/analytic-solution-for-matrix-factorization-using-alternating-least-squares/1073170#1073170}}
#'   \item{\url{http://activisiongamescience.github.io/2016/01/11/Implicit-Recommender-Systems-Biased-Matrix-Factorization/}}
#'   \item{\url{http://datamusing.info/blog/2015/01/07/implicit-feedback-and-collaborative-filtering/}}
#'   \item{\url{https://jessesw.com/Rec-System/}}
#'   \item{\url{http://danielnee.com/2016/09/collaborative-filtering-using-alternating-least-squares/}}
#'   \item{\url{http://www.benfrederickson.com/matrix-factorization/}}
#'   \item{\url{http://www.benfrederickson.com/fast-implicit-matrix-factorization/}}
#' }
#' @format \code{R6Class} object.
#' @section Usage:
#' For usage details see \bold{Methods, Arguments and Examples} sections.
#' \preformatted{
#'   model = WRMF$new(rank = 10L, lambda = 0,
#'                   feedback = c("implicit", "explicit"),
#'                   init_stdv = 0.01,
#'                   n_threads = parallel::detectCores(),
#'                   non_negative = FALSE,
#'                   solver = c("conjugate_gradient", "cholesky"),
#'                   cg_steps = 3L,
#'                   components = NULL)
#'   model$fit_transform(x, n_iter = 5L, ...)
#'   model$predict(x, k, not_recommend = x, ...)
#'   model$components
#'   model$add_scorers(x_train, x_cv, specs = list("map10" = "map@@10"), ...)
#'   model$remove_scorer(name)
#' }
#' @section Methods:
#' \describe{
#'   \item{\code{$new(rank = 10L, lambda = 0, feedback = c("implicit", "explicit"),
#'                    init_stdv = 0.01, n_threads = parallel::detectCores(), non_negative = FALSE,
#'                    solver = c("conjugate_gradient", "cholesky"), cg_steps = 3L,
#'                    components = NULL) }}{ creates matrix
#'     factorization model model with \code{rank} latent factors. If \code{components} is provided then initialize
#'     item embeddings with its values.}
#'   \item{\code{$fit_transform(x, n_iter = 5L, ...)}}{ fits model to
#'     an input user-item matrix. (preferably in "dgCMatrix" format).
#'     For implicit feedback \code{x} should be a confidence matrix which corresponds to \code{1 + alpha * r_ui} in original paper.
#'     Usually \code{r_ui} corresponds to the number of interactions of user \code{u} and item \code{i}.
#'     For explicit feedback values in \code{x} represents ratings.
#'     \bold{Returns factor matrix for users of size \code{n_users * rank}}}
#'   \item{\code{$predict(x, k, not_recommend = x, ...)}}{predict \code{top k}
#'     item ids for users \code{x} (= column names from the matrix passed to \code{fit_transform()} method).
#'     Users features should be defined the same way as they were defined in training data - as \bold{sparse matrix}
#'     of confidence values (implicit feedback) or ratings (explicit feedback).
#'     Column names (=item ids) should be in the same order as in the \code{fit_transform()}.}
#'   \item{\code{$add_scorers(x_train, x_cv, specs = list("map10" = "map@@10"), ...)}}{add a metric to watchlist.
#'   Metric will be evaluated after each ALS interation. At the moment following metrices are supported:
#'     \bold{"loss"}, \bold{"map@@k"}, \bold{"ndcg@@k"}, where \bold{k} is some integer. For example \code{map@@10}.}
#'   \item{\code{$remove_scorer(name)}}{remove a metric from watchlist}
#'   \item{\code{$components}}{item factors matrix of size \code{rank * n_items}}
#'   \item{n_threads}{\code{numeric} default number of threads to use during training and prediction
#'   (if OpenMP is available).}

#'}
#' @section Arguments:
#' \describe{
#'  \item{model}{A \code{WRMF} model.}
#'  \item{x}{An input sparse user-item matrix(of class \code{dgCMatrix}).
#'  For explicit feedback should consists of ratings.
#'  For implicit feedback all positive interactions should be filled with \bold{confidence} values.
#'  Missed interactions should me zeros/empty.
#'  So for simple case case when \code{confidence = 1 + alpha * x}}
#'  \item{x_train}{An input user-item \bold{relevance} matrix. Used during evaluation of \code{map@@k}, \code{ndcg@@k}
#'    Should have the same shape as corresponding confidence matrix \code{x_cv}.
#'    Values are used as "relevance" in ndgc calculation}
#'  \item{x_cv}{user-item matrix used for validation (ground-truth observations)}
#'  \item{name}{\code{character} - user-defined name of the scorer. For example "ndcg-scorer-1"}
#'  \item{rank}{\code{integer} - number of latent factors}
#'  \item{lambda}{\code{numeric} - regularization parameter}
#'  \item{feedback}{\code{character} - feedback type - one of \code{c("implicit", "explicit")}}
#'  \item{solver}{\code{character} - solver for "implicit feedback" problem.
#'     One of \code{c("conjugate_gradient", "cholesky")}.
#'     Usually approximate \code{"conjugate_gradient"} is significantly faster and solution is
#'     on par with exact \code{"cholesky"}}
#'  \item{cg_steps}{\code{integer > 0} - max number of internal steps in conjugate gradient
#'     (if "conjugate_gradient" solver used). \code{cg_steps = 3} by default.
#'     Controls precision of linear equation solution at the each ALS step. Usually no need to tune this parameter.}
#'  \item{preprocess}{\code{function} = \code{identity()} by default. User spectified function which will be applied to user-item interaction matrix
#'     before running matrix factorization (also applied in inference time before making predictions). For example we may
#'     want to normalize each row of user-item matrix to have 1 norm. Or apply \code{log1p()} to discount large counts.
#'     This essentially corresponds to the "confidence" function from (Hu, Koren, Volinsky)'2008 paper \url{http://yifanhu.net/PUB/cf.pdf}}
#'  \item{n_threads}{\code{numeric} default number of threads to use during training and prediction
#'  (if OpenMP is available).}
#'  \item{not_recommend}{\code{sparse matrix} or \code{NULL} - points which items should be excluided from recommendations for a user.
#'    By default it excludes previously seen/consumed items.}
#'  \item{convergence_tol}{{\code{numeric = -Inf} defines early stopping strategy. We stop fitting
#'     when one of two following conditions will be satisfied: (a) we have used
#'     all iterations, or (b) \code{loss_previous_iter / loss_current_iter - 1 < convergence_tol}}}
#'  \item{init_stdv}{\code{numeric} standart deviation for initialization of the initial latent matrices}
#'  \item{...}{other arguments. Not used at the moment}
#' }
#' @export
WRMF = R6::R6Class(
  inherit = mlapi::mlapiDecomposition,
  classname = "AlternatingLeastSquares",
  public = list(
    n_threads = NULL,
    initialize = function(rank = 10L,
                          lambda = 0,
                          feedback = c("implicit", "explicit"),
                          n_threads = parallel::detectCores(),
                          init_stdv = 0.01,
                          non_negative = FALSE,
                          solver = c("conjugate_gradient", "cholesky"),
                          cg_steps = 3L,
                          components = NULL,
                          preprocess = identity,
                          ...) {
      stopifnot(is.null(components) || is.matrix(components))
      private$components_ = components
      solver = match.arg(solver)
      private$feedback = match.arg(feedback)
      if(solver == "conjugate_gradient" && private$feedback == "explicit")
        flog.warn("only 'cholesky' is available for 'explicit' feedback")

      if(solver == "cholesky") private$solver_code = 0L
      if(solver == "conjugate_gradient") private$solver_code = 1L

      if(solver == "cholesky") flog.info("'cg_steps' ignored for 'cholesky' solver")
      stopifnot(is.integer(cg_steps) && length(cg_steps) == 1)
      private$cg_steps = cg_steps

      private$set_internal_matrix_formats(sparse = "dgCMatrix", dense = NULL)
      private$lambda = as.numeric(lambda)
      private$init_stdv = as.numeric(init_stdv)
      private$rank = as.integer(rank)
      stopifnot(is.function(preprocess))
      private$preprocess = preprocess

      private$scorers = new.env(hash = TRUE, parent = emptyenv())
      self$n_threads = n_threads
      private$non_negative = non_negative
    },
    fit_transform = function(x, n_iter = 5L, convergence_tol = -Inf, ...) {

      # x = confidense matrix, not ratings/interactions matrix!
      # we expect user already transformed it
      # default choice will be
      # x = 1 + alpha * r

      flog.debug("convert input to %s if needed", private$internal_matrix_formats$sparse)
      c_ui = private$check_convert_input(x)
      c_ui = private$preprocess(c_ui)
      # strore item_ids in order to use them in predict method
      private$item_ids = colnames(c_ui)

      flog.debug("check items in input are not negative")
      stopifnot(all(c_ui@x >= 0))

      flog.debug("making antoher matrix for convenient traverse by users - transposing input matrix")
      c_iu = t(c_ui)

      # init
      n_user = nrow(c_ui)
      n_item = ncol(c_ui)

      private$U = matrix(0.0, ncol = n_user, nrow = private$rank)

      if(is.null(private$components_)) {
        private$components_ = matrix(rnorm(n_item * private$rank, 0, private$init_stdv), ncol = n_item, nrow = private$rank)
      } else {
        stopifnot(is.matrix(private$components_))
        stopifnot(ncol(private$components_) == n_item)
        stopifnot(nrow(private$components_) == private$rank)
      }

      Lambda = diag(x = private$lambda, nrow = private$rank, ncol = private$rank)

      trace_values = vector("numeric", n_iter)

      flog.info("starting factorization with %d threads", self$n_threads)
      trace_lst = vector("list", n_iter)
      loss_prev_iter = Inf
      # iterate
      for (i in seq_len(n_iter)) {

        flog.debug("iter %d by item", i)
        stopifnot(ncol(private$U) == ncol(c_iu))
        if (private$feedback == "implicit") {
          # private$U will be modified in place
          loss = als_implicit(c_iu, private$components_, private$U, n_threads = self$n_threads,
                       lambda = private$lambda,
                       solver = private$solver_code, cg_steps = private$cg_steps)
        } else if (private$feedback == "explicit") {
          private$U = private$solver_explicit_feedback(c_iu, private$components_)
        }
        # if need non-negative matrix factorization - just set all negative values to zero
        if(private$non_negative)
          private$U[private$U < 0] = 0

        flog.debug("iter %d by user", i)
        stopifnot(ncol(private$components_) == ncol(c_ui))
        if (private$feedback == "implicit") {
          # private$components_ will be modified in place
          loss = als_implicit(c_ui, private$U, private$components_, n_threads = self$n_threads,
                              lambda = private$lambda,
                              private$solver_code, private$cg_steps)
        } else if (private$feedback == "explicit") {
          private$components_ = private$solver_explicit_feedback(c_ui, private$U)
        }
        # if need non-negative matrix factorization - just set all negative values to zero
        if(private$non_negative)
          private$components_[private$components_ < 0] = 0

        #------------------------------------------------------------------------
        # calculate some metrics if needed in order to diagnose convergence
        #------------------------------------------------------------------------
        if (private$feedback == "explicit")
          loss = als_loss_explicit(c_ui, private$U, private$components_, private$lambda, self$n_threads);


        j = 1L
        trace_scors_string = ""
        trace_iter = NULL
        # check if we have scorers
        if(length(private$scorers) > 0) {
          trace_iter = vector("list", length(names(private$scorers)))
          max_k = max(vapply(private$scorers, function(x) as.integer(x[["k"]]), -1L))
          preds = do.call(function(...) self$predict(x = private$cv_data$train, k = max_k, ...),  private$scorers_ellipsis)
          for(sc in names(private$scorers)) {
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
        flog.info("iter %d loss = %.4f %s", i, loss, trace_scors_string)
        if(loss_prev_iter / loss - 1 < convergence_tol) {
          flog.info("Converged after %d iterations", i)
          break
        }
        loss_prev_iter = loss
        #------------------------------------------------------------------------
      }

      data.table::setattr(private$components_, "dimnames", list(NULL, colnames(x)))

      res = t(private$U)
      private$U = NULL
      setattr(res, "trace", rbindlist(trace_lst))
      setattr(res, "dimnames", list(rownames(x), NULL))
      res
    },
    # project new users into latent user space - just make ALS step given fixed items matrix
    transform = function(x, ...) {
      stopifnot(ncol(x) == ncol(private$components_))
      # allocate result matrix - will be modified in place

      x = private$check_convert_input(x)
      x = private$preprocess(x)

      if(private$feedback == "implicit") {
        res = matrix(0, nrow = private$rank, ncol = nrow(x))

        als_implicit(t(x), private$components_, res, n_threads = self$n_threads,
                     lambda = private$lambda,
                     private$solver_code, private$cg_steps)

      } else if(private$feedback == "explicit")
        res = private$solver_explicit_feedback(t(x), private$components_)
      else
        stop(sprintf("don't know how to work with feedback = '%s'", private$feedback))
      if(private$non_negative)
        res[res < 0] = 0
      res = t(res)
      data.table::setattr(res, "dimnames", list(rownames(x), NULL))
      res
    },
    predict = function(x, k, not_recommend = x, ...) {
      stopifnot(private$item_ids == colnames(x))
      stopifnot(is.null(not_recommend) || inherits(not_recommend, "sparseMatrix"))
      if(!is.null(not_recommend))
        not_recommend = as(not_recommend, "dgCMatrix")
      m = nrow(x)

      # transform user features into latent space
      # calculate scores for each item
      # user_item_score = self$transform(x) %*% private$components_
      indices = dotprod_top_k(self$transform(x), private$components_, k, self$n_threads, not_recommend)
      data.table::setattr(indices, "dimnames", list(rownames(x), NULL))
      data.table::setattr(indices, "indices", NULL)

      if(!is.null(private$item_ids)) {
        predicted_item_ids = private$item_ids[indices]
        data.table::setattr(predicted_item_ids, "dim", dim(indices))
        data.table::setattr(predicted_item_ids, "dimnames", list(rownames(x), NULL))
        data.table::setattr(indices, "indices", predicted_item_ids)
      }

      indices
    },
    add_scorers = function(x_train, x_cv, specs = list("map10" = "map@10"), ...) {
      stopifnot(data.table::uniqueN(names(specs)) == length(specs))
      private$cv_data = list(train = x_train, cv = x_cv)
      private$scorers_ellipsis = list(...)
      for(scorer_name in names(specs)) {
        # check scorer exists
        if(exists(scorer_name, where = private$scorers, inherits = FALSE))
          stop(sprintf("scorer with name '%s' already exists", scorer_name))

        metric = specs[[scorer_name]]
        scorer_placeholder = list("scorer_function" = NULL, "k" = NULL)

        if (length(grep(pattern = "(ndcg|map)\\@[[:digit:]]+", x = metric)) != 1 )
          stop(sprintf("don't know how add '%s' metric. Only 'loss', 'map@k', 'ndcg@k' are supported", metric))

        scorer_conf = strsplit(metric, "@", T)[[1]]
        scorer_placeholder[["k"]] = as.integer(tail(scorer_conf, 1))

        scorer_fun = scorer_conf[[1]]
        if(scorer_fun == "map")
          scorer_placeholder[["scorer_function"]] =
            function(predictions, ...) mean(ap_k(predictions, private$cv_data$cv, ...), na.rm = T)
        if(scorer_fun == "ndcg")
          scorer_placeholder[["scorer_function"]] =
            function(predictions, ...) mean(ndcg_k(predictions, private$cv_data$cv, ...), na.rm = T)

        private$scorers[[scorer_name]] = scorer_placeholder
      }
    },
    remove_scorer = function(scorer_name) {
      if(!exists(scorer_name, where = private$scorers))
        stop(sprintf("can't find scorer '%s'", scorer_name))
      rm(list = scorer_name, envir = private$scorers)
    },
    finalize = function() {
      rm(list = names(private$scorers), envir = private$scorers)
      private$scorers = NULL
      gc();
    }
  ),
  private = list(
    solver_code = NULL,
    cg_steps = NULL,
    scorers = NULL,
    lambda = NULL,
    init_stdv = NULL,
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
    item_ids = NULL,
    cv_data = NULL,
    scorers_ellipsis = NULL,
    #------------------------------------------------------------
    solver_explicit_feedback = function(R, X) {
      XtX = tcrossprod(X) + diag(x = private$lambda, nrow = private$rank, ncol = private$rank)
      solve(XtX, as(X %*% R, "matrix"))
    }
  )
)
