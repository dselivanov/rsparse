#' @title Base class for matrix factorization recommenders
#' @description All matrix factorization recommenders inherit from this class
#' @keywords internal
MatrixFactorizationRecommender = R6::R6Class(
  classname = "MatrixFactorizationRecommender",
  public = list(
    #' @field components item embeddings
    components = NULL,
    #' @field global_mean global mean (for centering values in explicit feedback)
    global_mean = 0.,
    #' @description recommends items for users
    #' @param x user-item interactions matrix (usually sparse - `Matrix::sparseMatrix`).Users are
    #' rows and items are columns
    #' @param k number of items to recommend
    #' @param not_recommend user-item matrix (sparse) which describes which items method should NOT
    #' recomment for each user. Usually this is same as `x` as we don't want to recommend items user
    #' already liked.
    #' @param items_exclude either integer indices or character identifiers of the items to not
    #' recommend to any user.
    #' @param ... not used at the moment
    predict = function(x, k, not_recommend = x, items_exclude = integer(0), ...) {
      items_exclude = unique(items_exclude)

      if (!(is.character(items_exclude) || is.integer(items_exclude)))
        stop("items_exclude should be one of character/integer")

      stopifnot(private$item_ids == colnames(x))
      stopifnot(is.null(not_recommend) || inherits(not_recommend, "sparseMatrix"))
      user_embeddings = self$transform(x)
      private$predict_internal(user_embeddings, self$components, k, not_recommend, items_exclude)
    }
  ),
  private = list(
    predict_internal = function(user_embeddings, item_embeddings, k, not_recommend, items_exclude = integer(0), ...) {

      logger$trace("MatrixFactorizationRecommender$predict(): calling `RhpcBLASctl::blas_set_num_threads(1)` (to avoid thread contention)")
      n_blas_threads_to_restore = RhpcBLASctl::get_num_cores()
      RhpcBLASctl::blas_set_num_threads(1)
      on.exit({
        logger$trace(
          "MatrixFactorizationRecommender$predict(): on exit blas_set_num_threads(%d)",
          n_blas_threads_to_restore
        )
        RhpcBLASctl::blas_set_num_threads(n_blas_threads_to_restore)
      })

      if (is.character(items_exclude)) {
        if (is.null(private$item_ids))
          stop("model doesn't contain item ids")
        items_exclude = match(items_exclude, private$item_ids)
        items_exclude = items_exclude[!is.na(items_exclude)]
      }
      if (is.integer(items_exclude) && length(items_exclude) > 0) {
        if (max(items_exclude) > ncol(item_embeddings))
          stop("some of items_exclude indices are bigger than number of items")
        logger$trace("found %d items to exclude for all recommendations", length(items_exclude))
      }

      if (!is.null(not_recommend))
        not_recommend = as(not_recommend, "RsparseMatrix")

      uids = rownames(user_embeddings)
      indices = find_top_product(user_embeddings, item_embeddings, k, not_recommend, items_exclude, self$global_mean)

      data.table::setattr(indices, "dimnames", list(uids, NULL))
      data.table::setattr(indices, "ids", NULL)

      if (!is.null(private$item_ids)) {
        predicted_item_ids = private$item_ids[indices]
        data.table::setattr(predicted_item_ids, "dim", dim(indices))
        data.table::setattr(predicted_item_ids, "dimnames", list(uids, NULL))
        data.table::setattr(indices, "ids", predicted_item_ids)
      }
      indices
    },
    get_similar_items = function(item_id, k = ncol(self$components), ... ) {
      stopifnot(is.character(item_id) && length(item_id) == 1)
      if (is.null(private$item_ids)) {
        stop("can't run 'get_similar_items()' - model doesn't have item ids (item_ids = NULL)")
      }
      if (is.null(private$components_l2)) {
        private$components_l2 = private$init_components_l2(...)
      }
      i = which(colnames(private$components_l2) == item_id)
      if (length(i) == 0) {
        stop(sprintf("There is no item with id = '%s' in the model.", item_id))
      }
      query_embedding = private$components_l2[, i]
      # dot-product to find cosine distance
      # both components_l2 and query_embedding should have L2 norm = 1
      # result is matrix with 1 row and n_items components
      # scores = (query_embedding %*% private$components_l2[, -i, drop= FALSE])[1, ]
      scores = (query_embedding %*% private$components_l2)
      dim(scores) = NULL
      # and also remove similarity with itself
      scores = scores[-i]
      ord = order(scores, decreasing = TRUE)
      if (k < length(ord))
        ord = ord[seq_len(k)]
      res = private$item_ids[ord]
      names(scores) = NULL
      attr(res, "scores") = scores[ord]
      res
    },

    item_ids = NULL,
    # prepare components
    init_components_l2 = function(force_init = FALSE) {
      if (is.null(private$components_l2) || force_init) {
        logger$trace("calculating components_l2")
        t(t(private$components_) / sqrt(colSums(private$components_ ^ 2)))
      }
    },
    # L2 normalized components
    components_ = NULL,
    components_l2 = NULL
  )
)
