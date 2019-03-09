BaseRecommender = R6::R6Class(
  inherit = mlapi::mlapiDecomposition,
  classname = "BaseRecommender",
  public = list(
    predict = function(x, k, not_recommend = x, items_exclude = integer(0), ...) {
      items_exclude = unique(items_exclude)

      if(!(is.character(items_exclude) || is.integer(items_exclude)))
        stop("items_exclude should be one of character/integer")

      stopifnot(private$item_ids == colnames(x))
      stopifnot(is.null(not_recommend) || inherits(not_recommend, "sparseMatrix"))
      m = nrow(x)

      user_embeddings = self$transform(x)
      private$predict_low_level(user_embeddings, private$components_, k, not_recommend, items_exclude)
    }
  ),
  private = list(
    predict_low_level = function(user_embeddings, item_embeddings, k, not_recommend, items_exclude = integer(0), ...) {

      flog.debug("BaseRecommender$predict(): calling `RhpcBLASctl::blas_set_num_threads(1)` (to avoid thread contention)")
      RhpcBLASctl::blas_set_num_threads(1)
      on.exit({
        n_physical_cores = RhpcBLASctl::get_num_cores()
        flog.debug("BaseRecommender$predict(): on exit `RhpcBLASctl::blas_set_num_threads(%d)` (=number of physical cores)", n_physical_cores)
        RhpcBLASctl::blas_set_num_threads(n_physical_cores)
      })

      if(is.character(items_exclude)) {
        if(is.null(private$item_ids))
          stop("model doesn't contain item ids")
        items_exclude = match(items_exclude, private$item_ids)
        items_exclude = items_exclude[!is.na(items_exclude)]
      }
      if(is.integer(items_exclude) && length(items_exclude) > 0) {
        if(max(items_exclude) > ncol(item_embeddings))
          stop("some of items_exclude indices are bigger than number of items")
        flog.debug("found %d items to exclude for all recommendations", length(items_exclude))
      }

      if(!is.null(not_recommend))
        not_recommend = as(not_recommend, "RsparseMatrix")

      uids = rownames(user_embeddings)
      indices = find_top_product(user_embeddings, item_embeddings, k, not_recommend, items_exclude)

      data.table::setattr(indices, "dimnames", list(uids, NULL))
      data.table::setattr(indices, "ids", NULL)

      if(!is.null(private$item_ids)) {
        predicted_item_ids = private$item_ids[indices]
        data.table::setattr(predicted_item_ids, "dim", dim(indices))
        data.table::setattr(predicted_item_ids, "dimnames", list(uids, NULL))
        data.table::setattr(indices, "ids", predicted_item_ids)
      }
      indices
    },
    get_similar_items = function(item_id, k = ncol(self$components), ... ) {
      stopifnot(is.character(item_id) && length(item_id) == 1)
      if(is.null(private$item_ids)) {
        stop("can't run 'get_similar_items()' - model doesn't have item ids (item_ids = NULL)")
      }
      if(is.null(private$components_l2)) {
        private$components_l2 = private$init_components_l2(...)
      }
      i = which(colnames(private$components_l2) == item_id)
      if(length(i) == 0) {
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
      if(k < length(ord))
        ord = ord[seq_len(k)]
      res = private$item_ids[ord]
      names(scores) = NULL
      attr(res, "scores") = scores[ord]
      res
    },

    item_ids = NULL,
    # prepare components
    init_components_l2 = function(force_init = FALSE) {
      if(is.null(private$components_l2) || force_init) {
        flog.debug("calculating components_l2")
        t(t(private$components_) / sqrt(colSums(private$components_ ^ 2)))
      }
    },
    # L2 normalized components
    components_l2 = NULL
  )
)
