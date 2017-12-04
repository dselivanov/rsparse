BaseRecommender = R6::R6Class(
  inherit = mlapi::mlapiDecomposition,
  classname = "BaseRecommender",
  public = list(
    n_threads = NULL,
    predict = function(x, k, not_recommend = x, ...) {
      stopifnot(private$item_ids == colnames(x))
      stopifnot(is.null(not_recommend) || inherits(not_recommend, "sparseMatrix"))
      if(!is.null(not_recommend))
        not_recommend = as(not_recommend, "RsparseMatrix")
      m = nrow(x)

      user_embeddings = self$transform(x)
      private$predict_low_level(user_embeddings, private$components_, k, not_recommend)
    }
  ),
  private = list(
    predict_low_level = function(user_embeddings, item_embeddings, k, not_recommend, ...) {

      if(isTRUE(self$n_threads > 1)) {
        flog.debug("BaseRecommender$predict(): calling `RhpcBLASctl::blas_set_num_threads(1)` (to avoid thread contention)")
        RhpcBLASctl::blas_set_num_threads(1)
        on.exit({
          n_physical_cores = RhpcBLASctl::get_num_cores()
          flog.debug("BaseRecommender$predict(): on exit `RhpcBLASctl::blas_set_num_threads(%d)` (=number of physical cores)", n_physical_cores)
          RhpcBLASctl::blas_set_num_threads(n_physical_cores)
        })
      }


      uids = rownames(user_embeddings)
      indices = find_top_product(user_embeddings, item_embeddings, k, self$n_threads, not_recommend)

      data.table::setattr(indices, "dimnames", list(uids, NULL))
      data.table::setattr(indices, "indices", NULL)

      if(!is.null(private$item_ids)) {
        predicted_item_ids = private$item_ids[indices]
        data.table::setattr(predicted_item_ids, "dim", dim(indices))
        data.table::setattr(predicted_item_ids, "dimnames", list(uids, NULL))
        data.table::setattr(indices, "indices", predicted_item_ids)
      }
      indices
    },
    item_ids = NULL
  )
)
