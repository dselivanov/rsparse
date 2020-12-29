context("WRMF")

logger = lgr::get_logger('rsparse')
logger$set_threshold('warn')

train = movielens100k[1:900, ]
cv = movielens100k[901:nrow(movielens100k), ]

test_that("test WRMF core", {
  p_impl = expand.grid(solver = c("cholesky", "nnls"),
                       feedback = c("implicit"),
                       lambda = c(0, 0.1, 1000),
                       with_user_item_bias = c(TRUE, FALSE),
                       precision = c("double", "float"),
                       stringsAsFactors = FALSE)
  p_impl_2 = expand.grid(solver = c("conjugate_gradient"),
                       feedback = c("implicit"),
                       lambda = c(0, 0.1, 1000),
                       with_user_item_bias = c(FALSE),
                       precision = c("double", "float"),
                       stringsAsFactors = FALSE)
  p_expl = expand.grid(solver = c("conjugate_gradient", "cholesky", "nnls"),
                       feedback = c("explicit"),
                       lambda = c(0.1, 1000),
                       with_user_item_bias = c(TRUE, FALSE),
                       precision = c("double", "float"),
                       stringsAsFactors = FALSE)
  params = rbind(p_impl, p_impl_2, p_expl)
  set.seed(1)
  for(i in 1:nrow(params)) {
    rank = sample(4:10, size = 1)
    K = sample(4:10, size = 1)

    solver = params$solver[[i]]
    feedback = params$feedback[[i]]
    lambda = params$lambda[[i]]
    with_user_item_bias = params$with_user_item_bias[[i]]
    precision = params$precision[[i]]
    rank_with_bias = rank + with_user_item_bias * 2
    fmd = c("testing WRMF with parameters: solver = '%s'",
            "feedback = '%s' lambda = %.3f, rank = %d,",
            "with_bias = %d, precision = %s")
    msg = sprintf(paste(fmd, collapse = " "),
                  solver, feedback, lambda,
                  rank, with_user_item_bias, precision)
    message(msg)
    model = WRMF$new(rank = rank,  lambda = lambda, feedback = feedback, solver = solver,
                     with_user_item_bias = with_user_item_bias, precision = precision)
    user_emb = model$fit_transform(train, n_iter = 5, convergence_tol = -1)

    # check dimensions
    expect_equal(dim(user_emb), c(nrow(train), rank_with_bias))
    expect_equal(rownames(user_emb), rownames(train))
    expect_equal(colnames(model$components), colnames(train))

    # check fit and fit_transform produce same results
    expect_equal(user_emb, model$transform(train))

    preds = model$predict(cv, k = K)
    expect_equal(rownames(preds), rownames(cv))
    expect_equal(dim(preds), c(nrow(cv), K))

    user_emb = model$transform(cv)
    expect_equal(dim(user_emb), c(nrow(cv), rank_with_bias))
    # check embeddings non-negative
    if(solver == "nnls") {
      expect_true(all(user_emb >= 0))
      expect_true(all(model$components >= 0))
    }
  }
})

test_that("test WRMF FLOAT", {
  params = expand.grid(solver = c("conjugate_gradient", "cholesky"),
                       feedback = c("implicit"),
                       lambda = c(0, 1000),
                       stringsAsFactors = FALSE)
  for(i in 1:nrow(params)) {
    rank = 8
    solver = params$solver[[i]]
    feedback = params$feedback[[i]]
    lambda = params$lambda[[i]]
    message(sprintf("testing WRMF FLOAT with parameters: solver = '%s' feedback = '%s' lambda = %.3f, rank = %d",
                    solver, feedback, lambda, rank))
    model = WRMF$new(rank = rank,  lambda = lambda, feedback = feedback, solver = solver, precision = "float")
    user_emb = model$fit_transform(train, n_iter = 5, convergence_tol = -1)
    expect_true(inherits(user_emb, "float32"))
    expect_true(inherits(model$components, "float32"))
  }
}
)
