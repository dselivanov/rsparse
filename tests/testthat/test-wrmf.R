context("WRMF")

futile.logger::flog.threshold(futile.logger::WARN)
train = movielens100k[1:900, , drop = F]
cv = movielens100k[901:nrow(movielens100k), , drop = F]

test_that("test WRMF core", {
  p_impl = expand.grid(solver = c("conjugate_gradient", "cholesky"),
                       feedback = c("implicit"),
                       nnmf = c(TRUE, FALSE),
                       lambda = c(0, 1000),
                       stringsAsFactors = FALSE)
  p_expl = expand.grid(solver = "cholesky",
                       feedback = c("explicit"),
                       lambda = c(0, 1000),
                       nnmf = c(TRUE, FALSE), stringsAsFactors = FALSE)
  params = rbind(p_impl, p_expl)
  set.seed(1)
  for(i in 1:nrow(params)) {
    rank = sample(4:10, size = 1)
    K = sample(4:10, size = 1)

    nnmf = params$nnmf[[i]]
    solver = params$solver[[i]]
    feedback = params$feedback[[i]]
    lambda = params$lambda[[i]]
    message(sprintf("testing WRMF with parameters: nnmf = %d solver = '%s' feedback = '%s' lambda = %.3f, rank = %d",
                    nnmf, solver, feedback, lambda, rank))
    model = WRMF$new(rank = rank,  lambda = lambda, feedback = feedback, non_negative = nnmf, solver = solver)
    user_emb = model$fit_transform(train, n_iter = 5, convergence_tol = -1)
    # check dimensions
    expect_equal(dim(user_emb), c(nrow(train), rank))
    expect_equal(rownames(user_emb), rownames(train))
    fit_trace = attr(user_emb, "trace")
    expect_gte(fit_trace$value[[1]], fit_trace$value[[2]])
    expect_equal(colnames(model$components), colnames(train))

    preds = model$predict(cv, k = K)
    expect_equal(rownames(preds), rownames(cv))
    expect_equal(dim(preds), c(nrow(cv), K))

    user_emb = model$transform(cv)
    expect_equal(dim(user_emb), c(nrow(cv), rank))
    # check embeddings non-negative
    if(nnmf) {
      expect_true(all(user_emb >= 0))
      expect_true(all(model$components >= 0))
    }
  }
})

if(rsparse:::SINGLE_PRECISION_LAPACK_AVAILABLE) {
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
}

test_that("test WRMF extra", {
  lambda = 0.1
  rank = 8
  nnmf = FALSE
  solver = "cholesky"
  n_iter = 10
  cv_split = train_test_split(cv)
  for(feedback in c("explicit", "implicit")) {
    model = WRMF$new(rank = rank,  lambda = lambda, feedback = feedback, non_negative = nnmf, solver = solver)
    user_emb = model$fit_transform(train, n_iter = n_iter, convergence_tol = 0.05)
    fit_trace = attr(user_emb, "trace")
    setDT(fit_trace)
    # should converge for less than 10 iter
    expect_lte(max(fit_trace$iter), n_iter)
  }

})
