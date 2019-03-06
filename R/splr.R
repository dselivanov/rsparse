# at the moment borrowed from softImpute package
# setClass("SparsePlusLowRank", representation(x = "sparseMatrix", a = "matrix", b = "matrix"))
# @export

# nocov start
splr = function(x, a = NULL, b = NULL) {
  # x + a %*% t(b)
  dx = dim(x)
  if(is.null(a))
    b = NULL
  if(is.null(b))
    a = NULL
  if(!is.null(a)){
    da = dim(a)
    db = dim(b)
    if(da[1] != dx[1])
      stop("number of rows of x not equal to number of rows of a")
    if(db[1] != dx[2])
      stop("number of columns of x not equal to number of rows of b")
    if(da[2] != db[2])
      stop("number of columns of a not equal to number of columns of b")
  }
  new("SparsePlusLowRank", x = x, a = a, b = b)
}

# @export
t.SparsePlusLowRank = function(x) new("SparsePlusLowRank", x = t(x@x), a = x@b, b = x@a)

splr_crossprod = function(x, y) {
  a = y@a
  b = y@b
  sx = y@x

  if(is.null(a) | is.null(b)) {
    crossprod(x, sx)
  } else {
    part1 = crossprod(x, sx)
    part2 = crossprod(x, a)
    part2 = part2 %*% t(b)
    part1 + part2
  }
}

# y is splr, x is a dense matrix
# setMethod("crossprod", signature(x = "matrix", y = "SparsePlusLowRank"), splr_crossprod)


.leftmult = function(x,y){
  #y is splr, x is matrix
  a=y@a
  b=y@b
  sx=y@x
  if(is.null(a)|is.null(b))x %*% sx
  else{
    part1 = x %*% sx
    part2 = x %*% a
    part2 = part2 %*% t(b)
    part1 + part2
  }
}
.rightmult = function(x,y){
  #x is splr, y is matrix
  a = x@a
  b = x@b
  sx = x@x
  if(is.null(a)|is.null(b))sx%*%y
  else{
    part1 = sx %*% y
    part2 = t(b) %*% y
    part2 = a %*% part2
    part1 + part2
  }
}

# setMethod("%*%", signature(x = "SparsePlusLowRank",y = "Matrix"), .rightmult)
# setMethod("%*%", signature(x = "Matrix", y = "SparsePlusLowRank"), .leftmult)
# setMethod("%*%", signature(x = "SparsePlusLowRank", y = "ANY"), .rightmult)
# setMethod("%*%", signature(x = "ANY", y = "SparsePlusLowRank"), .leftmult)

# setMethod("dim", signature(x = "SparsePlusLowRank"), function(x) dim(x@x), valueClass = "integer")

.rsum  =function(x, ...){
  #x is SparsePlusLowRank matrix
  rx = rowSums(x@x)
  cb = colSums(x@b)
  drop(rx + x@a %*% cb)
}
# setMethod("rowSums", "SparsePlusLowRank", .rsum)

.csum = function(x, ...){
  #x is SparsePlusLowRank matrix
  cx = colSums(x@x)
  ca = colSums(x@a)
  drop( cx + x@b %*% ca)
}
# setMethod("colSums", "SparsePlusLowRank", .csum)

.rmean = function(x, ...){
  #x is SparsePlusLowRank matrix
  rx = rowMeans(x@x)
  cb = colMeans(x@b)
  drop(rx + x@a %*% cb)
}
# setMethod("rowMeans", "SparsePlusLowRank", .rmean)

.cmean = function(x, ...){
  #x is SparsePlusLowRank matrix
  cx = colMeans(x@x)
  ca = colMeans(x@a)
  drop(cx + x@b %*% ca)
}
# setMethod("colMeans", "SparsePlusLowRank", .cmean)

as.matrix.splr = function(x, ...)  as.matrix(x@x) + x@a %*% t(x@b)
# setMethod("as.matrix","SparsePlusLowRank", as.matrix.splr)

# nocov end
