# at the moment borrowed from softImpute package
setClass("SparseplusLowRank", representation(x = "sparseMatrix", a = "matrix", b = "matrix"))

#' @export
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
  new("SparseplusLowRank", x = x, a = a, b = b)
}

setMethod("crossprod", signature(x = "matrix", y = "SparseplusLowRank"),
          # y is splr, x is a dense matrix
          function(x, y) {
            {
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
          })


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

setMethod("%*%", signature(x = "SparseplusLowRank",y = "Matrix"), .rightmult)
setMethod("%*%", signature(x = "Matrix", y = "SparseplusLowRank"), .leftmult)
setMethod("%*%", signature(x = "SparseplusLowRank", y = "ANY"), .rightmult)
setMethod("%*%", signature(x = "ANY", y = "SparseplusLowRank"), .leftmult)

setMethod("dim", signature(x = "SparseplusLowRank"),
          function(x) dim(x@x), valueClass = "integer")

.rsum  =function(x, ...){
  #x is SparseplusLowRank matrix
  rx = rowSums(x@x)
  cb = colSums(x@b)
  drop(rx + x@a %*% cb)
}
setMethod("rowSums", "SparseplusLowRank", .rsum)

.csum = function(x, ...){
  #x is SparseplusLowRank matrix
  cx = colSums(x@x)
  ca = colSums(x@a)
  drop( cx + x@b %*% ca)
}
setMethod("colSums", "SparseplusLowRank", .csum)

.rmean = function(x, ...){
  #x is SparseplusLowRank matrix
  rx = rowMeans(x@x)
  cb = colMeans(x@b)
  drop(rx + x@a %*% cb)
}
setMethod("rowMeans", "SparseplusLowRank", .rmean)

.cmean = function(x, ...){
  #x is SparseplusLowRank matrix
  cx = colMeans(x@x)
  ca = colMeans(x@a)
  drop(cx + x@b %*% ca)
}
setMethod("colMeans", "SparseplusLowRank", .cmean)

as.matrix.splr = function(x, ...)  as.matrix(x@x) + x@a %*% t(x@b)
setMethod("as.matrix","SparseplusLowRank", as.matrix.splr)
