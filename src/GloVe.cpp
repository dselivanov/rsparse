#include "rsparse.h"
using namespace Rcpp;

template <class T>
class GloveFit {
public:
  GloveFit(const List &params) {
    this->vocab_size = as<size_t>(params["vocab_size"]);
    this->word_vec_size = as<size_t>(params["word_vec_size"]);
    this->x_max = as<uint32_t>(params["x_max"]);
    this->learning_rate = as<T>(params["learning_rate"]);
    this->alpha = as<T>(params["alpha"]);
    this->lambda = as<T>(params["lambda"]);

    Rcpp::List init = as<List>(params["initial"]);

    Rcpp::NumericMatrix w_i_init = as<NumericMatrix>(init["w_i"]);
    Rcpp::NumericMatrix w_j_init = as<NumericMatrix>(init["w_j"]);
    Rcpp::NumericVector b_i_init = as<NumericVector>(init["b_i"]);
    Rcpp::NumericVector b_j_init = as<NumericVector>(init["b_j"]);

    // Rcpp::IntegerMatrix w_i_init = as<IntegerMatrix>(init["w_i"]);
    // Rcpp::IntegerMatrix w_j_init = as<IntegerMatrix>(init["w_j"]);
    // Rcpp::IntegerVector b_i_init = as<IntegerVector>(init["b_i"]);
    // Rcpp::IntegerVector b_j_init = as<IntegerVector>(init["b_j"]);

    T * w_i_ptr = reinterpret_cast<T *>(&w_i_init[0]);
    T * w_j_ptr = reinterpret_cast<T *>(&w_j_init[0]);
    T * b_i_ptr = reinterpret_cast<T *>(&b_i_init[0]);
    T * b_j_ptr = reinterpret_cast<T *>(&b_j_init[0]);

    this->w_i = arma::Mat<T>(w_i_ptr, w_i_init.nrow(), w_i_init.ncol(), false, false);
    this->w_j = arma::Mat<T>(w_j_ptr, w_j_init.nrow(), w_j_init.ncol(), false, false);

    this->b_i = arma::Col<T>(b_i_ptr, b_i_init.size(), false, false);
    this->b_j = arma::Col<T>(b_j_ptr, b_j_init.size(), false, false);

    this->grad_sq_b_i = arma::Col<T>(vocab_size, arma::fill::ones);
    this->grad_sq_b_j = arma::Col<T>(vocab_size, arma::fill::ones);

    this->grad_sq_w_i = arma::Mat<T>(w_i_init.nrow(), w_i_init.ncol() , arma::fill::ones);
    this->grad_sq_w_j = arma::Mat<T>(w_j_init.nrow(), w_j_init.ncol() , arma::fill::ones);
  }
  static inline int is_odd(size_t ind) { return ind & 1; }

  inline T weighting_fun(T x, T x_max, T alpha) {
    if(x < x_max)
      return pow(x / x_max, alpha);
    else
      return 1.0;
  }

  T partial_fit(const arma::Col<int> &x_irow,
                const arma::Col<int> &x_icol,
                const arma::Col<double> &x_val,
                const arma::Col<int> &iter_order,
                int n_threads);
private:
  size_t vocab_size, word_vec_size;
  uint32_t x_max;
  T learning_rate;
  // initial learning rate
  T alpha;
  // word vecrtors
  arma::Mat<T> w_i, w_j;
  // vector<vector<float> > w_i, w_j;
  // word biases
  arma::Col<T> b_i, b_j;
  // word vectors square gradients
  arma::Mat<T> grad_sq_w_i, grad_sq_w_j;
  // word biases square gradients
  arma::Col<T> grad_sq_b_i, grad_sq_b_j;

  // variables used if we will repform fit with L1 regularization
  int FLAG_DO_L1_REGURARIZATION;
  T lambda;
  arma::Col<T> time;
  arma::Mat<T> u_w_i;
};

template<typename T>
T GloveFit<T>::partial_fit(
            const arma::Col<int> &x_irow,
            const arma::Col<int> &x_icol,
            const arma::Col<double> &x_val,
            const arma::Col<int> &iter_order,
            int n_threads) {
  size_t nnz = x_irow.size();
  T global_cost = 0.0;

  int flag_do_shuffle = 0;

  if ( iter_order.size() == x_irow.size() )
    flag_do_shuffle = 1;

  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) reduction(+:global_cost)
  #endif
  for (size_t i = 0; i < nnz; i++) {

    T weight, cost_inner, cost, grad_b_i, grad_b_j;
    uint32_t x_irow_i, x_icol_i, i_iter_order;

    if ( flag_do_shuffle )
      // subtract 1 here, because we expecte 1-based indices from R
      // sample.int() returns 1-based shuffling indices
      i_iter_order = iter_order [ i ] - 1;
    else
      i_iter_order = i;
    x_irow_i = x_irow[ i_iter_order ];
    x_icol_i = x_icol[ i_iter_order ];
    weight = weighting_fun(x_val[ i_iter_order ], x_max, this->alpha);

    auto w_j_f = this->w_j.col(x_icol_i);
    auto w_i_f = this->w_i.col(x_irow_i);
    // arma::Col<float> w_j_f = arma::conv_to<arma::fvec>::from(this->w_j.col(x_icol_i));
    // arma::Col<float> w_i_f = arma::conv_to<arma::fvec>::from(this->w_i.col(x_irow_i));

    cost_inner = arma::dot(w_i_f, w_j_f) +
        this->b_i[ x_irow_i ] +
        this->b_j[ x_icol_i ] -
        log( x_val[ i_iter_order ] );

    //clip cost for numerical stability
    if (cost_inner > CLIP_VALUE)
      cost_inner = CLIP_VALUE;
    else if (cost_inner < -CLIP_VALUE)
      cost_inner = -CLIP_VALUE;

    cost = weight * cost_inner;

    // add cost^2
    global_cost += cost * cost_inner;

    //Compute gradients for bias terms
    grad_b_i = cost;
    grad_b_j = cost;

    arma::Col<T> grad_w_i = cost * w_j_f;
    arma::Col<T> grad_w_j = cost * w_i_f;

    // Perform adaptive updates for word vectors
    // main
    this->w_i.col(x_irow_i) -= learning_rate * grad_w_i / sqrt( this->grad_sq_w_i.col(x_irow_i) );
    // context
    this->w_j.col(x_icol_i) -= learning_rate * grad_w_j / sqrt( this->grad_sq_w_j.col(x_icol_i)) ;

    // Update squared gradient sums for word vectors
    // main
    this->grad_sq_w_i.col(x_irow_i) += grad_w_i % grad_w_i;
    // context
    this->grad_sq_w_j.col(x_icol_i) += grad_w_j % grad_w_j;

    // Perform adaptive updates for bias terms

    this->b_i[ x_irow_i ] -= (learning_rate * grad_b_i / sqrt( this->grad_sq_b_i[ x_irow_i ] ));
    this->b_j[ x_icol_i ] -= (learning_rate * grad_b_j / sqrt( this->grad_sq_b_j[ x_icol_i ] ));

    // Update squared gradient sums for biases
    this->grad_sq_b_i[ x_irow_i ] += grad_b_i * grad_b_i;
    this->grad_sq_b_j[ x_icol_i ] += grad_b_j * grad_b_j;
  }
  return 0.5 * global_cost;
}



// [[Rcpp::export]]
SEXP cpp_glove_create(const List &params) {
  GloveFit<double> *glove = new GloveFit<double>(params);
  XPtr< GloveFit<double> > ptr(glove, true);
  return ptr;
}

// [[Rcpp::export]]
double cpp_glove_partial_fit(SEXP ptr,
                             const IntegerVector &x_irow,
                             const IntegerVector &x_icol,
                             const NumericVector &x_val,
                             const IntegerVector &iter_order,
                             int n_threads = 1) {
  XPtr< GloveFit<double> > glove(ptr);
  double res = glove->partial_fit(x_irow, x_icol, x_val, iter_order, n_threads);
  return res;
}

