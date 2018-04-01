#include "rsparse.h"

inline float clip(float x) {
  float sign = x < 0.0 ? -1.0:1.0;
  return(fabs(x) < CLIP_VALUE ? x : sign * CLIP_VALUE);
}

class FMParam {
public:
  FMParam();
  FMParam(float learning_rate_w,
          float learning_rate_v,
          int rank,
          float lambda_w, float lambda_v,
          std::string task_name,
          int intercept):
    learning_rate_w(learning_rate_w),
    learning_rate_v(learning_rate_v),
    rank(rank),
    lambda_w(lambda_w),
    lambda_v(lambda_v),
    intercept(intercept) {
    if ( task_name == "classification")
      this->task = CLASSIFICATION;
    else if( task_name == "regression")
      this->task = REGRESSION;
    else throw(Rcpp::exception("can't match task code - not in (1=CLASSIFICATION, 2=REGRESSION)"));
  }
  int task = 0;
  float learning_rate_w;
  float learning_rate_v;

  int n_features;
  int rank;

  float lambda_w;
  float lambda_v;
  int intercept = 1;

  arma::fvec w0;
  arma::fvec w;
  arma::fvec grad_w2;

  arma::fmat v;
  arma::fmat grad_v2;

  float link_function(float x) {
    if(this->task == CLASSIFICATION)
      return(1.0 / ( 1.0 + exp(-x)));
    if(this->task == REGRESSION)
      return(x);
    return(x);
    throw(Rcpp::exception("no link function"));
  }
  float loss(float pred, float actual) {

    if(this->task == CLASSIFICATION)
      return(-log( this->link_function(pred * actual) ));

    if(this->task == REGRESSION)
      return((pred - actual) * (pred - actual));

    return(-log( this->link_function(pred * actual) ));
    throw(Rcpp::exception("no loss function"));
  }
  void init_weights(Rcpp::IntegerVector &w0_R, Rcpp::IntegerVector &w_R, Rcpp::IntegerMatrix &v_R,
                    Rcpp::IntegerVector &grad_w2_R, Rcpp::IntegerMatrix &grad_v2_R) {
    this->w0 = arma::fvec((float *)w0_R.begin(), 1, false, false);
    // number of features equal to number of input weights
    this->n_features = w_R.size();

    this->v = arma::fmat((float *)v_R.begin(), v_R.nrow(), v_R.ncol(), false, false);
    this->grad_v2 = arma::fmat((float *)grad_v2_R.begin(), grad_v2_R.nrow(), grad_v2_R.ncol(), false, false);

    this->w = arma::fvec((float *)w_R.begin(), w_R.size(), false, false);
    this->grad_w2 = arma::fvec((float *)grad_w2_R.begin(), grad_w2_R.size(), false, false);
  }
};

class FMModel {
public:
  FMModel(FMParam *params): params(params) {};
  FMParam *params;

  float fm_predict_internal(const uint32_t *nnz_index, const double *nnz_value, int offset_start, int offset_end) {
    float res = this->params->w0[0];
    // add linear terms
    #ifdef _OPENMP
    #pragma omp simd
    #endif
    for(int j = offset_start; j < offset_end; j++) {
      uint32_t feature_index = nnz_index[j];
      res += this->params->w[feature_index] * (float)nnz_value[j];
    }
    float res_pair_interactions = 0.0;
    // add interactions
    for(int f = 0; f < this->params->rank; f++) {
      float s1 = 0.0;
      float s2 = 0.0;
      float prod;
      arma::subview_row<float> vf = this->params->v.row(f);
      #ifdef _OPENMP
      #pragma omp simd
      #endif
      for(int j = offset_start; j < offset_end; j++) {
        int feature_index = nnz_index[j];
        prod = vf[feature_index] * nnz_value[j];
        s1  += prod;
        s2  += prod * prod;
      }
      res_pair_interactions += s1 * s1 - s2;
    }
    return(res + 0.5 * res_pair_interactions);
  }

  Rcpp::NumericVector fit_predict(const Rcpp::S4 &m,
                                  const Rcpp::NumericVector &y_R,
                                  const Rcpp::NumericVector &w_R,
                                  int n_threads = 1,
                                  int do_update = 1) {
    const double *y = y_R.begin();
    const double *w = w_R.begin();

    dMappedCSR x = extract_mapped_csr(m);
    Rcpp::NumericVector y_hat_R(x.n_rows);

    // get pointers to not touch R API
    double *y_hat = y_hat_R.begin();

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) schedule(guided, 1000)
    #endif
    for(uint32_t i = 0; i < x.n_rows; i++) {
      uint32_t p1 = x.row_ptrs[i];
      uint32_t p2 = x.row_ptrs[i + 1];
      float y_hat_raw = this->fm_predict_internal(x.col_indices, x.values, p1, p2);
      // prediction
      y_hat[i] = this->params->link_function(y_hat_raw);
      // fitting
      if(do_update) {
        //------------------------------------------------------------------
        // first part of d_L/d_theta -  intependent of parameters theta
        float dL;
        if(this->params->task == CLASSIFICATION)
          dL = (this->params->link_function(y_hat_raw * y[i]) - 1) * y[i];
        else if(this->params->task == REGRESSION )
          dL = 2 * (y_hat_raw - y[i]);
        else
          throw(Rcpp::exception("task not defined in FMModel::fit_predict()"));
        // mult by error-weight of the sample
        dL *= w[i];
        //------------------------------------------------------------------
        // update w0
        if(this->params->intercept)
          this->params->w0 -= this->params->learning_rate_w * dL;
        for( uint32_t p = p1; p < p2; p++) {
          uint32_t feature_index  = x.col_indices[p];
          float feature_value = x.values[p];

          float grad_w = clip(feature_value * dL + 2 * this->params->lambda_w);

          this->params->w[feature_index] -= this->params->learning_rate_w * grad_w / sqrt(this->params->grad_w2[feature_index]);
          // update sum gradient squre
          this->params->grad_w2[feature_index] += grad_w * grad_w;

          // pairwise interactions
          //------------------------------------------------------------------------
          arma::fvec grad_v_k(-this->params->v.col(feature_index) * feature_value);
          for(uint32_t k = 0; k < p2 - p1; k++) {
            uint32_t index = x.col_indices[p1 + k];
            float val = x.values[p1 + k];
            // same as
            // grad_v_k += this->params->v.col(index) * val;
            // but faster - not sure why not vectorized
            float *v_ptr = this->params->v.colptr(index);
            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for(int f = 0; f < this->params->rank; f++)
              grad_v_k[f] += v_ptr[f] * val;
          }

          arma::fvec grad_v = dL * (feature_value * grad_v_k) + 2 * this->params->v.col(feature_index) * this->params->lambda_v;

          #ifdef _OPENMP
          #pragma omp simd
          #endif
          for(size_t i = 0; i < grad_v.size(); i++) grad_v[i] = clip(grad_v[i]);

          this->params->v.col(feature_index) -= this->params->learning_rate_v * grad_v / sqrt(this->params->grad_v2.col(feature_index));
          this->params->grad_v2.col(feature_index) += grad_v % grad_v;
        }
      }
    }
    return(y_hat_R);
  }
};

// [[Rcpp::export]]
SEXP fm_create_param(float learning_rate_w,
                     float learning_rate_v,
                     int rank,
                     float lambda_w,
                     float lambda_v,
                     Rcpp::IntegerVector &w0_R,
                     Rcpp::IntegerVector &w_R,
                     Rcpp::IntegerMatrix &v_R,
                     Rcpp::IntegerVector &grad_w2_R,
                     Rcpp::IntegerMatrix &grad_v2_R,
                     const Rcpp::String task,
                     int intercept) {
  FMParam * param = new FMParam(learning_rate_w, learning_rate_v, rank, lambda_w, lambda_v, task, intercept);
  param->init_weights(w0_R,  w_R, v_R, grad_w2_R, grad_v2_R);
  Rcpp::XPtr< FMParam> ptr(param, true);
  return ptr;
}

// [[Rcpp::export]]
SEXP fm_create_model(SEXP params_ptr) {
  Rcpp::XPtr<FMParam> params(params_ptr);
  FMModel *model = new FMModel(params);
  Rcpp::XPtr< FMModel> model_ptr(model, true);
  return model_ptr;
}

// [[Rcpp::export]]
void fill_float_matrix_randn(Rcpp::IntegerMatrix &x, double stdev = 0.001) {
  arma::fmat res = arma::fmat((float *)x.begin(), x.nrow(), x.ncol(), false, false);
  res.randn();
  res *= stdev;
}

// [[Rcpp::export]]
void fill_float_matrix(Rcpp::IntegerMatrix &x, double val) {
  arma::fmat res = arma::fmat((float *)x.begin(), x.nrow(), x.ncol(), false, false);
  res.fill(float(val));
}

// [[Rcpp::export]]
void fill_float_vector_randn(Rcpp::IntegerVector &x, double stdev = 0.001) {
  arma::fvec res = arma::fvec((float *)x.begin(), x.size(), false, false);
  res.randn();
  res *= stdev;
}

// [[Rcpp::export]]
void fill_float_vector(Rcpp::IntegerVector &x, double val) {
  arma::fvec res = arma::fvec((float *)x.begin(), x.size(), false, false);
  res.fill(float(val));
}

// [[Rcpp::export]]
Rcpp::NumericVector fm_partial_fit(SEXP ptr, const Rcpp::S4 &X,
                             const Rcpp::NumericVector &y,
                             const Rcpp::NumericVector &w,
                             int n_threads = 1,
                             int do_update = 1) {
  Rcpp::XPtr<FMModel> model(ptr);
  return(model->fit_predict(X, y, w, n_threads, do_update));
}

// checks if external pointer invalid
// [[Rcpp::export]]
int is_invalid_ptr(SEXP sexp_ptr) {
  Rcpp::XPtr<SEXP> ptr(sexp_ptr);
  return (ptr.get() == NULL);
}
