#include "rsparse.h"

#define ADAGRAD 0
#define RMSPROP 1

#define BPR 0
#define WARP 1

#define IDENTITY 0
#define SIGMOID 1

#define EPS 1e-10

const arma::uword get_random_user(const arma::uword n_user);
const arma::uword get_positive_item_index(const arma::uword max_index);
const arma::uword get_negative_candidate(const arma::uword n_item);

template <typename T>
T positive_item_relevance(const MappedCSR<T>& x, const arma::uword u,
                          const arma::uword i);
template <typename T>
const arma::uvec positive_items(const MappedCSR<T>& x, const arma::uword u);

template <typename T>
T rank_loss(const T x) {
  return std::log1p(x + 1);
}

template <typename T>
T sigmoid(T x) {
  return 1.0 / (1.0 + std::exp(-x));
}

// binary_search - check if element is in the sorted array
// if yes - return true - skip sgd update
bool is_negative(const arma::uvec& arr, arma::uword x) {
  int l = 0;
  int r = arr.size();
  while (l <= r) {
    size_t m = l + (r - l) / 2;

    // Check if x is present at mid
    if (arr[m] == x) return false;

    // If x greater, ignore left half
    if (arr[m] < x) l = m + 1;

    // If x is smaller, ignore right half
    else
      r = m - 1;
  }

  // if we reach here, then element was
  // not present
  return true;
}

inline const arma::uword get_random_user(const arma::uword n_user) {
  return (arma::randi(arma::distr_param(0, n_user - 1)));
}

inline const arma::uword get_negative_candidate(const arma::uword n_item) {
  return (arma::randi(arma::distr_param(0, n_item - 1)));
}

template <typename T>
T positive_item_relevance(const MappedCSR<T>& x, const arma::uword u,
                          const arma::uword i) {
  const arma::uword p1 = x.row_ptrs[u];
  return (x.values[p1 + i]);
}

template <typename T>
const arma::uvec positive_items(const MappedCSR<T>& x, const arma::uword u) {
  const arma::uword p1 = x.row_ptrs[u];
  const arma::uword p2 = x.row_ptrs[u + 1];
  arma::uvec idx = arma::uvec(&x.col_indices[p1], p2 - p1, false, true);
  return (idx);
}

const arma::uword get_positive_item_index(const arma::uword max_index) {
  arma::uword index_random_element = arma::randi(arma::distr_param(0, max_index));
  return (index_random_element);
}

template <typename T>
T get_grad_square_acc(const arma::Col<T>& grad, const T grad_square_acc, arma::uword rank,
                      T gamma, const arma::uword optimizer) {
  T res = grad_square_acc;
  T grad_square =
      arma::dot(grad, grad) / rank;  //  mean of the squared gradient per embedding
  if (optimizer == ADAGRAD) {
    res += grad_square;
  } else {
    if (optimizer == RMSPROP) {
      res = gamma * grad_square_acc + (1 - gamma) * grad_square;
    }
  }
  return (res);
}

template <typename T, typename T2>
void rankmf_solver(const MappedCSR<T2>& x,
                   // embeddings
                   arma::Mat<T>& W,  // user latent factors rank * n_user_features
                   arma::Mat<T>& H,  // item latent factors rank * n_item_features
                   // accumulated squared gradients for Adagrad.RMSprop
                   arma::Col<T>& W2_grad,  // user latent factors n_user_features
                   arma::Col<T>& H2_grad,  // item latent factors n_item_features
                   const MappedCSR<T2>& user_features, const MappedCSR<T2>& item_features,
                   const arma::uword rank, const arma::uword n_updates,
                   const T learning_rate,
                   const T gamma,  // gamma in RMSprop - fraction of squared gradient to
                                   // take from moving average
                   const T lambda_user, const T lambda_item_positive,
                   const T lambda_item_negative, const arma::uword n_threads,
                   bool update_items = true, const arma::uword loss = BPR,
                   const arma::uword kernel = IDENTITY,
                   arma::uword max_negative_samples = 50, const T margin = 0.1,
                   const arma::uword optimizer = 0,  // ADAGRAD
                   const arma::uword report_progress = 10) {
  double TRACK;
  if (report_progress <= 0 || report_progress > 100) {
    TRACK = arma::datum::inf;
  } else {
    TRACK = (n_updates / n_threads / (100 / report_progress));
  }
  const arma::uword n_user = x.n_rows;
  const arma::uword n_item = x.n_cols;
  max_negative_samples = std::min(max_negative_samples, n_item);
  const double WARP_RANK_NORMALIZER = rank_loss(double(n_item));

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
  {
    size_t n_correct = 0, n_positive = 0, n_negative = 0;
#ifdef _OPENMP
#pragma omp for schedule(static, GRAIN_SIZE)
#endif
    for (arma::uword iter = 0; iter < n_updates; ++iter) {
      if (n_positive >= TRACK) {
        if (is_master()) {
          Rcpp::Rcout.precision(3);
          Rcpp::Rcout.width(10);
          Rcpp::Rcout << currentDateTime();
          Rcpp::Rcout.width(10);
          Rcpp::Rcout << 100 * (iter + 1.0) / n_updates << "%";
          Rcpp::Rcout.width(10);
          Rcpp::Rcout << " AUC:" << double(n_correct) / n_positive;
          Rcpp::Rcout.width(10);
          Rcpp::Rcout << " negative_oversampling:" << double(n_negative) / (n_positive)
                      << std::endl;
        }
        n_correct = n_positive = n_negative = 0;
      }

      n_positive++;
      const arma::uword u = get_random_user(n_user);
      const std::pair<arma::uvec, arma::Col<T2>> user_features_nnz =
          user_features.get_row(u);
      const arma::uvec user_features_nnz_indices = user_features_nnz.first;
      // const arma::Col<T> w_u = W.cols(user_features_nnz_indices) *
      // user_features_nnz.second;
      const arma::Col<T> w_u =
          W.cols(user_features_nnz_indices) *
          arma::conv_to<arma::Col<T>>::from(user_features_nnz.second);

      const arma::uvec idx = positive_items(x, u);
      if (idx.is_empty()) continue;  // user with no positive items
      auto id = get_positive_item_index(idx.size() - 1);
      const arma::uword i = idx[id];
      const std::pair<arma::uvec, arma::Col<T2>> item_features_pos =
          item_features.get_row(i);
      const arma::uvec item_features_pos_nnz_indices = item_features_pos.first;
      // const arma::Col<T> h_i = H.cols(item_features_pos_nnz_indices) *
      // item_features_pos.second;
      const arma::Col<T> h_i =
          H.cols(item_features_pos_nnz_indices) *
          arma::conv_to<arma::Col<T>>::from(item_features_pos.second);

      arma::Col<T> h_j;
      std::pair<arma::uvec, arma::Col<T2>> item_features_neg;
      arma::uvec item_features_neg_nnz_indices;

      arma::uword j = 0, k = 0, correct_samples = 0;
      bool skip = true;

      // hj_adjust, hi_adjust = 1 if f(w, h) = arma::dot(w, h)
      // if f(w, h) = SIGMOID(arma::dot(w, h)) they are derivatives of a SIGMOID function
      T r_ui = 1, r_uj = 1, hj_adjust = 1, hi_adjust = 1;
      double weight = 1;
      for (k = 0; k < max_negative_samples; k++) {
        j = get_negative_candidate(n_item);
        // continue if we've sampled a really negative element
        if (is_negative(idx, j)) {
          n_negative++;
          item_features_neg = item_features.get_row(j);
          item_features_neg_nnz_indices = item_features_neg.first;
          // h_j = H.cols(item_features_neg_nnz_indices) * item_features_neg.second;
          h_j = H.cols(item_features_neg_nnz_indices) *
                arma::conv_to<arma::Col<T>>::from(item_features_neg.second);

          r_ui = arma::dot(w_u, h_i);
          r_uj = arma::dot(w_u, h_j);

          if (kernel == SIGMOID) {
            r_ui = sigmoid(r_ui);
            r_uj = sigmoid(r_uj);
            hj_adjust = r_uj * (1 - r_uj);
            hi_adjust = r_ui * (1 - r_ui);
          }

          T distance = r_uj - r_ui;

          if (distance < 0) {
            correct_samples++;
            // for AUC estimation
            if (k == 0) {
              n_correct += 1;
            }
          }
          weight = sigmoid<T>(distance);
          if (loss == BPR) {
            skip = false;
            break;
          } else {  // WAPR loss
            // don't update on easy negatives
            if (distance + margin >= 0) {
              skip = false;
              weight *= rank_loss((double(n_item) - 1.0) / double(k + 1)) /
                        WARP_RANK_NORMALIZER;
              break;
            }
          }
        }
      }

      if (!skip) {
        for (auto i = 0; i < user_features_nnz_indices.size(); i++) {
          auto id = user_features_nnz_indices[i];
          arma::Col<T> grad = weight * (hj_adjust * h_j - hi_adjust * h_i);
          const T grad_square_acc =
              get_grad_square_acc(grad, W2_grad[id], rank, gamma, optimizer);

          W.col(id) -= learning_rate * grad / sqrt(grad_square_acc + EPS);
          if (lambda_user > 0) {
            W.col(id) -= learning_rate * lambda_user * w_u;
          }
          // keep sum of square of gradients per embedding
          W2_grad[id] = grad_square_acc;
        }
        if (update_items) {
          for (auto i = 0; i < item_features_pos_nnz_indices.size(); i++) {
            auto id = item_features_pos_nnz_indices[i];
            arma::Col<T> grad = -weight * hi_adjust * w_u;
            const T grad_square_acc =
                get_grad_square_acc(grad, H2_grad[id], rank, gamma, optimizer);
            H.col(id) -= learning_rate * grad / sqrt(grad_square_acc + EPS);
            if (lambda_item_positive > 0) {
              H.col(id) -= learning_rate * lambda_item_positive * h_i;
            }
            // keep sum of square of gradients per embedding
            H2_grad[id] = grad_square_acc;
          }

          for (auto i = 0; i < item_features_neg_nnz_indices.size(); i++) {
            auto id = item_features_neg_nnz_indices[i];
            arma::Col<T> grad = weight * hj_adjust * w_u;
            const T grad_square_acc =
                get_grad_square_acc(grad, H2_grad[id], rank, gamma, optimizer);
            H.col(id) -= learning_rate * grad / sqrt(grad_square_acc + EPS);
            if (lambda_item_negative > 0) {
              H.col(id) -= learning_rate * lambda_item_negative * h_j;
            }
            // keep sum of square of gradients per embedding
            H2_grad[id] = grad_square_acc;
          }
        }
      }
    }
  }
}

// [[Rcpp::export]]
void rankmf_solver_double(
    const Rcpp::S4& x_r,
    arma::Mat<double>& W,        // user latent factors rank * n_user_features
    arma::Mat<double>& H,        // item latent factors rank * n_item_features
    arma::Col<double>& W2_grad,  // user accumulated squared gradients for Adagrad.RMSprop
    arma::Col<double>& H2_grad,  // item accumulated squared gradients for Adagrad.RMSprop
    const Rcpp::S4& user_features_r, const Rcpp::S4& item_features_r,
    const arma::uword rank, const arma::uword n_updates, double learning_rate = 0.01,
    double gamma = 1, double lambda_user = 0.0, double lambda_item_positive = 0.0,
    double lambda_item_negative = 0.0, const arma::uword n_threads = 1,
    bool update_items = true,
    const arma::uword loss = 0,    // BPR
    const arma::uword kernel = 0,  // IDENTITY
    arma::uword max_negative_samples = 50, double margin = 0.1,
    const arma::uword optimizer = 0,  // ADAGRAD
    const arma::uword report_progress = 10) {
  const dMappedCSR x = extract_mapped_csr(x_r);
  const dMappedCSR user_features = extract_mapped_csr(user_features_r);
  const dMappedCSR item_features = extract_mapped_csr(item_features_r);
  rankmf_solver<double, double>(x, W, H, W2_grad, H2_grad, user_features, item_features,
                                rank, n_updates, learning_rate, gamma, lambda_user,
                                lambda_item_positive, lambda_item_negative, n_threads,
                                update_items, loss, kernel, max_negative_samples, margin,
                                optimizer, report_progress);
}

// [[Rcpp::export]]
void rankmf_solver_float(
    const Rcpp::S4& x_r,
    Rcpp::S4& W,        // user latent factors rank * n_user_features
    Rcpp::S4& H,        // item latent factors rank * n_item_features
    Rcpp::S4& W2_grad,  // user accumulated squared gradients for Adagrad.RMSprop
    Rcpp::S4& H2_grad,  // item accumulated squared gradients for Adagrad.RMSprop
    const Rcpp::S4& user_features_r, const Rcpp::S4& item_features_r,
    const arma::uword rank, const arma::uword n_updates, float learning_rate = 0.01,
    float gamma = 1, float lambda_user = 0.0, float lambda_item_positive = 0.0,
    float lambda_item_negative = 0.0, const arma::uword n_threads = 1,
    bool update_items = true,
    const arma::uword loss = 0,    // BPR
    const arma::uword kernel = 0,  // IDENTITY
    arma::uword max_negative_samples = 50, float margin = 0.1,
    const arma::uword optimizer = 0,  // ADAGRAD
    const arma::uword report_progress = 10) {
  const dMappedCSR x = extract_mapped_csr(x_r);
  const dMappedCSR user_features = extract_mapped_csr(user_features_r);
  const dMappedCSR item_features = extract_mapped_csr(item_features_r);

  Rcpp::IntegerMatrix W_data = W.slot("Data");
  Rcpp::IntegerMatrix H_data = H.slot("Data");
  arma::fmat W_float = arma::fmat(reinterpret_cast<float*>(&W_data[0]), W_data.nrow(),
                                  W_data.ncol(), false, true);
  arma::fmat H_float = arma::fmat(reinterpret_cast<float*>(&H_data[0]), H_data.nrow(),
                                  H_data.ncol(), false, true);

  Rcpp::IntegerVector W2_grad_data = W2_grad.slot("Data");
  Rcpp::IntegerVector H2_grad_data = H2_grad.slot("Data");

  arma::fvec W2_grad_float = arma::fvec(reinterpret_cast<float*>(&W2_grad_data[0]),
                                        W2_grad_data.length(), false, true);
  arma::fvec H2_grad_float = arma::fvec(reinterpret_cast<float*>(&H2_grad_data[0]),
                                        H2_grad_data.length(), false, true);

  rankmf_solver<float, double>(
      x, W_float, H_float, W2_grad_float, H2_grad_float, user_features, item_features,
      rank, n_updates, learning_rate, gamma, lambda_user, lambda_item_positive,
      lambda_item_negative, n_threads, update_items, loss, kernel, max_negative_samples,
      margin, optimizer, report_progress);
}
