#include <cmath>
#include <cstddef>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
  /**
   * A C++ version of the softmax regression epoch code.  This should run a
   * single epoch over the data defined by X and y (and sizes m,n,k), and
   * modify theta in place.  Your function will probably want to allocate
   * (and then delete) some helper arrays to store the logits and gradients.
   *
   * Args:
   *     X (const float *): pointer to X data, of size m*n, stored in row
   *          major (C) format
   *     y (const unsigned char *): pointer to y data, of size m
   *     theta (float *): pointer to theta data, of size n*k, stored in row
   *          major (C) format
   *     m (size_t): number of examples
   *     n (size_t): input dimension
   *     k (size_t): number of classes
   *     lr (float): learning rate / SGD step size
   *     batch (int): SGD minibatch size
   *
   * Returns:
   *     (None)
   */

  /// BEGIN YOUR CODE
  for (size_t start = 0; start < m; start += batch) {
    size_t end = std::min(start + batch, m);
    size_t bsize = end - start;

    // 1. Z = X_batch @ theta，(bsize, k) = (bsize, n) * (n, k)
    std::vector<float> Z(bsize * k, 0.0f);
    for (size_t i = 0; i < bsize; ++i) {
      for (size_t j = 0; j < k; ++j) {
        float sum = 0.0f;
        for (size_t d = 0; d < n; ++d) {
          sum += X[(start + i) * n + d] * theta[d * k + j];
        }
        Z[i * k + j] = sum;
      }
    }

    // 2. Compute softmax w/o stable numerics
    for (size_t i = 0; i < bsize; ++i) {
      float sum = 0.0;
      for (size_t j = 0; j < k; ++j) {
        sum += std::exp(Z[i * k + j]);
      }
      for (size_t j = 0; j < k; ++j) {
        Z[i * k + j] = std::exp(Z[i * k + j]) / sum;
      }
    }

    // 3. dtheta = X_batch^T @ (Z - I_y) / bsize,
    // (n, k) = (bsize, n)^T * (bsize, k)
    for (size_t i = 0; i < bsize; ++i) {
      Z[i * k + y[start + i]] -= 1.0f; // essentially Z - I_y
    }
    for (size_t d = 0; d < n; ++d) {
      for (size_t j = 0; j < k; ++j) {
        float grad = 0.0f;
        for (size_t i = 0; i < bsize; ++i) {
          grad += X[(start + i) * n + d] * Z[i * k + j];
        }
        theta[d * k + j] -= lr * grad / bsize;
      }
    }
  }
  /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
  m.def(
      "softmax_regression_epoch_cpp",
      [](py::array_t<float, py::array::c_style> X,
         py::array_t<unsigned char, py::array::c_style> y,
         py::array_t<float, py::array::c_style> theta, float lr, int batch) {
        softmax_regression_epoch_cpp(
            static_cast<const float *>(X.request().ptr),
            static_cast<const unsigned char *>(y.request().ptr),
            static_cast<float *>(theta.request().ptr), X.request().shape[0],
            X.request().shape[1], theta.request().shape[1], lr, batch);
      },
      py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"),
      py::arg("batch"));
}
