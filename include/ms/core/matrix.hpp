#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {

enum class StorageOrder { ColMajor, RowMajor };

template<typename S, StorageOrder Order = StorageOrder::ColMajor,
         template<typename> class Alloc = std::allocator>
class Matrix {
public:
    using value_type = S;
    using reference = S&;
    using const_reference = const S&;
    using allocator_type = Alloc<S>;

    Matrix() : m_rows_(0), m_cols_(0) {}

    Matrix(size_t rows, size_t cols)
        : m_rows_(rows), m_cols_(cols), m_data_(rows * cols) {}

    Matrix(size_t rows, size_t cols, S init)
        : m_rows_(rows), m_cols_(cols), m_data_(rows * cols, init) {}

    Matrix(std::initializer_list<std::initializer_list<S>> rows)
        : m_rows_(rows.size()), m_cols_(0) {
        if (m_rows_ > 0) {
            m_cols_ = rows.begin()->size();
        }
        m_data_.resize(m_rows_ * m_cols_);
        size_t i = 0;
        for (const auto& row : rows) {
            if (row.size() != m_cols_) {
                throw std::invalid_argument("Matrix initializer rows must have equal length");
            }
            size_t j = 0;
            for (S v : row) {
                m_data_[index(i, j)] = v;
                ++j;
            }
            ++i;
        }
    }

    size_t rows() const { return m_rows_; }
    size_t cols() const { return m_cols_; }
    size_t size() const { return m_data_.size(); }
    bool empty() const { return m_data_.empty(); }

    S* data() { return m_data_.data(); }
    const S* data() const { return m_data_.data(); }

    reference operator()(size_t i, size_t j) {
        return m_data_[index(i, j)];
    }

    const_reference operator()(size_t i, size_t j) const {
        return m_data_[index(i, j)];
    }

    void print() const {
        for (size_t i = 0; i < m_rows_; ++i) {
            std::cout << "[ ";
            for (size_t j = 0; j < m_cols_; ++j) {
                std::cout << (*this)(i, j);
                if (j + 1 < m_cols_) {
                    std::cout << ", ";
                }
            }
            std::cout << " ]";
            if (i + 1 < m_rows_) {
                std::cout << "\n";
            }
        }
        std::cout << std::endl;
    }

private:
    size_t index(size_t i, size_t j) const {
        if constexpr (Order == StorageOrder::RowMajor) {
            return i * m_cols_ + j;
        } else {
            return j * m_rows_ + i;
        }
    }

    size_t m_rows_;
    size_t m_cols_;
    std::vector<S, Alloc<S>> m_data_;
};

template<typename S>
using ColMatrix = Matrix<S, StorageOrder::ColMajor, std::allocator>;

template<typename S>
using RowMatrix = Matrix<S, StorageOrder::RowMajor, std::allocator>;

template<typename S, StorageOrder OA, StorageOrder OB, template<typename> class AllocA, template<typename> class AllocB>
Matrix<S, OA, AllocA> operator+(
    const Matrix<S, OA, AllocA>& A,
    const Matrix<S, OB, AllocB>& B) {
    Matrix<S, OA, AllocA> C(A.rows(), A.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            C(i, j) = A(i, j) + B(i, j);
        }
    }
    return C;
}

template<typename S, StorageOrder OA, StorageOrder OB, template<typename> class AllocA, template<typename> class AllocB>
Matrix<S, OA, AllocA> operator-(
    const Matrix<S, OA, AllocA>& A,
    const Matrix<S, OB, AllocB>& B) {
    Matrix<S, OA, AllocA> C(A.rows(), A.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            C(i, j) = A(i, j) - B(i, j);
        }
    }
    return C;
}

template<typename S, StorageOrder OA = StorageOrder::ColMajor, StorageOrder OB = StorageOrder::ColMajor,
         StorageOrder OC = StorageOrder::ColMajor,
         template<typename> class AllocA = std::allocator,
         template<typename> class AllocB = std::allocator,
         template<typename> class AllocC = std::allocator>
Matrix<S, OC, AllocC> operator*(
    const Matrix<S, OA, AllocA>& A,
    const Matrix<S, OB, AllocB>& B) {
    Matrix<S, OC, AllocC> C(A.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t k = 0; k < A.cols(); ++k) {
            for (size_t j = 0; j < B.cols(); ++j) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return C;
}

template<typename S, StorageOrder Order, template<typename> class Alloc, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
Matrix<S, Order, Alloc> operator*(const Matrix<S, Order, Alloc>& A, Scalar scalar) {
    Matrix<S, Order, Alloc> C(A.rows(), A.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            C(i, j) = A(i, j) * static_cast<S>(scalar);
        }
    }
    return C;
}

template<typename S, StorageOrder Order, template<typename> class Alloc, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
Matrix<S, Order, Alloc> operator*(Scalar scalar, const Matrix<S, Order, Alloc>& A) {
    return A * static_cast<S>(scalar);
}

} // namespace ms
