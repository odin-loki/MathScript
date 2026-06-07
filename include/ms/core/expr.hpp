// MathScript Expression Templates
// Lazy evaluation for matrix expressions

#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"

namespace ms::expr {

// Base CRTP class for all matrix expressions
template<typename Derived>
class MatExpr {
public:
    const Derived& self() const { return static_cast<const Derived&>(*this); }

    size_t rows() const { return self().rows_impl(); }
    size_t cols() const { return self().cols_impl(); }

    // Force evaluation to an owning Matrix
    template<typename S, StorageOrder O, template<typename> class A>
    operator Matrix<S, O, A>() const {
        Matrix<S, O, A> result(rows(), cols());
        for (size_t r = 0; r < rows(); ++r)
            for (size_t c = 0; c < cols(); ++c) {
                result(r, c) = self().eval_at(r, c);
            }
        return result;
    }
};

// Add expression
template<typename L, typename R>
class MatAdd : public MatExpr<MatAdd<L, R>> {
    const MatExpr<L>& l_;
    const MatExpr<R>& r_;
public:
    MatAdd(const MatExpr<L>& l, const MatExpr<R>& r) : l_(l.self()), r_(r.self()) {}
    size_t rows_impl() const { return l_.rows(); }
    size_t cols_impl() const { return l_.cols(); }
    auto   eval_at(size_t r, size_t c) const { return l_.eval_at(r, c) + r_.eval_at(r, c); }
};

// Scale expression
template<typename M, typename S>
class MatScale : public MatExpr<MatScale<M, S>> {
    const MatExpr<M>& m_;
    S s_;
public:
    MatScale(const MatExpr<M>& m, S s) : m_(m.self()), s_(s) {}
    size_t rows_impl() const { return m_.rows(); }
    size_t cols_impl() const { return m_.cols(); }
    auto   eval_at(size_t r, size_t c) const { return m_.eval_at(r, c) * s_; }
};

// Operators
template<typename L, typename R>
auto operator+(const MatExpr<L>& l, const MatExpr<R>& r) {
    return MatAdd<L, R>(l, r);
}

template<typename M, typename S>
auto operator*(const MatExpr<M>& m, S s) {
    return MatScale<M, S>(m, s);
}

} // namespace ms::expr