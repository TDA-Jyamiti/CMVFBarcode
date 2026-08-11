// reduction_pair.hpp
#pragma once

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "bit_matrix.hpp"

/*
  `reduction_pair` packages two square binary matrices, `R` and `U`, that are edited in lockstep.

  Think of `R` as the matrix currently being reduced, and `U` as the companion
  matrix that receives the same column operations. When constructed from an
  input matrix `D`, we begin with

      R = D
      U = I

  and whenever we add one column into another, we do so in both matrices:

      R.col(j) <- R.col(j) + R.col(i)
      U.col(j) <- U.col(j) + U.col(i)

  Thus, during ordinary reduction, `U` records the accumulated column
  operations applied to the original matrix.

  Convention
  ----------
  Rows are viewed as in an ordinary printed matrix: row 0 is at the top and row
  indices increase downward. Hence `low(j)` means "the lowest 1 in column j",
  i.e. the largest row index containing a 1. For a zero column, `low(j) == -1`.

  The routine `reduce()` performs the standard left-to-right reduction over
  GF(2). Scanning columns from left to right, it repeatedly clears the current
  column's lowest 1 whenever an earlier column already uses that same pivot row:

      if low(j) = p and an earlier column i also has low(i) = p, then do column j += column i

  After reduction, each pivot row is used by at most one column.

  Example reduced matrix
  ----------------------
      columns →      0   1   2   3   4
      rows
        0            .   .   .   .   .
        1            .   1   .   .   .
        2            .   .   .   1   .
        3            .   .   .   .   .
        4            .   .   .   .   .

      low(1) = 1
      low(3) = 2

  The returned `reduction_result` summarizes this reduced form:

    - `low[j]` is the pivot row of column `j`, or -1 if column `j` is zero;
    - `pivot_column[r]` is the unique column whose pivot row is `r`, or `invalid_index` if no column pivots on row `r`;
    - `valid` contains the columns `j` that are zero and are not used as pivot rows by any other column.

  In the example above:

      low          = [-1, 1, -1, 2, -1]
      pivot_column = [invalid_index, 1, 3, invalid_index, invalid_index]
      valid        = {0, 4}

  Besides column reduction, this type also supports simultaneous adjacent row/column swaps and one-step growth/shrink at either boundary.
  These operations keep `R` and `U` synchronized so that higher-level algorithms can edit the ordered basis without rebuilding everything from scratch.
*/

class reduction_pair {
public:
  static constexpr std::size_t invalid_index = static_cast<std::size_t>(-1);

  struct reduction_result { std::vector<int> low; std::vector<std::size_t> pivot_column; std::vector<std::size_t> valid; };

  // Public, non-owning façade for one of the two internal matrices.
  //
  // Client code uses these as
  //
  //   pair.R(row, col)
  //   pair.U(row, col)
  //
  // The view is intentionally narrow: it exposes entry access and the
  // dimension, but not structural operations. Operations that must keep R and U
  // synchronized remain members of reduction_pair.
  //
  // The view stores a pointer, not a reference, so it can be rebound by
  // construction after a reduction_pair move.
  class matrix_view {
  public:
    matrix_view(const matrix_view&) = default;
    matrix_view(matrix_view&&) = default;
    matrix_view& operator=(const matrix_view&) = delete;
    matrix_view& operator=(matrix_view&&) = delete;

    inline std::size_t dimension() const {
      assert(m_);
      return m_->dimension();
    }

    inline bool operator()(std::size_t row, std::size_t col) const {
      assert(m_);
      return std::as_const(*m_)(row, col);
    }

    inline matrix::bit_ref operator()(std::size_t row, std::size_t col) {
      assert(m_);
      return (*m_)(row, col);
    }

  private:
    friend class reduction_pair;
    constexpr explicit matrix_view(matrix& m) noexcept : m_(&m) {}
    matrix* m_ = nullptr;
  };

  // Tiny proxy enabling: `pair.col(dst) += pair.col(src);`
  class column_proxy {
  public:
    inline void operator+=(const column_proxy& src) const {
      assert(owner_);
      assert(src.owner_ == owner_);
      owner_->add(src.j_, j_);
    }

  private:
    friend class reduction_pair;
    constexpr column_proxy(reduction_pair& owner, std::size_t j) noexcept : owner_(&owner), j_(j) {}
    reduction_pair* owner_ = nullptr;
    std::size_t j_ = 0;
  };

private:
  // Owned storage. Client code reaches these through the public views below.
  matrix R_;
  matrix U_;

public:
  // R_ and U_ must be declared before these views so the views bind to live storage.
  matrix_view R{R_};
  matrix_view U{U_};

  reduction_pair() = default;

  explicit reduction_pair(std::size_t n) : R_(n), U_(make_identity_(n)) {}

  explicit reduction_pair(matrix&& D) : R_(std::move(D)), U_(make_identity_(R_.dimension())) {}

  reduction_pair(const reduction_pair&) = delete;
  reduction_pair& operator=(const reduction_pair&) = delete;

  reduction_pair(reduction_pair&& other) noexcept : R_(std::move(other.R_)), U_(std::move(other.U_)) {}

  reduction_pair& operator=(reduction_pair&& other) noexcept {
    if (this == &other) return *this;
    R_ = std::move(other.R_);
    U_ = std::move(other.U_);
    return *this;
  }

  inline std::size_t dimension() const { return R_.dimension(); }
  inline int low(std::size_t j) const { return R_.column_at(j).pivot(); }

  // Return a later column j with low(j) == row, or -1 if none exists.
  // If R is reduced, this column is unique.
  inline int column_with_low(std::size_t row) const {
    for (std::size_t j = row + 1; j < dimension(); ++j) if (low(j) == static_cast<int>(row)) return static_cast<int>(j);
    return -1;
  }

  // Return a lightweight handle naming column `j`, for expressions like
  //
  //   col(dst) += col(src)
  //
  // which performs `dst <- dst + src` in both R and U.
  inline column_proxy col(std::size_t j) { return column_proxy(*this, j); }

  // Perform the column operation
  //
  //   column dst <- column dst + column src
  //
  // in both R and U.
  inline void add(std::size_t source, std::size_t destination) { R_.add_column(source, destination); U_.add_column(source, destination); }

  // Swap adjacent indices `k` and `k+1` on both axes in both matrices.
  inline void swap_adjacent(std::size_t k) { R_.swap_adjacent(k); U_.swap_adjacent(k); }

  // Insert a new shell at the upper-left corner.
  inline void grow_upper_left() {
    R_.grow_upper_left();
    U_.grow_upper_left();
    if (U_.dimension()) U_(0, 0) = true;
  }

  // Insert a new shell at the lower-right corner.
  inline void grow_lower_right() {
    R_.grow_lower_right();
    U_.grow_lower_right();
    if (U_.dimension()) U_(dimension() - 1, dimension() - 1) = true;
  }

  // Remove the first row/column pair from both R and U.
  inline void shrink_upper_left() {
    assert(R_.dimension() && U_.dimension());
    R_.shrink_upper_left();
    U_.shrink_upper_left();
  }

  // Remove the last row/column pair from both R and U.
  inline void shrink_lower_right() {
    assert(R_.dimension() && U_.dimension());
    R_.shrink_lower_right();
    U_.shrink_lower_right();
  }

  // Reduce R from left to right using earlier columns to clear repeated pivot rows.
  //
  // U receives the same column operations, so on return it records the full
  // change of basis applied during the reduction.
  //
  // The returned summary is computed from the reduced form of R.
  reduction_result reduce() {
    reduction_result out;
    out.low.assign(dimension(), -1);
    out.pivot_column.assign(dimension(), invalid_index);

    for (std::size_t j = 0; j < dimension(); ++j) {
      int p = low(j);
      while (p >= 0 && out.pivot_column[static_cast<std::size_t>(p)] != invalid_index) {
        add(out.pivot_column[static_cast<std::size_t>(p)], j);
        p = low(j);
      }

      out.low[j] = p;
      if (p >= 0) out.pivot_column[static_cast<std::size_t>(p)] = j;
    }

    out.valid.clear();
    out.valid.reserve(dimension());
    for (std::size_t j = 0; j < dimension(); ++j) if (out.low[j] == -1 && out.pivot_column[j] == invalid_index) out.valid.push_back(j);
    return out;
  }

private:
  static matrix make_identity_(std::size_t n) {
    matrix I(n);
    for (std::size_t i = 0; i < n; ++i) I(i, i) = true;
    return I;
  }
};
