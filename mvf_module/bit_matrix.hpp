// bit_matrix.hpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>
#include <cassert>
#include <memory>
#include <algorithm>
#include <bit>

/*  ____  ____________  ______  __________  _____  __
   / __ )/  _/_  __/  |/  /   |/_  __/ __ \/  _/ |/ /
  / __  |/ /  / / / /|_/ / /| | / / / /_/ // / |   / 
 / /_/ // /  / / / /  / / ___ |/ / / _, _// / /   |  
/_____/___/ /_/ /_/  /_/_/  |_/_/ /_/ |_/___//_/|_|  

  HOW IT WORKS
  ------------
  The matrix is stored column-wise in packed 64-bit words. Every column shares
  a common column_layout describing the current logical row range and its
  placement inside the allocated word storage.

  For a given layout:
    - logical_size is the number of logical rows,
    - bit_offset is the bit position of logical row 0 in the first active word,
    - word_count is the number of active words,
    - word_capacity is the allocated number of words per column,
    - word_offset is the starting index of the active word window.

  Hence logical row r is stored at physical bit index bit_offset + r within
  the active word window beginning at word_offset.

  Each column maintains a cached pivot_row, equal to the largest logical row
  containing a 1. For a zero-column pivot_row == -1.

  The matrix stores its columns in an overallocated circular buffer, allowing
  insertion/removal at the left or right edge without shifting all columns.

  In short, the following diagrams communicate the structure:

  => One column's word storage (each box is one 64-bit word)

    +-----------+-----------+-----+-----------+-----------+-----------+-----+-----------+-----------+-----+-----------+
    |  unused   |  unused   | ... |  unused   |  active   |  active   | ... |  active   |  unused   | ... |  unused   |
    +-----------+-----------+-----+-----------+-----------+-----------+-----+-----------+-----------+-----+-----------+
    ^           ^                                   ^                                   ^
    0           1                              word_offset                   word_offset + word_count - 1

  => First active word in a column

    higher bit indices
    +-----------------------------------------------+
    | active bits                                   |
    |                                               |
    |                                               |
    | ...                                           |
    | row 2                                         |
    | row 1                                         |
    | row 0   stored at bit index bit_offset        |
    +-----------------------------------------------+
    | inactive bits                                 |
    | bit indices bit_offset - 1 down to 0          |
    +-----------------------------------------------+
    lower bit indices

  => The columns vector, used as a circular buffer (each box is one column object)

    +-----------+-----------+-----+-----------+-----------+-----------+-----+-----------+-----------+-----+-----------+
    |   col a   | col a+1   | ... | col N-1   |  unused   |  unused   | ... |   col 0   |   col 1   | ... | col a-1   |
    +-----------+-----------+-----+-----------+-----------+-----------+-----+-----------+-----------+-----+-----------+
    ^                                                                       ^
    0                                                                       col_offset

    col 0, col 1, ..., col column_count - 1 are stored starting at col_offset,
    wrapping around the end of the vector if necessary.


  EXTERNAL INTERFACE
  ------------------
  The type supports the operations needed by reduction-style algorithms on
  square binary matrices:

    - construction of an n x n zero matrix,
    - entry access by (row, column),
    - column addition over GF(2),
    - simultaneous adjacent row/column swap,
    - growth/shrink by one row/column pair at either boundary,
    - direct access to individual columns when lower-level control is needed.

  This is an internal data structure specialized for mutation-heavy binary
  matrix algorithms, not a general-purpose matrix API.
*/

static inline int highest_set_bit_index(std::uint64_t word) {
  assert(word != 0);
  return static_cast<int>(std::bit_width(word) - 1);
}

static inline std::size_t words_for_bits(unsigned bit_offset, std::size_t logical_size) {
  return logical_size ? ((static_cast<std::size_t>(bit_offset) + logical_size + 63) >> 6) : 0;
}

struct column_layout {
  std::size_t word_offset = 0;
  std::size_t word_count = 0;
  std::size_t word_capacity = 0;
  std::size_t logical_size = 0;
  unsigned bit_offset = 0;
};

struct column {

  // Small-buffer optimization: columns with at most this many words keep
  // their storage inline and avoid a heap allocation.
  static constexpr std::size_t small_word_capacity = 4;

  // Inline word storage used when the active layout fits in small_word_capacity
  // words. For larger layouts, storage is heap-allocated instead.
  std::uint64_t small_words[small_word_capacity];

  // Pointer to the current word buffer. This is either:
  //   - small_words, or
  //   - a heap allocation of layout.word_capacity words.
  std::uint64_t* word_data = small_words;

  // Shared row-layout metadata owned by the enclosing matrix.
  // All live columns in the same matrix point at the same layout object.
  const column_layout* layout_ptr = nullptr;

  // Cached pivot row: the lowest logical row containing a 1.
  // For the zero column, pivot_row == -1.
  int pivot_row = -1;

  // Default-constructed column is detached: no layout, inline storage,
  // and pivot_row == -1.
  column() = default;

  // Copying is disabled. A column owns storage and is intended to move
  // together with its matrix-managed layout.
  column(const column&) = delete;
  column& operator=(const column&) = delete;

  // Move construction delegates to move assignment. This is safe because the
  // default member initializers have already put *this into detached inline
  // storage before the constructor body runs.
  column(column&& other) noexcept { *this = std::move(other); }

  // Move assignment.
  //
  // Cases:
  //   - if `other` uses inline storage, copy `other.small_words` into this
  //     object's inline buffer;
  //   - if `other` uses heap storage, steal the heap pointer and reset
  //     `other.word_data` to `other.small_words`.
  //
  // In either case, transfer `layout_ptr` and `pivot_row`, and leave `other`
  // detached and safely destructible.
  column& operator=(column&& other) noexcept {
    if (this == &other) return *this;
    if (word_data != small_words) delete[] word_data;

    const bool other_inline = (other.word_data == other.small_words);
    word_data = other_inline ? small_words : std::exchange(other.word_data, other.small_words);
    if (other_inline) std::memcpy(small_words, other.small_words, sizeof(small_words));

    layout_ptr = std::exchange(other.layout_ptr, nullptr);
    pivot_row = std::exchange(other.pivot_row, -1);
    return *this;
  }

  // Destroy the column, releasing heap storage if this column is not using
  // its inline buffer.
  ~column() { if (word_data != small_words) delete[] word_data; }

  // Construct a zero column for layout.
  explicit column(const column_layout* layout) {
    attach_layout(layout);
    allocate_for_layout();
    clear_active_window();
  }

  // Set the shared layout pointer.
  inline void attach_layout(const column_layout* layout) { layout_ptr = layout; }

  // Return the logical row count.
  inline std::size_t size() const { assert(layout_ptr); return layout_ptr->logical_size; }

  // Return the bit at logical row `row`.
  //
  // The logical row is translated to a physical bit position using the shared
  // layout: physical_bit = layout.bit_offset + row
  inline bool get(std::size_t row) const {
    assert(layout_ptr);
    const column_layout& layout = *layout_ptr;
    assert(row < layout.logical_size);

    [[maybe_unused]] const std::size_t physical_bit = static_cast<std::size_t>(layout.bit_offset) + row;
    return (word_data[layout.word_offset + (physical_bit >> 6)] >> (physical_bit & 63)) & 1;
  }

  // Set the bit at logical row `row`, maintaining the cached pivot.
  //
  // If a 1 is inserted below the current pivot, pivot_row is updated directly.
  // If the current pivot bit is cleared, the column rescans upward to find
  // the next lowest 1. Clearing any non-pivot bit leaves pivot_row unchanged.
  inline void set(std::size_t row, bool value) {
    assert(layout_ptr);
    const column_layout& layout = *layout_ptr;
    assert(row < layout.logical_size);

    [[maybe_unused]] const std::size_t physical_bit = static_cast<std::size_t>(layout.bit_offset) + row;
    std::uint64_t& word = word_data[layout.word_offset + (physical_bit >> 6)];
    const std::uint64_t mask = std::uint64_t(1) << (physical_bit & 63);

    if (((word & mask) != 0) == value) return;
    word = value ? (word | mask) : (word & ~mask);

    if (value) {
      if (pivot_row < static_cast<int>(row)) pivot_row = static_cast<int>(row);
      return;
    }

    if (pivot_row == static_cast<int>(row)) pivot_row = row ? refresh_pivot_up_to(row - 1) : -1;
  }

  // Return the cached pivot row, or -1 if the column is zero.
  inline int pivot() const { return pivot_row; }

  // Add `other` into this column over GF(2), updating the cached pivot.
  //
  // Method:
  //   1. XOR the active word window word-by-word.
  //   2. If the two columns had different pivots, the lower of those two pivot
  //      rows cannot cancel, so the new pivot is the lower one; numerically,
  //      this is max(self_pivot, other_pivot).
  //   3. If they had the same pivot, that pivot bit cancels, so rescan upward
  //      from the row just above it to find the new lowest 1.
  inline void xor_assign(const column& other) {
    assert(layout_ptr);
    assert(other.layout_ptr);
    assert(layout_ptr == other.layout_ptr);

    const column_layout& layout = *layout_ptr;
    const int self_pivot = pivot_row;
    const int other_pivot = other.pivot_row;

    for (std::size_t i = 0; i < layout.word_count; ++i) word_data[layout.word_offset + i] ^= other.word_data[layout.word_offset + i];

    if (self_pivot != other_pivot) {
      pivot_row = std::max(self_pivot, other_pivot);
      return;
    }

    pivot_row = self_pivot > 0 ? refresh_pivot_up_to(static_cast<std::size_t>(self_pivot - 1)) : -1;
  }

  // Swap the bits in logical rows `row` and `row + 1`.
  //
  // The caller must also provide the physical location of logical row `row`
  // inside the active word window:
  //
  //   physical_bit = layout.bit_offset + row
  //   word_index   = physical_bit >> 6
  //   bit_index    = physical_bit & 63
  //
  // Method:
  //   - If bit_index != 63, both bits lie in the same word. Extract the 2-bit
  //     pattern at positions row and row+1, and toggle both bits iff that
  //     pattern is 01 or 10.
  //   - If bit_index == 63, the pair straddles two words. Swap bit 63 of the
  //     lower word with bit 0 of the next word.
  inline void swap_adjacent_rows(std::size_t row, std::size_t word_index, unsigned bit_index) {
    assert(layout_ptr);
    const column_layout& layout = *layout_ptr;
    assert(row + 1 < layout.logical_size);
    [[maybe_unused]] const std::size_t physical_bit = static_cast<std::size_t>(layout.bit_offset) + row;
    assert(word_index == (physical_bit >> 6));
    assert(bit_index == static_cast<unsigned>(physical_bit & 63));
    assert(bit_index != 63 || word_index + 1 < layout.word_count);

    const int r = static_cast<int>(row);
    const int r1 = r + 1;
    std::uint64_t* const words = word_data + layout.word_offset;

    auto swap_bits = [](std::uint64_t& a, std::uint64_t amask, std::uint64_t& b, std::uint64_t bmask) {
      const bool abit = (a & amask) != 0;
      const bool bbit = (b & bmask) != 0;
      if (abit != bbit) { a ^= amask; b ^= bmask; }
      return abit;
    };

    const bool row_bit = (bit_index != 63)
      ? swap_bits(words[word_index], std::uint64_t(1) << bit_index, words[word_index], std::uint64_t(1) << (bit_index + 1))
      : swap_bits(words[word_index], std::uint64_t(1) << 63, words[word_index + 1], std::uint64_t(1));

    if (pivot_row == r) pivot_row = r1;
    else if (pivot_row == r1) pivot_row = row_bit ? r1 : r;
  }

  // Choose storage for the current layout.
  //
  // If the layout fits in the small-buffer optimization, use the inline buffer
  // `small_words`; otherwise allocate `layout.word_capacity` words on the heap.
  //
  // This helper only selects/acquires the buffer. It does not initialize the
  // active words, and it does not release any existing heap buffer. It is
  // therefore intended for freshly constructed or detached columns.
  inline void allocate_for_layout() {
    assert(layout_ptr);
    const column_layout& layout = *layout_ptr;
    word_data = (layout.word_capacity > small_word_capacity) ? new std::uint64_t[layout.word_capacity] : small_words;
  }

  // Zero the entire active word window and reset the cached pivot.
  inline void clear_active_window() {
    assert(layout_ptr);
    const column_layout& layout = *layout_ptr;
    if (layout.word_count) std::memset(word_data + layout.word_offset, 0, layout.word_count * sizeof(std::uint64_t));
    pivot_row = -1;
  }

  // Recompute the pivot using only logical rows 0 through `row`.
  //
  // Equivalently, return the lowest logical row j <= row such that the
  // bit in row j is 1. If no such row exists, return -1.
  //
  // Method:
  //   1. Locate logical row `row` in the active word window.
  //   2. In that starting word, mask off all bits strictly below row `row`.
  //   3. If the starting word is also the first active word, mask off inactive
  //      low bits below `layout.bit_offset`.
  //   4. Scan upward word-by-word until a nonzero word is found.
  //   5. In the first nonzero candidate word, the highest set physical bit is
  //      the desired pivot; convert that physical bit index back to a logical row.
  inline int refresh_pivot_up_to(std::size_t row) const {
    assert(layout_ptr);
    const column_layout& layout = *layout_ptr;
    assert(row < layout.logical_size);

    [[maybe_unused]] const std::size_t physical_bit = static_cast<std::size_t>(layout.bit_offset) + row;
    std::size_t word_index = physical_bit >> 6;
    const unsigned bit_index = static_cast<unsigned>(physical_bit & 63);

    std::uint64_t word = word_data[layout.word_offset + word_index] & (bit_index == 63 ? ~std::uint64_t(0) : ((std::uint64_t(1) << (bit_index + 1)) - 1));
    if (!word_index && layout.bit_offset) word &= (~std::uint64_t(0) << layout.bit_offset);

    for (;;) {
      if (word) return static_cast<int>((word_index << 6) + highest_set_bit_index(word) - layout.bit_offset);
      if (!word_index) return -1;
      --word_index;
      word = word_data[layout.word_offset + word_index];
      if (!word_index && layout.bit_offset) word &= (~std::uint64_t(0) << layout.bit_offset);
    }
  }
};

struct matrix {

  struct bit_ref {
    // Proxy reference to one matrix bit.
    //
    // Because bits are packed into 64-bit words, the matrix cannot return a
    // true `bool&`. Instead it returns this lightweight object, which forwards
    // reads and writes through column::get/set.
    //
    // Converting to bool reads the current bit value. Assigning from bool writes
    // that value into the matrix. Assigning from another bit_ref copies the
    // referenced bit value, so expressions like `m(i, j) = m(k, l);` behave as expected;
    // that is, the value of `m(k,l)` will be written to `m(i,j)`.
    //
    // Note that `auto x = m(i, j);` stores a proxy, while `bool x = m(i, j);`
    // stores the current bit value.
    column* column_ptr;
    std::size_t row;

    inline operator bool() const { return column_ptr->get(row); }
    inline bit_ref& operator=(bool value) { column_ptr->set(row, value); return *this; }
    inline bit_ref& operator=(const bit_ref& other) { return *this = static_cast<bool>(other); }
  };

  // Shared row-layout metadata for all live columns.
  //
  // This is heap-allocated so that its address remains stable across matrix
  // moves. Every live column in this matrix stores a raw pointer to this object.
  std::unique_ptr<column_layout> layout_storage;

  // Overallocated circular buffer of column objects.
  //
  // The live logical columns occupy column_count consecutive slots beginning at
  // col_offset, wrapping around the end of the vector if necessary.
  std::vector<column> columns;

  // Number of live logical columns.
  std::size_t column_count = 0;

  // Physical index in `columns` at which logical column 0 is stored.
  std::size_t col_offset = 0;

  // Default construction yields an empty detached matrix:
  //
  //   - no layout object,
  //   - no column storage,
  //   - column_count == 0.
  matrix() = default;

  matrix(const matrix&) = delete;
  matrix& operator=(const matrix&) = delete;

  // Move construction transfers ownership of the shared layout object and the
  // column buffer, then resets the source matrix to the empty-detached state.
  matrix(matrix&& other) noexcept
    : layout_storage(std::move(other.layout_storage)),
      columns(std::move(other.columns)),
      column_count(other.column_count),
      col_offset(other.col_offset)
  {
    other.reset_to_empty();
  }

  // Move assignment does the same, first releasing any resources currently owned
  // by *this through the normal move-assignment behavior of unique_ptr/vector.
  matrix& operator=(matrix&& other) noexcept {
    if (this == &other) return *this;

    layout_storage = std::move(other.layout_storage);
    columns = std::move(other.columns);
    column_count = other.column_count;
    col_offset = other.col_offset;

    other.reset_to_empty();
    return *this;
  }

  // Construct an n x n zero matrix.
  //
  // Representation strategy:
  //
  //   1. Allocate one shared column_layout describing the logical row range.
  //   2. Choose a per-column word capacity with slack on both sides, and place
  //      the active word window in the middle of that storage.
  //   3. Allocate an overlarge circular buffer of column objects, and place the
  //      n live logical columns in the middle of that buffer as well.
  //
  // The extra slack at both the word level and the column-buffer level supports
  // cheap growth at either boundary.
  explicit matrix(std::size_t n) {
    if (!n) return;
    layout_storage = std::make_unique<column_layout>();
    column_layout& layout = *layout_storage;
    layout.logical_size = n;
    layout.bit_offset = 0;
    layout.word_count = words_for_bits(layout.bit_offset, layout.logical_size);
    layout.word_capacity = (layout.word_count > column::small_word_capacity) ? (layout.word_count * 2 + 2) : column::small_word_capacity;
    layout.word_offset = (layout.word_capacity - layout.word_count) / 2;
    column_count = n;
    std::size_t storage_capacity = n * 2 + 2;
    columns.resize(storage_capacity);
    col_offset = (storage_capacity - column_count) / 2;
    for (std::size_t column_index = 0; column_index != column_count; ++column_index) columns[col_offset + column_index] = column(layout_storage.get());
  }

  // Return the logical matrix dimension.
  inline std::size_t dimension() const { return column_count; }

  // Translate a logical column index through the circular buffer.
  inline std::size_t physical_column_index(std::size_t logical_index) const {
    assert(logical_index < column_count);
    std::size_t index = col_offset + logical_index;
    return (index < columns.size()) ? index : (index - columns.size());
  }

  // Return logical column `col`, resolving through the circular buffer.
  inline column& column_at(std::size_t col) { return columns[physical_column_index(col)]; }
  inline const column& column_at(std::size_t col) const { return columns[physical_column_index(col)]; }

  // Access logical entry (row, column): const returns bool, non-const returns a writable proxy.
  inline bool operator()(std::size_t row, std::size_t column_index) const { return column_at(column_index).get(row); }
  inline bit_ref operator()(std::size_t row, std::size_t column_index) { return bit_ref{ &column_at(column_index), row }; }

  // Apply `fn` to each live logical column in order.
  template <typename F>
  inline void for_each_column(F&& fn) {
    if (!column_count) return;

    std::size_t cap = columns.size();
    std::size_t first_chunk = cap - col_offset;
    if (first_chunk > column_count) first_chunk = column_count;

    for (std::size_t i = 0; i != first_chunk; ++i) fn(columns[col_offset + i]);

    std::size_t remaining = column_count - first_chunk;
    for (std::size_t i = 0; i != remaining; ++i) fn(columns[i]);
  }

  // Over GF(2), perform destination <- destination XOR source.
  inline void add_column(std::size_t source, std::size_t destination) {
    column_at(destination).xor_assign(column_at(source));
  }

  // Grow or shrink the square matrix by one zero row / zero column pair at a
  // boundary.
  //
  // In matrix form, with M the old matrix:
  //
  //   grow_upper_left:   M -> [ 0 | 0 ]
  //                           [---+---]
  //                           [ 0 | M ]
  //
  //   grow_lower_right:  M -> [ M | 0 ]
  //                           [---+---]
  //                           [ 0 | 0 ]
  //
  //   shrink_upper_left: remove row 0 and column 0
  //   shrink_lower_right: remove the last row and last column
  //
  // If the matrix is empty, either growth operation creates the 1x1 zero matrix.
  inline void grow_upper_left() {
    if (!column_count) { *this = matrix(1); return; }
    push_top_all(false);
    push_front_column(column(layout_storage.get()));
  }

  inline void grow_lower_right() {
    if (!column_count) { *this = matrix(1); return; }
    push_bottom_all(false);
    push_back_column(column(layout_storage.get()));
  }

  inline void shrink_upper_left() {
    assert(column_count);
    if (column_count == 1) { reset_to_empty(); return; }
    pop_front_column();
    pop_top_all();
  }

  inline void shrink_lower_right() {
    assert(column_count);
    if (column_count == 1) { reset_to_empty(); return; }
    pop_back_column();
    pop_bottom_all();
  }

  // Simultaneously swap logical indices k and k+1 on both axes.
  // Equivalently: swap columns k/k+1, then swap rows k/k+1 in every column.
  inline void swap_adjacent(std::size_t k) {
    assert(column_count);
    assert(k + 1 < column_count);

    std::swap(column_at(k), column_at(k + 1));

    const column_layout& layout = *layout_storage;
    const std::size_t physical_bit_index = static_cast<std::size_t>(layout.bit_offset) + k;
    const std::size_t word_index = physical_bit_index >> 6;
    const unsigned bit_index = static_cast<unsigned>(physical_bit_index & 63);

    for_each_column([&](column& col) { col.swap_adjacent_rows(k, word_index, bit_index); });
  }

  // Ensure the circular column buffer can hold `required_count` live columns
  // while still leaving at least one unused slot afterward.
  //
  // This helper is used immediately before inserting a column at the front or
  // back. In current usage, callers pass
  //
  //   required_count = column_count + 1
  //
  // so the no-op condition
  //
  //   columns.size() >= required_count + 1
  //
  // means: "after the pending insertion, the ring buffer will still contain at
  // least one spare physical slot."
  //
  // If more space is needed, allocate a larger buffer, move the live logical
  // columns into it in logical order, and recenter them. Recentering restores
  // slack on both sides of the live range, which keeps future front/back
  // insertions cheap.
  //
  // Growth policy:
  //   - normally grow to old_capacity * 2 + 2,
  //   - if starting from empty, use required_count * 2 + 2,
  //   - then clamp up to at least required_count + 1.
  inline void ensure_column_storage_capacity(std::size_t required_count) {
    if (columns.size() >= required_count + 1) return;

    std::size_t old_capacity = columns.size();
    std::size_t new_capacity = old_capacity ? (old_capacity * 2 + 2) : (required_count * 2 + 2);
    if (new_capacity < required_count + 1) new_capacity = required_count + 1;

    std::vector<column> new_columns;
    new_columns.resize(new_capacity);

    std::size_t new_offset = (new_capacity - column_count) / 2;

    for (std::size_t logical_index = 0; logical_index != column_count; ++logical_index)
      new_columns[new_offset + logical_index] = std::move(columns[physical_column_index(logical_index)]);

    columns.swap(new_columns);
    col_offset = new_offset;
  }

  // Circular-buffer column-range editors.
  //
  // These helpers manipulate only the live logical column interval inside
  // `columns`. The live columns always occupy `column_count` consecutive
  // physical slots beginning at `col_offset`, modulo `columns.size()`.
  //
  // They preserve the logical order of existing columns, but do not modify the
  // shared row layout. They are the column-side half of grow/shrink operations.

  // Insert `col` as the new logical column 0.
  inline void push_front_column(column&& col) {
    ensure_column_storage_capacity(column_count + 1);

    col_offset = col_offset ? (col_offset - 1) : (columns.size() - 1);
    columns[col_offset] = std::move(col);
    ++column_count;
  }

  // Append `col` as the new last logical column.
  inline void push_back_column(column&& col) {
    ensure_column_storage_capacity(column_count + 1);

    std::size_t index = col_offset + column_count;
    if (index >= columns.size()) index -= columns.size();

    columns[index] = std::move(col);
    ++column_count;
  }

  // Remove and destroy logical column 0.
  inline void pop_front_column() {
    assert(column_count);

    columns[col_offset] = column();
    col_offset = col_offset + 1;
    if (col_offset == columns.size()) col_offset = 0;
    --column_count;
  }

  // Remove and destroy the last logical column.
  inline void pop_back_column() {
    assert(column_count);

    std::size_t index = col_offset + (column_count - 1);
    if (index >= columns.size()) index -= columns.size();

    columns[index] = column();
    --column_count;
  }

  // Restore the canonical empty-detached state.
  inline void reset_to_empty() {
    std::vector<column>().swap(columns);
    layout_storage.reset();
    column_count = 0;
    col_offset = 0;
  }

  // Ensure there is at least one spare word before the active word window.
  inline void ensure_word_space_top() {
    column_layout& layout = *layout_storage;
    if (layout.word_offset) return;
    recenter_words(layout.word_capacity ? layout.word_capacity * 2 : column::small_word_capacity * 2);
  }

  // Ensure there is at least one spare word after the active word window.
  inline void ensure_word_space_bottom() {
    column_layout& layout = *layout_storage;
    if (layout.word_offset + layout.word_count < layout.word_capacity) return;
    recenter_words(layout.word_capacity ? layout.word_capacity * 2 : column::small_word_capacity * 2);
  }

    // Reallocate every live column's word buffer and recenter the active word
  // window inside the new storage.
  //
  // Input:
  //   - `new_capacity` is the requested per-column word capacity.
  //
  // Normalization:
  //   - the actual capacity is clamped to at least `layout.word_count + 2`, so
  //     the recentered active window has at least one spare word on each side.
  //
  // Layout update:
  //   - the active words are copied unchanged,
  //   - the new active window begins at
  //         new_offset = (new_capacity - layout.word_count) / 2,
  //   - layout.word_capacity and layout.word_offset are updated accordingly.
  //
  // Logical row numbering, bit values, and cached pivots are unchanged.
  // Only the physical placement of the active word window changes.
  //
  // Words outside the active window are not initialized and have no semantic
  // meaning.
  inline void recenter_words(std::size_t new_capacity) {
    column_layout& layout = *layout_storage;
    if (new_capacity < layout.word_count + 2) new_capacity = layout.word_count + 2;

    std::size_t new_offset = (new_capacity - layout.word_count) / 2;

    for_each_column([&](column& col) {
      std::uint64_t* new_data = (new_capacity <= column::small_word_capacity) ? col.small_words : new std::uint64_t[new_capacity];

      if (layout.word_count)
        std::memcpy(new_data + new_offset, col.word_data + layout.word_offset, layout.word_count * sizeof(std::uint64_t));

      if (col.word_data != col.small_words && new_data != col.word_data) delete[] col.word_data;

      col.word_data = new_data;
    });

    layout.word_capacity = new_capacity;
    layout.word_offset = new_offset;
  }

  // Row-window editors.
  //
  // These helpers apply the same row insertion/removal operation to every live
  // column by modifying the shared row layout and then updating the affected bits
  // in each column's active word window.
  //
  // There are four variants:
  //
  //   - push_top_all(value):    prepend one logical row, initialized to `value`
  //   - push_bottom_all(value): append  one logical row, initialized to `value`
  //   - pop_top_all():          remove logical row 0
  //   - pop_bottom_all():       remove the last logical row
  //
  // Top vs. bottom are not symmetric at the pivot level:
  //
  //   - Top insertion/removal renumbers every surviving row, so cached pivots can
  //     often be updated by a uniform shift.
  //   - Bottom insertion/removal leaves surviving row numbers unchanged, so
  //     cached pivots usually stay fixed; only columns touching the bottom edge
  //     need special handling.
  inline void push_top_all(bool value) {
    column_layout& layout = *layout_storage;

    if (!layout.bit_offset) {
      ensure_word_space_top();
      --layout.word_offset;
      layout.bit_offset = 63;
    } else {
      --layout.bit_offset;
    }

    ++layout.logical_size;

    std::size_t new_word_count = words_for_bits(layout.bit_offset, layout.logical_size);
    if (new_word_count != layout.word_count) layout.word_count = new_word_count;

    std::uint64_t mask = std::uint64_t(1) << layout.bit_offset;

    for_each_column([&](column& col) {
      if (value) col.word_data[layout.word_offset] |= mask;
      else col.word_data[layout.word_offset] &= ~mask;

      if (col.pivot_row != -1) ++col.pivot_row;
      else if (value) col.pivot_row = 0;
    });
  }

  inline void push_bottom_all(bool value) {
    column_layout& layout = *layout_storage;

    std::size_t new_word_count = words_for_bits(layout.bit_offset, layout.logical_size + 1);
    if (new_word_count != layout.word_count) {
      ensure_word_space_bottom();
      layout.word_count = new_word_count;
    }

    ++layout.logical_size;
    std::size_t row = layout.logical_size - 1;

    for_each_column([&](column& col) {
      std::uint64_t& word = col.word_data[layout.word_offset + ((static_cast<std::size_t>(layout.bit_offset) + row) >> 6)];
      std::uint64_t mask = std::uint64_t(1) << ((static_cast<std::size_t>(layout.bit_offset) + row) & 63);
      if (value) word |= mask;
      else word &= ~mask;

      if (value) col.pivot_row = static_cast<int>(row);
    });
  }

  inline void pop_top_all() {
    column_layout& layout = *layout_storage;
    assert(layout.logical_size);

    if (layout.bit_offset == 63) { ++layout.word_offset; layout.bit_offset = 0; }
    else ++layout.bit_offset;

    --layout.logical_size;
    layout.word_count = words_for_bits(layout.bit_offset, layout.logical_size);

    for_each_column([&](column& col) {
      if (col.pivot_row == -1) return;
      if (col.pivot_row == 0) { col.pivot_row = -1; return; }
      --col.pivot_row;
    });
  }

  inline void pop_bottom_all() {
    column_layout& layout = *layout_storage;
    assert(layout.logical_size);

    std::size_t old_last_row = layout.logical_size - 1;

    --layout.logical_size;
    layout.word_count = words_for_bits(layout.bit_offset, layout.logical_size);

    for_each_column([&](column& col) {
      if (col.pivot_row == static_cast<int>(old_last_row))
        col.pivot_row = layout.logical_size ? col.refresh_pivot_up_to(layout.logical_size - 1) : -1;
    });
  }
};
