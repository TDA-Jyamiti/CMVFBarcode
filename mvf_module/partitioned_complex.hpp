// partitioned_complex.hpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <utility>
#include <cassert>
#include <cstring>
#include <limits>
#include <initializer_list>
#include <stdexcept>

#include "graph_utilities.hpp"
#include "reduction_pair.hpp"

// PARTITIONED SIMPLICIAL COMPLEX
// ==============================
//
// `partitioned_complex` owns one fixed abstract simplicial complex together with a mutable partition
// of its simplices. Simplex vertex lists are packed into `vertex_pool`; parallel offset/count arrays
// locate each list, and `codim1_face_graph` stores the fixed boundary incidence in CSR form. A hash
// index supports face lookup while the complex is finalized.
//
// FIXED SIMPLEX STORAGE
// ---------------------
//
// For `s0 = {0}`, `s1 = {1}`, and `s2 = {0, 1}`, the packed representation is
//
//     simplex id              s0          s1              s2
//     (offset, count)       (0, 1)      (1, 1)          (2, 2)
//
//     vertex_pool      +-----------+-----------+-------------------+
//                      |     0     |     1     |       0, 1        |
//                      +-----------+-----------+-------------------+
//                            s0          s1              s2
//
// and the fixed codimension-one face graph contains
//
//                         s0 <--------- s2 ---------> s1
//
// `simplex_to_class` and `simplex_pos_in_class` locate every simplex in an alive `class_state`. A
// class is one partition block: it keeps its simplices in the current filtration order, owns the
// reduced boundary matrix for that order, and maps homology-bearing zero columns to entries in the
// append-only `features` history. Dead class slots are recycled through `free_class_ids`.
//
// MUTABLE CLASS LAYOUT
// --------------------
//
// Every simplex has a two-way address into its live class. For a class `c` ordered as
// `[s2, s5, s7]`:
//
//     simplex_to_class[s5] = c
//                 |
//                 v
//     classes[c].simplices       +--------+--------+--------+
//                                |   s2   |   s5   |   s7   |
//                                +--------+--------+--------+
//     class position / matrix axis   0        1        2
//                                             ^
//                                             |
//                           simplex_pos_in_class[s5] = 1
//
// The same positions index both axes of `R` and `U` and align with the feature slots:
//
//     class column              0        1        2
//     feature_of_col          [f8]     [npos]    [f11]
//                               |                   |
//                               v                   v
//     features[f8]  = { ... cls = c, column = 0 ... }
//     features[f11] = { ... cls = c, column = 2 ... }
//
// `finalize()` builds faces, coalesces cyclic partition classes, constructs each reduced matrix, and
// creates the initial live features. Translation then changes the partition only through adjacent
// swaps, singleton splits, and singleton merges. Each edit advances `step` and updates matrices,
// positions, feature ownership, births, and deaths in lockstep.

struct partitioned_complex {
  using vertex_id  = std::uint32_t;
  using simplex_id = std::uint32_t;
  using class_id   = std::uint32_t;

  static constexpr simplex_id invalid_simplex = std::numeric_limits<simplex_id>::max();
  static constexpr class_id   invalid_class   = std::numeric_limits<class_id>::max();

  static constexpr std::size_t npos = static_cast<std::size_t>(-1);

  using time_type = std::uint64_t;
  static constexpr time_type time_infty = std::numeric_limits<time_type>::max();

  struct feature {
    time_type birth = 0;
    time_type death = time_infty;
    class_id  cls   = invalid_class;
    std::uint32_t column = 0;
    std::uint32_t dimension = 0;

    inline bool alive() const { return death == time_infty; }
  };

  std::uint64_t step = 0;
  std::vector<feature> features;

  std::vector<vertex_id>     vertex_pool;
  std::vector<std::uint32_t> simplex_vertex_offset;
  std::vector<std::uint32_t> simplex_vertex_count;

  csr_graph codim1_face_graph;

  std::vector<class_id> simplex_to_class;

  struct class_state : reduction_pair {
    bool alive = true;
    std::vector<simplex_id> simplices;
    std::vector<std::size_t> feature_of_col;

    inline std::size_t size() const { return simplices.size(); }

    inline void deactivate() {
      alive = false;
      simplices.clear();
      feature_of_col.clear();
      static_cast<reduction_pair&>(*this) = reduction_pair();
    }
  };
  std::vector<class_state> classes;

  std::vector<std::uint32_t> simplex_pos_in_class;
  std::vector<class_id> free_class_ids;

  struct simplex_index {
    std::vector<std::uint64_t> hashes;
    std::vector<simplex_id>    ids;
    std::size_t mask = 0;

    inline void clear() { hashes.clear(); ids.clear(); mask = 0; }

    inline void reset(std::size_t expected) {
      std::size_t cap = 1;
      while (cap < expected * 2 + 8) cap <<= 1;
      hashes.assign(cap, 0);
      ids.assign(cap, invalid_simplex);
      mask = cap - 1;
    }
  } index;

  partitioned_complex() = default;
  partitioned_complex(const partitioned_complex&) = delete;
  partitioned_complex& operator=(const partitioned_complex&) = delete;
  partitioned_complex(partitioned_complex&&) noexcept = default;
  partitioned_complex& operator=(partitioned_complex&&) noexcept = default;

  inline void clear() {
    step = 0;
    features.clear();

    vertex_pool.clear();
    simplex_vertex_offset.clear();
    simplex_vertex_count.clear();
    codim1_face_graph = csr_graph();

    simplex_to_class.clear();
    classes.clear();
    simplex_pos_in_class.clear();
    free_class_ids.clear();

    index.clear();
  }

  inline void reserve_simplices(std::size_t simplex_n, std::size_t total_vertex_n) {
    simplex_vertex_offset.reserve(simplex_n);
    simplex_vertex_count.reserve(simplex_n);
    vertex_pool.reserve(total_vertex_n);
  }

  inline std::size_t simplex_count() const { return simplex_vertex_offset.size(); }
  inline std::size_t class_count_total() const { return classes.size(); }

  inline simplex_id add_simplex_sorted(const vertex_id* verts, std::uint32_t count) {
    if (!count || !verts) throw std::invalid_argument("add_simplex_sorted: simplex is empty");
    if (simplex_vertex_offset.size() >= invalid_simplex) throw std::length_error("add_simplex_sorted: too many simplices");
    if (vertex_pool.size() > std::numeric_limits<std::uint32_t>::max() - static_cast<std::size_t>(count)) throw std::length_error("add_simplex_sorted: vertex pool too large");
#ifndef NDEBUG
    for (std::uint32_t i = 1; i < count; ++i) assert(verts[i - 1] < verts[i]);
#endif
    simplex_id id = static_cast<simplex_id>(simplex_vertex_offset.size());
    simplex_vertex_offset.push_back(static_cast<std::uint32_t>(vertex_pool.size()));
    simplex_vertex_count.push_back(count);
    vertex_pool.insert(vertex_pool.end(), verts, verts + count);
    return id;
  }

  inline simplex_id add_simplex_sorted(std::initializer_list<vertex_id> verts) {
    return add_simplex_sorted(verts.begin(), static_cast<std::uint32_t>(verts.size()));
  }

  inline void set_simplex_class(simplex_id s, class_id c) {
    if (simplex_to_class.size() != simplex_count())
      simplex_to_class.assign(simplex_count(), invalid_class);
    simplex_to_class[s] = c;
  }

  inline void finalize(bool require_all_faces = true) {
    build_index_();
    build_codim1_faces_(require_all_faces);
    complete_partition_map_();
    coarsen_partition_from_map_();
    rebuild_classes_from_map_();
    build_algebra_and_features_();
  }

  inline bool finalized() const {
    return (codim1_face_graph.vertex_count() == simplex_count()) &&
           (simplex_to_class.size() == simplex_count()) &&
           (simplex_pos_in_class.size() == simplex_count()) &&
           !index.hashes.empty();
  }

  inline const vertex_id* simplex_vertices_data(simplex_id s) const {
    return vertex_pool.data() + simplex_vertex_offset[s];
  }
  inline std::uint32_t simplex_vertices_size(simplex_id s) const { return simplex_vertex_count[s]; }

  inline std::uint32_t simplex_dimension(simplex_id s) const {
    std::uint32_t n = simplex_vertex_count[s];
    return n ? (n - 1) : 0;
  }

  inline bool simplex_contains(simplex_id sup, simplex_id sub) const {
    const vertex_id* a = simplex_vertices_data(sup);
    std::uint32_t an = simplex_vertices_size(sup);
    const vertex_id* b = simplex_vertices_data(sub);
    std::uint32_t bn = simplex_vertices_size(sub);
    if (bn > an) return false;
    return std::includes(a, a + an, b, b + bn);
  }

  inline const simplex_id* codim1_faces_data(simplex_id s) const {
    return codim1_face_graph.edges.empty() ? nullptr : codim1_face_graph.edges.data() + codim1_face_graph.edge_begin(s);
  }
  inline std::uint32_t codim1_faces_size(simplex_id s) const { return static_cast<std::uint32_t>(codim1_face_graph.degree(s)); }

  inline bool boundary_query(simplex_id s, simplex_id face) const {
    if (simplex_vertices_size(s) != simplex_vertices_size(face) + 1) return false;
    const simplex_id* f = codim1_faces_data(s);
    std::uint32_t fn = codim1_faces_size(s);
    for (std::uint32_t i = 0; i < fn; ++i) if (f[i] == face) return true;
    return false;
  }

  inline simplex_id find_simplex_sorted(const vertex_id* verts, std::uint32_t count) const {
    assert(index.mask && "call finalize() before lookups");
#ifndef NDEBUG
    for (std::uint32_t i = 1; i < count; ++i) assert(verts[i - 1] < verts[i]);
#endif
    return find_simplex_(verts, count);
  }

  inline class_id class_of(simplex_id s) const { return simplex_to_class[s]; }

  inline bool class_alive(class_id c) const {
    return (static_cast<std::size_t>(c) < classes.size()) && classes[c].alive;
  }

  inline const simplex_id* class_simplices_data(class_id c) const {
    assert(finalized() && "call finalize() before class queries");
    assert(class_alive(c) && "class is dead");
    return classes[c].simplices.data();
  }
  inline std::uint32_t class_simplices_size(class_id c) const {
    assert(finalized() && "call finalize() before class queries");
    assert(class_alive(c) && "class is dead");
    return static_cast<std::uint32_t>(classes[c].simplices.size());
  }

  inline std::vector<simplex_id> compute_exit_set(class_id c) const {
    assert(finalized() && "call finalize() before exit-set queries");
    assert(class_alive(c) && "compute_exit_set on dead class");

    std::vector<simplex_id> exit;

    const simplex_id* ss = class_simplices_data(c);
    std::uint32_t sn = class_simplices_size(c);

    exit.reserve(static_cast<std::size_t>(sn) * 4);

    for (std::uint32_t i = 0; i < sn; ++i) {
      simplex_id s = ss[i];
      const simplex_id* ff = codim1_faces_data(s);
      std::uint32_t fn = codim1_faces_size(s);
      for (std::uint32_t j = 0; j < fn; ++j) {
        simplex_id f = ff[j];
        if (simplex_to_class[f] != c) exit.push_back(f);
      }
    }

    auto by_dim_then_id = [&](simplex_id a, simplex_id b) {
      std::uint32_t da = simplex_vertices_size(a);
      std::uint32_t db = simplex_vertices_size(b);
      if (da != db) return da < db;
      return a < b;
    };

    std::sort(exit.begin(), exit.end(), by_dim_then_id);
    exit.erase(std::unique(exit.begin(), exit.end()), exit.end());

    return exit;
  }

  inline void swap_adjacent(class_id c, std::size_t p) {
    assert(finalized() && "call finalize() before swapping");
    if (!class_alive(c)) throw std::runtime_error("swap_adjacent: class dead");

    class_state& C = classes[c];
    const std::size_t n = C.size();
    const std::size_t q = p + 1;

    if (!(q < n) || C.R(p, q))
      throw std::runtime_error("swap_adjacent: out of range or illegal (R[p,p+1]==1)");

    ++step;

    const simplex_id sp = C.simplices[p];
    const simplex_id sq = C.simplices[q];

    const std::uint32_t dim_p = simplex_dimension(sp);
    const std::uint32_t dim_q = simplex_dimension(sq);

    auto swap_index_only = [&]() {
      simplex_id a = C.simplices[p];
      simplex_id b = C.simplices[q];
      std::swap(C.simplices[p], C.simplices[q]);
      simplex_pos_in_class[a] = static_cast<std::uint32_t>(q);
      simplex_pos_in_class[b] = static_cast<std::uint32_t>(p);
      C.swap_adjacent(p);
    };

    if (dim_p != dim_q) {
      swap_index_only();
      swap_feature_slots_(C, p, q);
      return;
    }

    const int low_p = C.low(p);
    const int low_q = C.low(q);
    const bool p_zero = (low_p < 0);
    const bool q_zero = (low_q < 0);

    int a = -1;
    int b = -1;
    for (std::size_t j = q; j < n; ++j) {
      const int pj = C.low(j);
      if (pj == static_cast<int>(p)) a = static_cast<int>(j);
      if (pj == static_cast<int>(q)) b = static_cast<int>(j);
      if (a >= 0 && b >= 0) break;
    }

    const bool b_hits_p = (b >= 0) && C.R(p, static_cast<std::size_t>(b));
    const bool need_clear = C.U(p, q);
    const bool need_fix_I_after_swap = need_clear && (low_p > low_q);

    if (need_clear) C.col(q) += C.col(p);

    swap_index_only();

    if (need_fix_I_after_swap) C.col(q) += C.col(p);

    if (a >= 0 && b >= 0 && b_hits_p) {
      const std::size_t donor  = static_cast<std::size_t>(std::min(a, b));
      const std::size_t target = static_cast<std::size_t>(std::max(a, b));
      if (donor != target) C.col(target) += C.col(donor);
    }

    if ((!p_zero) && q_zero) {
      if (!need_clear) move_feature_slot_(C, q, p);
    } else if (p_zero && (!q_zero)) {
      move_feature_slot_(C, p, q);
    } else if (p_zero) {
      if (a >= 0 && b < 0) move_feature_slot_(C, q, p);
      else if (a < 0 && b >= 0 && !b_hits_p) move_feature_slot_(C, p, q);
    }
  }

  inline class_id split_right(class_id c) {
    assert(finalized() && "call finalize() before splitting");
    if (!class_alive(c)) throw std::runtime_error("split_right: class dead");

    const std::size_t n = classes[c].size();
    if (n < 2) throw std::runtime_error("split_right: class size < 2");

    ++step;

    const std::size_t last = n - 1;
    const simplex_id moved = classes[c].simplices[last];

    const int low_last = classes[c].low(last);
    const bool was_zero = (low_last < 0);

    const class_id new_c = create_empty_class();
    class_state& C = classes[c];
    class_state& N = classes[new_c];
    N.simplices.clear();
    N.simplices.push_back(moved);
    N.feature_of_col.assign(1, npos);
    static_cast<reduction_pair&>(N) = reduction_pair(1);

    simplex_to_class[moved] = new_c;
    simplex_pos_in_class[moved] = 0;

    if (was_zero) {
      const std::size_t fidx = C.feature_of_col[last];
      if (fidx == npos) throw std::logic_error("split_right: expected feature on zero column");

      N.feature_of_col[0] = fidx;
      features[fidx].cls = new_c;
      features[fidx].column = 0;
    }

    C.simplices.pop_back();
    C.feature_of_col.pop_back();
    C.shrink_lower_right();

    if (!was_zero) {
      feature fa;
      fa.birth = step;
      fa.death = time_infty;
      fa.cls = new_c;
      fa.column = 0;
      fa.dimension = simplex_dimension(moved);

      const std::size_t ia = features.size();
      features.push_back(fa);
      N.feature_of_col[0] = ia;

      const std::size_t tgt = static_cast<std::size_t>(low_last);
      if (tgt >= C.size()) throw std::logic_error("split_right: target out of range");
      if (C.feature_of_col[tgt] != npos)
        throw std::logic_error("split_right: target already has a feature");

      const simplex_id tgt_s = C.simplices[tgt];

      feature fb;
      fb.birth = step;
      fb.death = time_infty;
      fb.cls = c;
      fb.column = static_cast<std::uint32_t>(tgt);
      fb.dimension = simplex_dimension(tgt_s);

      const std::size_t ib = features.size();
      features.push_back(fb);
      C.feature_of_col[tgt] = ib;
    }

    return new_c;
  }

  inline class_id split_left(class_id c) {
    assert(finalized() && "call finalize() before splitting");
    if (!class_alive(c)) throw std::runtime_error("split_left: class dead");

    const std::size_t n = classes[c].size();
    if (n < 2) throw std::runtime_error("split_left: class size < 2");

    ++step;

    const simplex_id moved = classes[c].simplices[0];
    const std::size_t f0   = classes[c].feature_of_col[0];

    const int targeting = classes[c].column_with_low(0);
    const bool was_targeted = (targeting >= 0);

    const class_id new_c = create_empty_class();
    class_state& C = classes[c];
    class_state& N = classes[new_c];
    N.simplices.clear();
    N.simplices.push_back(moved);
    N.feature_of_col.assign(1, npos);
    static_cast<reduction_pair&>(N) = reduction_pair(1);

    simplex_to_class[moved] = new_c;
    simplex_pos_in_class[moved] = 0;

    if (!was_targeted) {
      if (f0 == npos) throw std::logic_error("split_left: expected feature on untargeted zero column");

      N.feature_of_col[0] = f0;
      features[f0].cls = new_c;
      features[f0].column = 0;
    } else {
      feature fa;
      fa.birth = step;
      fa.death = time_infty;
      fa.cls = new_c;
      fa.column = 0;
      fa.dimension = simplex_dimension(moved);

      const std::size_t ia = features.size();
      features.push_back(fa);
      N.feature_of_col[0] = ia;
    }

    C.simplices.erase(C.simplices.begin());
    rebuild_positions_for_class_(c, 0);

    C.feature_of_col.erase(C.feature_of_col.begin());
    C.shrink_upper_left();

    if (was_targeted) {
      const std::size_t new_target = static_cast<std::size_t>(targeting - 1);
      if (new_target >= C.size()) throw std::logic_error("split_left: target out of range");
      if (C.feature_of_col[new_target] != npos)
        throw std::logic_error("split_left: target already has a feature");

      const simplex_id tgt_s = C.simplices[new_target];

      feature fb;
      fb.birth = step;
      fb.death = time_infty;
      fb.cls = c;
      fb.column = static_cast<std::uint32_t>(new_target);
      fb.dimension = simplex_dimension(tgt_s);

      const std::size_t ib = features.size();
      features.push_back(fb);
      C.feature_of_col[new_target] = ib;
    }

    for (std::size_t col = 0; col < C.feature_of_col.size(); ++col) {
      const std::size_t idx = C.feature_of_col[col];
      if (idx != npos) features[idx].column = static_cast<std::uint32_t>(col);
    }

    return new_c;
  }

  inline void merge_left(class_id target, class_id singleton) {
    assert(finalized() && "call finalize() before merging");
    if (target == singleton) throw std::runtime_error("merge_left: target == singleton");
    if (!class_alive(target) || !class_alive(singleton))
      throw std::runtime_error("merge_left: target or singleton dead");

    class_state& T = classes[target];
    class_state& S = classes[singleton];

    if (S.size() != 1) throw std::runtime_error("merge_left: singleton size != 1");

    const simplex_id s = S.simplices[0];

    const std::size_t moved_fidx = S.feature_of_col.empty() ? npos : S.feature_of_col[0];
    if (moved_fidx == npos)
      throw std::logic_error("merge_left: expected feature on singleton column");

    ++step;

    S.simplices.clear();
    T.simplices.insert(T.simplices.begin(), s);
    simplex_to_class[s] = target;
    rebuild_positions_for_class_(target, 0);
    delete_empty_class(singleton);

    T.grow_upper_left();
    T.feature_of_col.insert(T.feature_of_col.begin(), npos);

    for (std::size_t col = 1; col < T.feature_of_col.size(); ++col) {
      const std::size_t idx = T.feature_of_col[col];
      if (idx != npos) features[idx].column = static_cast<std::uint32_t>(col);
    }

    T.feature_of_col[0] = moved_fidx;
    features[moved_fidx].cls = target;
    features[moved_fidx].column = 0;

    const std::size_t n = T.size();
    if (static_cast<std::size_t>(class_simplices_size(target)) != n)
      throw std::logic_error("merge_left: target size mismatch after merge");
    if (T.simplices[0] != s) throw std::logic_error("merge_left: unexpected inserted simplex");

    std::vector<std::int8_t> hits(n, static_cast<std::int8_t>(-1));
    if (n) hits[0] = 0;

    if (n) T.R(0, 0) = false;

    for (std::size_t col = 1; col < n; ++col) {
      if (simplex_dimension(T.simplices[col]) != simplex_dimension(s) + 1) {
        T.R(0, col) = false;
        continue;
      }

      std::uint8_t parity = 0;

      for (std::size_t row = 1; row <= col; ++row) {
        if (!T.U(row, col)) continue;

        std::int8_t h = hits[row];
        if (h < 0) {
          const simplex_id t = T.simplices[row];
          if (simplex_dimension(t) == (simplex_dimension(s) + 1) && boundary_query(t, s)) h = 1;
          else h = 0;
          hits[row] = h;
        }

        parity ^= static_cast<std::uint8_t>(h);
      }

      T.R(0, col) = (parity != 0);
    }

    std::size_t c1 = npos;
    for (std::size_t col = 1; col < n; ++col) {
      if (T.low(col) == 0) { c1 = col; break; }
    }

    if (c1 != npos) {
      for (std::size_t col = c1 + 1; col < n; ++col) {
        if (T.low(col) == 0) T.col(col) += T.col(c1);
      }

      if (T.feature_of_col[c1] != npos) {
        const std::size_t f = T.feature_of_col[c1];
        T.feature_of_col[c1] = npos;
        features[f].death = step;
      }

      if (T.feature_of_col[0] == npos)
        throw std::logic_error("merge_left: expected feature on inserted column 0");
      {
        const std::size_t f = T.feature_of_col[0];
        T.feature_of_col[0] = npos;
        features[f].death = step;
      }
    }
  }

  inline void merge_right(class_id target, class_id singleton) {
    assert(finalized() && "call finalize() before merging");
    if (target == singleton) throw std::runtime_error("merge_right: target == singleton");
    if (!class_alive(target) || !class_alive(singleton))
      throw std::runtime_error("merge_right: target or singleton dead");

    class_state& T = classes[target];
    class_state& S = classes[singleton];

    if (S.size() != 1) throw std::runtime_error("merge_right: singleton size != 1");

    const simplex_id s = S.simplices[0];

    const std::size_t moved_fidx = S.feature_of_col.empty() ? npos : S.feature_of_col[0];
    if (moved_fidx == npos)
      throw std::logic_error("merge_right: expected feature on singleton column");

    ++step;

    S.simplices.clear();
    T.simplices.push_back(s);
    simplex_to_class[s] = target;
    simplex_pos_in_class[s] = static_cast<std::uint32_t>(T.simplices.size() - 1);
    delete_empty_class(singleton);

    const std::size_t old_n = T.dimension();
    T.grow_lower_right();
    T.feature_of_col.push_back(npos);

    const std::size_t n = T.dimension();
    if (n != old_n + 1) throw std::logic_error("merge_right: size mismatch after grow");

    const std::size_t new_col = n - 1;

    T.feature_of_col[new_col] = moved_fidx;
    features[moved_fidx].cls = target;
    features[moved_fidx].column = static_cast<std::uint32_t>(new_col);

    if (T.simplices[new_col] != s) throw std::logic_error("merge_right: unexpected appended simplex");

    const simplex_id* ff = codim1_faces_data(s);
    const std::uint32_t fn = codim1_faces_size(s);

    for (std::uint32_t k = 0; k < fn; ++k) {
      const simplex_id f = ff[k];
      if (class_of(f) != target) continue;
      const std::size_t row = static_cast<std::size_t>(simplex_pos_in_class[f]);
      if (row >= n) throw std::logic_error("merge_right: face row out of range");
      T.R(row, new_col) = true;
    }

    std::vector<std::size_t> pivot_column(n, npos);
    for (std::size_t j = 0; j < new_col; ++j) {
      const int p = T.low(j);
      if (p >= 0) pivot_column[static_cast<std::size_t>(p)] = j;
    }

    for (;;) {
      const int p = T.low(new_col);
      if (p < 0) break;
      const std::size_t i = pivot_column[static_cast<std::size_t>(p)];
      if (i == npos) break;
      T.col(new_col) += T.col(i);
    }

    const int low_new = T.low(new_col);
    if (low_new >= 0) {
      const std::size_t tgt = static_cast<std::size_t>(low_new);
      if (tgt >= n) throw std::logic_error("merge_right: pivot out of range");

      if (T.feature_of_col[new_col] == npos)
        throw std::logic_error("merge_right: expected feature on new column");
      {
        const std::size_t f = T.feature_of_col[new_col];
        T.feature_of_col[new_col] = npos;
        features[f].death = step;
      }

      if (T.feature_of_col[tgt] == npos)
        throw std::logic_error("merge_right: expected feature on target column");
      {
        const std::size_t f = T.feature_of_col[tgt];
        T.feature_of_col[tgt] = npos;
        features[f].death = step;
      }
    }
  }

  inline std::vector<class_id> coarsen_partition_map(std::vector<class_id> map) const {
    const std::size_t S = simplex_count();
    if (!S) return {};
    if (codim1_face_graph.vertex_count() != S) throw std::logic_error("coarsen_partition_map: codimension-one faces not built");

    if (map.size() != S) {
      map.resize(S);
      for (simplex_id s = 0; s < static_cast<simplex_id>(S); ++s) map[s] = s;
      return map;
    }

    class_id next = 0;
    for (class_id c : map)
      if (c != invalid_class && c + 1 > next) next = c + 1;
    for (simplex_id s = 0; s < static_cast<simplex_id>(S); ++s)
      if (map[s] == invalid_class) map[s] = next++;

    class_id max_c = 0;
    for (class_id c : map) if (c > max_c) max_c = c;
    const class_id C = max_c + 1;
    if (!C) return map;

    std::vector<std::uint64_t> packed;
    for (simplex_id s = 0; s < static_cast<simplex_id>(S); ++s) {
      const class_id cs = map[s];
      const simplex_id* faces = codim1_faces_data(s);
      const std::uint32_t face_n = codim1_faces_size(s);
      for (std::uint32_t i = 0; i < face_n; ++i) {
        const simplex_id face = faces[i];
        if (face == invalid_simplex) throw std::logic_error("coarsen_partition_map: invalid simplex in face list");
        const class_id cf = map[face];
        if (cs != cf) packed.push_back((std::uint64_t(cs) << 32) | cf);
      }
    }

    csr_graph graph = csr_graph::from_packed_edges(static_cast<csr_graph::vertex_id>(C), packed);
    const csr_graph::component_map components = graph.strongly_connected_components();
    if (components.count == C) return map;

    std::vector<simplex_id> component_min(components.count, invalid_simplex);
    for (simplex_id s = 0; s < static_cast<simplex_id>(S); ++s) {
      const class_id component = components.of_vertex[map[s]];
      if (s < component_min[component]) component_min[component] = s;
    }

    std::vector<class_id> order(components.count), component_to_class(components.count, 0);
    for (class_id component = 0; component < components.count; ++component) order[component] = component;
    std::sort(order.begin(), order.end(), [&](class_id a, class_id b) {
      return component_min[a] != component_min[b] ? component_min[a] < component_min[b] : a < b;
    });

    for (class_id c = 0; c < components.count; ++c) component_to_class[order[c]] = c;
    for (simplex_id s = 0; s < static_cast<simplex_id>(S); ++s) map[s] = component_to_class[components.of_vertex[map[s]]];
    return map;
  }

  inline void coarsen_partition() {
    assert((codim1_face_graph.vertex_count() == simplex_count()) && "call finalize() first (faces needed)");
    complete_partition_map_();
    coarsen_partition_from_map_();
    rebuild_classes_from_map_();
    build_algebra_and_features_();
  }

  inline void delete_empty_class(class_id c) {
    assert(finalized() && "call finalize() before editing classes");
    if (!class_alive(c)) return;

    if (!classes[c].simplices.empty())
      throw std::logic_error("delete_empty_class: class not empty");

    classes[c].deactivate();
    free_class_ids.push_back(c);
  }

  inline class_id create_empty_class() {
    assert(finalized() && "call finalize() before editing classes");
    class_id id = invalid_class;

    if (!free_class_ids.empty()) {
      id = free_class_ids.back();
      free_class_ids.pop_back();
      classes[id].alive = true;
      classes[id].simplices.clear();
      classes[id].feature_of_col.clear();
      static_cast<reduction_pair&>(classes[id]) = reduction_pair();
    } else {
      if (classes.size() >= invalid_class) throw std::length_error("create_empty_class: too many classes");
      id = static_cast<class_id>(classes.size());
      classes.emplace_back();
    }
    return id;
  }

  inline void swap_feature_slots_(class_state& C, std::size_t p, std::size_t q) {
    const std::size_t fp = C.feature_of_col[p];
    const std::size_t fq = C.feature_of_col[q];

    std::swap(C.feature_of_col[p], C.feature_of_col[q]);

    if (fp != npos) features[fp].column = static_cast<std::uint32_t>(q);
    if (fq != npos) features[fq].column = static_cast<std::uint32_t>(p);
  }

  inline void move_feature_slot_(class_state& C, std::size_t src, std::size_t dst) {
    if (src == dst) return;

    const std::size_t idx = C.feature_of_col[src];
    if (idx == npos) return;

#ifndef NDEBUG
    assert(C.feature_of_col[dst] == npos);
#endif
    if (C.feature_of_col[dst] != npos)
      throw std::logic_error("move_feature_slot: destination already has a feature");

    C.feature_of_col[src] = npos;
    C.feature_of_col[dst] = idx;
    features[idx].column = static_cast<std::uint32_t>(dst);
  }

  inline void build_algebra_and_features_() {
    step = 0;
    features.clear();

    for (class_id c = 0; c < static_cast<class_id>(classes.size()); ++c) {
      class_state& C = classes[c];
      if (!C.alive) {
        C.feature_of_col.clear();
        static_cast<reduction_pair&>(C) = reduction_pair();
        continue;
      }

      const std::size_t n = C.simplices.size();
      matrix D(n);

      for (std::size_t col = 0; col < n; ++col) {
        simplex_id s = C.simplices[col];
        const simplex_id* ff = codim1_faces_data(s);
        const std::uint32_t fn = codim1_faces_size(s);

        for (std::uint32_t k = 0; k < fn; ++k) {
          simplex_id f = ff[k];
          if (class_of(f) != c) continue;

          const std::size_t row = static_cast<std::size_t>(simplex_pos_in_class[f]);
          assert(row < n);
          D(row, col) = true;
        }
      }

      static_cast<reduction_pair&>(C) = reduction_pair(std::move(D));
      C.feature_of_col.assign(n, npos);

      auto red = C.reduce();

      for (std::size_t col : red.valid) {
        feature ft;
        ft.birth = step;
        ft.death = time_infty;
        ft.cls = c;
        ft.column = static_cast<std::uint32_t>(col);
        ft.dimension = simplex_dimension(C.simplices[col]);

        const std::size_t idx = features.size();
        features.push_back(ft);

        assert(C.feature_of_col[col] == npos);
        C.feature_of_col[col] = idx;
      }
    }
  }

  static inline std::uint64_t hash_vertices_(const vertex_id* v, std::uint32_t n) noexcept {
    std::uint64_t h = 14695981039346656037ull;
    for (std::uint32_t i = 0; i < n; ++i) { h ^= (std::uint64_t)v[i]; h *= 1099511628211ull; }
    h ^= (std::uint64_t)n * 0x9e3779b97f4a7c15ull;
    return h ? h : 1ull;
  }

  inline bool vertices_equal_(simplex_id s, const vertex_id* v, std::uint32_t n) const {
    if (simplex_vertices_size(s) != n) return false;
    const vertex_id* a = simplex_vertices_data(s);
    return std::memcmp(a, v, n * sizeof(vertex_id)) == 0;
  }

  inline void build_index_() {
    index.reset(simplex_count());
    for (simplex_id s = 0; s < simplex_count(); ++s) {
      const vertex_id* v = simplex_vertices_data(s);
      std::uint32_t n = simplex_vertices_size(s);
      std::uint64_t h = (hash_vertices_(v, n) | 1ull);

      std::size_t pos = static_cast<std::size_t>(h) & index.mask;
      for (;;) {
        if (!index.hashes[pos]) { index.hashes[pos] = h; index.ids[pos] = s; break; }
        if (index.hashes[pos] == h && vertices_equal_(index.ids[pos], v, n)) {
          throw std::runtime_error("duplicate simplex vertices");
        }
        pos = (pos + 1) & index.mask;
      }
    }
  }

  inline simplex_id find_simplex_(const vertex_id* v, std::uint32_t n) const {
    std::uint64_t h = (hash_vertices_(v, n) | 1ull);
    std::size_t pos = static_cast<std::size_t>(h) & index.mask;
    for (;;) {
      std::uint64_t slot = index.hashes[pos];
      if (!slot) return invalid_simplex;
      if (slot == h) {
        simplex_id cand = index.ids[pos];
        if (vertices_equal_(cand, v, n)) return cand;
      }
      pos = (pos + 1) & index.mask;
    }
  }

  inline void build_codim1_faces_(bool require_all_faces) {
    std::size_t total_faces = 0;
    std::uint32_t max_n = 0;
    for (simplex_id s = 0; s < simplex_count(); ++s) {
      std::uint32_t n = simplex_vertices_size(s);
      max_n = std::max(max_n, n);
      if (n > 1) total_faces += n;
    }
    std::vector<csr_graph::unweighted_edge> face_edges;
    face_edges.reserve(total_faces);

    std::vector<vertex_id> scratch;
    scratch.resize(max_n ? (max_n - 1) : 0);

    for (simplex_id s = 0; s < simplex_count(); ++s) {
      const vertex_id* v = simplex_vertices_data(s);
      std::uint32_t n = simplex_vertices_size(s);
      if (n <= 1) continue;

      scratch.resize(n - 1);
      for (std::uint32_t drop = 0; drop < n; ++drop) {
        if (drop) std::memcpy(scratch.data(), v, drop * sizeof(vertex_id));
        if (drop + 1 < n)
          std::memcpy(scratch.data() + drop, v + drop + 1, (n - drop - 1) * sizeof(vertex_id));

        simplex_id f = find_simplex_(scratch.data(), n - 1);
        if (f == invalid_simplex) {
          if (require_all_faces) throw std::runtime_error("missing codim-1 face simplex");
          else continue;
        }
        face_edges.push_back({ s, f });
      }
    }
    codim1_face_graph = csr_graph::from_edges(static_cast<csr_graph::vertex_id>(simplex_count()), face_edges);
  }

  inline void complete_partition_map_() {
    const std::size_t S = simplex_count();
    if (!S) { simplex_to_class.clear(); return; }

    if (simplex_to_class.size() != S) {
      simplex_to_class.resize(S);
      for (simplex_id s = 0; s < S; ++s) simplex_to_class[s] = s;
      return;
    }

    class_id next = 0;
    for (class_id c : simplex_to_class)
      if (c != invalid_class && c + 1 > next) next = c + 1;

    for (simplex_id s = 0; s < S; ++s)
      if (simplex_to_class[s] == invalid_class)
        simplex_to_class[s] = next++;
  }

  inline void coarsen_partition_from_map_() {
    simplex_to_class = coarsen_partition_map(std::move(simplex_to_class));
  }

  inline void rebuild_classes_from_map_() {
    const std::size_t S = simplex_count();

    classes.clear();
    free_class_ids.clear();
    simplex_pos_in_class.clear();

    if (!S) return;

    class_id max_c = 0;
    for (class_id c : simplex_to_class) if (c > max_c) max_c = c;
    const std::size_t C = static_cast<std::size_t>(max_c) + 1;

    classes.resize(C);

    std::vector<std::uint32_t> counts(C, 0);
    for (simplex_id s = 0; s < S; ++s) ++counts[simplex_to_class[s]];

    for (std::size_t c = 0; c < C; ++c) {
      class_state& cs = classes[c];
      cs.alive = (counts[c] != 0);
      cs.simplices.clear();
      cs.simplices.reserve(counts[c]);
      cs.feature_of_col.clear();
      static_cast<reduction_pair&>(cs) = reduction_pair();
      if (!cs.alive) free_class_ids.push_back(static_cast<class_id>(c));
    }

    for (simplex_id s = 0; s < S; ++s) {
      class_id c = simplex_to_class[s];
      classes[c].simplices.push_back(s);
    }

    for (std::size_t c = 0; c < C; ++c) {
      if (!classes[c].alive) continue;
      auto& v = classes[c].simplices;
      std::sort(v.begin(), v.end(), [&](simplex_id a, simplex_id b) {
        std::uint32_t da = simplex_vertices_size(a);
        std::uint32_t db = simplex_vertices_size(b);
        if (da != db) return da < db;
        return a < b;
      });
    }

    simplex_pos_in_class.assign(S, 0);
    for (std::size_t c = 0; c < C; ++c) {
      if (!classes[c].alive) continue;
      const auto& v = classes[c].simplices;
      for (std::size_t i = 0; i < v.size(); ++i) {
        simplex_pos_in_class[v[i]] = static_cast<std::uint32_t>(i);
      }
    }
  }

  inline void rebuild_positions_for_class_(class_id c, std::size_t start) {
    assert(class_alive(c));
    auto& v = classes[c].simplices;
    for (std::size_t i = start; i < v.size(); ++i)
      simplex_pos_in_class[v[i]] = static_cast<std::uint32_t>(i);
  }
};
