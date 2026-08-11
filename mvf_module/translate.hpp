// translate.hpp
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "graph_utilities.hpp"
#include "partitioned_complex.hpp"
#include "smp_io.hpp"

// PARTITION TRANSLATION
// =====================
//
// Translation edits one finalized partition into another while updating the reduced boundary
// matrices and barcode features after every elementary operation. A maximum-weight overlap matching
// first selects persistent source/target block pairs. Their sorted simplex intersections become
// cores, which adjacent legal swaps make contiguous inside each source block. Everything outside a
// core is split into singleton blocks; unmatched targets take one singleton as a seed; and the
// remaining singletons merge into their target blocks in face-compatible left/right order.
//
//     source partition A
//             |
//             v
//     maximum-weight overlap matching with target B
//             |
//             v
//     matched core = A_i intersect B_j
//             |
//             v
//     reorder source A_i as [ LEFT | CORE | RIGHT ]
//             |
//             v
//     keep each core; split every outside simplex into a singleton
//             |
//             +---- unmatched B_j takes one destined singleton as its seed
//             |
//             v
//     merge LEFT (high dimension first), then RIGHT (low dimension first)
//             |
//             v
//       target partition B
//
// Core vectors are sorted before any helper performs binary search. At completion, every simplex is
// checked against the requested target label, so a failed invariant is reported rather than leaving
// a silently different partition.

static inline void translate_error_(const char* msg) { throw std::runtime_error(msg); }

static inline void face_coface_to_core_(const partitioned_complex& pc, partitioned_complex::simplex_id x, const std::vector<partitioned_complex::simplex_id>& core, bool& face, bool& coface)
{
  face = coface = false;
  const std::uint32_t xn = pc.simplex_vertices_size(x);

  for (partitioned_complex::simplex_id k : core) {
    const std::uint32_t kn = pc.simplex_vertices_size(k);
    if (!face   && xn < kn && pc.simplex_contains(k, x)) face = true;
    if (!coface && xn > kn && pc.simplex_contains(x, k)) coface = true;
    if (face && coface) break;
  }
}

static inline bool core_contains_simplex_(const std::vector<partitioned_complex::simplex_id>& core, partitioned_complex::simplex_id s) {
  return std::binary_search(core.begin(), core.end(), s);
}

struct singleton_rec {
  partitioned_complex::class_id cls = partitioned_complex::invalid_class;
  partitioned_complex::class_id dest = partitioned_complex::invalid_class;
  partitioned_complex::simplex_id x = partitioned_complex::invalid_simplex;
  std::uint32_t dim = 0;
};

struct merge_rec {
  partitioned_complex::class_id target = partitioned_complex::invalid_class;
  partitioned_complex::class_id singleton = partitioned_complex::invalid_class;
  partitioned_complex::simplex_id x = partitioned_complex::invalid_simplex;
  std::uint32_t dim = 0;
};

static inline void transpose_core_contiguous_(partitioned_complex& pc, partitioned_complex::class_id c, const std::vector<partitioned_complex::simplex_id>& core)
{
  if (core.empty()) return;
  if (!pc.class_alive(c)) translate_error_("transpose: class dead");

  const std::size_t n = pc.classes[c].simplices.size();
  if (!n) translate_error_("transpose: empty alive class");

  std::size_t lo = partitioned_complex::npos, hi = 0;
  for (std::size_t p = 0; p < n; ++p) {
    if (core_contains_simplex_(core, pc.classes[c].simplices[p])) {
      if (lo == partitioned_complex::npos) lo = p;
      hi = p;
    }
  }

  if (lo == partitioned_complex::npos) translate_error_("transpose: core simplex not found in host class");

  std::vector<std::uint8_t> key(n, 2); // 0 LEFT, 1 CORE, 2 RIGHT

  for (std::size_t p = 0; p < n; ++p) {
    const auto x = pc.classes[c].simplices[p];
    if (core_contains_simplex_(core, x)) {
      key[p] = 1;
      continue;
    }

    bool face = false, coface = false;
    face_coface_to_core_(pc, x, core, face, coface);

    if (face && !coface) key[p] = 0;
    else if (coface && !face) key[p] = 2;
    else {
      if (p < lo) key[p] = 0;
      else if (p > hi) key[p] = 2;
      else key[p] = ((p - lo) < (hi - p)) ? 0 : 2;
    }
  }

  for (;;) {
    bool any = false;
    for (std::size_t p = 0; p + 1 < n; ++p) {
      if (key[p] <= key[p + 1]) continue;
      pc.swap_adjacent(c, p);
      std::swap(key[p], key[p + 1]);
      any = true;
    }
    if (!any) break;
  }
}

static inline singleton_rec make_singleton_record_(partitioned_complex& pc, partitioned_complex::class_id c, const std::vector<partitioned_complex::class_id>& dest)
{
  if (!pc.class_alive(c) || pc.class_simplices_size(c) != 1) translate_error_("make_singleton_record: expected size-1 alive class");

  const auto x = pc.class_simplices_data(c)[0];
  singleton_rec out;
  out.cls = c;
  out.dest = dest[static_cast<std::size_t>(x)];
  out.x = x;
  out.dim = pc.simplex_dimension(x);
  return out;
}

static inline void split_outside_core_to_singletons_(partitioned_complex& pc, partitioned_complex::class_id c, const std::vector<partitioned_complex::simplex_id>& core, const std::vector<partitioned_complex::class_id>& dest, std::vector<singleton_rec>& out)
{
  if (!pc.class_alive(c)) translate_error_("split: class dead");

  if (core.empty()) {
    while (pc.class_simplices_size(c) > 1) {
      const auto new_c = pc.split_right(c);
      out.push_back(make_singleton_record_(pc, new_c, dest));
    }
    out.push_back(make_singleton_record_(pc, c, dest));
    return;
  }

  while (!core_contains_simplex_(core, pc.class_simplices_data(c)[0])) {
    const auto new_c = pc.split_left(c);
    out.push_back(make_singleton_record_(pc, new_c, dest));
  }

  while (!core_contains_simplex_(core, pc.class_simplices_data(c)[pc.class_simplices_size(c) - 1])) {
    const auto new_c = pc.split_right(c);
    out.push_back(make_singleton_record_(pc, new_c, dest));
  }
}

static inline std::size_t pick_seed_singleton_index_(const std::vector<singleton_rec>& ss, partitioned_complex::class_id dest_class)
{
  std::size_t best = partitioned_complex::npos;
  for (std::size_t i = 0; i < ss.size(); ++i) {
    if (ss[i].dest != dest_class) continue;
    if (best == partitioned_complex::npos || ss[i].x < ss[best].x || (ss[i].x == ss[best].x && ss[i].cls < ss[best].cls)) best = i;
  }
  return best;
}

static inline void translate_partition(partitioned_complex& pc0, const std::vector<partitioned_complex::class_id>& dest)
{
  using simplex_id = partitioned_complex::simplex_id;
  using class_id   = partitioned_complex::class_id;

  if (!pc0.finalized()) translate_error_("translate: pc0 must be finalized");
  const std::size_t S = pc0.simplex_count();
  if (dest.size() != S) translate_error_("translate: dest size mismatch");

  class_id max_b = 0;
  for (class_id b : dest) {
    if (b == partitioned_complex::invalid_class) translate_error_("translate: dest contains invalid_class");
    if (b > max_b) max_b = b;
  }
  const std::size_t n = static_cast<std::size_t>(max_b) + 1;
  if (n > S) translate_error_("translate: dest class id out of range");

  std::vector<class_id> alive;
  alive.reserve(pc0.classes.size());
  for (class_id c = 0; c < static_cast<class_id>(pc0.classes.size()); ++c)
    if (pc0.class_alive(c)) alive.push_back(c);

  const std::size_t m = alive.size();
  if (!m) translate_error_("translate: no alive classes");

  std::vector<int> dense(pc0.classes.size(), -1);
  for (std::size_t i = 0; i < m; ++i) dense[alive[i]] = static_cast<int>(i);

  std::vector<int> match_A;
  match_from_labels(
    S, m, n,
    [&](std::size_t s) -> std::uint32_t {
      const class_id ca = pc0.class_of(static_cast<simplex_id>(s));
      if (static_cast<std::size_t>(ca) >= dense.size() || dense[ca] < 0) translate_error_("translate: simplex in dead or out-of-range class");
      return static_cast<std::uint32_t>(dense[ca]);
    },
    [&](std::size_t s) -> std::uint32_t {
      const class_id b = dest[static_cast<simplex_id>(s)];
      if (static_cast<std::size_t>(b) >= n) translate_error_("translate: dest class OOB");
      return static_cast<std::uint32_t>(b);
    },
    match_A);

  std::vector<class_id> host_of_B(n, partitioned_complex::invalid_class);
  std::vector<std::vector<simplex_id>> core_of_host(pc0.classes.size());

  for (std::size_t i = 0; i < m; ++i) {
    const int j = match_A[i];
    if (j < 0) continue;
    const std::size_t jj = static_cast<std::size_t>(j);
    host_of_B[jj] = alive[i];
  }

  for (simplex_id s = 0; s < static_cast<simplex_id>(S); ++s) {
    const class_id a = pc0.class_of(s);
    const class_id b = dest[s];
    const int i = dense[a];
    if (i >= 0 && match_A[static_cast<std::size_t>(i)] == static_cast<int>(b)) core_of_host[a].push_back(s);
  }

  for (class_id c : alive)
    if (!core_of_host[c].empty()) std::sort(core_of_host[c].begin(), core_of_host[c].end());

  for (class_id c : alive) transpose_core_contiguous_(pc0, c, core_of_host[c]);

  std::vector<singleton_rec> singletons;
  singletons.reserve(S);
  for (class_id c : alive)
    if (pc0.class_alive(c)) split_outside_core_to_singletons_(pc0, c, core_of_host[c], dest, singletons);

  for (std::size_t j = 0; j < n; ++j) {
    if (host_of_B[j] != partitioned_complex::invalid_class) continue;

    const std::size_t idx = pick_seed_singleton_index_(singletons, static_cast<class_id>(j));
    if (idx == partitioned_complex::npos) translate_error_("translate: unmatched target class has no singleton to seed host");

    host_of_B[j] = singletons[idx].cls;
    if (static_cast<std::size_t>(host_of_B[j]) >= core_of_host.size()) core_of_host.resize(static_cast<std::size_t>(host_of_B[j]) + 1);
    core_of_host[host_of_B[j]].assign(1, singletons[idx].x);

    singletons.erase(singletons.begin() + static_cast<std::ptrdiff_t>(idx));
  }

  std::sort(singletons.begin(), singletons.end(), [](const singleton_rec& a, const singleton_rec& b) {
    if (a.dest != b.dest) return a.dest < b.dest;
    if (a.dim != b.dim) return a.dim < b.dim;
    if (a.x != b.x) return a.x < b.x;
    return a.cls < b.cls;
  });

  std::vector<merge_rec> left, right;
  left.reserve(singletons.size());
  right.reserve(singletons.size());

  for (const auto& s : singletons) {
    const class_id H = host_of_B[s.dest];
    if (H == partitioned_complex::invalid_class || H == s.cls || !pc0.class_alive(H) || !pc0.class_alive(s.cls)) translate_error_("translate: invalid host/singleton state");

    bool face = false, coface = false;
    face_coface_to_core_(pc0, s.x, core_of_host[H], face, coface);

    merge_rec mr;
    mr.target = H;
    mr.singleton = s.cls;
    mr.x = s.x;
    mr.dim = s.dim;
    (face && !coface ? left : right).push_back(mr);
  }

  std::sort(left.begin(), left.end(), [](const merge_rec& a, const merge_rec& b) {
    if (a.dim != b.dim) return a.dim > b.dim;
    if (a.x != b.x) return a.x < b.x;
    if (a.singleton != b.singleton) return a.singleton < b.singleton;
    return a.target < b.target;
  });

  for (const auto& mr : left)
    if (pc0.class_alive(mr.singleton)) pc0.merge_left(mr.target, mr.singleton);

  std::sort(right.begin(), right.end(), [](const merge_rec& a, const merge_rec& b) {
    if (a.dim != b.dim) return a.dim < b.dim;
    if (a.x != b.x) return a.x < b.x;
    if (a.singleton != b.singleton) return a.singleton < b.singleton;
    return a.target < b.target;
  });

  for (const auto& mr : right)
    if (pc0.class_alive(mr.singleton)) pc0.merge_right(mr.target, mr.singleton);

  for (simplex_id s = 0; s < static_cast<simplex_id>(S); ++s)
    if (pc0.class_of(s) != host_of_B[static_cast<std::size_t>(dest[s])]) translate_error_("translate: final partition mismatch vs dest");
}

static inline void translate_partition_to_smp(partitioned_complex& pc0, const char* path)
{
  std::vector<partitioned_complex::class_id> raw;
  if (!load_smp_partition_map_checked(path, pc0, raw)) throw std::runtime_error("translate_partition_to_smp: failed to parse file or simplex mismatch");
  translate_partition(pc0, pc0.coarsen_partition_map(std::move(raw)));
}
