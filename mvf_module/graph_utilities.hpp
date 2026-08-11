// graph_utilities.hpp
#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

// SHARED GRAPH UTILITIES
// ======================
//
// `csr_graph` is the single sparse graph representation used throughout the library. It represents
// the ordered codimension-one faces of each simplex, directed relations between multivector classes,
// and weighted relations between the classes of two successive partitions. The sections below record
// its data layout and explain the major algorithms and project-specific terminology used with it.
//
// CSR STORAGE
// -----------
//
// Vertices are the consecutive 32-bit integers `0, 1, ..., n - 1`. Compressed sparse row storage
// concatenates their outgoing adjacency lists in `edges`; the targets of vertex `v` occupy
//
//     edges[offset[v] ... offset[v + 1])
//
// Thus a nonempty graph's `offset` has `n + 1` entries, starts at zero, is nondecreasing, and ends at
// `edges.size()`. The canonical zero-vertex graph instead leaves all three arrays empty.
// For example, the adjacency lists
//
//     0 : 1, 4
//     1 :
//     2 : 0, 3, 4
//     3 : 3
//     4 :
//
// are stored as
//
//     offset = [0, 2, 2, 5, 6, 6]
//     edges  = [1, 4, 0, 3, 4, 3]
//
// `weights` is either empty, giving every edge implicit weight one, or parallel to `edges`, with
// `weights[i]` belonging to the target in `edges[i]`:
//
//     edge slot     0  1  2  3  4  5
//     edges        [1, 4, 0, 3, 4, 3]
//     weights      [7, 2, 5, 4, 9, 1]
//
// Unweighted graphs therefore pay for only one 32-bit target per edge; weighted graphs add one
// aligned 32-bit weight per edge.
//
//
// MAXIMUM-WEIGHT BIPARTITE MATCHING
// ---------------------------------
//
// Matching pairs classes without using a class more than once and maximizes the sum of their overlap
// weights. `match_of_left[a]` is the right class paired with left class `a`, or `-1` if unmatched.
// Positive-edge connected components are independent, so the implementation finds them by stack
// traversal and solves each separately. It sorts and densely relabels the component's left and right
// ids before solving, reducing solver state and fixing traversal order.
//
// The local score for pairing left `i` with right `j` is
//
//     rank(i, j)  = i * right_count + j + 1
//     BIG         = left_count * right_count * max(left_count, right_count) + 1
//     score(i, j) = overlap_weight(i, j) * BIG - rank(i, j)
//
// `BIG` exceeds the largest possible total rank penalty, so gaining one unit of overlap always
// dominates every tie-break penalty. Equal-overlap solutions therefore prefer smaller total rank;
// remaining ties follow sorted row and reconstruction order. Missing and zero-weight edges are not
// selected.
//
// A component with one vertex on either side is handled by a direct maximum scan. Other components
// use subset dynamic programming when either side has at most 18 vertices and Hungarian assignment
// otherwise.
//
// SUBSET DYNAMIC PROGRAMMING
// --------------------------
//
// When the right side has at most 18 vertices, a bit mask records its used vertices. `F(i, mask)`
// is the best encoded total obtainable from left vertices `i, i + 1, ...`; it considers leaving `i`
// unmatched or pairing it with each available neighbor `j`:
//
//     F(i, mask) = max(F(i + 1, mask),
//                      score(i, j) + F(i + 1, mask union {j}) for every available edge (i, j))
//
// Memoization evaluates each state once and a second walk reconstructs the choices. If the left
// side is the small one, the symmetric recurrence processes right vertices and masks used left
// vertices. The state count is `O(L * 2^R)` or `O(R * 2^L)`, depending on the masked side.
//
// HUNGARIAN ALGORITHM
// -------------------
//
// With both sides larger than 18, the code forms a square `N x N` matrix, `N = max(L, R)`, padding
// the smaller side with zero-score dummy vertices that represent unmatched classes. It converts
// score maximization to minimum-cost assignment by setting
//
//     cost(i, j) = maximum_matrix_score - score(i, j).
//
// The Hungarian algorithm adds rows one at a time, maintaining row and column potentials and growing
// a least-reduced-cost alternating path until the current assignment can be augmented. Dummy and
// nonpositive-score pairs are discarded afterward. This gives an exact maximum-overlap matching in
// `O(N^3)` time and `O(N^2)` storage.
struct csr_graph {
  using vertex_id = std::uint32_t;
  using weight_type = std::uint32_t;
  using edge_index = std::uint32_t;

  struct weighted_edge {
    vertex_id source = 0;
    vertex_id target = 0;
    weight_type weight = 0;
  };

  struct unweighted_edge {
    vertex_id source = 0;
    vertex_id target = 0;
  };

  struct component_map {
    std::vector<vertex_id> of_vertex;
    vertex_id count = 0;
  };

  std::vector<edge_index> offset;
  std::vector<vertex_id> edges;
  std::vector<weight_type> weights;

  inline std::size_t vertex_count() const { return offset.empty() ? 0 : offset.size() - 1; }
  inline std::size_t edge_count() const { return edges.size(); }
  inline bool weighted() const { return !weights.empty(); }
  inline edge_index edge_begin(vertex_id vertex) const { assert(static_cast<std::size_t>(vertex) < vertex_count()); return offset[vertex]; }
  inline edge_index edge_end(vertex_id vertex) const { assert(static_cast<std::size_t>(vertex) < vertex_count()); return offset[static_cast<std::size_t>(vertex) + 1]; }
  inline vertex_id edge_target(edge_index edge) const { assert(static_cast<std::size_t>(edge) < edges.size()); return edges[edge]; }
  inline weight_type edge_weight(edge_index edge) const { assert(static_cast<std::size_t>(edge) < edges.size()); return weights.empty() ? 1 : weights[edge]; }
  inline std::size_t degree(vertex_id vertex) const { return static_cast<std::size_t>(edge_end(vertex) - edge_begin(vertex)); }

  static inline csr_graph from_packed_edges(vertex_id vertex_count, std::vector<std::uint64_t>& packed_edges) {
    csr_graph graph;
    if (!vertex_count) {
      if (!packed_edges.empty()) throw std::runtime_error("csr_graph::from_packed_edges: edge endpoint out of range");
      return graph;
    }

    graph.offset.assign(static_cast<std::size_t>(vertex_count) + 1, 0);
    if (packed_edges.empty()) return graph;

    std::sort(packed_edges.begin(), packed_edges.end());
    packed_edges.erase(std::unique(packed_edges.begin(), packed_edges.end()), packed_edges.end());
    if (packed_edges.size() > std::numeric_limits<edge_index>::max()) throw std::runtime_error("csr_graph::from_packed_edges: too many edges");

    for (std::uint64_t packed_edge : packed_edges) {
      const vertex_id source = static_cast<vertex_id>(packed_edge >> 32);
      const vertex_id target = static_cast<vertex_id>(packed_edge & 0xffffffffu);
      if (source >= vertex_count || target >= vertex_count) throw std::runtime_error("csr_graph::from_packed_edges: edge endpoint out of range");
      ++graph.offset[static_cast<std::size_t>(source) + 1];
    }

    std::partial_sum(graph.offset.begin(), graph.offset.end(), graph.offset.begin());
    graph.edges.resize(packed_edges.size());
    std::vector<edge_index> next(graph.offset.begin(), graph.offset.end() - 1);

    for (std::uint64_t packed_edge : packed_edges) {
      const vertex_id source = static_cast<vertex_id>(packed_edge >> 32);
      graph.edges[next[source]++] = static_cast<vertex_id>(packed_edge & 0xffffffffu);
    }
    return graph;
  }

  static inline csr_graph from_edges(vertex_id vertex_count, const std::vector<unweighted_edge>& input, bool add_reverse = false) {
    csr_graph graph;
    if (!vertex_count) {
      if (!input.empty()) throw std::runtime_error("csr_graph::from_edges: edge endpoint out of range");
      return graph;
    }

    std::size_t stored_edge_count = input.size();
    if (add_reverse)
      for (const unweighted_edge& edge : input) stored_edge_count += (edge.source != edge.target);
    if (stored_edge_count > std::numeric_limits<edge_index>::max()) throw std::runtime_error("csr_graph::from_edges: too many edges");

    graph.offset.assign(static_cast<std::size_t>(vertex_count) + 1, 0);
    for (const unweighted_edge& edge : input) {
      if (edge.source >= vertex_count || edge.target >= vertex_count) throw std::runtime_error("csr_graph::from_edges: edge endpoint out of range");
      ++graph.offset[static_cast<std::size_t>(edge.source) + 1];
      if (add_reverse && edge.source != edge.target) ++graph.offset[static_cast<std::size_t>(edge.target) + 1];
    }

    std::partial_sum(graph.offset.begin(), graph.offset.end(), graph.offset.begin());
    graph.edges.resize(stored_edge_count);
    std::vector<edge_index> next(graph.offset.begin(), graph.offset.end() - 1);
    for (const unweighted_edge& edge : input) {
      graph.edges[next[edge.source]++] = edge.target;
      if (add_reverse && edge.source != edge.target) graph.edges[next[edge.target]++] = edge.source;
    }
    return graph;
  }

  static inline csr_graph from_weighted_edges(vertex_id vertex_count, const std::vector<weighted_edge>& input, bool add_reverse = false) {
    csr_graph graph;
    if (!vertex_count) {
      if (!input.empty()) throw std::runtime_error("csr_graph::from_weighted_edges: edge endpoint out of range");
      return graph;
    }

    std::size_t stored_edge_count = input.size();
    if (add_reverse)
      for (const weighted_edge& edge : input) stored_edge_count += (edge.source != edge.target);
    if (stored_edge_count > std::numeric_limits<edge_index>::max()) throw std::runtime_error("csr_graph::from_weighted_edges: too many edges");

    graph.offset.assign(static_cast<std::size_t>(vertex_count) + 1, 0);
    for (const weighted_edge& edge : input) {
      if (edge.source >= vertex_count || edge.target >= vertex_count) throw std::runtime_error("csr_graph::from_weighted_edges: edge endpoint out of range");
      ++graph.offset[static_cast<std::size_t>(edge.source) + 1];
      if (add_reverse && edge.source != edge.target) ++graph.offset[static_cast<std::size_t>(edge.target) + 1];
    }

    std::partial_sum(graph.offset.begin(), graph.offset.end(), graph.offset.begin());
    graph.edges.resize(stored_edge_count);
    graph.weights.resize(stored_edge_count);
    std::vector<edge_index> next(graph.offset.begin(), graph.offset.end() - 1);

    for (const weighted_edge& edge : input) {
      edge_index slot = next[edge.source]++;
      graph.edges[slot] = edge.target;
      graph.weights[slot] = edge.weight;
      if (add_reverse && edge.source != edge.target) {
        slot = next[edge.target]++;
        graph.edges[slot] = edge.source;
        graph.weights[slot] = edge.weight;
      }
    }

    std::vector<std::uint64_t> row;
    for (vertex_id vertex = 0; vertex < vertex_count; ++vertex) {
      const edge_index begin = graph.offset[vertex], end = graph.offset[static_cast<std::size_t>(vertex) + 1];
      if (end - begin < 2) continue;
      row.clear();
      row.reserve(end - begin);
      for (edge_index edge = begin; edge < end; ++edge) row.push_back((std::uint64_t(graph.edges[edge]) << 32) | graph.weights[edge]);
      std::sort(row.begin(), row.end());
      for (edge_index edge = begin; edge < end; ++edge) { graph.edges[edge] = static_cast<vertex_id>(row[edge - begin] >> 32); graph.weights[edge] = static_cast<weight_type>(row[edge - begin]); }
    }
    return graph;
  }

  inline csr_graph reversed() const {
    csr_graph reverse;
    if (!vertex_count()) return reverse;

    reverse.offset.assign(vertex_count() + 1, 0);
    reverse.edges.resize(edges.size());
    if (!weights.empty()) reverse.weights.resize(weights.size());
    for (vertex_id target : edges) ++reverse.offset[static_cast<std::size_t>(target) + 1];

    std::partial_sum(reverse.offset.begin(), reverse.offset.end(), reverse.offset.begin());
    std::vector<edge_index> next(reverse.offset.begin(), reverse.offset.end() - 1);
    for (vertex_id source = 0; source < static_cast<vertex_id>(vertex_count()); ++source) {
      for (edge_index edge = edge_begin(source); edge < edge_end(source); ++edge) {
        const edge_index slot = next[edge_target(edge)]++;
        reverse.edges[slot] = source;
        if (!weights.empty()) reverse.weights[slot] = weights[edge];
      }
    }
    return reverse;
  }

  inline component_map strongly_connected_components() const {
    static constexpr vertex_id unassigned = std::numeric_limits<vertex_id>::max();
    component_map components;
    components.of_vertex.assign(vertex_count(), unassigned);
    if (!vertex_count()) return components;

    const csr_graph reverse = reversed();
    std::vector<std::uint8_t> seen(vertex_count(), 0);
    std::vector<vertex_id> finish_order;
    finish_order.reserve(vertex_count());

    struct depth_first_frame { vertex_id vertex = 0; edge_index next_edge = 0; };
    std::vector<depth_first_frame> depth_first_stack;
    for (vertex_id start = 0; start < static_cast<vertex_id>(vertex_count()); ++start) {
      if (seen[start]) continue;
      depth_first_stack.clear();
      depth_first_stack.push_back({ start, edge_begin(start) });
      seen[start] = 1;

      while (!depth_first_stack.empty()) {
        depth_first_frame& frame = depth_first_stack.back();
        if (frame.next_edge < edge_end(frame.vertex)) {
          const vertex_id neighbor = edge_target(frame.next_edge++);
          if (!seen[neighbor]) { seen[neighbor] = 1; depth_first_stack.push_back({ neighbor, edge_begin(neighbor) }); }
        } else {
          finish_order.push_back(frame.vertex);
          depth_first_stack.pop_back();
        }
      }
    }

    std::vector<vertex_id> search_stack;
    for (std::size_t order = finish_order.size(); order; ) {
      const vertex_id start = finish_order[--order];
      if (components.of_vertex[start] != unassigned) continue;

      components.of_vertex[start] = components.count;
      search_stack.clear();
      search_stack.push_back(start);
      while (!search_stack.empty()) {
        const vertex_id vertex = search_stack.back();
        search_stack.pop_back();
        for (edge_index edge = reverse.edge_begin(vertex); edge < reverse.edge_end(vertex); ++edge) {
          const vertex_id neighbor = reverse.edge_target(edge);
          if (components.of_vertex[neighbor] == unassigned) { components.of_vertex[neighbor] = components.count; search_stack.push_back(neighbor); }
        }
      }
      ++components.count;
    }
    return components;
  }
};

static constexpr std::size_t matching_dp_mask_limit = 18;

static inline std::int64_t tie_big_(std::size_t left_n, std::size_t right_n) {
  const std::int64_t ln = static_cast<std::int64_t>(left_n);
  const std::int64_t rn = static_cast<std::int64_t>(right_n);
  return ln * rn * std::max<std::int64_t>(ln, rn) + 1;
}

static inline std::int64_t encoded_score_(std::uint32_t weight, std::uint32_t rank, std::int64_t big) {
  return static_cast<std::int64_t>(weight) * big - static_cast<std::int64_t>(rank);
}

static inline std::uint32_t matching_rank_(std::size_t left, std::size_t right, std::size_t right_n) {
  return static_cast<std::uint32_t>(left * right_n + right + 1);
}

static inline void solve_component_dp_small_right_(const csr_graph& graph, std::size_t left_n, std::size_t right_n, std::vector<int>& local_match_left)
{
  const std::size_t state_n = std::size_t(1) << right_n;
  const std::int64_t NEG = std::numeric_limits<std::int64_t>::min() / 4;
  const std::int64_t BIG = tie_big_(left_n, right_n);
  std::vector<std::int64_t> memo((left_n + 1) * state_n, NEG);
  std::vector<std::uint8_t> seen((left_n + 1) * state_n, 0);
  auto idx = [&](std::size_t i, std::size_t mask) { return i * state_n + mask; };

  struct solver {
    const csr_graph& graph;
    std::size_t left_n;
    std::size_t right_n;
    std::int64_t BIG;
    std::vector<std::int64_t>& memo;
    std::vector<std::uint8_t>& seen;
    decltype(idx)& idx_fn;

    std::int64_t go(std::size_t i, std::size_t mask) {
      const std::size_t k = idx_fn(i, mask);
      if (seen[k]) return memo[k];
      seen[k] = 1;
      if (i == left_n) return memo[k] = 0;

      std::int64_t best = go(i + 1, mask);
      for (csr_graph::edge_index edge = graph.edge_begin(static_cast<csr_graph::vertex_id>(i)); edge < graph.edge_end(static_cast<csr_graph::vertex_id>(i)); ++edge) {
        const std::size_t j = static_cast<std::size_t>(graph.edge_target(edge)) - left_n;
        const std::size_t bit = std::size_t(1) << j;
        if (mask & bit) continue;
        const std::int64_t cand = encoded_score_(graph.edge_weight(edge), matching_rank_(i, j, right_n), BIG) + go(i + 1, mask | bit);
        if (cand > best) best = cand;
      }
      return memo[k] = best;
    }
  } solve{ graph, left_n, right_n, BIG, memo, seen, idx };

  local_match_left.assign(left_n, -1);
  (void)solve.go(0, 0);
  std::size_t mask = 0;
  for (std::size_t i = 0; i < left_n; ++i) {
    const std::int64_t best = solve.go(i, mask);
    std::int64_t best_match = NEG;
    int best_j = -1;

    for (csr_graph::edge_index edge = graph.edge_begin(static_cast<csr_graph::vertex_id>(i)); edge < graph.edge_end(static_cast<csr_graph::vertex_id>(i)); ++edge) {
      const std::size_t j = static_cast<std::size_t>(graph.edge_target(edge)) - left_n;
      const std::size_t bit = std::size_t(1) << j;
      if (mask & bit) continue;
      const std::int64_t cand = encoded_score_(graph.edge_weight(edge), matching_rank_(i, j, right_n), BIG) + solve.go(i + 1, mask | bit);
      if (cand > best_match) { best_match = cand; best_j = static_cast<int>(j); }
    }

    if (best_match > solve.go(i + 1, mask) && best_match == best) {
      local_match_left[i] = best_j;
      mask |= std::size_t(1) << static_cast<std::size_t>(best_j);
    }
  }
}

static inline void solve_component_dp_small_left_(const csr_graph& graph, std::size_t left_n, std::size_t right_n, std::vector<int>& local_match_left)
{
  const std::size_t state_n = std::size_t(1) << left_n;
  const std::int64_t NEG = std::numeric_limits<std::int64_t>::min() / 4;
  const std::int64_t BIG = tie_big_(left_n, right_n);
  std::vector<std::int64_t> memo((right_n + 1) * state_n, NEG);
  std::vector<std::uint8_t> seen((right_n + 1) * state_n, 0);
  auto idx = [&](std::size_t j, std::size_t mask) { return j * state_n + mask; };

  struct solver {
    const csr_graph& graph;
    std::size_t left_n;
    std::size_t right_n;
    std::int64_t BIG;
    std::vector<std::int64_t>& memo;
    std::vector<std::uint8_t>& seen;
    decltype(idx)& idx_fn;

    std::int64_t go(std::size_t j, std::size_t mask) {
      const std::size_t k = idx_fn(j, mask);
      if (seen[k]) return memo[k];
      seen[k] = 1;
      if (j == right_n) return memo[k] = 0;

      std::int64_t best = go(j + 1, mask);
      const csr_graph::vertex_id vertex = static_cast<csr_graph::vertex_id>(left_n + j);
      for (csr_graph::edge_index edge = graph.edge_begin(vertex); edge < graph.edge_end(vertex); ++edge) {
        const std::size_t i = graph.edge_target(edge);
        const std::size_t bit = std::size_t(1) << i;
        if (mask & bit) continue;
        const std::int64_t cand = encoded_score_(graph.edge_weight(edge), matching_rank_(i, j, right_n), BIG) + go(j + 1, mask | bit);
        if (cand > best) best = cand;
      }
      return memo[k] = best;
    }
  } solve{ graph, left_n, right_n, BIG, memo, seen, idx };

  local_match_left.assign(left_n, -1);
  (void)solve.go(0, 0);
  std::size_t mask = 0;
  for (std::size_t j = 0; j < right_n; ++j) {
    const std::int64_t best = solve.go(j, mask);
    std::int64_t best_match = NEG;
    int best_i = -1;
    const csr_graph::vertex_id vertex = static_cast<csr_graph::vertex_id>(left_n + j);

    for (csr_graph::edge_index edge = graph.edge_begin(vertex); edge < graph.edge_end(vertex); ++edge) {
      const std::size_t i = graph.edge_target(edge);
      const std::size_t bit = std::size_t(1) << i;
      if (mask & bit) continue;
      const std::int64_t cand = encoded_score_(graph.edge_weight(edge), matching_rank_(i, j, right_n), BIG) + solve.go(j + 1, mask | bit);
      if (cand > best_match) { best_match = cand; best_i = static_cast<int>(i); }
    }

    if (best_match > solve.go(j + 1, mask) && best_match == best) {
      local_match_left[static_cast<std::size_t>(best_i)] = static_cast<int>(j);
      mask |= std::size_t(1) << static_cast<std::size_t>(best_i);
    }
  }
}

static inline void solve_component_hungarian_(const csr_graph& graph, std::size_t left_n, std::size_t right_n, std::vector<int>& local_match_left)
{
  const std::size_t N = std::max(left_n, right_n);
  const std::int64_t BIG = tie_big_(left_n, right_n);
  std::vector<std::int64_t> score(N * N, 0);
  auto at = [&](std::size_t i, std::size_t j) -> std::int64_t& { return score[i * N + j]; };

  for (std::size_t i = 0; i < left_n; ++i)
    for (csr_graph::edge_index edge = graph.edge_begin(static_cast<csr_graph::vertex_id>(i)); edge < graph.edge_end(static_cast<csr_graph::vertex_id>(i)); ++edge) {
      const std::size_t j = static_cast<std::size_t>(graph.edge_target(edge)) - left_n;
      at(i, j) = encoded_score_(graph.edge_weight(edge), matching_rank_(i, j, right_n), BIG);
    }

  std::int64_t max_score = 0;
  for (std::size_t i = 0; i < left_n; ++i)
    for (std::size_t j = 0; j < right_n; ++j)
      if (at(i, j) > max_score) max_score = at(i, j);

  std::vector<std::int64_t> cost(N * N, max_score);
  auto cat = [&](std::size_t i, std::size_t j) -> std::int64_t& { return cost[i * N + j]; };
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j)
      cat(i, j) = max_score - at(i, j);

  const std::int64_t INF = std::numeric_limits<std::int64_t>::max() / 4;
  std::vector<std::int64_t> u(N + 1, 0), v(N + 1, 0);
  std::vector<std::size_t> p(N + 1, 0), way(N + 1, 0);
  for (std::size_t i = 1; i <= N; ++i) {
    p[0] = i;
    std::size_t j0 = 0;
    std::vector<std::int64_t> minv(N + 1, INF);
    std::vector<std::uint8_t> used(N + 1, 0);

    do {
      used[j0] = 1;
      const std::size_t i0 = p[j0];
      std::int64_t delta = INF;
      std::size_t j1 = 0;

      for (std::size_t j = 1; j <= N; ++j) {
        if (used[j]) continue;
        const std::int64_t cur = cat(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }

      for (std::size_t j = 0; j <= N; ++j)
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else minv[j] -= delta;
      j0 = j1;
    } while (p[j0] != 0);

    do {
      const std::size_t j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  local_match_left.assign(left_n, -1);
  for (std::size_t j = 1; j <= N; ++j) {
    if (!p[j]) continue;
    const std::size_t i = p[j] - 1;
    const std::size_t col = j - 1;
    if (i < left_n && col < right_n && at(i, col) > 0) local_match_left[i] = static_cast<int>(col);
  }
}

template <typename left_fn, typename right_fn>
static inline csr_graph build_overlap_graph(std::size_t simplex_count, std::size_t left_count, std::size_t right_count, left_fn&& left_of_simplex, right_fn&& right_of_simplex)
{
  if (left_count > std::numeric_limits<csr_graph::vertex_id>::max() || right_count > std::numeric_limits<csr_graph::vertex_id>::max() - left_count) throw std::runtime_error("build_overlap_graph: too many vertices");

  std::vector<std::uint64_t> packed;
  packed.reserve(simplex_count);
  for (std::size_t simplex = 0; simplex < simplex_count; ++simplex) {
    const std::size_t left_value = static_cast<std::size_t>(left_of_simplex(simplex));
    const std::size_t right_value = static_cast<std::size_t>(right_of_simplex(simplex));
    if (left_value >= left_count || right_value >= right_count) throw std::runtime_error("build_overlap_graph: class label out of range");
    const std::uint32_t left = static_cast<std::uint32_t>(left_value);
    const std::uint32_t right = static_cast<std::uint32_t>(right_value);
    packed.push_back((std::uint64_t(left) << 32) | right);
  }
  std::sort(packed.begin(), packed.end());

  std::vector<csr_graph::weighted_edge> overlap;
  overlap.reserve(packed.size());
  for (std::size_t i = 0; i < packed.size(); ) {
    std::size_t j = i + 1;
    while (j < packed.size() && packed[j] == packed[i]) ++j;
    overlap.push_back({ static_cast<std::uint32_t>(packed[i] >> 32), static_cast<std::uint32_t>(left_count + static_cast<std::uint32_t>(packed[i])), static_cast<std::uint32_t>(j - i) });
    i = j;
  }
  return csr_graph::from_weighted_edges(static_cast<csr_graph::vertex_id>(left_count + right_count), overlap, true);
}

static inline void max_weight_matching_sparse(std::size_t left_count, std::size_t right_count, const csr_graph& overlap_graph, std::vector<int>& match_of_left)
{
  match_of_left.assign(left_count, -1);
  if (!left_count || !right_count || !overlap_graph.edge_count()) return;
  if (overlap_graph.vertex_count() != left_count + right_count) throw std::runtime_error("max_weight_matching_sparse: graph size mismatch");

  std::vector<std::uint8_t> seen(overlap_graph.vertex_count(), 0);
  std::vector<csr_graph::vertex_id> stack;
  std::vector<int> right_local(right_count, -1);
  stack.reserve(64);

  for (csr_graph::vertex_id left0 = 0; left0 < static_cast<csr_graph::vertex_id>(left_count); ++left0) {
    if (seen[left0] || !overlap_graph.degree(left0)) continue;

    std::vector<std::uint32_t> left_ids, right_ids;
    stack.clear();
    stack.push_back(left0);
    seen[left0] = 1;
    while (!stack.empty()) {
      const csr_graph::vertex_id vertex = stack.back();
      stack.pop_back();
      if (vertex < left_count) left_ids.push_back(vertex);
      else right_ids.push_back(static_cast<std::uint32_t>(vertex - left_count));

      for (csr_graph::edge_index edge = overlap_graph.edge_begin(vertex); edge < overlap_graph.edge_end(vertex); ++edge) {
        if (!overlap_graph.edge_weight(edge)) continue;
        const csr_graph::vertex_id neighbor = overlap_graph.edge_target(edge);
        if (!seen[neighbor]) { seen[neighbor] = 1; stack.push_back(neighbor); }
      }
    }

    std::sort(left_ids.begin(), left_ids.end());
    std::sort(right_ids.begin(), right_ids.end());
    if (right_ids.empty()) continue;
    const std::size_t left_n = left_ids.size(), right_n = right_ids.size();
    for (std::size_t j = 0; j < right_n; ++j) right_local[right_ids[j]] = static_cast<int>(j);

    std::vector<csr_graph::weighted_edge> component_edges;
    for (std::size_t i = 0; i < left_n; ++i) {
      const csr_graph::vertex_id left = left_ids[i];
      for (csr_graph::edge_index edge = overlap_graph.edge_begin(left); edge < overlap_graph.edge_end(left); ++edge) {
        if (!overlap_graph.edge_weight(edge)) continue;
        const csr_graph::vertex_id target = overlap_graph.edge_target(edge);
        if (target < left_count || target >= left_count + right_count) throw std::runtime_error("max_weight_matching_sparse: non-bipartite edge");
        const int j = right_local[target - left_count];
        if (j < 0) throw std::runtime_error("max_weight_matching_sparse: incomplete component map");
        component_edges.push_back({ static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(left_n + static_cast<std::size_t>(j)), overlap_graph.edge_weight(edge) });
      }
    }
    csr_graph component_graph = csr_graph::from_weighted_edges(static_cast<csr_graph::vertex_id>(left_n + right_n), component_edges, true);

    std::vector<int> local_match_left;
    if (left_n == 1) {
      local_match_left.assign(1, -1);
      std::uint32_t best_weight = 0;
      for (csr_graph::edge_index edge = component_graph.edge_begin(0); edge < component_graph.edge_end(0); ++edge) {
        const int j = static_cast<int>(component_graph.edge_target(edge) - left_n);
        if (component_graph.edge_weight(edge) > best_weight || (component_graph.edge_weight(edge) == best_weight && (local_match_left[0] < 0 || j < local_match_left[0]))) { best_weight = component_graph.edge_weight(edge); local_match_left[0] = j; }
      }
    } else if (right_n == 1) {
      local_match_left.assign(left_n, -1);
      std::uint32_t best_weight = 0;
      int best_i = -1;
      const csr_graph::vertex_id right = static_cast<csr_graph::vertex_id>(left_n);
      for (csr_graph::edge_index edge = component_graph.edge_begin(right); edge < component_graph.edge_end(right); ++edge) {
        const int i = static_cast<int>(component_graph.edge_target(edge));
        if (component_graph.edge_weight(edge) > best_weight || (component_graph.edge_weight(edge) == best_weight && (best_i < 0 || i < best_i))) { best_weight = component_graph.edge_weight(edge); best_i = i; }
      }
      if (best_i >= 0) local_match_left[static_cast<std::size_t>(best_i)] = 0;
    } else if (right_n <= matching_dp_mask_limit) solve_component_dp_small_right_(component_graph, left_n, right_n, local_match_left);
    else if (left_n <= matching_dp_mask_limit) solve_component_dp_small_left_(component_graph, left_n, right_n, local_match_left);
    else solve_component_hungarian_(component_graph, left_n, right_n, local_match_left);

    for (std::size_t i = 0; i < left_n; ++i)
      if (local_match_left[i] >= 0) match_of_left[left_ids[i]] = static_cast<int>(right_ids[static_cast<std::size_t>(local_match_left[i])]);
    for (std::uint32_t right : right_ids) right_local[right] = -1;
  }
}

template <typename left_fn, typename right_fn>
static inline void match_from_labels(std::size_t simplex_count, std::size_t left_class_count, std::size_t right_class_count, left_fn&& left_of_simplex, right_fn&& right_of_simplex, std::vector<int>& match_of_left)
{
  if (!left_class_count || !right_class_count) {
    for (std::size_t simplex = 0; simplex < simplex_count; ++simplex) { (void)left_of_simplex(simplex); (void)right_of_simplex(simplex); }
    match_of_left.assign(left_class_count, -1);
    return;
  }
  csr_graph graph = build_overlap_graph(simplex_count, left_class_count, right_class_count, std::forward<left_fn>(left_of_simplex), std::forward<right_fn>(right_of_simplex));
  max_weight_matching_sparse(left_class_count, right_class_count, graph, match_of_left);
}
