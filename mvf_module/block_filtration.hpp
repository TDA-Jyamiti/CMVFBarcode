// block_filtration.hpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "partitioned_complex.hpp"
#include "smp_io.hpp"

// VALIDATED BLOCK-FILTRATION TRANSLATION
// ======================================
//
// The general translator connects arbitrary partitions by preserving maximum-overlap cores. This
// translator instead realizes exactly the simplex-wise expansion of one atomic block split from the
// paper. If B = W disjoint-union V with W closed in B and V open in B, first transpose B stably into
// [W | V]. Writing the resulting order on V as v0,...,vk, the recursive expansion flattens to
//
//     B  --RS(vk)-->  B\vk | vk  --RS(v{k-1})--> ... --> W | v0 | ... | vk
//                                                        |
//                                                        +--RM(v1),...,RM(vk)--> W | V
//
// A merge executes this sequence backwards. Thus an atomic transition uses exactly 2|V|-1 right
// splits/merges, while the stable transpositions implement the intervening change of basis. The
// endpoint quotient discards features born and killed wholly inside this expansion.
// An inverted adjacent pair [v in V, w in W] cannot be an incidence because W is closed, so each
// transposition is legal. Relative orders are preserved and the merged order is carried forward;
// the next stable transpositions, rather than an endpoint re-sort, realize the paper's gamma map.

struct block_partition {
  using class_id = partitioned_complex::class_id;

  std::vector<class_id> block_of;
  std::size_t block_count = 0;
};

struct atomic_refinement {
  block_partition::class_id parent = partitioned_complex::invalid_class;
  block_partition::class_id child_0 = partitioned_complex::invalid_class;
  block_partition::class_id child_1 = partitioned_complex::invalid_class;
};

static inline void block_filtration_error_(const char* message) { throw std::runtime_error(message); }

static inline block_partition canonical_block_partition(std::vector<partitioned_complex::class_id> block_of)
{
  using class_id = partitioned_complex::class_id;

  std::vector<class_id> canonical(block_of.size(), partitioned_complex::invalid_class);
  std::size_t block_count = 0;
  for (class_id& block : block_of) {
    if (static_cast<std::size_t>(block) >= canonical.size()) block_filtration_error_("partition contains an invalid block id");
    if (canonical[block] == partitioned_complex::invalid_class) canonical[block] = static_cast<class_id>(block_count++);
    block = canonical[block];
  }
  return { std::move(block_of), block_count };
}

static inline bool is_block_partition(const partitioned_complex& complex, const block_partition& partition)
{
  if (partition.block_of.size() != complex.simplex_count()) return false;
  return canonical_block_partition(complex.coarsen_partition_map(partition.block_of)).block_of == partition.block_of;
}

static inline bool is_one_block_merge_(const block_partition& fine, const block_partition& coarse, atomic_refinement* refinement)
{
  using class_id = block_partition::class_id;

  if (fine.block_of.size() != coarse.block_of.size() || fine.block_count != coarse.block_count + 1) return false;

  std::vector<class_id> image(fine.block_count, partitioned_complex::invalid_class);
  for (std::size_t s = 0; s < fine.block_of.size(); ++s) {
    const class_id source = fine.block_of[s], target = coarse.block_of[s];
    if (source >= fine.block_count || target >= coarse.block_count) return false;
    if (image[source] == partitioned_complex::invalid_class) image[source] = target;
    else if (image[source] != target) return false;
  }

  std::vector<class_id> first_child(coarse.block_count, partitioned_complex::invalid_class);
  atomic_refinement found;
  for (class_id child = 0; child < static_cast<class_id>(fine.block_count); ++child) {
    const class_id parent = image[child];
    if (parent == partitioned_complex::invalid_class) return false;
    if (first_child[parent] == partitioned_complex::invalid_class) first_child[parent] = child;
    else {
      if (found.parent != partitioned_complex::invalid_class) return false;
      found = { parent, first_child[parent], child };
    }
  }
  if (found.parent == partitioned_complex::invalid_class) return false;
  if (refinement) *refinement = found;
  return true;
}

static inline bool is_atomic_block_transition(const block_partition& first, const block_partition& second)
{
  return is_one_block_merge_(first, second, nullptr) || is_one_block_merge_(second, first, nullptr);
}

static inline partitioned_complex::simplex_id first_simplex_in_block_(const block_partition& partition, block_partition::class_id block)
{
  for (partitioned_complex::simplex_id s = 0; s < static_cast<partitioned_complex::simplex_id>(partition.block_of.size()); ++s)
    if (partition.block_of[s] == block) return s;
  return partitioned_complex::invalid_simplex;
}

static inline bool block_closed_in_parent_(const partitioned_complex& complex, const block_partition& fine, const block_partition& coarse, block_partition::class_id child, block_partition::class_id parent)
{
  for (partitioned_complex::simplex_id s = 0; s < static_cast<partitioned_complex::simplex_id>(fine.block_of.size()); ++s) {
    if (fine.block_of[s] != child) continue;
    const partitioned_complex::simplex_id* faces = complex.codim1_faces_data(s);
    const std::uint32_t face_count = complex.codim1_faces_size(s);
    for (std::uint32_t i = 0; i < face_count; ++i)
      if (coarse.block_of[faces[i]] == parent && fine.block_of[faces[i]] != child) return false;
  }
  return true;
}

static inline void closed_open_children_(const partitioned_complex& complex, const block_partition& fine, const block_partition& coarse, const atomic_refinement& refinement, block_partition::class_id& closed, block_partition::class_id& open)
{
  const bool closed_0 = block_closed_in_parent_(complex, fine, coarse, refinement.child_0, refinement.parent);
  const bool closed_1 = block_closed_in_parent_(complex, fine, coarse, refinement.child_1, refinement.parent);
  if (!closed_0 && !closed_1) block_filtration_error_("atomic refinement has no child closed in its parent");

  const auto first_0 = first_simplex_in_block_(fine, refinement.child_0), first_1 = first_simplex_in_block_(fine, refinement.child_1);
  closed = (closed_0 && (!closed_1 || first_0 < first_1)) ? refinement.child_0 : refinement.child_1;
  open = closed == refinement.child_0 ? refinement.child_1 : refinement.child_0;
}

static inline std::size_t transpose_closed_open_(partitioned_complex& complex, partitioned_complex::class_id parent, const block_partition& fine, block_partition::class_id closed, block_partition::class_id open)
{
  const std::size_t size = complex.class_simplices_size(parent);
  std::vector<std::uint8_t> side(size, 0);
  std::size_t open_count = 0;
  for (std::size_t i = 0; i < size; ++i) {
    const auto block = fine.block_of[complex.class_simplices_data(parent)[i]];
    if (block != closed && block != open) block_filtration_error_("atomic split parent contains an unexpected simplex");
    side[i] = static_cast<std::uint8_t>(block == open);
    open_count += side[i];
  }

  for (std::size_t i = 1; i < size; ++i)
    for (std::size_t p = i; p && side[p - 1] > side[p]; --p) { complex.swap_adjacent(parent, p - 1); std::swap(side[p - 1], side[p]); }
  return open_count;
}

static inline void split_block_(partitioned_complex& complex, const block_partition& fine, block_partition::class_id closed, block_partition::class_id open)
{
  const auto closed_simplex = first_simplex_in_block_(fine, closed), open_simplex = first_simplex_in_block_(fine, open);
  if (closed_simplex == partitioned_complex::invalid_simplex || open_simplex == partitioned_complex::invalid_simplex) block_filtration_error_("atomic split has an empty child");
  const auto parent = complex.class_of(closed_simplex);
  if (complex.class_of(open_simplex) != parent) block_filtration_error_("atomic split children do not share one parent");

  const std::size_t open_count = transpose_closed_open_(complex, parent, fine, closed, open);
  if (!open_count || open_count == complex.class_simplices_size(parent)) block_filtration_error_("atomic split has an empty child");
  std::vector<partitioned_complex::class_id> singletons;
  singletons.reserve(open_count);
  for (std::size_t i = 0; i < open_count; ++i) singletons.push_back(complex.split_right(parent));

  const auto open_host = singletons.back();
  for (std::size_t i = singletons.size() - 1; i; --i) complex.merge_right(open_host, singletons[i - 1]);
}

static inline void merge_blocks_(partitioned_complex& complex, const block_partition& fine, block_partition::class_id closed, block_partition::class_id open)
{
  const auto closed_simplex = first_simplex_in_block_(fine, closed), open_simplex = first_simplex_in_block_(fine, open);
  if (closed_simplex == partitioned_complex::invalid_simplex || open_simplex == partitioned_complex::invalid_simplex) block_filtration_error_("atomic merge has an empty child");
  const auto closed_host = complex.class_of(closed_simplex), open_host = complex.class_of(open_simplex);
  if (closed_host == open_host) block_filtration_error_("atomic merge children are not distinct");

  std::vector<partitioned_complex::class_id> singletons;
  while (complex.class_simplices_size(open_host) > 1) singletons.push_back(complex.split_right(open_host));
  complex.merge_right(closed_host, open_host);
  for (auto it = singletons.rbegin(); it != singletons.rend(); ++it) complex.merge_right(closed_host, *it);
}

static inline void translate_block_partition(partitioned_complex& complex, const std::vector<partitioned_complex::class_id>& destination_map)
{
  if (!complex.finalized()) block_filtration_error_("block translation requires a finalized complex");
  const block_partition current = canonical_block_partition(complex.simplex_to_class), destination = canonical_block_partition(destination_map);
  if (!is_block_partition(complex, current)) block_filtration_error_("source is not a block partition");
  if (!is_block_partition(complex, destination)) block_filtration_error_("destination is not a block partition");

  atomic_refinement refinement;
  block_partition::class_id closed = partitioned_complex::invalid_class, open = partitioned_complex::invalid_class;
  if (is_one_block_merge_(destination, current, &refinement)) {
    closed_open_children_(complex, destination, current, refinement, closed, open);
    split_block_(complex, destination, closed, open);
  } else if (is_one_block_merge_(current, destination, &refinement)) {
    closed_open_children_(complex, current, destination, refinement, closed, open);
    merge_blocks_(complex, current, closed, open);
  } else block_filtration_error_("transition is not exactly one binary block merge or split");

  if (canonical_block_partition(complex.simplex_to_class).block_of != destination.block_of) block_filtration_error_("block translation ended at the wrong partition");
}

static inline void translate_block_partition_to_smp(partitioned_complex& complex, const char* path)
{
  std::vector<partitioned_complex::class_id> raw;
  if (!load_smp_partition_map_checked(path, complex, raw)) block_filtration_error_("translate_block_partition_to_smp: failed to parse file or simplex mismatch");
  translate_block_partition(complex, raw);
}
