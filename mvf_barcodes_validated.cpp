#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mvf_module/partitioned_complex.hpp"
#include "mvf_module/smp_io.hpp"

// VALIDATED BLOCK-FILTRATION INPUT
// ================================
//
// This is the restrictive companion to `mvf_barcodes.cpp`, which accepts any sequence of
// multivector fields on one fixed simplicial complex. Here the raw inputs must form a zigzag
// filtration of block partitions in the sense of the motivating preprint arXiv:2608.06507:
//
//                       B^0 <==> B^1 <==> ... <==> B^T
//
// Each arrow means that one adjacent partition refines the other. This program requires the more
// specific case in which every arrow is exactly one binary merge or split, and hence an atomic
// refinement in the terminology of the preprint.
//
// The first input establishes the fixed simplicial complex. Every frame is read again as a raw
// partition so validation sees the blocks before `partitioned_complex` coalesces cyclic classes.
// Block labels have no semantic meaning, so each map is canonically relabeled by the first simplex in
// every block. A frame is valid exactly when SCC coalescing preserves this canonical map; this also
// rules out nonconvex blocks, whose face-poset interval leaves and reenters the block.
//
// For adjacent frames, a fine partition is one binary merge away from a coarse partition precisely
// when it has one extra block and every fine block lies wholly inside one coarse block. Because all
// blocks are nonempty, the one-block count difference then forces exactly two fine blocks to share a
// coarse image. Testing this relation in both directions admits either one merge or one split.
//
// A one-merge refinement has this block-image shape:
//
//     fine F0 -----+----> coarse C0
//                  |
//     fine F1 -----+
//     fine F2 ----------> coarse C1
//     fine F3 ----------> coarse C2
//
// Every fine block has one image; exactly one coarse block has two. Reversing the arrows gives one
// split.

struct block_partition {
  using class_id = partitioned_complex::class_id;

  std::vector<class_id> block_of;
  std::size_t block_count = 0;
};

static inline block_partition canonical_partition_(std::vector<partitioned_complex::class_id> block_of)
{
  using class_id = partitioned_complex::class_id;

  std::vector<class_id> canonical(block_of.size(), partitioned_complex::invalid_class);
  std::size_t block_count = 0;
  for (class_id& block : block_of) {
    if (static_cast<std::size_t>(block) >= canonical.size()) throw std::runtime_error("partition contains an invalid block id");
    if (canonical[block] == partitioned_complex::invalid_class) canonical[block] = static_cast<class_id>(block_count++);
    block = canonical[block];
  }
  return { std::move(block_of), block_count };
}

static inline bool is_one_merge_(const block_partition& fine, const block_partition& coarse)
{
  if (fine.block_of.size() != coarse.block_of.size() || fine.block_count != coarse.block_count + 1) return false;

  std::vector<block_partition::class_id> image(fine.block_count, partitioned_complex::invalid_class);
  for (std::size_t s = 0; s < fine.block_of.size(); ++s) {
    const auto source = fine.block_of[s];
    const auto target = coarse.block_of[s];
    if (image[source] == partitioned_complex::invalid_class) image[source] = target;
    else if (image[source] != target) return false;
  }
  return true;
}

static inline void print_usage_(std::ostream& out, const char* argv0)
{
  out << "usage: " << argv0 << " frame_0000.smp frame_0001.smp ...\n";
}

int main(int argc, char** argv)
{
  try {
    if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) { print_usage_(std::cout, argv[0]); return 0; }
    if (argc < 2) { print_usage_(std::cerr, argv[0]); return 1; }

    partitioned_complex complex;
    if (!load_smp(argv[1], complex)) throw std::runtime_error(std::string(argv[1]) + ": invalid .smp file");

    block_partition previous;
    for (int i = 1; i < argc; ++i) {
      std::vector<partitioned_complex::class_id> raw;
      if (!load_smp_partition_map_checked(argv[i], complex, raw)) throw std::runtime_error(std::string(argv[i]) + ": invalid .smp file or simplex mismatch");

      block_partition current = canonical_partition_(std::move(raw));
      const block_partition coarsened = canonical_partition_(complex.coarsen_partition_map(current.block_of));
      if (current.block_of != coarsened.block_of) throw std::runtime_error(std::string(argv[i]) + ": SCC coalescing changes the partition");

      if (i > 1 && !is_one_merge_(previous, current) && !is_one_merge_(current, previous))
        throw std::runtime_error(std::string(argv[i - 1]) + " -> " + argv[i] + ": transition is not exactly one binary merge or split");
      previous = std::move(current);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "mvf_barcodes_validated: " << e.what() << '\n';
    return 1;
  }
}
