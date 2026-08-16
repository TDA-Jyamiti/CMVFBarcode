#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mvf_module/block_filtration.hpp"

// VALIDATED BLOCK-FILTRATION INPUT
// ================================
//
// This is the restrictive companion to `cmvf_barcodes.cpp`, which accepts any sequence of
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

      block_partition current = canonical_block_partition(std::move(raw));
      if (!is_block_partition(complex, current)) throw std::runtime_error(std::string(argv[i]) + ": SCC coalescing changes the partition");

      if (i > 1 && !is_atomic_block_transition(previous, current))
        throw std::runtime_error(std::string(argv[i - 1]) + " -> " + argv[i] + ": transition is not exactly one binary merge or split");
      previous = std::move(current);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cmvf_barcodes_validated: " << e.what() << '\n';
    return 1;
  }
}
