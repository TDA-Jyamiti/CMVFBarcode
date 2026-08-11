// barcodes.hpp
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "partitioned_complex.hpp"

// ENDPOINT-QUOTIENT BARCODES
// ==========================
//
// Features are born and killed at internal translation steps. `endpoint_steps` stores the internal
// step reached at each input frame, and every event in the translation from frame `i` to frame
// `i + 1` is projected to `i + 1`. A feature born and killed within that same open-left,
// closed-right step interval is suppressed entirely. Infinite deaths remain infinite and are written
// as JSON `null` by the command-line entry point.
//
//                         translation i -> i + 1
//
//     endpoint_steps[i]  (---------*-----------*---------]  endpoint_steps[i + 1]
//                                  birth       death
//                                     \_________/
//                                  both become i + 1
//
// Any event `t` with `endpoint_steps[i] < t <= endpoint_steps[i + 1]` maps to frame `i + 1`. If the
// two marked events belong to the same feature, its zero-length quotient interval is omitted.

struct barcode_interval {
  std::uint32_t dimension = 0;
  partitioned_complex::time_type birth = 0;
  partitioned_complex::time_type death = partitioned_complex::time_infty;
};

static inline void validate_endpoint_steps_(const std::vector<partitioned_complex::time_type>& endpoint_steps)
{
  if (endpoint_steps.empty()) throw std::invalid_argument("endpoint steps are empty");
  for (std::size_t i = 1; i < endpoint_steps.size(); ++i)
    if (endpoint_steps[i - 1] > endpoint_steps[i]) throw std::invalid_argument("endpoint steps are not sorted");
}

static inline partitioned_complex::time_type endpoint_index_(partitioned_complex::time_type step, const std::vector<partitioned_complex::time_type>& endpoint_steps)
{
  if (step <= endpoint_steps.front()) return 0;
  const auto it = std::lower_bound(endpoint_steps.begin(), endpoint_steps.end(), step);
  if (it == endpoint_steps.end()) return static_cast<std::uint64_t>(endpoint_steps.size() - 1);
  return static_cast<std::uint64_t>(it - endpoint_steps.begin());
}

static inline void sort_barcodes(std::vector<barcode_interval>& bars)
{
  std::sort(bars.begin(), bars.end(), [](const barcode_interval& a, const barcode_interval& b) {
    if (a.dimension != b.dimension) return a.dimension < b.dimension;
    if (a.birth != b.birth) return a.birth < b.birth;
    const bool a_alive = a.death == partitioned_complex::time_infty;
    const bool b_alive = b.death == partitioned_complex::time_infty;
    if (a_alive != b_alive) return !a_alive;
    if (!a_alive && a.death != b.death) return a.death < b.death;
    return false;
  });
}

static inline std::vector<barcode_interval> endpoint_quotient_barcodes(const partitioned_complex& pc, const std::vector<partitioned_complex::time_type>& endpoint_steps)
{
  validate_endpoint_steps_(endpoint_steps);
  std::vector<barcode_interval> out;
  out.reserve(pc.features.size());

  for (const auto& ft : pc.features) {
    barcode_interval b;
    b.dimension = ft.dimension;
    b.birth = endpoint_index_(ft.birth, endpoint_steps);
    b.death = (ft.death == partitioned_complex::time_infty) ? partitioned_complex::time_infty : endpoint_index_(ft.death, endpoint_steps);
    if (b.death != partitioned_complex::time_infty && b.birth == b.death && b.birth) {
      const std::size_t endpoint = static_cast<std::size_t>(b.birth);
      if (ft.birth > endpoint_steps[endpoint - 1] && ft.death <= endpoint_steps[endpoint]) continue;
    }
    out.push_back(b);
  }

  sort_barcodes(out);
  return out;
}
