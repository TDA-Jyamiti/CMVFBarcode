// smp_io.hpp
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "partitioned_complex.hpp"

// SMP INPUT
// =========
//
// An `.smp` file contains a numbered simplex section followed by a numbered multivector section.
// Simplex records must be dense and ordered because their ids are shared by every frame in a
// sequence. Multivector record ids are labels only; they must be unique, but their values and order
// do not affect the partition. Every listed block is nonempty, blocks may not overlap, and an
// unlisted simplex becomes its own singleton block.
//
// Parsing is strict: every noncomment record must be consumed completely, the multivector header is
// required, and a failed load leaves the caller's output unchanged. `load_smp()` finalizes the
// complex and coalesces cyclic classes. `load_smp_partition_map_checked()` instead returns the raw
// partition after confirming that the simplex section exactly matches a finalized reference
// complex. The raw map is what callers use when they must detect whether SCC coalescing changes a
// partition.

struct smp_file_data {
  std::vector<std::vector<std::uint32_t>> simplex_vertices;
  std::vector<std::vector<std::uint32_t>> multivectors;
};

static inline const char* smp_skip_ws_(const char* p) {
  while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
  return p;
}

static inline bool smp_is_multivectors_header_(const char* p) {
  p = smp_skip_ws_(p);
  if (*p++ != '#') return false;
  p = smp_skip_ws_(p);

  constexpr char label[] = "multivectors";
  if (std::strncmp(p, label, sizeof(label) - 1)) return false;
  p = smp_skip_ws_(p + sizeof(label) - 1);
  if (*p == '#') p = smp_skip_ws_(p + 1);
  return !*p;
}

static inline bool smp_parse_u32_(const char*& p, std::uint32_t& out) {
  p = smp_skip_ws_(p);
  if (*p < '0' || *p > '9') return false;

  std::uint32_t value = 0;
  do {
    const std::uint32_t digit = static_cast<std::uint32_t>(*p - '0');
    if (value > (std::numeric_limits<std::uint32_t>::max() - digit) / 10) return false;
    value = value * 10 + digit;
    ++p;
  } while (*p >= '0' && *p <= '9');

  out = value;
  return true;
}

static inline bool smp_parse_u32_list_(const char* p, std::vector<std::uint32_t>& out) {
  out.clear();
  for (;;) {
    std::uint32_t x = 0;
    if (!smp_parse_u32_(p, x)) return false;
    out.push_back(x);
    p = smp_skip_ws_(p);
    if (!*p) return true;
    if (*p != ',') return false;
    ++p;
  }
}

static inline bool smp_parse_file_(const char* path, smp_file_data& out) {
  std::ifstream in(path);
  if (!in) return false;

  smp_file_data parsed;
  std::vector<std::uint32_t> multivector_ids;
  std::vector<std::uint32_t> list;
  std::string line;
  bool in_multivectors = false;

  while (std::getline(in, line)) {
    const char* p = smp_skip_ws_(line.c_str());
    if (*p == '#') {
      if (smp_is_multivectors_header_(p)) in_multivectors = true;
      continue;
    }

    if (const std::size_t pos = line.find('#'); pos != std::string::npos) line.resize(pos);
    p = smp_skip_ws_(line.c_str());
    if (!*p) continue;

    std::uint32_t idx = 0;
    if (!smp_parse_u32_(p, idx)) return false;
    p = smp_skip_ws_(p);
    if (*p != ':' || !smp_parse_u32_list_(p + 1, list)) return false;

    std::sort(list.begin(), list.end());
    if (std::adjacent_find(list.begin(), list.end()) != list.end()) return false;

    if (!in_multivectors) {
      if (parsed.simplex_vertices.size() >= std::numeric_limits<std::uint32_t>::max() || idx != static_cast<std::uint32_t>(parsed.simplex_vertices.size())) return false;
      parsed.simplex_vertices.push_back(list);
    } else {
      multivector_ids.push_back(idx);
      parsed.multivectors.push_back(list);
    }
  }

  if (in.bad() || !in_multivectors || parsed.simplex_vertices.empty()) return false;
  std::sort(multivector_ids.begin(), multivector_ids.end());
  if (std::adjacent_find(multivector_ids.begin(), multivector_ids.end()) != multivector_ids.end()) return false;

  out = std::move(parsed);
  return true;
}

static inline bool smp_partition_map_(const smp_file_data& data, std::vector<partitioned_complex::class_id>& out) {
  using class_id = partitioned_complex::class_id;

  const std::size_t S = data.simplex_vertices.size();
  if (S > std::numeric_limits<class_id>::max() || data.multivectors.size() > S) return false;

  std::vector<class_id> simplex_to_class(S, partitioned_complex::invalid_class);
  class_id next = 0;
  for (const auto& multivector : data.multivectors) {
    for (std::uint32_t s : multivector) {
      if (static_cast<std::size_t>(s) >= S || simplex_to_class[s] != partitioned_complex::invalid_class) return false;
      simplex_to_class[s] = next;
    }
    ++next;
  }

  for (partitioned_complex::simplex_id s = 0; s < static_cast<partitioned_complex::simplex_id>(S); ++s)
    if (simplex_to_class[s] == partitioned_complex::invalid_class) simplex_to_class[s] = next++;

  out = std::move(simplex_to_class);
  return true;
}

static inline bool same_vertices_sorted_(const partitioned_complex& ref, partitioned_complex::simplex_id s, const std::vector<std::uint32_t>& sorted_vertices) {
  using vertex_id = partitioned_complex::vertex_id;
  const std::uint32_t size = ref.simplex_vertices_size(s);
  if (size != static_cast<std::uint32_t>(sorted_vertices.size())) return false;
  const vertex_id* vertices = ref.simplex_vertices_data(s);
  for (std::uint32_t i = 0; i < size; ++i)
    if (vertices[i] != static_cast<vertex_id>(sorted_vertices[i])) return false;
  return true;
}

static inline bool load_smp(const char* path, partitioned_complex& out, bool require_all_faces = true) {
  smp_file_data data;
  if (!smp_parse_file_(path, data)) return false;

  partitioned_complex parsed;
  for (const auto& vertices : data.simplex_vertices) parsed.add_simplex_sorted(vertices.data(), static_cast<std::uint32_t>(vertices.size()));
  if (!smp_partition_map_(data, parsed.simplex_to_class)) return false;

  try {
    parsed.finalize(require_all_faces);
  } catch (...) {
    return false;
  }
  out = std::move(parsed);
  return true;
}

static inline bool load_smp_partition_map_checked(const char* path, const partitioned_complex& ref, std::vector<partitioned_complex::class_id>& out_map_raw) {
  using simplex_id = partitioned_complex::simplex_id;

  const std::size_t S_ref = ref.simplex_count();
  if (!S_ref) return false;

  smp_file_data data;
  if (!smp_parse_file_(path, data) || data.simplex_vertices.size() != S_ref) return false;
  for (simplex_id s = 0; s < static_cast<simplex_id>(S_ref); ++s)
    if (!same_vertices_sorted_(ref, s, data.simplex_vertices[s])) return false;
  return smp_partition_map_(data, out_map_raw);
}
