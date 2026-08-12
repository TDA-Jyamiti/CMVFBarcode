#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mvf_module/mvf_module.hpp"

static inline void print_usage_(std::ostream& out, const char* argv0)
{
  out << "usage: " << argv0 << " [--pretty] [--no-quotient] frame_0000.smp frame_0001.smp ...\n";
}

static inline void write_time_(std::ostream& out, partitioned_complex::time_type t)
{
  if (t == partitioned_complex::time_infty) out << "null";
  else out << t;
}

static inline std::vector<barcode_interval> raw_barcodes_(const partitioned_complex& pc)
{
  std::vector<barcode_interval> out;
  out.reserve(pc.features.size());
  for (const auto& ft : pc.features) out.push_back({ ft.dimension, ft.birth, ft.death });
  sort_barcodes(out);
  return out;
}

static inline void write_barcode_(std::ostream& out, const barcode_interval& b, bool pretty)
{
  if (pretty) out << "{ \"dimension\": " << b.dimension << ", \"birth\": ";
  else out << "{\"dimension\":" << b.dimension << ",\"birth\":";
  write_time_(out, b.birth);
  if (pretty) out << ", \"death\": ";
  else out << ",\"death\":";
  write_time_(out, b.death);
  if (pretty) out << " }";
  else out << '}';
}

static inline void write_barcodes_(std::ostream& out, const std::vector<barcode_interval>& barcodes, bool pretty)
{
  if (!pretty) {
    out << "{\"schema\":\"mvf_barcodes_v1\",\"barcodes\":[";
    for (std::size_t i = 0; i < barcodes.size(); ++i) { if (i) out << ','; write_barcode_(out, barcodes[i], false); }
    out << "]}\n";
    return;
  }

  out << "{\n";
  out << "  \"schema\": \"mvf_barcodes_v1\",\n";
  out << "  \"barcodes\": [\n";
  for (std::size_t i = 0; i < barcodes.size(); ++i) {
    out << "    ";
    write_barcode_(out, barcodes[i], true);
    out << (i + 1 == barcodes.size() ? "\n" : ",\n");
  }
  out << "  ]\n";
  out << "}\n";
}

int main(int argc, char** argv)
{
  try {
    bool pretty = false;
    bool no_quotient = false;
    std::vector<std::string> paths;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "-h" || arg == "--help") { print_usage_(std::cout, argv[0]); return 0; }
      if (arg == "--pretty") { pretty = true; continue; }
      if (arg == "--no-quotient") { no_quotient = true; continue; }
      paths.push_back(arg);
    }
    if (paths.empty()) { print_usage_(std::cerr, argv[0]); return 1; }

    partitioned_complex pc;
    if (!load_smp(paths.front().c_str(), pc)) throw std::runtime_error(std::string("failed to load ") + paths.front());

    std::vector<partitioned_complex::time_type> endpoint_steps;
    endpoint_steps.reserve(paths.size());
    endpoint_steps.push_back(pc.step);

    for (std::size_t i = 1; i < paths.size(); ++i) {
      translate_partition_to_smp(pc, paths[i].c_str());
      endpoint_steps.push_back(pc.step);
    }

    write_barcodes_(std::cout, no_quotient ? raw_barcodes_(pc) : endpoint_quotient_barcodes(pc, endpoint_steps), pretty);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cmvf_barcodes: " << e.what() << '\n';
    return 1;
  }
}
