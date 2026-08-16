# CMVFBarcode

A small C++20 tool and header-only library for computing barcode intervals from a sequence of combinatorial multivector-field partitions on a fixed simplicial complex.

The name **CMVFBarcode** deliberately gives “CM” two readings: **Conley–Morse**, for the theory behind the persistence barcode, and **Combinatorial Multi-**, for the multivector fields supplied as input.

The barcode program reads a sequence of `.smp` files, translates the first partition through the later partitions, and writes barcode data as JSON. Its default mode connects arbitrary partitions; `--validated` instead computes the paper-defined sequence for an atomic merge/split filtration of block partitions. A second program validates such filtrations without computing a barcode. The package also includes a small Python utility for generating planar test data from sampled vector fields.

## Paper and implementation status

This software is an efficient implementation of the algorithm presented in [Computing Conley-Morse Persistence Barcode Efficiently by Updating Matrix Decompositions](https://arxiv.org/abs/2608.06507) by Tamal K. Dey, Andrew Haas, and Michał Lipiński.

## Group information

This project was developed by the [Computational Geometry and Topological Data Analysis (CGTDA) research group](https://www.cs.purdue.edu/homes/tamaldey/CGTDAwebsite/) at Purdue University, with Andrew Haas as the primary software developer. The group is led by [Prof. Tamal K. Dey](https://www.cs.purdue.edu/homes/tamaldey/).

## Build

Requirements:

- a C++20 compiler;
- Python 3.10 or newer to run the tests or regenerate example data;
- NumPy when regenerating example data, including the temporary large fixture used by tests in a fresh checkout.

```bash
make
```

This builds:

```bash
build/cmvf_barcodes
build/cmvf_barcodes_validated
```

Run the end-to-end barcode checks with:

```bash
make test
```

If `examples/large_grid_14` is absent, the test script generates that fixture in a temporary directory and removes it afterward.

## Run

Use one or more `.smp` frames from the same fixed simplicial complex; e.g.

```bash
./build/cmvf_barcodes examples/small_endpoint_quotient/frame_*.smp
```

The default output is compact JSON; e.g.

```json
{"schema":"mvf_barcodes_v1","barcodes":[{"dimension":0,"birth":0,"death":1},{"dimension":0,"birth":1,"death":3},{"dimension":0,"birth":2,"death":null},{"dimension":1,"birth":2,"death":3}]}
```

Use `--pretty` for indented output; e.g.

```bash
./build/cmvf_barcodes --pretty examples/small_endpoint_quotient/frame_*.smp
```

Here `birth` and `death` are input-frame indices, and intervals are half-open: a feature is present at frame `i` when `birth <= i < death`. A `null` death means the feature is still alive after the final input.

The `mvf_barcodes_v1` schema name is retained as a stable compatibility identifier for existing consumers.

### Validated block-filtration mode

Use `--validated` when the inputs form the atomic zigzag filtration of block partitions required by the paper; e.g.

```bash
./build/cmvf_barcodes --validated \
  test/data/paper_atomic_0000.smp \
  test/data/paper_atomic_0001.smp
```

Every raw frame must survive SCC coarsening unchanged, and every consecutive pair must differ by exactly one binary block split or merge. For a split `B = W ⊔ V`, the implementation identifies the child `W` closed in `B`, stably orders the parent as `[W | V]`, and executes the paper's recursive maximal-simplex expansion with elementary right splits and merges. The reverse sequence implements a merge; transpositions between differently adapted block orders carry the required changes of basis.

Without `--validated`, `cmvf_barcodes` retains the general maximum-overlap translator. That mode deliberately accepts arbitrary multivector-field sequences which need not be related by block refinements, so its connecting route is not the restrictive block-filtration construction.

Loading coalesces raw blocks that belong to the same strongly connected component of the quotient block graph. Validated mode rejects a frame if this changes its raw partition; general mode continues with the coarsened partition.

## Endpoint quotienting

The reported intervals use the endpoint-quotient convention. During the translation from input `i` to input `i + 1`, every birth or death that occurs inside the translation is reported at `i + 1`. Features born and killed entirely inside one translation are omitted.

Use `--no-quotient` to report raw internal translation steps instead; e.g.

```bash
./build/cmvf_barcodes --no-quotient examples/small_endpoint_quotient/frame_*.smp
```

In this mode, `birth` and `death` are implementation-dependent elementary-operation step numbers rather than input-frame indices. The schema identifier remains `mvf_barcodes_v1` for compatibility.

## Validate block sequences

`cmvf_barcodes_validated` checks the same input conditions as `cmvf_barcodes --validated`, but remains silent and does not compute a barcode. Unlike the default general barcode mode, it requires every adjacent pair to be related by refinement in one direction or the other. Every input must describe the same simplex-id-preserving complex, and every partition must remain unchanged when its quotient block graph is coalesced by strongly connected components. Block membership is compared independently of multivector record labels.

Each adjacent pair must differ by exactly one operation: either two blocks merge into one, or one block splits into exactly two. The program is silent and returns zero on success; malformed input, a non-block partition, a complex mismatch, or a non-binary transition produces a diagnostic and a nonzero status.

For example, the last three small frames form one split followed by one merge:

```bash
./build/cmvf_barcodes_validated \
  examples/small_endpoint_quotient/frame_0001.smp \
  examples/small_endpoint_quotient/frame_0002.smp \
  examples/small_endpoint_quotient/frame_0003.smp
```

## Input format

The fundamental input is a `.smp` file. Each file has a simplex section followed by a multivector section; e.g.

```text
# simplices
0:0
1:1
2:0,1
# multivectors
0:0,2
1:1
```

The simplex section assigns dense, ordered integer ids to abstract simplices and lists their constituent vertices. The multivector section has uniquely labeled records that may list any nonempty blocks by simplex id. Record labels and order do not affect block membership. In general mode, record order supplies internal class ids and can affect which connecting route is selected when maximum-overlap matchings are tied; validated mode is invariant to record labels and order. Simplices omitted from the multivector section are treated as singleton multivectors.

All files passed in one run must describe the same simplicial complex, with the same simplex ids. Only the partition is allowed to change from frame to frame.

## Library use

The headers can also be used directly:

```cpp
#include <stdexcept>
#include <vector>

#include "mvf_module/mvf_module.hpp"

int main()
{
  partitioned_complex pc;
  if (!load_smp("examples/small_endpoint_quotient/frame_0000.smp", pc))
    throw std::runtime_error("failed to load initial frame");

  std::vector<partitioned_complex::time_type> endpoints;
  endpoints.push_back(pc.step);

  translate_partition_to_smp(pc, "examples/small_endpoint_quotient/frame_0001.smp");
  endpoints.push_back(pc.step);

  auto bars = endpoint_quotient_barcodes(pc, endpoints);
}
```

For a validated atomic block filtration, replace `translate_partition_to_smp` with `translate_block_partition_to_smp`. The latter rejects a source or destination that is not an SCC-stable block partition, and rejects a destination that is not one atomic refinement away from the current partition.

## Generating example data

The small checked-in example and the larger generated example can be written with:

```bash
make generate-examples
```

or directly:

```bash
python3 -B mvf_module/utils/generate_demo_sequences.py
```

This writes:

```text
examples/small_endpoint_quotient/
examples/large_grid_14/
```

The generator code lives in:

```text
mvf_module/utils/field_loom.py
mvf_module/utils/fields.py
mvf_module/utils/generate_demo_sequences.py
```

`field_loom.py` builds a rectangular triangulated grid, samples a time-dependent planar vector field, discretizes each sampled field into a multivector partition, and writes one `.smp` file per time sample. `fields.py` contains reusable vector-field equations. To add another field, add a small factory returning a callable of the form:

```python
field_t(point, t) -> vector
```

Then pass it to `generate_sequence`.

Generation removes existing `frame_*.smp` files from the destination directory before writing the replacement sequence, so use a dedicated output directory.

Run custom generator scripts from the repository root with `PYTHONPATH=mvf_module/utils python3 -B your_script.py` so the utility modules are importable without creating bytecode files.

A minimal generator sketch is:

```python
import numpy as np

from field_loom import build_grid_complex, generate_sequence
from fields import rotating_sink

complex_2d = build_grid_complex(top_left=(-1.0, 1.0), bottom_right=(1.0, -1.0), resolution=14)
times = np.linspace(0.0, 2.4, 7)

generate_sequence(
    "examples/large_grid_14",
    complex_2d,
    rotating_sink(strength=0.65, angular_speed=1.15),
    times,
    angle_threshold_radians=0.25,
    bind_boundary=True,
)
```

## License

CMVFBarcode is available under the short attribution license in [LICENSE](LICENSE). Users are asked to cite the paper and credit its authors and the CGTDA research group at Purdue University when the software contributes to published or distributed work.

## Citation

The mathematical motivation for this software comes from the following preprint

```bibtex
@misc{dey2026computing,
  title={Computing Conley-Morse Persistence Barcode Efficiently by Updating Matrix Decompositions},
  author={Tamal K. Dey and Andrew Haas and Michał Lipiński},
  year={2026},
  eprint={2608.06507},
  archivePrefix={arXiv},
  primaryClass={math.AT},
  url={https://arxiv.org/abs/2608.06507}
}
```
