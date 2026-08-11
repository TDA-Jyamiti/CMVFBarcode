# MVF barcodes

A small C++20 tool and header-only library for computing barcode intervals from a sequence of multivector-field partitions on a fixed simplicial complex.

The barcode program reads a sequence of `.smp` files, translates the first partition through the later partitions, and writes barcode data as JSON. A second program validates binary merge/split zigzag filtrations of block partitions. The package also includes a small Python utility for generating planar test data from sampled vector fields.

## Paper and implementation status

This software is a loose, independent adaptation inspired by [Computing Conley-Morse Persistence Barcode Efficiently by Updating Matrix Decompositions](https://arxiv.org/abs/2608.06507) by Tamal K. Dey, Andrew Haas, and Michał Lipiński.

## Group information

This project was developed by the [Computational Geometry and Topological Data Analysis (CGTDA) research group](https://www.cs.purdue.edu/homes/tamaldey/CGTDAwebsite/) at Purdue University, with Andrew Haas as the primary software developer. The group is led by [Prof. Tamal K. Dey](https://www.cs.purdue.edu/homes/tamaldey/).

## Build

Requirements:

- a C++20 compiler;
- Python 3 and NumPy, only if regenerating the example data sets or running the tests when the large generated fixture is absent.

```bash
make
```

This builds:

```bash
build/mvf_barcodes
build/mvf_barcodes_validated
```

Run the end-to-end barcode checks with:

```bash
make test
```

If `examples/large_grid_14` is absent, the test script generates that fixture in a temporary directory and removes it afterward.

## Run

Use one or more `.smp` frames from the same fixed simplicial complex; e.g.

```bash
./build/mvf_barcodes examples/small_endpoint_quotient/frame_*.smp
```

The default output is compact JSON; e.g.

```json
{"schema":"mvf_barcodes_v1","barcodes":[{"dimension":0,"birth":0,"death":1},{"dimension":0,"birth":1,"death":3},{"dimension":0,"birth":2,"death":null},{"dimension":1,"birth":2,"death":3}]}
```

Use `--pretty` for indented output; e.g.

```bash
./build/mvf_barcodes --pretty examples/small_endpoint_quotient/frame_*.smp
```

Here `birth` and `death` are input-frame indices. A `null` death means the feature is still alive after the final input.

## Endpoint quotienting

The reported intervals use the endpoint-quotient convention. During the translation from input `i` to input `i + 1`, every birth or death that occurs inside the translation is reported at `i + 1`. Features born and killed entirely inside one translation are omitted.

Use `--no-quotient` to report raw internal translation steps instead; e.g.

```bash
./build/mvf_barcodes --no-quotient examples/small_endpoint_quotient/frame_*.smp
```

## Validate block sequences

`mvf_barcodes_validated` checks that raw `.smp` inputs form a zigzag filtration of block partitions in the sense of the motivating [preprint](https://arxiv.org/abs/2608.06507). Unlike `mvf_barcodes`, which accepts any sequence of multivector fields on the same complex, this program requires every adjacent pair to be related by refinement in one direction or the other. Every input must describe the same simplex-id-preserving complex, and every partition must remain unchanged when its quotient block graph is coalesced by strongly connected components. Block membership is compared independently of multivector record labels.

Each adjacent pair must differ by exactly one operation: either two blocks merge into one, or one block splits into exactly two. The program is silent and returns zero on success; malformed input, a non-block partition, a complex mismatch, or a non-binary transition produces a diagnostic and a nonzero status.

For example, the last three small frames form one split followed by one merge:

```bash
./build/mvf_barcodes_validated \
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

The simplex section assigns dense, ordered integer ids to abstract simplices and lists their constituent vertices. The multivector section has uniquely labeled records that may list any nonempty blocks by simplex id. Record labels and order do not affect block identity. Simplices omitted from the multivector section are treated as singleton multivectors.

All files passed in one run must describe the same simplicial complex, with the same simplex ids. Only the partition is allowed to change from frame to frame.

All homology and matrix-reduction computations use coefficients in `GF(2)`.

## Library use

The headers can also be used directly:

```cpp
#include <cassert>
#include <vector>

#include "mvf_module/mvf_module.hpp"

int main()
{
  partitioned_complex pc;
  assert(load_smp("examples/small_endpoint_quotient/frame_0000.smp", pc));

  std::vector<partitioned_complex::time_type> endpoints;
  endpoints.push_back(pc.step);

  translate_partition_to_smp(pc, "examples/small_endpoint_quotient/frame_0001.smp");
  endpoints.push_back(pc.step);

  auto bars = endpoint_quotient_barcodes(pc, endpoints);
}
```

## Shared graph representation

`mvf_module/graph_utilities.hpp` contains the library's graph representation and graph algorithms. Its reusable `csr_graph` stores outgoing adjacency in compressed sparse row form: `offset[v]` and `offset[v + 1]` delimit the targets of vertex `v` in `edges`. Unweighted graphs leave the parallel `weights` array empty, so they retain four bytes of target storage per edge; weighted graphs allocate one aligned 32-bit weight per edge.

The same object is used for the simplicial codimension-one incidence graph, the directed class graph whose strong components are coalesced, the symmetric weighted overlap graph used for partition matching, and the densely relabeled matching components. Builders, reversal, iterative strong-component decomposition, overlap construction, and matching are all kept in this one header.

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

This project is available under the [MIT License](LICENSE).

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
