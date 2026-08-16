#!/usr/bin/env bash
set -euo pipefail
mkdir -p build/test

large_dir=examples/large_grid_14
if [[ ! -d "$large_dir" ]]; then
  generated_dir=$(mktemp -d)
  trap 'rm -rf "$generated_dir"' EXIT
  large_dir=$generated_dir/large_grid_14
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mvf_module/utils python3 - "$large_dir" <<'PY'
import sys
from pathlib import Path

from generate_demo_sequences import write_large

write_large(Path(sys.argv[1]))
PY
fi

./build/cmvf_barcodes --help >/dev/null
./build/cmvf_barcodes_validated --help >/dev/null

if ./build/cmvf_barcodes >/dev/null 2>&1; then
  echo 'cmvf_barcodes unexpectedly accepted missing input' >&2
  exit 1
fi
if ./build/cmvf_barcodes_validated >/dev/null 2>&1; then
  echo 'cmvf_barcodes_validated unexpectedly accepted missing input' >&2
  exit 1
fi

./build/cmvf_barcodes examples/small_endpoint_quotient/frame_*.smp > build/test/small_endpoint.json
./build/cmvf_barcodes --pretty examples/small_endpoint_quotient/frame_*.smp > build/test/small_pretty.json
./build/cmvf_barcodes --no-quotient examples/small_endpoint_quotient/frame_*.smp > build/test/small_raw.json
./build/cmvf_barcodes "$large_dir"/frame_*.smp > build/test/large_endpoint.json
./build/cmvf_barcodes --validated test/data/paper_atomic_0000.smp test/data/paper_atomic_0001.smp > build/test/paper_atomic.json
./build/cmvf_barcodes --validated --no-quotient test/data/paper_atomic_0000.smp test/data/paper_atomic_0001.smp > build/test/paper_atomic_raw.json

./build/cmvf_barcodes_validated \
  examples/small_endpoint_quotient/frame_0001.smp \
  examples/small_endpoint_quotient/frame_0002.smp \
  examples/small_endpoint_quotient/frame_0003.smp

if ./build/cmvf_barcodes_validated \
  examples/small_endpoint_quotient/frame_0000.smp \
  examples/small_endpoint_quotient/frame_0001.smp \
  > build/test/nonbinary.out 2> build/test/nonbinary.err; then
  echo 'validator unexpectedly accepted a non-binary transition' >&2
  exit 1
fi

if ./build/cmvf_barcodes_validated \
  examples/small_endpoint_quotient/frame_0001.smp \
  examples/small_endpoint_quotient/frame_0001.smp \
  > build/test/unchanged.out 2> build/test/unchanged.err; then
  echo 'validator unexpectedly accepted an unchanged transition' >&2
  exit 1
fi

python3 - <<'PY'
import json
from pathlib import Path

small = json.loads(Path('build/test/small_endpoint.json').read_text())
pretty = json.loads(Path('build/test/small_pretty.json').read_text())
raw = json.loads(Path('build/test/small_raw.json').read_text())
large = json.loads(Path('build/test/large_endpoint.json').read_text())
paper = json.loads(Path('build/test/paper_atomic.json').read_text())
paper_raw = json.loads(Path('build/test/paper_atomic_raw.json').read_text())

assert small['schema'] == 'mvf_barcodes_v1'
assert large['schema'] == 'mvf_barcodes_v1'
assert set(small) == {'schema', 'barcodes'}
assert set(large) == {'schema', 'barcodes'}
assert small['barcodes'] == [
    {'dimension': 0, 'birth': 0, 'death': 1},
    {'dimension': 0, 'birth': 1, 'death': 3},
    {'dimension': 0, 'birth': 2, 'death': None},
    {'dimension': 1, 'birth': 2, 'death': 3},
]
assert pretty == small
assert raw == {
    'schema': 'mvf_barcodes_v1',
    'barcodes': [
        {'dimension': 0, 'birth': 0, 'death': 2},
        {'dimension': 0, 'birth': 1, 'death': 4},
        {'dimension': 0, 'birth': 3, 'death': None},
        {'dimension': 1, 'birth': 1, 'death': 2},
        {'dimension': 1, 'birth': 3, 'death': 4},
    ],
}
assert len(large['barcodes']) == 14
assert paper == {
    'schema': 'mvf_barcodes_v1',
    'barcodes': [
        {'dimension': 0, 'birth': 0, 'death': None},
        {'dimension': 1, 'birth': 0, 'death': None},
        {'dimension': 2, 'birth': 0, 'death': None},
    ],
}
assert paper_raw == {
    'schema': 'mvf_barcodes_v1',
    'barcodes': [
        {'dimension': 0, 'birth': 0, 'death': None},
        {'dimension': 0, 'birth': 5, 'death': 7},
        {'dimension': 1, 'birth': 0, 'death': None},
        {'dimension': 1, 'birth': 5, 'death': 7},
        {'dimension': 2, 'birth': 0, 'death': None},
    ],
}

for out in (small, large):
    for b in out['barcodes']:
        assert set(b) == {'dimension', 'birth', 'death'}
        assert isinstance(b['dimension'], int)
        assert isinstance(b['birth'], int)
        assert b['death'] is None or isinstance(b['death'], int)
        assert b['death'] is None or b['birth'] <= b['death']

print('tests passed')
PY
