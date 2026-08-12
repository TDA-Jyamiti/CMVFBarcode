#!/usr/bin/env python3

# Regenerate the demo `.smp` input sequences.
#
# An example of how to use `field_loom.py`.
# It writes the tiny endpoint-quotient test sequence by hand, then writes a
# larger sampled-grid sequence from a rotating sink vector field.
#
# Run from the repository root with:
#
#     python3 mvf_module/utils/generate_demo_sequences.py
#
# Then run, for example:
#
#     ./build/cmvf_barcodes examples/small_endpoint_quotient/frame_*.smp
#     ./build/cmvf_barcodes examples/large_grid_14/frame_*.smp

from pathlib import Path

import numpy as np

from field_loom import build_grid_complex, generate_sequence, simplices_header, multivectors_header
from fields import rotating_sink


root: Path = Path(__file__).resolve().parents[2]
examples_dir: Path = root / "examples"

small_frames: list[str] = [
    "0:0,2\n1:1\n",
    "0:0\n1:1,2\n",
    "0:0\n1:1\n2:2\n",
    "0:0,2\n1:1\n",
]


def write_small(out_dir: Path) -> None:
    header: str = simplices_header + "0:0\n1:1\n2:0,1\n" + multivectors_header
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(out_dir.glob("frame_*.smp")):
        path.unlink()

    for i, mv_section in enumerate(small_frames):
        (out_dir / f"frame_{i:04d}.smp").write_text(header + mv_section)


def write_large(out_dir: Path) -> None:
    resolution: int = 14
    complex_2d = build_grid_complex(top_left=(-1.0, 1.0), bottom_right=(1.0, -1.0), resolution=resolution)
    times = np.linspace(0.0, 2.4, 7)

    generate_sequence(
        out_dir,
        complex_2d,
        rotating_sink(strength=0.65, angular_speed=1.15),
        times,
        angle_threshold_radians=0.25,
        bind_boundary=True,
    )


def main() -> None:
    write_small(examples_dir / "small_endpoint_quotient")
    write_large(examples_dir / "large_grid_14")
    print(f"wrote examples under {examples_dir}")


if __name__ == "__main__":
    main()
