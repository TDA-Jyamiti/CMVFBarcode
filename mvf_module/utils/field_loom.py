#
# Quick planar test-data generator for CMVFBarcode.
#
# In short, this file provides a set of tools to discretize a time varying vector valued equation
# over a rectangular region of R^2. For each timestep, a `.smp` file is generated. These may be parsed
# in sequence by the main utility.
#
# The basic use is:
#
#     from field_loom import build_grid_complex, generate_sequence
#     from fields import rotating_sink
#
#     complex_2d = build_grid_complex(top_left=(-1.0, 1.0), bottom_right=(1.0, -1.0), resolution=14)
#     times = np.linspace(0.0, 2.4, 7)
#     generate_sequence("examples/large_grid_14", complex_2d, rotating_sink(strength=0.65, angular_speed=1.15), times)
#
# The discretization is very simple. At each vertex, a
# sampled vector is compared with the incident edge directions. If it points
# closely enough along an edge, the vertex is paired with that edge. Otherwise
# the incident triangles are tested, and the vertex may be paired with a triangle
# when the vector lies in the appropriate tangent cone. Edges are then tested
# against incident triangles using the average of field values at their endpoints.
# At the end, all boundary simplices are optionally bound together in a single multivector.
#
# A key idea is that we begin by assuming each simplex exists in a singleton multivector. Each time a pairing occurs,
# the multivectors of the paired simplices are merged, and convexity of the resulting multivector is checked. If convexity fails,
# new multivectors are merged in, and the check is repeated recursively. We iterate this to a fixed point upon each merge.
#
# Vector fields are ordinary callables of the form
#
#     field_t(point, t) -> vector
#
# where `point` is a NumPy vector in R^2 and `t` is the sampled time. See
# `fields.py` for examples and for the convention used when adding new fields.

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, TypeAlias

import numpy as np
from numpy.typing import NDArray

float_array: TypeAlias = NDArray[np.float64]
point: TypeAlias = float_array
vector: TypeAlias = float_array
simplex: TypeAlias = tuple[int, ...]
point_field: TypeAlias = Callable[[point], vector]
time_field: TypeAlias = Callable[[point, float], vector]

simplices_header = """# ============================= #
# simplices                     #
#   index: abstract simplex     #
# ============================= #
"""

multivectors_header = """# ===================================== #
# multivectors                          #
#   index: constituent simplex indices  #
# ===================================== #
"""


def simplex_key(vertices: simplex) -> tuple[int, simplex]:
    return (len(vertices), vertices)


def as_vector(value: object) -> vector:
    arr = np.asarray(value, dtype=float)
    assert arr.shape == (2,)
    return arr


@dataclass(frozen=True)
class grid_complex_2d:
    points: list[point]
    simplices: list[simplex]
    simplex_to_id: dict[simplex, int]
    dimensions: list[int]
    vertex_ids: list[int]
    edge_ids: list[int]
    triangle_ids: list[int]
    vertex_to_edges: list[list[int]]
    vertex_to_triangles: list[list[int]]
    edge_to_triangles: dict[int, list[int]]
    boundary_edges: list[int]
    boundary_vertices: list[int]

    @property
    def simplex_count(self) -> int:
        return len(self.simplices)

    def simplex_id(self, vertices: Iterable[int]) -> int:
        return self.simplex_to_id[tuple(sorted(vertices))]

    def write_simplices_section(self, fh: TextIO) -> None:
        fh.write(simplices_header)
        for idx, vertices in enumerate(self.simplices):
            fh.write(f"{idx}:{','.join(map(str, vertices))}\n")


def _grid_points(top_left: tuple[float, float], bottom_right: tuple[float, float], resolution: int) -> list[point]:
    assert resolution >= 2
    x_values = np.linspace(float(top_left[0]), float(bottom_right[0]), int(resolution))
    y_values = np.linspace(float(bottom_right[1]), float(top_left[1]), int(resolution))
    return [np.array([x, y], dtype=float) for x in x_values for y in y_values]


def _grid_triangles(resolution: int) -> list[simplex]:
    assert resolution >= 2
    grid_size = int(resolution)
    triangles: list[simplex] = []
    for row in range(grid_size - 1):
        for col in range(grid_size - 1):
            v00 = row * grid_size + col
            v01 = row * grid_size + (col + 1)
            v10 = (row + 1) * grid_size + col
            v11 = (row + 1) * grid_size + (col + 1)
            triangles.append(tuple(sorted((v00, v10, v11))))
            triangles.append(tuple(sorted((v00, v11, v01))))
    return triangles


def build_grid_complex(top_left: tuple[float, float], bottom_right: tuple[float, float], resolution: int) -> grid_complex_2d:
    points = _grid_points(top_left, bottom_right, resolution)
    vertices: list[simplex] = [(i,) for i in range(len(points))]
    triangles = sorted(set(_grid_triangles(resolution)), key=simplex_key)
    edge_set: set[simplex] = set()
    for a, b, c in triangles:
        edge_set.add(tuple(sorted((a, b))))
        edge_set.add(tuple(sorted((a, c))))
        edge_set.add(tuple(sorted((b, c))))
    edges = sorted(edge_set, key=simplex_key)
    simplices = vertices + edges + triangles
    simplex_to_id = {vertices: i for i, vertices in enumerate(simplices)}
    dimensions = [len(vertices) - 1 for vertices in simplices]
    vertex_ids = list(range(len(vertices)))
    edge_ids = list(range(len(vertices), len(vertices) + len(edges)))
    triangle_ids = list(range(len(vertices) + len(edges), len(simplices)))
    vertex_to_edges: list[list[int]] = [[] for _ in vertex_ids]
    vertex_to_triangles: list[list[int]] = [[] for _ in vertex_ids]
    edge_to_triangles: dict[int, list[int]] = {eid: [] for eid in edge_ids}
    for eid in edge_ids:
        a, b = simplices[eid]
        vertex_to_edges[a].append(eid)
        vertex_to_edges[b].append(eid)
    for tid in triangle_ids:
        tri = simplices[tid]
        for v in tri:
            vertex_to_triangles[v].append(tid)
        for e in ((tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])):
            edge_to_triangles[simplex_to_id[tuple(sorted(e))]].append(tid)
    for arr in vertex_to_edges: arr.sort(key=lambda sid: simplices[sid])
    for arr in vertex_to_triangles: arr.sort(key=lambda sid: simplices[sid])
    for eid in edge_to_triangles: edge_to_triangles[eid].sort(key=lambda sid: simplices[sid])
    boundary_edges = sorted([eid for eid, tris in edge_to_triangles.items() if len(tris) == 1], key=lambda sid: simplices[sid])
    boundary_vertices = sorted({v for eid in boundary_edges for v in simplices[eid]})
    return grid_complex_2d(points, simplices, simplex_to_id, dimensions, vertex_ids, edge_ids, triangle_ids,
                           vertex_to_edges, vertex_to_triangles, edge_to_triangles, boundary_edges, boundary_vertices)


class partition_2d:
    def __init__(self, complex_2d: grid_complex_2d) -> None:
        self.complex = complex_2d
        self.class_of: list[int] = list(range(complex_2d.simplex_count))
        self.members: dict[int, set[int]] = {i: {i} for i in range(complex_2d.simplex_count)}
        self.next_class_id = complex_2d.simplex_count

    def _expand_vertex_triangle_closure(self, members: set[int]) -> set[int]:
        simplices = self.complex.simplices
        verts = [sid for sid in members if self.complex.dimensions[sid] == 0]
        tris = [sid for sid in members if self.complex.dimensions[sid] == 2]
        out = set(members)
        for vid in verts:
            v = simplices[vid][0]
            for tid in tris:
                tri = simplices[tid]
                if v in tri:
                    for w in tri:
                        if w != v:
                            out.add(self.complex.simplex_id((v, w)))
        return out

    def add_class(self, simplex_ids: Iterable[int]) -> None:
        class_ids = {self.class_of[sid] for sid in simplex_ids}
        merged = set().union(*(self.members[cid] for cid in class_ids)) if class_ids else set()
        while True:
            closure = self._expand_vertex_triangle_closure(merged)
            expanded_ids = {self.class_of[sid] for sid in closure}
            expanded = set().union(*(self.members[cid] for cid in expanded_ids)) if expanded_ids else set()
            if expanded == merged:
                break
            merged = expanded
        new_id = self.next_class_id
        self.next_class_id += 1
        for cid in {self.class_of[sid] for sid in merged}:
            self.members.pop(cid, None)
        self.members[new_id] = merged
        for sid in merged:
            self.class_of[sid] = new_id

    def bind_boundary(self) -> None:
        boundary = set(self.complex.boundary_edges) | set(self.complex.boundary_vertices)
        if boundary:
            self.add_class(boundary)

    def canonical_classes(self) -> list[list[int]]:
        classes = [sorted(members) for members in self.members.values()]
        classes.sort(key=lambda cls: min(cls))
        return classes

    def write_smp(self, path: Path | str) -> None:
        smp_path = Path(path)
        with smp_path.open("w") as fh:
            self.complex.write_simplices_section(fh)
            fh.write(multivectors_header)
            for i, members in enumerate(self.canonical_classes()):
                fh.write(f"{i}:{','.join(map(str, members))}\n")


def discretize_field(complex_2d: grid_complex_2d, field: point_field, *, angle_threshold_radians: float = 0.0875, bind_boundary: bool = True) -> partition_2d:
    part = partition_2d(complex_2d)
    points, simplices = complex_2d.points, complex_2d.simplices
    for vid in complex_2d.vertex_ids:
        point_at_vertex = points[simplices[vid][0]]
        field_vector = as_vector(field(point_at_vertex))
        field_norm = float(np.linalg.norm(field_vector))
        if field_norm == 0.0:
            continue
        unit_field_vector = field_vector / field_norm
        paired = False
        v = simplices[vid][0]
        for eid in complex_2d.vertex_to_edges[v]:
            a, b = simplices[eid]
            other = b if a == v else a
            direction = points[other] - point_at_vertex
            direction = direction / np.linalg.norm(direction)
            dot = float(np.clip(np.dot(unit_field_vector, direction), -1.0, 1.0))
            if float(np.arccos(dot)) < angle_threshold_radians:
                part.add_class((vid, eid))
                paired = True
                break
        if not paired:
            for tid in complex_2d.vertex_to_triangles[v]:
                tri = simplices[tid]
                others = [points[w] for w in tri if w != v]
                edge1 = others[0] - point_at_vertex
                edge2 = others[1] - point_at_vertex
                cross_edge1_field = float(np.cross(edge1, unit_field_vector))
                cross_edge1_edge2 = float(np.cross(edge1, edge2))
                cross_edge2_field = float(np.cross(edge2, unit_field_vector))
                cross_edge2_edge1 = -cross_edge1_edge2
                if cross_edge1_field * cross_edge2_field != 0 and cross_edge1_field * cross_edge1_edge2 > 0 and cross_edge2_field * cross_edge2_edge1 > 0:
                    part.add_class((vid, tid))
                    break
    for eid in complex_2d.edge_ids:
        a, b = simplices[eid]
        pa, pb = points[a], points[b]
        edge_vector = pa - pb
        first_vector = as_vector(field(pa))
        second_vector = as_vector(field(pb))
        if float(np.cross(edge_vector, first_vector)) * float(np.cross(edge_vector, second_vector)) < 0:
            continue
        midpoint_vector = first_vector + second_vector
        midpoint_norm = float(np.linalg.norm(midpoint_vector))
        if midpoint_norm == 0.0:
            continue
        unit_midpoint_vector = midpoint_vector / midpoint_norm
        for tid in complex_2d.edge_to_triangles[eid]:
            tri = simplices[tid]
            opposite = next(v for v in tri if v not in simplices[eid])
            if float(np.cross(edge_vector, unit_midpoint_vector)) * float(np.cross(edge_vector, points[opposite] - pa)) > 0:
                part.add_class((eid, tid))
                break
    if bind_boundary:
        part.bind_boundary()
    return part


def generate_sequence(out_dir: Path | str, complex_2d: grid_complex_2d, field_t: time_field, times: Iterable[float], *, angle_threshold_radians: float = 0.0875, bind_boundary: bool = True) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for path in sorted(out_path.glob("frame_*.smp")):
        path.unlink()
    written: list[Path] = []
    for i, t in enumerate(times):
        part = discretize_field(complex_2d, lambda p, _t=t: field_t(p, _t), angle_threshold_radians=angle_threshold_radians, bind_boundary=bind_boundary)
        path = out_path / f"frame_{i:04d}.smp"
        part.write_smp(path)
        written.append(path)
    return written
