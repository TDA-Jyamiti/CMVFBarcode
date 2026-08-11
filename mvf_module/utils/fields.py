"""Time-dependent vector-field equations for `field_loom`.

Every public factory in this file returns a `time_field`: a callable of the form

    field(point, t) -> np.ndarray shape (2,)

where `point` is a two-dimensional NumPy point and `t` is the sequence parameter.  This uniform convention is what lets
`generate_sequence` accept any of these equations directly.  To add another equation, write a small factory with typed
parameters, return a nested `field_t(point, t)` function, and convert the output through `as_vector` or an explicit
`np.array(..., dtype=float)`.  Keep the parameters numeric and keyword-friendly so sequences can be scripted tersely.
"""

from __future__ import annotations

import numpy as np

from field_loom import as_vector, point, point_field, time_field, vector


def stationary(field: point_field) -> time_field:
    def field_t(p: point, t: float) -> vector:
        return as_vector(field(p))
    return field_t


def constant_flow(direction: tuple[float, float] = (1.0, 0.0), strength: float = 1.0) -> time_field:
    direction_vector = as_vector(direction)
    direction_norm = float(np.linalg.norm(direction_vector))
    assert direction_norm != 0.0
    unit_direction = direction_vector / direction_norm
    def field_t(p: point, t: float) -> vector:
        return strength * unit_direction
    return field_t


def radial_sink(strength: float = 1.0, center: tuple[float, float] = (0.0, 0.0)) -> time_field:
    center_vector = as_vector(center)
    def field_t(p: point, t: float) -> vector:
        return -strength * (p - center_vector)
    return field_t


def radial_source(strength: float = 1.0, center: tuple[float, float] = (0.0, 0.0)) -> time_field:
    center_vector = as_vector(center)
    def field_t(p: point, t: float) -> vector:
        return strength * (p - center_vector)
    return field_t


def vortex(strength: float = 1.0, center: tuple[float, float] = (0.0, 0.0)) -> time_field:
    center_vector = as_vector(center)
    def field_t(p: point, t: float) -> vector:
        x, y = p - center_vector
        return np.array([strength * y, -strength * x], dtype=float)
    return field_t


def saddle(strength: float = 1.0, center: tuple[float, float] = (0.0, 0.0)) -> time_field:
    center_vector = as_vector(center)
    def field_t(p: point, t: float) -> vector:
        x, y = p - center_vector
        return np.array([strength * x, -strength * y], dtype=float)
    return field_t


def rotating_sink(strength: float = 1.0, angular_speed: float = 1.0) -> time_field:
    def field_t(p: point, t: float) -> vector:
        x, y = float(p[0]), float(p[1])
        return np.array([-strength * x + angular_speed * np.cos(t) * y, -strength * y - angular_speed * np.sin(t) * x], dtype=float)
    return field_t


def interpolate_fields(field_a: time_field, field_b: time_field, start: float = 0.0, stop: float = 1.0) -> time_field:
    assert stop != start
    def field_t(p: point, t: float) -> vector:
        u = (t - start) / (stop - start)
        return (1.0 - u) * as_vector(field_a(p, t)) + u * as_vector(field_b(p, t))
    return field_t
