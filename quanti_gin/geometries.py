"""Geometry generators for the predefined hydrogen benchmarks."""

import numpy as np
from numpy.typing import NDArray


def parse_geometry_string(geometry_string: str) -> NDArray[np.float64]:
    coordinates = []
    for line in geometry_string.split("\n"):
        parts = line.split()
        if not parts:
            continue
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coordinates)


def generate_geometry_string(coordinates: NDArray[np.float64], element="h") -> str:
    geom = ""
    for coordinate in coordinates:
        geom += f"{element} {coordinate[0]:f} {coordinate[1]:f} {coordinate[2]:f}\n"
    return geom


def random_hydrogen_cluster(count: int, max_distance: float, rng: np.random.Generator = None) -> NDArray:
    """
    Random cluster: every new atom is placed at a random offset (up to max_distance in each direction)
    from a randomly chosen atom that is already placed, keeping at least 0.5 angstrom to all atoms.
    """
    if rng is None:
        rng = np.random.default_rng()
    coordinates = np.array([[0, 0, 0]], dtype=np.float64, ndmin=2)
    for i in range(1, count):
        prev_coord = rng.choice(coordinates)
        min_distance = 0
        # make sure we have a minimum of 0.5 angstrom between the atom cores
        while min_distance <= 0.5:
            new_coord = prev_coord + (rng.random(3) * max_distance)
            min_distance = np.min(np.linalg.norm(new_coord - coordinates, axis=1))
        coordinates = np.append(coordinates, new_coord.reshape(1, 3), axis=0)
    return coordinates


def hydrogen_chain(count: int, distance: float) -> NDArray:
    """Linear chain along x with equidistant atoms."""
    return np.array([[i * distance, 0.0, 0.0] for i in range(count)])


def hydrogen_ring(count: int, distance: float) -> NDArray:
    """Regular polygon in the xy plane with nearest-neighbour distance `distance`."""
    radius = distance / (2 * np.sin(np.pi / count))
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.array([[radius * np.cos(angle), radius * np.sin(angle), 0.0] for angle in angles])
