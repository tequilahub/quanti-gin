import pytest
import numpy as np
from quanti_gin.shared import (
    generate_min_global_distance_edges,
    random_matching,
    generate_all_possible_edges,
    total_distance,
    brute_force,
    nearest_insertion,
    two_opt,
    simulated_annealing,
    genetic_algorithm,
    minimum_weight_perfect_performance,
)


def is_valid_matching(edges, num_atoms):
    if len(edges) != num_atoms // 2:
        return False
    atoms = set()
    for edge in edges:

        a, b = int(edge[0]), int(edge[1])
        if a == b or len(edge) != 2:
            return False

        if a in atoms or b in atoms:
            return False
        atoms.add(a)
        atoms.add(b)
    return True


def test_random_matching():
    num_atoms = 6
    edges = random_matching(num_atoms)
    assert is_valid_matching(edges, num_atoms)


def test_generate_all_possible_edges():
    edges = generate_all_possible_edges([0, 1, 2, 3])
    assert len(edges) > 0
    for edge in edges:
        assert is_valid_matching(edge, 4)


def test_generate_all_possible_edges_odd_atoms():

    edges = generate_all_possible_edges([0, 1, 2])
    assert edges == []


def test_total_distance():
    coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    edges = [(0, 1), (2, 3)]
    assert total_distance(edges, coordinates) == pytest.approx(2.0)


@pytest.fixture
def fixed_coordinates():
    rand = np.random.default_rng(seed=42)
    return rand.random((6, 3)) * 10


@pytest.mark.parametrize(
    "heuristic",
    [
        brute_force,
        two_opt,
        simulated_annealing,
        genetic_algorithm,
        minimum_weight_perfect_performance,
    ],
)
def test_heuristic_return_valid_edges(heuristic, fixed_coordinates):
    num_atoms = 6
    edges = heuristic(num_atoms, fixed_coordinates)
    assert is_valid_matching(edges, num_atoms)


def test_nearest_insertion(fixed_coordinates):
    num_atoms = 6
    edges = nearest_insertion(fixed_coordinates)
    assert is_valid_matching(edges, num_atoms)


def test_generate_min_global_distance_edges(fixed_coordinates):
    num_atoms = 6
    edges = generate_min_global_distance_edges(fixed_coordinates)
    assert is_valid_matching(edges, num_atoms)
