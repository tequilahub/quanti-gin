import pytest
import numpy as np
from quanti_gin.data_generator import DataGenerator


def test_geometry_roundtrip():
    num_atoms = 6
    coordinates = np.random.rand(num_atoms, 3) * 10
    geometry = DataGenerator.generate_geometry_string(coordinates)
    coordinates_recovered = DataGenerator.parse_geometry_string(geometry)
    assert coordinates.shape == coordinates_recovered.shape
    assert np.allclose(coordinates, coordinates_recovered, atol=1e-6)


def test_generate_coordinates_count():
    coordinates = DataGenerator.generate_coordinates(count=6, max_distance=2)
    assert len(coordinates) == 6
    assert coordinates.shape == (6, 3)


def test_generate_initial_guess_shape():
    num_atoms = 4
    coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    edges = [(0, 1), (2, 3)]

    matrix = DataGenerator.generate_initial_guess_from_edges(coordinates, edges)
    assert matrix.shape == (num_atoms, num_atoms)


def test_generate_jobs():
    jobs = DataGenerator.generate_jobs(number_of_atoms=4, number_of_jobs=3)
    assert len(jobs) == 3

    for job in jobs:
        assert job.geometry is not None
        assert job.coordinates.shape == (4, 3)


def test_generate_initial_guess_odd():
    coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError):
        DataGenerator.generate_initial_guess_from_edges(coordinates, [(0, 1)])


def test_parse_geometry_string():
    geometry = "h 0.0 0.0 0.0\nh 1.0 0.0 0.0\nh 0.0 1.0 0.0\nh 1.0 1.0 0.0"
    expected_coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    coordinates = DataGenerator.parse_geometry_string(geometry)
    assert coordinates.shape == expected_coordinates.shape
    assert np.allclose(coordinates, expected_coordinates, atol=1e-6)
