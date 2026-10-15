import numpy as np
import pytest

from quanti_gin import BENCHMARKS, load_benchmark
from quanti_gin.geometries import (
    generate_geometry_string,
    hydrogen_chain,
    hydrogen_ring,
    parse_geometry_string,
    random_hydrogen_cluster,
)
from quanti_gin.references import compute_references


def test_geometry_roundtrip():
    coordinates = np.random.rand(6, 3) * 10
    recovered = parse_geometry_string(generate_geometry_string(coordinates))
    assert np.allclose(coordinates, recovered, atol=1e-6)


def test_random_cluster():
    coordinates = random_hydrogen_cluster(
        6, max_distance=2, rng=np.random.default_rng(1)
    )
    assert coordinates.shape == (6, 3)
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=-1)
    assert np.all(distances[np.triu_indices(6, 1)] > 0.5)


@pytest.mark.parametrize("generator", [hydrogen_chain, hydrogen_ring])
def test_scan_geometries(generator):
    coordinates = generator(6, 1.2)
    assert coordinates.shape == (6, 3)
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=-1)
    assert np.min(distances[np.triu_indices(6, 1)]) == pytest.approx(1.2)


def test_random_geometries_reproducible():
    benchmark = BENCHMARKS["h4_random"]
    assert benchmark.make_geometries() == benchmark.make_geometries()


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_dataset(name):
    benchmark = BENCHMARKS[name]
    data = load_benchmark(name)
    assert len(data) == len(benchmark.make_geometries())
    assert {"id", "parameter", "geometry", *benchmark.references} <= set(data.columns)
    if "fci" in data:
        fci = data["fci"].dropna()
        # variational: FCI below HF everywhere it was computed
        assert np.all(fci <= data.loc[fci.index, "hf"] + 1e-8)


def test_dataset_matches_geometries():
    benchmark = BENCHMARKS["h4_chain"]
    data = load_benchmark("h4_chain")
    parameter, geometry = benchmark.make_geometries()[3]
    assert data["parameter"][3] == pytest.approx(parameter)
    assert np.allclose(
        parse_geometry_string(data["geometry"][3]), parse_geometry_string(geometry)
    )


def test_fci_reference_symmetric_ring():
    # square H4: the default FCI guess converges to an excited state
    geometry = generate_geometry_string(hydrogen_ring(4, 1.0))
    references = compute_references(geometry, "sto-3g")
    assert references["fci"] == pytest.approx(-1.915107, abs=1e-6)
