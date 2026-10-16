import pytest
import numpy as np
import sunrise as sn
from quanti_gin.data_generator import DataGenerator
from quanti_gin.shared import list_datasets, load_dataset
from sunrise.graph_selection import run_spa_optimization

# pytestmark = pytest.mark.integration


@pytest.fixture
def square_coordinates():
    return np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])


def test_execute_job():
    jobs = DataGenerator.generate_jobs(
        number_of_atoms=4, number_of_jobs=1, method="fci"
    )
    result = DataGenerator.execute_job(jobs[0])
    assert "result" in result


def test_spa_pipeline(square_coordinates):
    geometry = DataGenerator.generate_geometry_string(square_coordinates)
    mol = sn.Molecule(geometry=geometry, basis_set="sto-3g")

    result = run_spa_optimization(molecule=mol, coordinates=square_coordinates)
    assert "energy" in result
    assert result["energy"] is not None


def test_datasets():
    names = list_datasets()
    assert "h4_chain" in names and "small_molecules" in names
    for name in names:
        data = load_dataset(name)
        assert {"id", "parameter", "geometry", "hf"} <= set(data.columns)
        if name.startswith("h"):
            assert (data["fci"] <= data["hf"] + 1e-8).all()
