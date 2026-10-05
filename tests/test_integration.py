import pytest
import numpy as np
import tequila as tq
from quanti_gin.data_generator import DataGenerator

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
    mol = tq.Molecule(geometry=geometry, basis_set="sto-3g")

    result = DataGenerator.run_spa_optimization(
        molecule=mol, coordinates=square_coordinates
    )
    assert "energy" in result
    assert result["energy"] is not None
