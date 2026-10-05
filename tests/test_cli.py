import pytest
import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "quanti_gin.data_generator", *args],
        capture_output=True,
        text=True,
    )


def test_cli_help():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "number of atoms" in result.stdout or "usage" in result.stdout.lower()


def test_cli_odd_number_of_atoms():
    result = run_cli("3", "1")
    assert result.returncode != 0


@pytest.mark.parametrize("num_atoms", range(2, 14, 2))
def test_valid_even_number_of_atoms(num_atoms):
    result = run_cli(str(num_atoms), "1")
    assert result.returncode == 0


def test_cli_generate_multiple_molecules():
    result = run_cli("4", "3")
    assert result.returncode == 0


@pytest.mark.parametrize("num_atoms", [-1, 0, 1, 3, 5, 7, 9])
def test_invalid_number_of_atoms(num_atoms):
    result = run_cli(str(num_atoms), "1")
    assert result.returncode != 0


def test_missing_number_of_atoms():
    result = run_cli()
    assert result.returncode != 0


def test_package_import():
    try:
        import quanti_gin
    except ImportError:
        pytest.fail("Failed to import quanti_gin package")
