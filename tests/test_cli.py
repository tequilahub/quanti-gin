import subprocess
import sys

import pytest


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "quanti_gin", *args],
        capture_output=True,
        text=True,
    )


def test_cli_help():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_cli_list():
    result = run_cli("list")
    assert result.returncode == 0
    assert "h4_random" in result.stdout
    assert "spa_blossom" in result.stdout


def test_cli_run(tmp_path):
    output = tmp_path / "results.csv"
    result = run_cli("run", "h4_chain", "-M", "spa_blossom", "--limit", "2", "-O", str(output))
    assert result.returncode == 0, result.stderr
    assert output.exists()


@pytest.mark.parametrize("args", [[], ["run"], ["run", "unknown"], ["run", "h4_chain", "-M", "unknown"]])
def test_cli_invalid(args):
    assert run_cli(*args).returncode != 0


def test_package_import():
    import quanti_gin  # noqa: F401
