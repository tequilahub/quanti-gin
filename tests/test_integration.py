import numpy as np
import pytest

from quanti_gin import run_benchmark, summarize
from quanti_gin.examples.custom_method import run_optimization


def test_run_benchmark():
    results = run_benchmark(
        "h4_random",
        methods=["spa_blossom", "spa_nearest_neighbour"],
        limit=2,
        progress=False,
    )
    assert len(results) == 4
    assert results["energy"].notna().all()
    # SPA is variational
    assert np.all(results["error"] > -1e-8)
    assert len(summarize(results)) == 2


def test_run_benchmark_fidelity():
    results = run_benchmark(
        "h4_chain",
        methods=["spa_blossom"],
        limit=1,
        calculate_fidelity=True,
        progress=False,
    )
    assert 0 < results["fidelity"][0] <= 1 + 1e-8


def test_custom_method():
    results = run_benchmark(
        "h4_chain", methods=[run_optimization], limit=1, progress=False
    )
    assert results["energy"].notna().all()


def test_failing_method_is_recorded():
    def broken(molecule, **kwargs):
        raise RuntimeError("broken method")

    results = run_benchmark("h4_chain", methods=[broken], limit=1, progress=False)
    assert "broken method" in results["exception"][0]


def test_unknown_reference():
    with pytest.raises(ValueError):
        run_benchmark("h4_chain", methods=["spa_blossom"], reference="ccsdt", limit=1)


def test_plot_results():
    import matplotlib

    matplotlib.use("Agg")
    from quanti_gin.visualization import plot_results

    results = run_benchmark(
        "h4_chain", methods=["spa_blossom"], limit=2, progress=False
    )
    assert len(plot_results(results, show=False)) == 2


def test_spa_clpo():
    # CH4, the first entry of the alkanes
    results = run_benchmark("alkanes", methods=["spa_clpo"], limit=1, progress=False)
    assert 0 < results["error"][0] < 0.05
