"""Plots for the results of `run_benchmark`."""

import matplotlib.pyplot as plt
import pandas as pd

from quanti_gin.benchmarks import BENCHMARKS


def _boxplot(data, column, methods, ax, ylabel):
    ax.boxplot(
        [data[data["method"] == m][column].dropna() for m in methods],
        tick_labels=methods,
        showmeans=True,
        meanline=True,
        medianprops=dict(color="black"),
        meanprops=dict(color="red"),
        widths=0.6,
    )
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)


def plot_results(results, methods=None, show=True):
    """
    Plot benchmark results.

    Parameters
    ----------
    results : pd.DataFrame or str
        results of `run_benchmark` or path to the csv file
    methods : list[str], optional
        methods to include, all if not given

    Plots
    -----
    1. Error to the reference energy per method (boxplot)
    2. Runtime per method (boxplot)
    3. Energies of the methods and the reference against the benchmark parameter
    """
    if isinstance(results, str):
        results = pd.read_csv(results)
    data = results.dropna(subset=["energy"])
    if methods is None:
        methods = sorted(data["method"].unique())
    data = data[data["method"].isin(methods)]

    figures = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _boxplot(
        data, "error", methods, axes[0], f"E - E({data['reference'].iloc[0]}) (Eh)"
    )
    axes[0].set_title("Error per method")
    _boxplot(data, "runtime", methods, axes[1], "runtime (s)")
    axes[1].set_title("Runtime per method")
    fig.tight_layout()
    figures.append(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    reference = data.drop_duplicates("id").sort_values("id")
    numeric = pd.api.types.is_numeric_dtype(reference["parameter"])
    x = reference["parameter"] if numeric else reference["id"]
    ax.plot(
        x,
        reference["reference_energy"],
        color="black",
        linewidth=2,
        label=reference["reference"].iloc[0],
    )
    for method in methods:
        df = data[data["method"] == method].sort_values("id")
        ax.plot(
            df["parameter"] if numeric else df["id"],
            df["energy"],
            marker="o",
            markersize=3,
            label=method,
        )
    if not numeric:
        ax.set_xticks(reference["id"], reference["parameter"], rotation=45)
    name = data["benchmark"].iloc[0]
    ax.set_xlabel(
        f"{name}: {BENCHMARKS[name].parameter if name in BENCHMARKS else 'parameter'}"
    )
    ax.set_ylabel("energy (Eh)")
    ax.legend()
    fig.tight_layout()
    figures.append(fig)

    if show:
        plt.show()
    return figures
