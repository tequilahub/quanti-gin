"""Run methods on a benchmark and compare them to the stored reference energies."""

import multiprocessing
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Sequence, Union

import numpy as np
import openfermion as of
import pandas as pd
import scipy
import sunrise as sn
import tequila as tq
from tqdm import tqdm

from quanti_gin.benchmarks import Benchmark, get_benchmark
from quanti_gin.geometries import parse_geometry_string
from quanti_gin.methods import get_method


@dataclass
class Job:
    benchmark: str
    entry: dict
    basis_set: str
    method_name: str
    method: Callable
    reference: str = "fci"
    calculate_fidelity: bool = False


def compute_fidelity(result: dict, reference_energy: float, num_states=10) -> float:
    """
    Fidelity of the state of the method with the ground state, summed over the (degenerate)
    eigenstates at the reference energy.
    """
    mol = result["molecule"]
    state = tq.simulate(result["circuit"], variables=result["variables"])

    H = of.linalg.get_sparse_operator(mol.make_hamiltonian().to_openfermion())
    num_states = min(num_states, H.shape[0] - 2)
    energies, vectors = scipy.sparse.linalg.eigsh(
        H, k=num_states, sigma=reference_energy
    )

    fidelity = 0.0
    for k in range(num_states):
        if abs(energies[k] - reference_energy) < 1e-6:
            fidelity += (
                abs(state.inner(tq.QubitWaveFunction.from_array(vectors[:, k]))) ** 2
            )
    return fidelity


def execute_job(job: Job) -> dict:
    entry = job.entry
    reference_energy = entry.get(job.reference, np.nan)
    row = {
        "benchmark": job.benchmark,
        "id": entry["id"],
        "parameter": entry["parameter"],
        "method": job.method_name,
        "reference": job.reference,
        "reference_energy": reference_energy,
    }
    try:
        molecule = sn.Molecule(
            geometry=entry["geometry"], basis_set=job.basis_set, backend="pyscf"
        )
        start = time.time()
        result = job.method(
            molecule, coordinates=parse_geometry_string(entry["geometry"])
        )
        row["runtime"] = time.time() - start
        row["energy"] = float(np.real(result["energy"]))
        row["error"] = row["energy"] - reference_energy
        if "edges" in result:
            row["edges"] = str(
                [tuple(int(i) for i in edge) for edge in result["edges"]]
            )
        if (
            job.calculate_fidelity
            and {"circuit", "variables", "molecule"} <= result.keys()
        ):
            row["fidelity"] = compute_fidelity(result, reference_energy)
    except Exception as e:
        row["exception"] = "".join(traceback.format_exception_only(type(e), e)).strip()
    return row


def make_jobs(
    benchmark: Union[str, Benchmark],
    methods: Sequence[Union[str, Callable]] = None,
    reference="fci",
    limit: int = None,
    calculate_fidelity=False,
) -> list[Job]:
    if isinstance(benchmark, str):
        benchmark = get_benchmark(benchmark)
    if not methods:
        methods = benchmark.default_methods
    if reference not in benchmark.references:
        raise ValueError(
            f"benchmark {benchmark.name} has no reference {reference}, available: {benchmark.references}"
        )

    data = benchmark.load()
    if limit is not None:
        data = data.head(limit)

    jobs = []
    for entry in data.to_dict("records"):
        for method in methods:
            if callable(method):
                method_name = f"{method.__module__}.{method.__name__}"
            else:
                method_name, method = method, get_method(method)
            jobs.append(
                Job(
                    benchmark=benchmark.name,
                    entry=entry,
                    basis_set=benchmark.basis_set,
                    method_name=method_name,
                    method=method,
                    reference=reference,
                    calculate_fidelity=calculate_fidelity,
                )
            )
    return jobs


def run_benchmark(
    benchmark: Union[str, Benchmark],
    methods: Sequence[Union[str, Callable]] = None,
    reference="fci",
    limit: int = None,
    calculate_fidelity=False,
    threads: int = None,
    progress=True,
) -> pd.DataFrame:
    """
    Run the methods on every entry of the benchmark.

    Parameters
    ----------
    benchmark :
        name of a predefined benchmark (see `list_benchmarks`) or a `Benchmark`
    methods :
        names from `quanti_gin.methods.METHODS` or callables `method(molecule, coordinates) -> dict`,
        the default methods of the benchmark if not given
    reference :
        reference energy the error is computed to
    limit :
        only run the first `limit` entries
    calculate_fidelity :
        compute the fidelity with the ground state (expensive)
    threads :
        run the jobs in a process pool with this many workers

    Returns
    -------
    pd.DataFrame with one row per entry and method
    """
    jobs = make_jobs(
        benchmark,
        methods,
        reference=reference,
        limit=limit,
        calculate_fidelity=calculate_fidelity,
    )
    if threads:
        with multiprocessing.Pool(threads) as p:
            rows = list(
                tqdm(p.imap(execute_job, jobs), total=len(jobs), disable=not progress)
            )
    else:
        rows = [
            execute_job(job) for job in tqdm(jobs, desc="running", disable=not progress)
        ]
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Error statistics per method"""
    grouped = results.groupby("method")
    summary = grouped["error"].agg(["mean", "median", "max"])
    summary["mean_runtime"] = grouped["runtime"].mean()
    summary["failed"] = grouped["energy"].apply(lambda x: int(x.isna().sum()))
    return summary.sort_values("mean")
