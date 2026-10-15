"""
Predefined benchmarks.

Every benchmark is a list of geometries together with reference energies that are expensive to
compute (FCI, CCSD(T)). Geometries and references are stored in `quanti_gin/data/<name>.csv` and
regenerated with `python -m quanti_gin generate <name>`.
"""

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from quanti_gin.geometries import (
    generate_geometry_string,
    hydrogen_chain,
    hydrogen_ring,
    random_hydrogen_cluster,
)
from quanti_gin.references import compute_references

DATA_DIR = Path(str(resources.files("quanti_gin") / "data"))

HYDROGEN_SIZES = [4, 6, 8, 10, 12]
DISTANCES = np.round(np.linspace(0.5, 3.5, 31), 2)
RANDOM_SAMPLES = 50

ALKANES = ["ch4", "c2h6", "c3h8", "c4h10", "c5h12", "c6h14", "c7h16", "c8h18"]
SMALL_MOLECULES = ["h2o", "nh3", "ch4", "hf", "h2co", "c2h4", "c2h2", "ch3oh"]


@dataclass
class Benchmark:
    name: str
    description: str
    # meaning of the `parameter` column (x-axis in the plots)
    parameter: str
    # returns a list of (parameter, geometry string)
    make_geometries: Callable[[], list]
    basis_set: str = "sto-3g"
    references: tuple = ("hf", "fci")
    # methods that are run when none are given
    default_methods: list = field(default_factory=list)

    @property
    def path(self) -> Path:
        return DATA_DIR / f"{self.name}.csv"

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"no dataset for benchmark {self.name}, generate it with `python -m quanti_gin generate {self.name}`"
            )
        return pd.read_csv(self.path)

    def generate(self, path=None, progress=True) -> pd.DataFrame:
        """Compute the reference energies for all geometries and write the dataset."""
        rows = []
        geometries = self.make_geometries()
        for i, (parameter, geometry) in enumerate(tqdm(geometries, desc=self.name, disable=not progress)):
            row = {"id": i, "parameter": parameter, "geometry": geometry}
            row.update(compute_references(geometry, self.basis_set, self.references))
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(self.path if path is None else path, index=False, float_format="%.10f")
        return df


def _random_geometries(n_atoms, samples=RANDOM_SAMPLES, max_distance=2.75, seed=None):
    rng = np.random.default_rng(n_atoms if seed is None else seed)
    return [
        (i, generate_geometry_string(random_hydrogen_cluster(n_atoms, max_distance=max_distance, rng=rng)))
        for i in range(samples)
    ]


def _scan_geometries(generator, n_atoms, distances=DISTANCES):
    return [(float(d), generate_geometry_string(generator(n_atoms, d))) for d in distances]


def _xyz_geometries(names):
    geometries = []
    for name in names:
        lines = (DATA_DIR / "geometries" / f"{name}.xyz").read_text().splitlines()
        n_atoms = int(lines[0])
        geometries.append((name, "\n".join(line.strip() for line in lines[2 : 2 + n_atoms]) + "\n"))
    return geometries


GRAPH_METHODS = ["spa_nearest_neighbour", "spa_nearest_insertion", "spa_two_opt", "spa_blossom", "spa_blossom_scaled"]

BENCHMARKS: dict[str, Benchmark] = {}


def register(benchmark: Benchmark):
    BENCHMARKS[benchmark.name] = benchmark
    return benchmark


for n in HYDROGEN_SIZES:
    register(
        Benchmark(
            name=f"h{n}_random",
            description=f"{RANDOM_SAMPLES} random H{n} clusters",
            parameter="sample",
            make_geometries=lambda n=n: _random_geometries(n),
            default_methods=GRAPH_METHODS,
        )
    )
    register(
        Benchmark(
            name=f"h{n}_chain",
            description=f"linear H{n} chain, distances {DISTANCES[0]}-{DISTANCES[-1]} angstrom",
            parameter="distance",
            make_geometries=lambda n=n: _scan_geometries(hydrogen_chain, n),
            default_methods=GRAPH_METHODS,
        )
    )
    register(
        Benchmark(
            name=f"h{n}_ring",
            description=f"H{n} ring, nearest-neighbour distances {DISTANCES[0]}-{DISTANCES[-1]} angstrom",
            parameter="distance",
            make_geometries=lambda n=n: _scan_geometries(hydrogen_ring, n),
            default_methods=GRAPH_METHODS,
        )
    )

register(
    Benchmark(
        name="alkanes",
        description="alkanes CH4-C8H18 at equilibrium",
        parameter="molecule",
        make_geometries=lambda: _xyz_geometries(ALKANES),
        references=("hf", "ccsdt", "fci"),
        default_methods=["spa_clpo"],
    )
)
register(
    Benchmark(
        name="small_molecules",
        description="small molecules at equilibrium",
        parameter="molecule",
        make_geometries=lambda: _xyz_geometries(SMALL_MOLECULES),
        references=("hf", "ccsdt", "fci"),
        default_methods=["spa_clpo"],
    )
)


def list_benchmarks() -> list[str]:
    return list(BENCHMARKS)


def get_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name}, available: {list_benchmarks()}")
    return BENCHMARKS[name]


def load_benchmark(name: str) -> pd.DataFrame:
    return get_benchmark(name).load()
