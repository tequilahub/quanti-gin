from dataclasses import dataclass
from os import PathLike
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Result:
    job_id: int
    method: str
    optimized_energy: float
    optimized_variables: list[float]
    edges: list[tuple[int, int]]


@dataclass
class DataFile:
    file_path: PathLike
    df: pd.DataFrame
    coordinates: NDArray[np.float64]
    method_results: dict[str, Result]


def read_result(job) -> Result:
    job_id = job["job_id"]
    method = job["method"]
    optimized_energy = job["optimized_energy"]
    if "optimized_variables" in job:
        optimized_variables = job["optimized_variables"]
        optimized_variable_counts = job["optimized_variable_count"]
        optimized_variable_cols = [
            [f"optimized_variable_{i}" for i in range(optimized_variable_count)]
            for optimized_variable_count in optimized_variable_counts
        ]

        for i, cols in enumerate(optimized_variable_cols):
            sample_variables = [job[col][i] for col in cols]
            optimized_variables.append(sample_variables)
    else:
        optimized_variables = []
    if "edges" in job:
        edges = job["edges"]
    else:
        edges = []
    return Result(
        job_id=job_id,
        method=method,
        optimized_energy=optimized_energy,
        optimized_variables=optimized_variables,
        edges=edges,
    )


def read_data_file(file_path: PathLike):
    # load the csv file into a pandas dataframe
    df = pd.read_csv(file_path)

    method_results = {}
    job_groups = df.groupby("method")
    for group_key, group in job_groups:
        method_results[group_key] = read_result(group)

    atom_counts = df["atom_count"]
    coordinate_cols = [
        [(f"x_{i}", f"y_{i}", f"z_{i}") for i in range(atom_count)]
        for atom_count in atom_counts
    ]

    coordinates = []
    for i, cols in enumerate(coordinate_cols):
        sample_coordinates = [
            (df[col[0]][i], df[col[1]][i], df[col[2]][i]) for col in cols
        ]
        coordinates.append(sample_coordinates)
    coordinates = np.array(coordinates)

    if "edge_0_start" in df:
        edges = []
        edge_counts = df["edge_count"]

        edge_cols = [
            [(f"edge_{i}_start", f"edge_{i}_end") for i in range(edge_count)]
            for edge_count in edge_counts
        ]

        for i, cols in enumerate(edge_cols):
            sample_edges = [(df[col[0]][i], df[col[1]][i]) for col in cols]
            edges.append(sample_edges)
    else:
        edges = []

    return DataFile(
        file_path=file_path,
        df=df,
        coordinates=coordinates,
        method_results=method_results,
    )


# reference energies (HF, FCI, CCSD(T)) of predefined geometries, stored in quanti_gin/data
DATA_DIR = Path(__file__).parent / "data"


def list_datasets():
    return sorted(path.stem for path in DATA_DIR.glob("*.csv"))


def load_dataset(name: str) -> pd.DataFrame:
    """columns: id, parameter (sample, distance in angstrom or molecule), geometry and the reference energies"""
    return pd.read_csv(DATA_DIR / f"{name}.csv")
