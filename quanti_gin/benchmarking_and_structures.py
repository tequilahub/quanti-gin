import quanti_gin
import quanti_gin.data_generator
import tequila as tq
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from quanti_gin.data_generator import DataGenerator
from quanti_gin.data_generator import Job
from quanti_gin.shared import minimum_weight_perfect_performance
from quanti_gin.shared import nearest_insertion
from quanti_gin.shared import generate_min_global_distance_edges
from quanti_gin.shared import brute_force
from quanti_gin.shared import simulated_annealing
from quanti_gin.shared import two_opt
from quanti_gin.shared import genetic_algorithm
from quanti_gin.shared import blossom_scaled
from quanti_gin.visualization_for_benchmarking import (
    benchmarking_data_visualize_matplotlib,
)

from quanti_gin.ant_colony_optimization import ant_colony_opt


heueristics = {
    "blossom": minimum_weight_perfect_performance,
    "nearest_insertion": nearest_insertion,
    "nearest_neighbour": generate_min_global_distance_edges,
    "simulated annealing": simulated_annealing,
    "2-opt": two_opt,
    "genetic algorithm": genetic_algorithm,
    "brute force": brute_force,
    "blossom scaled": blossom_scaled,
    "ant colony": ant_colony_opt,
}


def visualize_molecule(coordinates: np.ndarray, geometry: str):
    """
    Simple 3D scatter plot of molecule structure.
    coordinates: Nx3 array of xyz positions.
    geometry: String for every atom of the form: f"h {coordinate[0]:f} {coordinate[1]:f} {coordinate[2]:f}\n"
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    xs = [c[0] for c in coordinates]
    ys = [c[1] for c in coordinates]
    zs = [c[2] for c in coordinates]

    ax.scatter(xs, ys, zs, s=120, c="skyblue", edgecolors="k")

    for i, (x, y, z) in enumerate(coordinates):
        ax.text(x, y, z, geometry[i], fontsize=12, color="black")

    ax.set_title("Molecular Structure")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def run_benchmark(num_atoms, num_jobs):
    """
    Benchmark heuristics on random molecular geometries.

    Parameters
    ----------
    num_atoms : int
        Number of atoms.
    num_jobs : int
        Number of molecules to test.

    Returns
    -------
    None
        Saves results to CSV.

    CSV structure
    -------------
    method : str
    energy : float
    edges : list[tuple[int, int]]
    runtime : float
    ground_state_energy : float
    energy_gap : float
    error : str (optional)
    """

    jobs = DataGenerator.generate_jobs(
        number_of_jobs=num_jobs, number_of_atoms=num_atoms
    )
    results = []

    # Optional: Visualize first molecule

    # first_job = jobs[0]
    # visualize_molecule(first_job.coordinates, first_job.geometry)

    for job in tqdm(jobs):
        coordinates = job.coordinates
        geometry = job.geometry
        mol = tq.Molecule(geometry=geometry, basis_set="sto-3g")

        for name, function in heueristics.items():
            try:
                if name in (
                    "blossom",
                    "simulated annealing",
                    "2-opt",
                    "genetic algorithm",
                    "brute force",
                    "blossom scaled",
                    "ant colony",
                ):
                    start_time = time.time()
                    edges = function(num_atoms, coordinates)

                    runtime = time.time() - start_time
                else:
                    start_time = time.time()
                    edges = function(coordinates)

                    runtime = time.time() - start_time

                result = DataGenerator.run_spa_optimization(
                    molecule=mol, coordinates=coordinates, edges=edges
                )

                # _, ground_state_energy = DataGenerator.get_ground_states(molecule = mol)
                # ground_state_energy = DataGenerator.run_fci_optimization(molecule=mol)
                ground_state_energy = mol.compute_energy("fci")
                energy = result["energy"]

                energy_gab = energy - ground_state_energy

                results.append(
                    {
                        "method": name,
                        "energy": energy,
                        "edges": edges,
                        "runtime": runtime,
                        "ground state energy": ground_state_energy,
                        "energy gab": energy_gab,
                    }
                )

            except Exception as e:

                results.append(
                    {
                        "method": name,
                        "energy": None,
                        "edges": None,
                        "runtime": None,
                        "ground state energy": None,
                        "energy gab": None,
                        "error": str(e),
                    }
                )

    data = pd.DataFrame(results)
    data.to_csv(f"benchmark_results_{num_atoms}.csv", index=False)


def run_benchmark_for_linear_molecules(
    num_atoms: int, num_jobs: int, axis="x", base_spacing=0.25
):
    """
    Benchmark heuristics on linear molecules.

    Parameters
    ----------
    num_atoms : int
    num_jobs : int
    axis : {"x","y","z"}, optional
    base_spacing : float, optional

    Returns
    -------
    None
        Saves results to CSV.

    CSV structure
    -------------
    method : str
    energy : float
    edges : list[tuple[int, int]]
    runtime : float
    ground_state_energy : float
    energy_gap : float
    error : str (optional)
    """

    assert axis in ("x", "y", "z")  # Axis must be 'x', 'y', or 'z'

    results = []

    for i in tqdm(range(1, num_jobs + 1)):
        # Vary spacing slightly for each job
        spacing = base_spacing + 0.05 * i
        coordinates = []
        for j in range(num_atoms):
            if axis == "x":
                coord = [j * spacing, 0.0, 0.0]
            elif axis == "y":
                coord = [0.0, j * spacing, 0.0]
            else:
                coord = [0.0, 0.0, j * spacing]
            coordinates.append(coord)

        coordinates = np.array(coordinates)
        geometry = DataGenerator.generate_geometry_string(coordinates)

        # visualize_molecule(coordinates, geometry)

        mol = tq.Molecule(geometry=geometry, basis_set="sto-3g")

        for name, function in heueristics.items():
            try:
                if name in (
                    "blossom",
                    "simulated annealing",
                    "2-opt",
                    "genetic algorithm",
                    "brute force",
                ):
                    start_time = time.time()
                    edges = function(num_atoms, coordinates)
                    runtime = time.time() - start_time

                else:
                    start_time = time.time()
                    edges = function(coordinates)
                    runtime = time.time() - start_time

                result = DataGenerator.run_spa_optimization(
                    molecule=mol, coordinates=coordinates, edges=edges
                )

                ground_state_energy = mol.compute_energy("fci")
                energy = result["energy"]

                energy_gab = energy - ground_state_energy

                results.append(
                    {
                        "method": name,
                        "energy": energy,
                        "edges": edges,
                        "runtime": runtime,
                        "ground state energy": ground_state_energy,
                        "energy gab": energy_gab,
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "method": name,
                        "energy": None,
                        "edges": None,
                        "runtime": None,
                        "ground state energy": None,
                        "energy gab": None,
                        "error": str(e),
                    }
                )

    data = pd.DataFrame(results)
    data.to_csv(f"benchmark_results_{num_atoms}_line_{axis}.csv", index=False)


def run_benchmark_for_ring_molecules(
    num_atoms, num_jobs, radius=1, radius_increase=0.25
):
    """
    Benchmark heuristics on ring molecules.

    Parameters
    ----------
    num_atoms : int
    num_jobs : int
    radius : float, optional
    radius_increase : float, optional

    Returns
    -------
    None
        Saves results to CSV.

    CSV structure
    -------------
    method : str
    energy : float
    edges : list[tuple[int, int]]
    runtime : float
    ground_state_energy : float
    energy_gap : float
    error : str (optional)
    """

    jobs = DataGenerator.generate_jobs(
        number_of_jobs=num_jobs, number_of_atoms=num_atoms
    )
    results = []

    for job in tqdm(jobs):

        angles = np.linspace(0, 2 * np.pi, num_atoms, endpoint=False)
        # angles = np.sort(np.random.uniform(0, 2 * np.pi, num_atoms))
        coordinates = np.array(
            [[radius * np.cos(angle), radius * np.sin(angle), 0.0] for angle in angles]
        )

        # print(radius)
        radius = radius + radius_increase

        formatted_lines = []
        for coord in coordinates:
            x, y, z = coord
            line = f"h {x:.6f} {y:.6f} {z:.6f}"
            formatted_lines.append(line)
        geometry = "\n".join(formatted_lines)

        # visualize_molecule(coordinates, geometry)

        mol = tq.Molecule(geometry=geometry, basis_set="sto-3g")

        for name, function in heueristics.items():
            try:
                if name in (
                    "blossom",
                    "simulated annealing",
                    "2-opt",
                    "genetic algorithm",
                    "brute force",
                ):
                    start_time = time.time()
                    edges = function(num_atoms, coordinates)

                    runtime = time.time() - start_time
                else:
                    start_time = time.time()
                    edges = function(coordinates)

                    runtime = time.time() - start_time

                result = DataGenerator.run_spa_optimization(
                    molecule=mol, coordinates=coordinates, edges=edges
                )

                ground_state_energy = mol.compute_energy("fci")
                energy = result["energy"]

                energy_gap = energy - ground_state_energy

                results.append(
                    {
                        "method": name,
                        "energy": energy,
                        "edges": edges,
                        "runtime": runtime,
                        "ground state energy": ground_state_energy,
                        "energy gab": energy_gap,
                    }
                )

            except Exception as e:

                results.append(
                    {
                        "method": name,
                        "energy": None,
                        "edges": None,
                        "runtime": None,
                        "ground state energy": None,
                        "energy gab": None,
                        "error": str(e),
                    }
                )

    data = pd.DataFrame(results)

    data.to_csv(f"benchmark_results_{num_atoms}_ring.csv", index=False)


# examples for testing

if __name__ == "__main__":

    # run_benchmark(num_atoms=10, num_jobs=10)
    # benchmarking_data_visualize_matplotlib("benchmark_results_10.csv")

    # run_benchmark_for_linear_molecules(num_atoms=4, num_jobs=200, axis="x")

    # run_benchmark_for_ring_molecules(num_atoms=10, num_jobs=200, radius=1)

    """
    benchmarking_data_visualize_matplotlib("benchmark_results_10.csv",
                                                methods_to_plot=["blossom", "nearest_insertion", "nearest_neighbour", "simulated annealing", "2-opt", "genetic algorithm", "brute force"],
                                                show_first_n_molecules=55)
    """
