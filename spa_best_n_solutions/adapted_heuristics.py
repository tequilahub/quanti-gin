import sunrise as sn
import pandas as pd

from quanti_gin.data_generator import DataGenerator
from sunrise.graph_selection import (
    genetic_algorithm_best_n_solutions,
    simulated_annealing_best_n_solutions,
    two_opt_best_n_solutions,
    random_blossom,
    random_blossom_scaled,
    run_spa_optimization,
)


heuristics = {
    "genetic_algorithm": genetic_algorithm_best_n_solutions,
    "simulated_annealing": simulated_annealing_best_n_solutions,
    "2-opt": two_opt_best_n_solutions,
    "blossom": random_blossom,
    "scaled blossom": random_blossom_scaled,
}

if __name__ == "__main__":
    # run_benchmark(num_atoms=6, num_jobs=5)
    results = []

    jobs = DataGenerator.generate_jobs(number_of_jobs=5, number_of_atoms=12)

    for job in jobs:
        print("new molecule")

        coordinates = job.coordinates
        geometry = job.geometry

        mol = sn.Molecule(geometry=geometry, basis_set="sto-3g")

        for name, function in heuristics.items():
            print(f"Running {name}...")
            edges = function(num_atoms=12, coordinates=coordinates)
            # print(edges)
            for rank, edge in enumerate(edges, start=1):
                print("new edge", edge)
                result = run_spa_optimization(
                    molecule=mol, coordinates=coordinates, edges=edge
                )

                ground_state_energy = mol.compute_energy("fci")
                energy = result["energy"]
                energy_gab = energy - ground_state_energy
                # print(energy)

                results.append(
                    {
                        "molecule_id": job.id,
                        "method": name,
                        "rank": rank,
                        "energy": energy,
                        "edges": edge,
                        "ground state energy": ground_state_energy,
                        "energy gab": energy_gab,
                    }
                )

    data = pd.DataFrame(results)
    data.to_csv(f"benchmark_results_{12}_spa_angles.csv", index=False)
