# quanti-gin

<img src=./quanti-gin.png width=200px alt="quanti-gin-logo">

A benchmark suite for quantum chemistry methods.
quanti-gin ships predefined benchmarks — sets of molecular geometries together with reference energies that
are expensive to compute (FCI, CCSD(T)) — and runs methods on them.
The methods themselves (SPA graph selection heuristics, SPA angle read-out, orbital optimization, ...) live in
[sunrise](https://github.com/tequilahub/sunrise), quanti-gin only benchmarks them.

## Installation

To install quanti-gin run `pip install quanti-gin`.
This installs sunrise and all other required packages.

## Usage

List the predefined benchmarks and methods:

    python -m quanti_gin list

Run methods on a benchmark:

    python -m quanti_gin run h6_random --methods spa_blossom spa_nearest_neighbour

This writes one row per geometry and method to `h6_random_results.csv` (energy, error to the reference energy,
runtime, selected SPA graph) and prints the error statistics per method.
Use `--limit N` to only run the first `N` geometries, `--fidelity` to also compute the fidelity with the ground
state and `-j N` to run in `N` processes. Without `--methods` the default methods of the benchmark are run.

Plot the results:

    python -m quanti_gin plot h6_random_results.csv

The same is available from python:

```python
import quanti_gin as qg

results = qg.run_benchmark("h8_ring", methods=["spa_blossom", "spa_two_opt"])
print(qg.summarize(results))
```

## Benchmarks

| benchmark | geometries | references |
| --- | --- | --- |
| `h{4,6,8,10,12}_random` | 50 random hydrogen clusters | HF, FCI |
| `h{4,6,8,10,12}_chain` | linear hydrogen chains, distances 0.5–3.5 Å | HF, FCI |
| `h{4,6,8,10,12}_ring` | hydrogen rings, nearest-neighbour distances 0.5–3.5 Å | HF, FCI |
| `alkanes` | CH4–C8H18 at equilibrium | HF, CCSD(T), FCI where feasible |
| `small_molecules` | H2O, NH3, CH4, HF, H2CO, C2H4, C2H2, CH3OH | HF, CCSD(T), FCI where feasible |

All benchmarks use the STO-3G basis. The datasets are stored in [quanti_gin/data](quanti_gin/data) and can be
regenerated with

    python -m quanti_gin generate [benchmark ...]

The reference energies are computed with PySCF without point group symmetry; for FCI several roots are
converged in the singlet sector and the lowest one is kept, since the default FCI guess misses the ground
state of symmetric geometries such as the rings.

## Custom methods

Any function `run_optimization(molecule, coordinates=..., **kwargs)` that returns a dict with the `"energy"`
can be benchmarked. It receives the sunrise molecule of each geometry.
If it also returns the `"circuit"`, its `"variables"` and the `"molecule"`, the fidelity can be computed.
An example can be found in [examples/custom_method.py](quanti_gin/examples/custom_method.py):

    python -m quanti_gin run h4_chain --custom-method quanti_gin/examples/custom_method.py

New benchmarks are added with `quanti_gin.register(quanti_gin.Benchmark(...))`.
