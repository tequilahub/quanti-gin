# quanti-gin

<img src=./quanti-gin.png width=200px alt="quanti-gin-logo">

A customizable data generator for quantum simulation.
Using this library, you can easily create larger sets of simulations and output data for optimizing molecule ground state.
Per default, we are using [Tequila](https://github.com/tequilahub/tequila) as a quantum backend.

Get started with the [tutorial](https://tequilahub.github.io/tequila-tutorials/)

## Installation

To install quanti-gin run `pip install quanti-gin`.
quanti-gin also needs the `devel` branch of [sunrise](https://github.com/tequilahub/sunrise), which is not on PyPI yet:

    pip install "project-sunrise @ git+https://github.com/tequilahub/sunrise.git@devel"

## Usage

For a basic data generation job use:

    python -m quanti_gin 4 100

The basic format of the command line is:

    python -m quanti_gin <number_of_atoms> <number_of_jobs>

If you want to learn about more parameters you can use help:

    python -m quanti_gin -h

## Customize the data generator

quanti-gin is designed so it can be easily customized with your heuristics for data generation.
You can create your own version by simply subclassing the `quanti_gin.DataGenerator` class.

A full example of this can be found in [examples/customized_generator.py](quanti_gin/examples/customized_generator.py).

You can run the example code by executing:

    python -m quanti_gin.examples.customized_generator

## Documentation and Visualization notebook

For a more detailed introduction and background, please take a look at the [Summary](https://nylser.github.io/quanti-gin/Summary.html).

Additionally, for more details on how to interpret and evaluate the generated data, take a look at the [Visualization notebook](./notebooks/Visualization.ipynb).

## Methods in sunrise and reference datasets

The SPA graph heuristics (formerly `quanti_gin.shared`, `quanti_gin.ant_colony_optimization` and the methods of
`spa_best_n_solutions/adapted_heuristics.py`) and `DataGenerator.run_spa_optimization` /
`generate_initial_guess_from_edges` moved to [sunrise](https://github.com/tequilahub/sunrise) as
`sunrise.graph_selection`, with the same names and signatures.
The SPA angle read-out (formerly `quanti_gin.spa_solve`) is now `sunrise.spafastprototype.spa_classical_solver`
(`spa_angles_for_graph`, `hcb_model`, `spa_solve_graph`), where `hcb_model` takes the molecule and
`spa_angles_for_graph` returns the angles as tequila variables.

Reference energies that are expensive to compute (HF, FCI and CCSD(T) in STO-3G) for predefined geometries
(random, linear and ring hydrogen clusters H4-H10, alkanes, small molecules) are stored in
[quanti_gin/data](quanti_gin/data):

```python
from quanti_gin.shared import list_datasets, load_dataset

print(list_datasets())
data = load_dataset("h6_chain")  # columns: id, parameter, geometry, hf, fci
```
