"""
Methods that can be benchmarked.

A method is a callable `method(molecule, coordinates=..., **kwargs) -> dict` that returns at
least the "energy". The methods themselves live in sunrise, this module only names them.
If the result also holds "circuit", "variables" and "molecule", the fidelity can be computed.
"""

import importlib
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import sunrise as sn
from sunrise.graph_selection import HEURISTICS, run_graph_spa


def spa_graph(molecule, coordinates=None, heuristic="nearest_neighbour", **kwargs):
    """SPA on the graph selected from the atomic positions, see `sunrise.graph_selection.run_graph_spa`"""
    return run_graph_spa(molecule, heuristic=heuristic, coordinates=coordinates)


def spa_clpo(molecule, **kwargs):
    """Orbital-optimized SPA on the CLPO graph, see `sunrise.SPAFP.run_spa`"""
    mol, edges = sn.CLPO.generate_CLPO_molecule_edges(molecule)
    opt = sn.SPAFP.run_spa(mol=mol, edges=edges, initial_guess=np.eye(mol.n_orbitals))
    return {"energy": opt.energy, "edges": edges}


METHODS: dict[str, Callable] = {f"spa_{name}": partial(spa_graph, heuristic=name) for name in HEURISTICS}
METHODS["spa_clpo"] = spa_clpo


def get_method(name: str) -> Callable:
    if name not in METHODS:
        raise ValueError(f"unknown method {name}, available: {list(METHODS)}")
    return METHODS[name]


def import_custom_method(path: Path) -> Callable:
    """Import `run_optimization` from a python file"""
    path = Path(path)
    sys.path.append(str(path.parent))

    custom_module = importlib.import_module(path.stem)
    if not hasattr(custom_module, "run_optimization"):
        raise ValueError(f"invalid file {path} given for custom method, does not contain run_optimization method")
    optimization_method = custom_module.run_optimization
    if not callable(optimization_method):
        raise ValueError(f"invalid file {path} given for custom method, run_optimization is not callable")
    return optimization_method
