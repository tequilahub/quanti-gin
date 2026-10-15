"""
Reference energies stored in the benchmark datasets.

The references are computed with PySCF directly and without point group symmetry. For symmetric
geometries (rings, chains, degenerate HF references) the default FCI guess can miss the ground
state, so several roots are converged in the singlet sector and the lowest one is kept.
"""

import math

import numpy as np
from pyscf import cc, fci, gto, scf

# skip FCI above this number of determinants
MAX_FCI_DETERMINANTS = 2e7


def _mean_field(geometry: str, basis_set: str):
    mol = gto.M(atom=geometry, basis=basis_set, unit="angstrom", symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mf


def compute_references(geometry: str, basis_set: str, references=("hf", "fci")) -> dict:
    mf = _mean_field(geometry, basis_set)
    result = {}
    for reference in references:
        if reference == "hf":
            result["hf"] = mf.e_tot
        elif reference == "fci":
            norb = mf.mo_coeff.shape[1]
            nalpha, nbeta = mf.mol.nelec
            n_determinants = math.comb(norb, nalpha) * math.comb(norb, nbeta)
            if n_determinants > MAX_FCI_DETERMINANTS:
                result["fci"] = np.nan
                continue
            solver = fci.addons.fix_spin_(fci.FCI(mf), ss=0)
            nroots = 3 if n_determinants <= 1e6 else 1
            energies, _ = solver.kernel(nroots=nroots)
            result["fci"] = float(np.min(energies))
        elif reference == "ccsdt":
            mycc = cc.CCSD(mf)
            mycc.kernel()
            result["ccsdt"] = mycc.e_tot + mycc.ccsd_t()
        else:
            raise ValueError(f"unknown reference {reference}")
    return result
