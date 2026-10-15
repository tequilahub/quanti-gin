import tequila as tq
import logging

logger = logging.getLogger(__name__)


def run_optimization(molecule, *args, **kwargs):
    """
    Custom method for `python -m quanti_gin run <benchmark> --custom-method quanti_gin/examples/custom_method.py`.

    Receives the sunrise molecule of a benchmark entry (and its coordinates as keyword) and
    returns a dict with at least the energy.
    """
    logger.info("running custom optimization method")
    H = molecule.make_hamiltonian()

    U = tq.gates.QCircuit()

    # structure of the ansatz:
    # 1. Ry rotations on all qubits
    for i in range(molecule.n_electrons * 2):
        U += tq.gates.Ry(target=i, angle="a" + str(i))
    depth = 2
    # 2. Entangling layer => Alternating: CNOTs on even and odd qubits
    # 3. Ry rotations on all qubits
    for d in range(depth):
        offset = d % 2
        iterations = molecule.n_electrons * 2
        for i in range(0, iterations - 1 - offset, 2):
            U += tq.gates.CNOT(control=offset + i, target=offset + i + 1)
        for i in range(iterations):
            U += tq.gates.Ry(target=i, angle=f"b{d}_{i}")

    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True)
    logger.info(f"{result.energy}")
    return {
        "energy": result.energy,
        "circuit": U,
        "variables": result.variables,
        "molecule": molecule,
    }
