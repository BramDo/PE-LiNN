from __future__ import annotations
from typing import Callable, Sequence, Optional
import numpy as np

try:
    from mitiq import cdr
except Exception as e:
    raise RuntimeError("This script requires `mitiq`. Install mitiq>=3.0.") from e


def mitigate_with_cdr(
    executor: Callable[[object], float],
    circuit: object,
    num_training_circuits: int = 30,
    generator: Optional[Callable[[object, int], Sequence[object]]] = None,
) -> float:
    """Run Clifford Data Regression (CDR) with Mitiq.

    Args:
        executor: Callable that maps a circuit to an expectation value (float).
        circuit:  The target circuit to mitigate.
        num_training_circuits: Number of training circuits to generate.
        generator: Optional training-set generator. If None, uses Mitiq's default generator.

    Returns:
        Mitigated expectation value.
    """
    # Use default training circuit generator if not provided
    if generator is None:
        # This uses mitiq.cdr to create near-Clifford training set
        def generator(circ, n):
            return cdr.generate_training_circuits(circ, n)
    return cdr.execute_with_cdr(
        circuit=circuit,
        executor=executor,
        num_training_circuits=num_training_circuits,
        generator=generator,
    )


def _example_qiskit_executor(observable=None):
    from qiskit_aer.primitives import Estimator
    from qiskit.quantum_info import SparsePauliOp
    est = Estimator()
    obs = observable or SparsePauliOp.from_list([("Z", 1.0)])

    def _exec(qc):
        return float(est.run(qc, obs).result().values[0])
    return _exec


def _demo():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.h(0); qc.rz(0.17, 0); qc.h(0)

    executor = _example_qiskit_executor()
    mitigated = mitigate_with_cdr(executor, qc, num_training_circuits=20)
    print("CDR mitigated value:", mitigated)


if __name__ == "__main__":
    _demo()
