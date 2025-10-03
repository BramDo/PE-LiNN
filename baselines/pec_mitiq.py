from __future__ import annotations
from typing import Callable, Optional
import numpy as np

try:
    from mitiq import pec
    from mitiq.pec.representations import LocalFolding
except Exception as e:
    raise RuntimeError("This script requires `mitiq`. Install mitiq>=3.0.") from e


def mitigate_with_pec(
    executor: Callable[[object], float],
    circuit: object,
    representation: Optional[pec.PauliRepresentation] = None,
    num_samples: int = 10_000,
) -> float:
    """Run Probabilistic Error Cancellation (PEC) with Mitiq.

    Args:
        executor: Callable mapping circuit->float (expectation value).
        circuit:  The target circuit.
        representation: A quasiprobability representation of the noisy operations.
                        If None, raises to prompt user-specified model.
        num_samples: Number of Monte Carlo samples for PEC.

    Returns:
        Mitigated expectation value.
    """
    if representation is None:
        raise ValueError(
            "PEC requires a noise-model-specific quasiprobability representation. "
            "Provide `representation` built from calibration."
        )
    return pec.execute_with_pec(
        circuit=circuit,
        executor=executor,
        representation=representation,
        num_samples=num_samples,
    )


def _example_stub():
    """Stub showing how a user would calibrate a representation and call PEC.
    The actual calibration is backend/noise specific and not provided here.
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1); qc.h(0); qc.rx(0.3, 0); qc.h(0)

    # User must create a `representation` via calibration against their backend:
    # representation = build_my_quasiprob_representation(backend)
    representation = None  # placeholder

    def executor(_qc):
        # Replace with real expectation evaluation
        return 0.0

    if representation is None:
        print("Provide a calibrated `representation` for PEC.")
        return

    mitigated = mitigate_with_pec(executor, qc, representation=representation, num_samples=5000)
    print("PEC mitigated value:", mitigated)


if __name__ == "__main__":
    _example_stub()
