"""
Closed-form physics for the CRy(theta) entangling probe attack on BB84.

All formulas here were derived analytically (sympy) and verified against
direct Cirq circuit simulation before being used in the main experiment.
See verification_log.txt for the derivation/verification trace.

Eve's ancilla starts in |0>, CRy(theta) is applied with the
data qubit as CONTROL and Eve's ancilla as TARGET.

"""

import numpy as np


def qber_z(theta: float) -> float:
    return 0.0


def qber_x(theta: float) -> float:
    return (1.0 - np.cos(theta / 2.0)) / 2.0


def eve_acc_z(theta: float) -> float:
    return 0.75 - np.cos(theta) / 4.0


def eve_acc_x(theta: float) -> float:
    return 0.5


def expected_qber(theta: float, p_x_basis: float = 0.5) -> float:
    """Eve does not know the basis when she commits to theta (she acts
    before reconciliation), so her relevant planning quantity is the
    QBER averaged over the basis prior."""
    return (1 - p_x_basis) * qber_z(theta) + p_x_basis * qber_x(theta)


def expected_eve_info(theta: float, p_x_basis: float = 0.5) -> float:
    """Expected fraction of correct bits Eve nets above a 50% baseline
    guess, averaged over basis prior. This is the information-gain term
    used in the cost function (0 = no better than guessing, up to 0.5 =
    perfect knowledge)."""
    acc = (1 - p_x_basis) * eve_acc_z(theta) + p_x_basis * eve_acc_x(theta)
    return acc - 0.5


if __name__ == "__main__":
    print("theta      QBER_Z   QBER_X   EveAcc_Z  EveAcc_X  E[QBER]  E[InfoGain]")
    for th in np.linspace(0, np.pi, 9):
        print(
            f"{th:6.3f}   {qber_z(th):6.3f}   {qber_x(th):6.3f}   "
            f"{eve_acc_z(th):6.3f}    {eve_acc_x(th):6.3f}    "
            f"{expected_qber(th):6.3f}   {expected_eve_info(th):6.3f}"
        )

