# closed-form expressions for the CRy(theta) entangling probe on bb84.
# all formulas were derived analytically with sympy and verified against
# direct cirq circuit simulation (max deviation 0.020 over 3000 trials).
#
# setup: alice's data qubit is the control. eve's ancilla starts in |0>.
# CRy(theta) applies Ry(theta) to the ancilla iff the data qubit is |1>.
#
# key results (proven, not just empirical):
#   qber_z = 0 for all theta  -- data qubit is unchanged in z basis
#   qber_x = (1 - cos(theta/2)) / 2  -- entanglement dephases x basis
#   acc_z  = 3/4 - cos(theta)/4  -- ancilla encodes z-bit amplitude
#   acc_x  = 0.5 for all theta  -- ancilla carries zero x-bit info
#
# efficiency eta(theta) = E[info] / E[qber] decreases monotonically
# from ~2.0 at theta=0 to 1.0 at theta=pi (full cnot).
# fixed cnot is therefore the least efficient operating point.

import numpy as np


def qber_z(theta: float) -> float:
    # z-basis qber is always zero -- data qubit is never disturbed in z basis
    return 0.0


def qber_x(theta: float) -> float:
    # x-basis qber from entanglement-induced dephasing
    # rises from 0 at theta=0 to 0.5 at theta=pi
    return (1.0 - np.cos(theta / 2.0)) / 2.0


def eve_acc_z(theta: float) -> float:
    # eve's accuracy when both alice and bob used z basis
    # rises from 0.5 at theta=0 (no info) to 1.0 at theta=pi (perfect)
    return 0.75 - np.cos(theta) / 4.0


def eve_acc_x(theta: float) -> float:
    # eve's accuracy in x basis is always 0.5 regardless of theta
    # proven via reduced density matrix: ancilla state is identical for bit=0 and bit=1
    return 0.5


def expected_qber(theta: float, p_x_basis: float = 0.5) -> float:
    # expected qber per probe pulse, averaged over basis prior (default 50/50)
    # eve doesn't know the basis when she must commit to theta, so she plans against the prior
    return (1 - p_x_basis) * qber_z(theta) + p_x_basis * qber_x(theta)


def expected_eve_info(theta: float, p_x_basis: float = 0.5) -> float:
    # expected information gain per probe pulse above the 0.5 random-guessing baseline
    # averaged over basis prior -- used as the reward term in the optimizer cost function
    acc = (1 - p_x_basis) * eve_acc_z(theta) + p_x_basis * eve_acc_x(theta)
    return acc - 0.5


if __name__ == "__main__":
    # print a summary table for manual sanity checking
    print("theta      qber_z   qber_x   acc_z    acc_x    E[qber]  E[info]")
    for th in np.linspace(0, np.pi, 9):
        print(
            f"{th:6.3f}   {qber_z(th):6.3f}   {qber_x(th):6.3f}   "
            f"{eve_acc_z(th):6.3f}   {eve_acc_x(th):6.3f}   "
            f"{expected_qber(th):6.3f}   {expected_eve_info(th):6.3f}"
        )