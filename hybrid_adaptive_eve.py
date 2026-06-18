"""
Hybrid Adaptive Eve models a BB84 eavesdropper combining photon-number-splitting (PNS) and a CRy(θ) probe.

For each pulse:
1. Photon number n ~ Poisson(μ) is estimated via imperfect QND measurement (F_qnd).
2. If n > 1, Eve performs a PNS attack using quantum memory with fidelity F_mem.
3. If n ≤ 1, Eve applies a CRy(θ) entangling probe, where θ is chosen by optimizing expected information gain under a QBER budget constraint.

The CRy interaction is simulated using Cirq circuits, while closed-form expressions in hybrid_physics.py are used only for optimizing θ.
"""

import random
import numpy as np
import cirq
from scipy.optimize import minimize_scalar

from hybrid_physics import expected_qber, expected_eve_info


class HybridAdaptiveEve:
    def __init__(self, mu=0.6, F_qnd=1.0, F_mem=1.0, qber_budget=0.11,
                 lam=50.0, seed=None):
        self.mu = mu
        self.F_qnd = F_qnd
        self.F_mem = F_mem
        self.qber_budget = qber_budget
        self.lam = lam  # penalty weight in the cost function

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.simulator = cirq.Simulator()
        self.q = cirq.NamedQubit("data")
        self.a = cirq.NamedQubit("ancilla")

        # Running tallies (sifted bits only, ex: alice_basis == bob_basis)
        self.alice_key = []
        self.bob_key = []
        self.eve_key = []          # Eve's guess for each sifted bit (or None)
        self.basis_log = []        # 'Z' or 'X' for each sifted bit
        self.path_log = []         # 'PNS', 'PROBE', or 'SKIP' for each sifted bit
        self.theta_log = []        # theta used, for PROBE pulses (None otherwise)
        self.n_log = []            # photon number sampled, for every pulse (not just sifted)

        self.running_errors = 0
        self.running_total = 0

    # helpers

    def _current_qber(self):
        if self.running_total == 0:
            return 0.0
        return self.running_errors / self.running_total

    def _cost_function(self, theta):
        info_gain = expected_eve_info(theta)
        current_qber = self._current_qber()
        headroom = self.qber_budget - current_qber
        disturbance = expected_qber(theta)

        if headroom <= 0:
            penalty = self.lam * disturbance
        else:
            excess = max(0.0, disturbance - headroom)
            penalty = self.lam * excess

        return -(info_gain - penalty)  # negative because we minimize

    def _choose_theta(self):
        # Solve the constrained optimization for this pulse's probe angle.
        result = minimize_scalar(self._cost_function, bounds=(0, np.pi), method='bounded')
        return float(result.x)

    # attack paths 

    def _run_pns(self, alice_bit, alice_basis, bob_basis):
        # Imperfect QND disturbance probability, asymmetric by basis
        # (matches Ashkenazy et al.'s finding that X-basis qubits are more
        # fragile to QND backaction than Z-basis qubits).
        p_disturb_z = (1 - self.F_qnd) / 4.0
        p_disturb_x = (1 - self.F_qnd) / 2.0
        p_disturb = p_disturb_x if alice_basis == 'X' else p_disturb_z

        bob_bit = alice_bit
        if alice_basis == bob_basis and random.random() < p_disturb:
            bob_bit = 1 - bob_bit  # induced error from imperfect QND

        # Eve's readout: with probability F_mem her stored photon survives
        # and she reads the correct bit; otherwise she is reduced to a
        # random guess.
        if random.random() < self.F_mem:
            eve_bit = alice_bit
        else:
            eve_bit = random.randint(0, 1)

        return bob_bit, eve_bit

    def _run_probe(self, theta, alice_bit, alice_basis, bob_basis):
        """CRy(theta) probe path, executed as an actual Cirq circuit."""
        circuit = cirq.Circuit()
        if alice_bit == 1:
            circuit.append(cirq.X(self.q))
        if alice_basis == 'X':
            circuit.append(cirq.H(self.q))

        circuit.append(cirq.ControlledGate(cirq.ry(theta)).on(self.q, self.a))

        if bob_basis == 'X':
            circuit.append(cirq.H(self.q))
        circuit.append(cirq.measure(self.q, key='bob'))

        # Eve measures her ancilla AFTER basis reconciliation, in the
        # basis she now knows was used (only matters for Z; in X her
        # ancilla carries no information regardless -- see
        # hybrid_physics.py docstring for the derivation).
        if alice_basis == 'X':
            circuit.append(cirq.H(self.a))
        circuit.append(cirq.measure(self.a, key='eve'))

        result = self.simulator.run(circuit, repetitions=1)
        bob_bit = int(result.measurements['bob'][0][0])
        eve_bit = int(result.measurements['eve'][0][0])
        return bob_bit, eve_bit

    # main loop

    def run_round(self):
        alice_bit = random.randint(0, 1)
        alice_basis = random.choice(['Z', 'X'])
        bob_basis = random.choice(['Z', 'X'])

        n = np.random.poisson(self.mu)
        self.n_log.append(n)

        if n > 1:
            path = 'PNS'
            bob_bit, eve_bit = self._run_pns(alice_bit, alice_basis, bob_basis)
            theta_used = None
        else:
            path = 'PROBE'
            theta_used = self._choose_theta()
            bob_bit, eve_bit = self._run_probe(theta_used, alice_bit, alice_basis, bob_basis)

        if alice_basis == bob_basis:
            self.alice_key.append(alice_bit)
            self.bob_key.append(bob_bit)
            self.eve_key.append(eve_bit)
            self.basis_log.append(alice_basis)
            self.path_log.append(path)
            self.theta_log.append(theta_used)

            self.running_total += 1
            if bob_bit != alice_bit:
                self.running_errors += 1

    def run_n_rounds(self, n_rounds):
        for _ in range(n_rounds):
            self.run_round()

    # metrics

    def summary(self):
        n_sifted = len(self.alice_key)
        if n_sifted == 0:
            return {}

        alice = np.array(self.alice_key)
        bob = np.array(self.bob_key)
        eve = np.array(self.eve_key)
        basis = np.array(self.basis_log)
        path = np.array(self.path_log)

        qber_total = np.mean(alice != bob)
        qber_z = np.mean(alice[basis == 'Z'] != bob[basis == 'Z']) if np.any(basis == 'Z') else float('nan')
        qber_x = np.mean(alice[basis == 'X'] != bob[basis == 'X']) if np.any(basis == 'X') else float('nan')

        eve_acc_total = np.mean(alice == eve)
        eve_acc_pns = np.mean(alice[path == 'PNS'] == eve[path == 'PNS']) if np.any(path == 'PNS') else float('nan')
        eve_acc_probe = np.mean(alice[path == 'PROBE'] == eve[path == 'PROBE']) if np.any(path == 'PROBE') else float('nan')

        coverage_pns = np.mean(path == 'PNS')
        coverage_probe = np.mean(path == 'PROBE')

        return {
            'n_sifted': n_sifted,
            'qber_total': qber_total,
            'qber_z': qber_z,
            'qber_x': qber_x,
            'eve_acc_total': eve_acc_total,
            'eve_acc_pns': eve_acc_pns,
            'eve_acc_probe': eve_acc_probe,
            'coverage_pns': coverage_pns,
            'coverage_probe': coverage_probe,
            'mean_theta_used': np.mean([t for t in self.theta_log if t is not None]) if any(t is not None for t in self.theta_log) else float('nan'),
        }


if __name__ == "__main__":
    eve = HybridAdaptiveEve(mu=0.6, F_qnd=1.0, F_mem=1.0, qber_budget=0.11, seed=42)
    eve.run_n_rounds(2000)
    import json
    print(json.dumps(eve.summary(), indent=2))

