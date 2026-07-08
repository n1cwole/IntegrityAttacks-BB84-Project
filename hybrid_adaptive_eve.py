# the main simulation class. models a hybrid adaptive eavesdropper on bb84
# that combines photon-number-splitting (pns) and a parameterized CRy(theta)
# entangling probe. per pulse, it:
#   1. draws photon count n ~ poisson(mu) via imperfect qnd (F_qnd)
#   2. if n > 1: runs pns -- siphon one photon, read it after reconciliation (F_mem)
#   3. if n <= 1: runs CRy(theta) probe -- theta is chosen by a constrained optimizer
#
# the optimizer maximizes expected information gain subject to a running qber budget.
# the CRy probe circuit is executed as a real cirq circuit each round.
# the closed-form formulas in hybrid_physics.py are used only by the optimizer.
#
# inheritance chain:
#   HybridAdaptiveEve  <-- base class (this file)
#     EvasiveHybridEve <-- adds basis-symmetrization noise injection
#       FullStackEve   <-- adds fiber depolarization

import random
import numpy as np
import cirq
from scipy.optimize import minimize_scalar
import json

from hybrid_physics import expected_qber, expected_eve_info


class HybridAdaptiveEve:

    def __init__(self, mu=0.6, F_qnd=1.0, F_mem=1.0, qber_budget=0.11,
                 lam=50.0, seed=None):
        # mu: mean photon number of the weak coherent pulse source
        self.mu = mu
        # F_qnd: qnd measurement fidelity (1.0 = ideal, 0.0 = fully noisy)
        self.F_qnd = F_qnd
        # F_mem: quantum memory fidelity (1.0 = perfect readout, 0.0 = random guess)
        self.F_mem = F_mem
        # qber_budget: maximum tolerable pooled qber before session aborting (standard 11%)
        self.qber_budget = qber_budget
        # lam: penalty weight in cost function -- verified invariant across {10..200}
        self.lam = lam
        # theta tracking for probe rounds only
        self.theta_used_list = []
        self.probe_count = 0

        # seed both python random and numpy for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # cirq objects: two named qubits and one simulator instance
        self.simulator = cirq.Simulator()
        self.q = cirq.NamedQubit("data")     # alice's transmitted qubit
        self.a = cirq.NamedQubit("ancilla")  # eve's probe ancilla

        # sifted-key logs (only rounds where alice_basis == bob_basis are appended)
        self.alice_key = []
        self.bob_key   = []
        self.eve_key   = []
        self.basis_log = []   # 'Z' or 'X' for each sifted round
        self.path_log  = []   # 'PNS' or 'PROBE' for each sifted round
        self.theta_log = []   # theta used (None for pns rounds)
        self.n_log     = []   # photon count for every round (not just sifted)

        # running totals for the optimizer's headroom calculation
        self.running_errors = 0
        self.running_total  = 0

# helpers
    def _current_qber(self):
        # pooled qber across all sifted rounds so far
        if self.running_total == 0:
            return 0.0
        return self.running_errors / self.running_total

    def _cost_function(self, theta):
        # objective for the per-pulse optimizer (negated because we minimize)
        # reward: expected information gain above 0.5 baseline
        # penalty: expected disturbance beyond the remaining headroom
        info_gain   = expected_eve_info(theta)
        headroom    = self.qber_budget - self._current_qber()
        disturbance = expected_qber(theta)

        if headroom <= 0:
            # already at or over budget -- penalize any disturbance at all
            penalty = self.lam * disturbance
        else:
            # penalize only the portion that would exceed headroom
            excess  = max(0.0, disturbance - headroom)
            penalty = self.lam * excess

        return -(info_gain - penalty)  # negative because scipy minimizes

    def _choose_theta(self):
        # find the optimal probe angle for this round using bounded scalar search
        result = minimize_scalar(self._cost_function, bounds=(0, np.pi), method='bounded')
        return float(result.x)

#attack paths
    def _run_pns(self, alice_bit, alice_basis, bob_basis):
        # pns path: eve siphons one photon and forwards the rest to bob
        # imperfect qnd disturbs the forwarded qubit with basis-asymmetric probability
        # (x basis is ~2x more fragile than z basis, per ashkenazy et al.)
        p_disturb = (1 - self.F_qnd) / 2.0 if alice_basis == 'X' else (1 - self.F_qnd) / 4.0

        bob_bit = alice_bit
        # apply qnd disturbance only on sifted rounds (same basis)
        if alice_basis == bob_basis and random.random() < p_disturb:
            bob_bit = 1 - bob_bit

        # eve reads her stored photon after reconciliation
        # with probability F_mem she gets the correct bit; otherwise she guesses
        eve_bit = alice_bit if random.random() < self.F_mem else random.randint(0, 1)

        return bob_bit, eve_bit

    def _run_probe(self, theta, alice_bit, alice_basis, bob_basis):
        # probe path: execute CRy(theta) as a real cirq circuit
        # alice encodes her qubit, eve applies the controlled rotation, bob measures
        circuit = cirq.Circuit()

        # alice prepares: encode the bit, then optionally rotate to x basis
        if alice_bit == 1:
            circuit.append(cirq.X(self.q))
        if alice_basis == 'X':
            circuit.append(cirq.H(self.q))

        # eve's intervention: entangle her ancilla with the data qubit
        circuit.append(cirq.ControlledGate(cirq.ry(theta)).on(self.q, self.a))

        # bob measures: optionally rotate back from x basis, then measure
        if bob_basis == 'X':
            circuit.append(cirq.H(self.q))
        circuit.append(cirq.measure(self.q, key='bob'))

        # eve measures her ancilla after basis reconciliation
        # in x basis she gains nothing (proven), but in z basis she gets amplitude info
        if alice_basis == 'X':
            circuit.append(cirq.H(self.a))
        circuit.append(cirq.measure(self.a, key='eve'))

        result  = self.simulator.run(circuit, repetitions=1)
        bob_bit = int(result.measurements['bob'][0][0])
        eve_bit = int(result.measurements['eve'][0][0])
        return bob_bit, eve_bit

#main loop
    def run_round(self):
        # one round of bb84: alice prepares, eve intercepts, bob measures
        alice_bit   = random.randint(0, 1)
        alice_basis = random.choice(['Z', 'X'])
        bob_basis   = random.choice(['Z', 'X'])

        # sample photon count from poisson distribution
        n = np.random.poisson(self.mu)
        self.n_log.append(n)

        if n > 1:
            # pns is viable -- siphon one photon and forward the rest
            path       = 'PNS'
            bob_bit, eve_bit = self._run_pns(alice_bit, alice_basis, bob_basis)
            theta_used = None
        else:
            # single or zero photon -- use the optimized CRy probe
            path       = 'PROBE'
            theta_used = self._choose_theta()
            bob_bit, eve_bit = self._run_probe(theta_used, alice_bit, alice_basis, bob_basis)
            self.theta_used_list.append(theta_used)
            self.probe_count += 1

        # sifting: only record rounds where bases match
        if alice_basis == bob_basis:
            self.alice_key.append(alice_bit)
            self.bob_key.append(bob_bit)
            self.eve_key.append(eve_bit)
            self.basis_log.append(alice_basis)
            self.path_log.append(path)
            self.theta_log.append(theta_used)

            # update running qber for optimizer headroom calculation
            self.running_total += 1
            if bob_bit != alice_bit:
                self.running_errors += 1

    def run_n_rounds(self, n_rounds):
        for _ in range(n_rounds):
            self.run_round()

#metrics
    def summary(self):
        # compute all key metrics over the sifted key
        n_sifted = len(self.alice_key)
        if n_sifted == 0:
            return {}

        alice = np.array(self.alice_key)
        bob   = np.array(self.bob_key)
        eve   = np.array(self.eve_key)
        basis = np.array(self.basis_log)
        path  = np.array(self.path_log)

        # pooled and per-basis qber
        qber_total = np.mean(alice != bob)
        qber_z = np.mean(alice[basis=='Z'] != bob[basis=='Z']) if np.any(basis=='Z') else float('nan')
        qber_x = np.mean(alice[basis=='X'] != bob[basis=='X']) if np.any(basis=='X') else float('nan')

        # eve's accuracy overall and split by attack path
        eve_acc_total = np.mean(alice == eve)
        eve_acc_pns   = np.mean(alice[path=='PNS']   == eve[path=='PNS'])   if np.any(path=='PNS')   else float('nan')
        eve_acc_probe = np.mean(alice[path=='PROBE'] == eve[path=='PROBE']) if np.any(path=='PROBE') else float('nan')

        # fraction of sifted bits that went through each attack path
        coverage_pns   = np.mean(path == 'PNS')
        coverage_probe = np.mean(path == 'PROBE')

        # mean theta across probe rounds only
        mean_theta = float(np.mean(self.theta_used_list)) if self.theta_used_list else float('nan')

        return {
            'n_sifted':       n_sifted,
            'qber_total':     qber_total,
            'qber_z':         qber_z,
            'qber_x':         qber_x,
            'eve_acc_total':  eve_acc_total,
            'eve_acc_pns':    eve_acc_pns,
            'eve_acc_probe':  eve_acc_probe,
            'coverage_pns':   coverage_pns,
            'coverage_probe': coverage_probe,
            'mean_theta_used': mean_theta,
        }


if __name__ == "__main__":
    import json
    # quick smoke test at default parameters
    eve = HybridAdaptiveEve(mu=0.6, F_qnd=1.0, F_mem=1.0, qber_budget=0.11, seed=42)
    eve.run_n_rounds(2000)
    print(json.dumps(eve.summary(), indent=2))