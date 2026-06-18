"""
EvasiveHybridEve extends HybridAdaptiveEve by adding a countermeasure against basis-resolved QBER monitoring.

The CRy(θ) probe introduces zero error in the Z basis, creating a detectable imbalance between QBER_Z and QBER_X. A basis-split statistical test can therefore identify the attack even when overall QBER remains below threshold.

To mitigate this, EvasiveHybridEve tracks QBER separately by basis and injects controlled synthetic bit-flip noise into Z-basis probe events when imbalance appears. This equalizes basis error rates while staying within the global QBER budget and without affecting Eve’s information gain.
"""

import random
import numpy as np

from hybrid_adaptive_eve import HybridAdaptiveEve


class EvasiveHybridEve(HybridAdaptiveEve):
    def __init__(self, *args, symmetrize=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetrize = symmetrize
        self.running_errors_z = 0
        self.running_total_z = 0
        self.running_errors_x = 0
        self.running_total_x = 0

    def _current_qber_by_basis(self):
        qz = self.running_errors_z / self.running_total_z if self.running_total_z > 0 else 0.0
        qx = self.running_errors_x / self.running_total_x if self.running_total_x > 0 else 0.0
        return qz, qx

    def _synthetic_flip_prob(self):
        if not self.symmetrize:
            return 0.0
        qz, qx = self._current_qber_by_basis()
        gap = qx - qz
        if gap <= 0:
            return 0.0
        return min(gap, self.qber_budget)

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

            # Basis-symmetrization: if this pulse lands in the Z basis and a
            # detectable gap has opened up between QBER_Z and QBER_X,
            # deliberately flip Bob's bit with the computed probability.
            # This is pure noise injection -- it does not affect what Eve
            # measured on her ancilla, only what Bob receives.
            if alice_basis == bob_basis == 'Z':
                p_flip = self._synthetic_flip_prob()
                if random.random() < p_flip:
                    bob_bit = 1 - bob_bit

        if alice_basis == bob_basis:
            self.alice_key.append(alice_bit)
            self.bob_key.append(bob_bit)
            self.eve_key.append(eve_bit)
            self.basis_log.append(alice_basis)
            self.path_log.append(path)
            self.theta_log.append(theta_used)

            self.running_total += 1
            error = bob_bit != alice_bit
            if error:
                self.running_errors += 1

            if alice_basis == 'Z':
                self.running_total_z += 1
                if error:
                    self.running_errors_z += 1
            else:
                self.running_total_x += 1
                if error:
                    self.running_errors_x += 1


if __name__ == "__main__":
    import json

    print("EvasiveHybridEve (symmetrize=True)")
    eve = EvasiveHybridEve(mu=0.6, F_qnd=1.0, F_mem=1.0, qber_budget=0.11, seed=99, symmetrize=True)
    eve.run_n_rounds(2000)
    print(json.dumps(eve.summary(), indent=2))

    print("\nComparison: HybridAdaptiveEve (no symmetrization)")
    eve2 = HybridAdaptiveEve(mu=0.6, F_qnd=1.0, F_mem=1.0, qber_budget=0.11, seed=99)
    eve2.run_n_rounds(2000)
    print(json.dumps(eve2.summary(), indent=2))

