"""
Baseline attacks for comparison against HybridAdaptiveEve.

PNSOnlyEve: only attacks pulses with n > 1 (standard PNS), does nothing on n <= 1 pulses 
            (matches our original PNS simulation's behavior exactly).

FixedCNOTEve: always applies a full CNOT probe (theta = pi) on every pulse regardless of photon number, ignoring PNS entirely
              (matches our original CNOT probe simulation's behavior).

Both share the same noise dials (F_qnd, F_mem) and circuit machinery as HybridAdaptiveEve so that comparisons are apples-to-apples.
The only difference is the decision policy.
"""

import random
import numpy as np
import cirq

from hybrid_adaptive_eve import HybridAdaptiveEve


class PNSOnlyEve(HybridAdaptiveEve):
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
            path = 'SKIP'
            bob_bit = alice_bit  # untouched, passes through perfectly
            eve_bit = None
            theta_used = None

        if alice_basis == bob_basis:
            self.alice_key.append(alice_bit)
            self.bob_key.append(bob_bit)
            self.eve_key.append(eve_bit if eve_bit is not None else random.randint(0, 1))
            self.basis_log.append(alice_basis)
            self.path_log.append(path)
            self.theta_log.append(theta_used)

            self.running_total += 1
            if bob_bit != alice_bit:
                self.running_errors += 1


class FixedCNOTEve(HybridAdaptiveEve):
    def run_round(self):
        alice_bit = random.randint(0, 1)
        alice_basis = random.choice(['Z', 'X'])
        bob_basis = random.choice(['Z', 'X'])

        n = np.random.poisson(self.mu)
        self.n_log.append(n)

        path = 'PROBE'
        theta_used = np.pi  # always full CNOT, no optimization, no PNS
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


if __name__ == "__main__":
    import json

    pns = PNSOnlyEve(mu=0.6, F_qnd=1.0, F_mem=1.0, seed=42)
    pns.run_n_rounds(1000)
    print("PNS-only:", json.dumps(pns.summary(), indent=2))

    cnot = FixedCNOTEve(mu=0.6, F_qnd=1.0, F_mem=1.0, seed=42)
    cnot.run_n_rounds(1000)
    print("\nFixed-CNOT-only:", json.dumps(cnot.summary(), indent=2))

