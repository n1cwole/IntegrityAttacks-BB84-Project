# two baseline eavesdropper strategies for comparison against HybridAdaptiveEve.
# both inherit from HybridAdaptiveEve so they share all noise dials (F_qnd, F_mem)
# and circuit infrastructure -- the only difference is the per-pulse decision policy.
#
#   PNSOnlyEve:   attacks pulses with n > 1 only (standard pns, no probe)
#                 single-photon and vacuum pulses pass through untouched
#                 matches the original pns simulation from the prior paper
#
#   FixedCNOTEve: applies a full cnot probe (theta = pi) on every pulse
#                 ignores photon number entirely -- no pns path at all
#                 matches the original cnot probe simulation from the prior paper

import random
import numpy as np
from hybrid_adaptive_eve import HybridAdaptiveEve


class PNSOnlyEve(HybridAdaptiveEve):
#standard pns with no probe on single-photon pulses

    def run_round(self):
        alice_bit   = random.randint(0, 1)
        alice_basis = random.choice(['Z', 'X'])
        bob_basis   = random.choice(['Z', 'X'])

        n = np.random.poisson(self.mu)
        self.n_log.append(n)

        if n > 1:
            # multi-photon: run pns
            path     = 'PNS'
            bob_bit, eve_bit = self._run_pns(alice_bit, alice_basis, bob_basis)
            theta_used = None
        else:
            # single or vacuum: do nothing, qubit passes through perfectly
            path     = 'SKIP'
            bob_bit  = alice_bit
            eve_bit  = None   # eve has no information on this round
            theta_used = None

        if alice_basis == bob_basis:
            self.alice_key.append(alice_bit)
            self.bob_key.append(bob_bit)
            # on skipped rounds, eve's guess is random (no information)
            self.eve_key.append(eve_bit if eve_bit is not None else random.randint(0, 1))
            self.basis_log.append(alice_basis)
            self.path_log.append(path)
            self.theta_log.append(theta_used)
            self.running_total += 1
            if bob_bit != alice_bit:
                self.running_errors += 1


class FixedCNOTEve(HybridAdaptiveEve):
#full cnot probe (theta=pi) applied unconditionally to every pulse.

    def run_round(self):
        alice_bit   = random.randint(0, 1)
        alice_basis = random.choice(['Z', 'X'])
        bob_basis   = random.choice(['Z', 'X'])

        n = np.random.poisson(self.mu)
        self.n_log.append(n)

        # always probe at maximum strength -- ignores photon number entirely
        path       = 'PROBE'
        theta_used = np.pi
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
    print("pns-only:", json.dumps(pns.summary(), indent=2))

    cnot = FixedCNOTEve(mu=0.6, F_qnd=1.0, F_mem=1.0, seed=42)
    cnot.run_n_rounds(1000)
    print("\nfixed-cnot:", json.dumps(cnot.summary(), indent=2))