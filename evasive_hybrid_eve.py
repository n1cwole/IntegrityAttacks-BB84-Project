# extends HybridAdaptiveEve with a basis-symmetrization evasion tactic.
#
# why this exists:
#   the CRy probe has qber_z = 0 for all theta (proven analytically). this creates a detectable signature: qber_z << qber_x in sifted bits.
#   a simple two-proportion z-test on per-basis error rates catches this within ~200 sifted bits, even when pooled qber passes the 11% threshold.
#
# what symmetrization does:
#   eve tracks qber_z and qber_x separately as the session runs.
#   when a gap opens up (qber_x > qber_z), she deliberately flips bob's bit with probability p_flip on probe-path z-basis events.
#   this pushes qber_z toward qber_x, narrowing the detectable gap.
#
# constraints and costs:
#   - p_flip is capped at qber_budget so pooled qber stays within 11%
#   - the injected flips carry zero information for eve (pure cost)
#   - overall eve accuracy drops ~1.5 percentage points
#   - the basis-split z-statistic crossing time is delayed ~4-5x
#   - detection is delayed, not prevented, within the tested range
#
# note: symmetrization only touches probe-path z-basis events.
# pns-path stats are completely unaffected (verified in run_all_validations.py).

import random
import numpy as np
from hybrid_adaptive_eve import HybridAdaptiveEve


class EvasiveHybridEve(HybridAdaptiveEve):

    def __init__(self, *args, symmetrize=True, **kwargs):
        super().__init__(*args, **kwargs)
        # flag to enable/disable the evasion tactic (default on)
        self.symmetrize = symmetrize
        # track qber separately by basis for the symmetrization decision
        self.running_errors_z = 0
        self.running_total_z  = 0
        self.running_errors_x = 0
        self.running_total_x  = 0

    def _current_qber_by_basis(self):
        # compute running qber for z and x basis independently
        qz = self.running_errors_z / self.running_total_z if self.running_total_z > 0 else 0.0
        qx = self.running_errors_x / self.running_total_x if self.running_total_x > 0 else 0.0
        return qz, qx

    def _synthetic_flip_prob(self):
        # compute how much z-basis noise to inject this round
        # p_flip = min(qber_x - qber_z, qber_budget)
        # zero if there is no gap or symmetrization is off
        if not self.symmetrize:
            return 0.0
        qz, qx = self._current_qber_by_basis()
        gap = qx - qz
        if gap <= 0:
            return 0.0
        return min(gap, self.qber_budget)

    def run_round(self):
        alice_bit   = random.randint(0, 1)
        alice_basis = random.choice(['Z', 'X'])
        bob_basis   = random.choice(['Z', 'X'])

        n = np.random.poisson(self.mu)
        self.n_log.append(n)

        if n > 1:
            # pns path: symmetrization does not touch this path at all
            path     = 'PNS'
            bob_bit, eve_bit = self._run_pns(alice_bit, alice_basis, bob_basis)
            theta_used = None
        else:
            # probe path: may inject z-basis noise after the circuit runs
            path       = 'PROBE'
            theta_used = self._choose_theta()
            bob_bit, eve_bit = self._run_probe(theta_used, alice_bit, alice_basis, bob_basis)

            # inject synthetic noise only on sifted z-basis probe events
            # this does not affect what eve measured on her ancilla
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
            error = (bob_bit != alice_bit)
            if error:
                self.running_errors += 1

            # update per-basis trackers for the symmetrization decision
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

    print("evasive (symmetrize=true):")
    e = EvasiveHybridEve(mu=0.6, F_qnd=1.0, F_mem=1.0, qber_budget=0.11, seed=99, symmetrize=True)
    e.run_n_rounds(2000)
    print(json.dumps(e.summary(), indent=2))

    print("\nnon-evasive (symmetrize=false):")
    e2 = EvasiveHybridEve(mu=0.6, F_qnd=1.0, F_mem=1.0, qber_budget=0.11, seed=99, symmetrize=False)
    e2.run_n_rounds(2000)
    print(json.dumps(e2.summary(), indent=2))