"""
DepolHybridEve: HybridAdaptiveEve extended with fiber depolarization.

Depolarization is applied to Bob's received qubit on the Alice->Bob link.
For the PNS path this means the qubit forwarded to Bob may be randomized by fiber noise independent of Eve's QND disturbance. 
For the probe path it means Bob's circuit output is post-processed with the depol model (because the Cirq circuit models an ideal channel; 
we add depol on top classically, which is equivalent for a depolarizing channel in the Z and X bases since it randomizes the output with probability P(L)).

Eve's stored photon is not affected by fiber depolarization.
"""

import random
import numpy as np

from hybrid_adaptive_eve import HybridAdaptiveEve
from depolarization import apply_depol, depol_prob


class DepolHybridEve(HybridAdaptiveEve):
    def __init__(self, *args, L_km=0.0, gamma=0.0008, **kwargs):
        super().__init__(*args, **kwargs)
        self.L_km = L_km
        self.gamma = gamma

    def _run_pns(self, alice_bit, alice_basis, bob_basis):
        bob_bit, eve_bit = super()._run_pns(alice_bit, alice_basis, bob_basis)
        # fiber depolarization on the forwarded photon (after Eve's split)
        if alice_basis == bob_basis:
            bob_bit = apply_depol(bob_bit, self.L_km, self.gamma)
        return bob_bit, eve_bit

    def _run_probe(self, theta, alice_bit, alice_basis, bob_basis):
        bob_bit, eve_bit = super()._run_probe(theta, alice_bit, alice_basis, bob_basis)
        # fiber depolarization on top of circuit result
        if alice_basis == bob_basis:
            bob_bit = apply_depol(bob_bit, self.L_km, self.gamma)
        return bob_bit, eve_bit