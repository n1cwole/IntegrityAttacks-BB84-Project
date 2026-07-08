"""
FullStackEve: complete model with all components active simultaneously.
  - Adaptive theta optimizer (HybridAdaptiveEve)
  - Basis-symmetrization countermeasure evasion (EvasiveHybridEve)
  - Fiber depolarization P(L) = 1 - exp(-gamma * L) (DepolHybridEve)
  - Imperfect QND fidelity F_qnd
  - Quantum memory decay F_mem

Use this for the final publication figures and full-stack validation.
"""
import numpy as np
from evasive_hybrid_eve import EvasiveHybridEve
from depolarization import apply_depol

class FullStackEve(EvasiveHybridEve):
    def __init__(self, *args, L_km=0.0, gamma=0.0008, **kwargs):
        super().__init__(*args, **kwargs)
        self.L_km = L_km
        self.gamma = gamma

    def _run_pns(self, alice_bit, alice_basis, bob_basis):
        bob_bit, eve_bit = super()._run_pns(alice_bit, alice_basis, bob_basis)
        if alice_basis == bob_basis:
            bob_bit = apply_depol(bob_bit, self.L_km, self.gamma)
        return bob_bit, eve_bit

    def _run_probe(self, theta, alice_bit, alice_basis, bob_basis):
        bob_bit, eve_bit = super()._run_probe(theta, alice_bit, alice_basis, bob_basis)
        if alice_basis == bob_basis:
            bob_bit = apply_depol(bob_bit, self.L_km, self.gamma)
        return bob_bit, eve_bit