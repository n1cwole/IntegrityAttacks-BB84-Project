# fiber depolarization model for the alice-to-bob channel.
#
# model: P(L) = 1 - exp(-gamma * L)
#   with probability P(L), the qubit received by bob is randomized to a uniformly random bit, independent of alice's encoding.
#
# this is a simplified classical bit-flip model. it does not capture full polarization-mode dispersion, but is adequate for a first-order distance dependence study.
#
# gamma = 0.0008 /km is calibrated so that P(200 km) ~= 0.147, consistent with near-future hardware coherence specifications.
#
# important: applied to the alice->bob fiber link only.
# eve's stored photon sits in local quantum memory and is NOT depolarized.

import numpy as np

# default attenuation-style constant in units of 1/km
GAMMA_DEFAULT = 0.0008


def depol_prob(L_km: float, gamma: float = GAMMA_DEFAULT) -> float:
    # probability that a qubit traveling L km is randomized
    return 1.0 - np.exp(-gamma * L_km)


def apply_depol(bit: int, L_km: float, gamma: float = GAMMA_DEFAULT) -> int:
    # return the bit after possible fiber depolarization
    # with probability depol_prob(L_km), the bit is replaced by a random 0 or 1
    if np.random.random() < depol_prob(L_km, gamma):
        return np.random.randint(0, 2)
    return bit


if __name__ == "__main__":
    # print depolarization probability at several distances
    print("distance (km)  P_depol")
    for L in [0, 25, 50, 100, 150, 200, 300]:
        print(f"  {L:>4} km      {depol_prob(L):.4f}")