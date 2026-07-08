# generates the data for figure G: the basis-split detection monitor.
#
# strategy: run each eve variant once to a large sample size per seed, then slice the accumulated data at multiple checkpoints.
# this is much faster than re-running the simulation from scratch per checkpoint.
#
# produces: countermeasure_results.json
#
# the basis-split test uses a two-proportion z-test comparing qber_z vs qber_x.
# z-statistic > 1.96 => p < 0.05 (detectable at standard significance)
# z-statistic > 3.0  => p < 0.001 (strongly detectable)
#
# key finding: non-evasive hybrid crosses z=3.0 at ~200 sifted bits.
# evasive hybrid delays this crossing to ~1000 bits.

import json
import numpy as np

from evasive_hybrid_eve import EvasiveHybridEve
from hybrid_adaptive_eve import HybridAdaptiveEve


def basis_split_ztest(alice, bob, basis):
    # two-proportion z-test: tests whether qber_x == qber_z
    # returns the signed z-statistic, or None if either basis has no data
    alice = np.array(alice)
    bob   = np.array(bob)
    basis = np.array(basis)

    z_mask = basis == 'Z'
    x_mask = basis == 'X'
    n_z, n_x = z_mask.sum(), x_mask.sum()
    if n_z == 0 or n_x == 0:
        return None

    err_z = np.sum(alice[z_mask] != bob[z_mask])
    err_x = np.sum(alice[x_mask] != bob[x_mask])
    p_pool = (err_z + err_x) / (n_z + n_x)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_z + 1 / n_x))
    if se == 0:
        return None
    return (err_x / n_x - err_z / n_z) / se


#sweep settings
#checkpoints: sifted-bit counts at which we compute the detection statistic
CHECKPOINTS = [200, 500, 1000, 2000, 4000]
# max_rounds needs to be ~2x the largest checkpoint to account for ~50% sifting
MAX_ROUNDS  = max(CHECKPOINTS) * 2
SEEDS       = [1, 2, 3]   # increase for tighter error bars

# run simulations

evasive_runs = []
plain_runs   = []

for seed in SEEDS:
    # evasive variant: basis-symmetrization noise injection enabled
    e1 = EvasiveHybridEve(mu=0.6, F_qnd=1.0, F_mem=1.0, qber_budget=0.11,
                          seed=seed, symmetrize=True)
    e1.run_n_rounds(MAX_ROUNDS)
    evasive_runs.append(e1)

    # plain hybrid: no symmetrization -- structural qber_z=0 signature is fully visible
    e2 = HybridAdaptiveEve(mu=0.6, F_qnd=1.0, F_mem=1.0, qber_budget=0.11, seed=seed)
    e2.run_n_rounds(MAX_ROUNDS)
    plain_runs.append(e2)

    print(f"seed {seed} done  "
          f"(evasive sifted={len(e1.alice_key)}, plain sifted={len(e2.alice_key)})")

# compute z-statistic at each checkpoint by slicing accumulated data

results = {
    'checkpoints':       CHECKPOINTS,
    'evasive_zstat':     [],
    'evasive_zstat_std': [],
    'plain_zstat':       [],
    'plain_zstat_std':   [],
}

for cp in CHECKPOINTS:
    z_ev, z_pl = [], []

    for e1, e2 in zip(evasive_runs, plain_runs):
        # slice to exactly cp sifted bits
        if len(e1.alice_key) >= cp:
            z = basis_split_ztest(e1.alice_key[:cp], e1.bob_key[:cp], e1.basis_log[:cp])
            if z is not None:
                z_ev.append(abs(z))

        if len(e2.alice_key) >= cp:
            z = basis_split_ztest(e2.alice_key[:cp], e2.bob_key[:cp], e2.basis_log[:cp])
            if z is not None:
                z_pl.append(abs(z))

    results['evasive_zstat'].append(float(np.mean(z_ev))  if z_ev else None)
    results['evasive_zstat_std'].append(float(np.std(z_ev))  if z_ev else None)
    results['plain_zstat'].append(float(np.mean(z_pl))   if z_pl else None)
    results['plain_zstat_std'].append(float(np.std(z_pl))   if z_pl else None)

    print(f"checkpoint={cp:5d}:  "
          f"evasive z={np.mean(z_ev):.2f}±{np.std(z_ev):.2f}  "
          f"plain z={np.mean(z_pl):.2f}±{np.std(z_pl):.2f}")

# save

with open('countermeasure_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nsaved countermeasure_results.json")