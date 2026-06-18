"""
Experimental sweep for Hybrid Adaptive Eve.

Generates four datasets, each averaged over multiple random seeds for smooth curves:

  1. mu-sweep:      compare PNS-only, Fixed-CNOT, Hybrid-Adaptive across mean photon number mu, at ideal hardware (F_qnd=F_mem=1)
  2. F_qnd-sweep:    Hybrid-Adaptive's QBER_Z and QBER_X as imperfect QND measurement fidelity degrades (F_mem held at 1.0)
  3. F_mem-sweep:    Hybrid-Adaptive's overall Eve accuracy as quantum memory fidelity degrades (F_qnd held at 1.0)
  4. theta-efficiency: closed-form info-gain-per-unit-QBER across theta
                     (no simulation needed, this is exact from hybrid_physics.py, included for the methodology section figure)

Results are saved as JSON so the visualization step can be re-run without re-simulating.
"""

import json
import numpy as np

from hybrid_adaptive_eve import HybridAdaptiveEve
from baselines import PNSOnlyEve, FixedCNOTEve
from hybrid_physics import expected_qber, expected_eve_info

N_ROUNDS = 2000
SEEDS = [1, 2, 3]


def average_runs(eve_class, n_rounds, seeds, **kwargs):
    #Run the same configuration across multiple seeds and average the summary metrics, so sweep curves are smooth rather than noisy.
    all_summaries = []
    for seed in seeds:
        eve = eve_class(seed=seed, **kwargs)
        eve.run_n_rounds(n_rounds)
        s = eve.summary()
        if s:
            all_summaries.append(s)

    if not all_summaries:
        return {}

    averaged = {}
    keys = all_summaries[0].keys()
    for k in keys:
        vals = [s[k] for s in all_summaries if not (isinstance(s[k], float) and np.isnan(s[k]))]
        averaged[k] = float(np.mean(vals)) if vals else float('nan')
    return averaged


def run_mu_sweep():
    mu_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    results = {'mu_values': mu_values, 'pns_only': [], 'fixed_cnot': [], 'hybrid': []}

    for mu in mu_values:
        results['pns_only'].append(
            average_runs(PNSOnlyEve, N_ROUNDS, SEEDS, mu=mu, F_qnd=1.0, F_mem=1.0)
        )
        results['fixed_cnot'].append(
            average_runs(FixedCNOTEve, N_ROUNDS, SEEDS, mu=mu, F_qnd=1.0, F_mem=1.0)
        )
        results['hybrid'].append(
            average_runs(HybridAdaptiveEve, N_ROUNDS, SEEDS, mu=mu, F_qnd=1.0, F_mem=1.0, qber_budget=0.11)
        )
        print(f"mu={mu} done")

    return results


def run_fqnd_sweep():
    fqnd_values = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    results = {'fqnd_values': fqnd_values, 'hybrid': []}

    for f in fqnd_values:
        results['hybrid'].append(
            average_runs(HybridAdaptiveEve, N_ROUNDS, SEEDS, mu=0.6, F_qnd=f, F_mem=1.0, qber_budget=0.11)
        )
        print(f"F_qnd={f} done")

    return results


def run_fmem_sweep():
    fmem_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
    results = {'fmem_values': fmem_values, 'hybrid': [], 'pns_only': []}

    for f in fmem_values:
        results['hybrid'].append(
            average_runs(HybridAdaptiveEve, N_ROUNDS, SEEDS, mu=0.6, F_qnd=1.0, F_mem=f, qber_budget=0.11)
        )
        results['pns_only'].append(
            average_runs(PNSOnlyEve, N_ROUNDS, SEEDS, mu=0.6, F_qnd=1.0, F_mem=f)
        )
        print(f"F_mem={f} done")

    return results


def run_theta_efficiency():
    thetas = np.linspace(0.001, np.pi, 50)
    info_gains = [expected_eve_info(t) for t in thetas]
    qbers = [expected_qber(t) for t in thetas]
    efficiency = [g / q if q > 0 else None for g, q in zip(info_gains, qbers)]
    return {
        'thetas': thetas.tolist(),
        'info_gains': info_gains,
        'qbers': qbers,
        'efficiency': efficiency,
    }


if __name__ == "__main__":
    print("Running mu-sweep...")
    mu_results = run_mu_sweep()

    print("\nRunning F_qnd-sweep...")
    fqnd_results = run_fqnd_sweep()

    print("\nRunning F_mem-sweep...")
    fmem_results = run_fmem_sweep()

    print("\nComputing theta-efficiency curve (closed-form, instant)...")
    theta_results = run_theta_efficiency()

    all_results = {
        'mu_sweep': mu_results,
        'fqnd_sweep': fqnd_results,
        'fmem_sweep': fmem_results,
        'theta_efficiency': theta_results,
    }

    with open('sweep_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nAll results saved to sweep_results.json")

