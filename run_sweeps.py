# generates all simulation data needed for the main paper figures.
# results are saved as json so make_figures.py can be re-run without re-running the (slow) simulations.
#
# what this produces:
#   sweep_results.json       mu-sweep, fqnd-sweep, fmem-sweep, theta-efficiency
#   depol_results.json       distance sweep (fiber depolarization)
#   boundary_results.json    net advantage landscape over (mu, F_mem)
#   fqnd_isolated.json       isolated fqnd effect using pns-only (no probe contamination)
#
# typical runtime: ~5-8 minutes total depending on hardware.
# increase N_ROUNDS and SEEDS for publication-quality error bars.

import json
import numpy as np

from hybrid_adaptive_eve import HybridAdaptiveEve
from baselines import PNSOnlyEve, FixedCNOTEve
from depol_hybrid_eve import DepolHybridEve
from hybrid_physics import expected_qber, expected_eve_info
from scipy.optimize import minimize_scalar

#run params
N_ROUNDS = 2000   # rounds per seed per configuration
SEEDS    = [1, 2, 3, 4]  # random seeds to average over


#helpers
def average_runs(eve_class, seeds, n_rounds=N_ROUNDS, **kwargs):
    # run the same configuration across multiple seeds and average the summary metrics.
    # nan values from empty paths (e.g. pns with no multi-photon pulses) are excluded.
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
    for k in all_summaries[0].keys():
        vals = [s[k] for s in all_summaries
                if not (isinstance(s[k], float) and np.isnan(s[k]))]
        averaged[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))} if vals else {'mean': float('nan'), 'std': float('nan')}
    return averaged


#sweep funcs
def run_mu_sweep():
    # compare pns-only, fixed-cnot, and hybrid-adaptive across mu at ideal hardware
    mu_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    results = {'mu_values': mu_values, 'pns_only': [], 'fixed_cnot': [], 'hybrid': []}
    for mu in mu_values:
        results['pns_only'].append(average_runs(PNSOnlyEve,         SEEDS, mu=mu, F_qnd=1.0, F_mem=1.0))
        results['fixed_cnot'].append(average_runs(FixedCNOTEve,     SEEDS, mu=mu, F_qnd=1.0, F_mem=1.0))
        results['hybrid'].append(average_runs(HybridAdaptiveEve,    SEEDS, mu=mu, F_qnd=1.0, F_mem=1.0, qber_budget=0.11))
        print(f"  mu={mu} done")
    return results


def run_fqnd_sweep():
    # hybrid-adaptive as qnd fidelity degrades (F_mem held at 1.0)
    fqnd_values = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    results = {'fqnd_values': fqnd_values, 'hybrid': []}
    for f in fqnd_values:
        results['hybrid'].append(average_runs(HybridAdaptiveEve, SEEDS, mu=0.6, F_qnd=f, F_mem=1.0, qber_budget=0.11))
        print(f"  F_qnd={f} done")
    return results


def run_fmem_sweep():
    # hybrid-adaptive and pns-only as memory fidelity degrades (F_qnd held at 1.0)
    fmem_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
    results = {'fmem_values': fmem_values, 'hybrid': [], 'pns_only': []}
    for f in fmem_values:
        results['hybrid'].append(average_runs(HybridAdaptiveEve, SEEDS, mu=0.6, F_qnd=1.0, F_mem=f, qber_budget=0.11))
        results['pns_only'].append(average_runs(PNSOnlyEve,       SEEDS, mu=0.6, F_qnd=1.0, F_mem=f))
        print(f"  F_mem={f} done")
    return results


def run_theta_efficiency():
    # closed-form only -- no simulation needed
    # eta(theta) = E[info] / E[qber], proves fixed cnot is least efficient
    thetas     = np.linspace(0.001, np.pi, 50)
    info_gains = [expected_eve_info(t) for t in thetas]
    qbers      = [expected_qber(t)     for t in thetas]
    efficiency = [g / q if q > 0 else None for g, q in zip(info_gains, qbers)]
    return {'thetas': thetas.tolist(), 'info_gains': info_gains,
            'qbers': qbers, 'efficiency': efficiency}


def run_depol_sweep():
    # hybrid-adaptive performance as fiber length increases (mu=0.6, ideal hardware)
    L_values = [0, 25, 50, 75, 100, 125, 150, 175, 200]
    results  = {'L_values': L_values, 'qber_total': [], 'qber_total_std': [],
                'qber_x': [], 'eve_acc_total': [], 'mean_theta': []}
    for L in L_values:
        qbers, qbersx, accs, thetas = [], [], [], []
        for seed in SEEDS:
            e = DepolHybridEve(mu=0.6, F_qnd=1.0, F_mem=1.0, qber_budget=0.11, seed=seed, L_km=L)
            e.run_n_rounds(N_ROUNDS)
            s = e.summary()
            qbers.append(s['qber_total'])
            qbersx.append(s['qber_x'])
            accs.append(s['eve_acc_total'])
            thetas.append(s['mean_theta_used'])
        results['qber_total'].append(float(np.mean(qbers)))
        results['qber_total_std'].append(float(np.std(qbers)))
        results['qber_x'].append(float(np.mean(qbersx)))
        results['eve_acc_total'].append(float(np.mean(accs)))
        results['mean_theta'].append(float(np.nanmean(thetas)))
        print(f"  L={L}km done")
    return results


def run_fqnd_isolated():
    # isolate fqnd effect with pns-only (no probe), high mu so all pulses are pns
    # x-basis should degrade ~2x faster than z-basis (validates asymmetric model)
    fqnd_values = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
    results = {'fqnd_values': fqnd_values, 'qber_z': [], 'qber_x': [], 'qber_total': []}
    for f_qnd in fqnd_values:
        qz, qx, qt = [], [], []
        for seed in SEEDS:
            eve = PNSOnlyEve(mu=3.0, F_qnd=f_qnd, F_mem=1.0, seed=seed)
            eve.run_n_rounds(4000)
            s = eve.summary()
            qz.append(s['qber_z'])
            qx.append(s['qber_x'])
            qt.append(s['qber_total'])
        results['qber_z'].append(float(np.mean(qz)))
        results['qber_x'].append(float(np.mean(qx)))
        results['qber_total'].append(float(np.mean(qt)))
        print(f"  F_qnd={f_qnd} isolated done")
    return results


def run_boundary():
    # net advantage landscape over (mu, F_mem) using verified closed-form model
    # uses operational theta at headroom=0.05 as the representative probe point
    def theta_at_headroom(h, lam=50):
        def cost(t):
            excess = max(0.0, expected_qber(t) - h)
            return -(expected_eve_info(t) - lam * excess)
        return minimize_scalar(cost, bounds=(0, np.pi), method='bounded').x

    def p_multi(mu):
        return 1 - np.exp(-mu) - mu * np.exp(-mu)

    theta_op = theta_at_headroom(0.05)
    info_op  = expected_eve_info(theta_op)

    mu_vals   = np.linspace(0.05, 2.0, 40)
    fmem_vals = np.linspace(0.0,  1.0, 40)
    grid      = np.zeros((len(mu_vals), len(fmem_vals)))

    for i, mu in enumerate(mu_vals):
        cp = p_multi(mu); cs = 1 - cp
        for j, fm in enumerate(fmem_vals):
            acc_pns   = 0.5 + 0.5 * fm
            acc_probe = 0.5 + info_op
            grid[i, j] = cp * (acc_pns - 0.5) + cs * (acc_probe - 0.5)

    return {
        'mu_vals': mu_vals.tolist(), 'fmem_vals': fmem_vals.tolist(),
        'grid_net': grid.tolist(), 'theta_op': float(theta_op), 'info_op': float(info_op),
        'note': 'net advantage = acc_eve - 0.5, computed from verified closed-form model'
    }


#main
if __name__ == "__main__":
    print("running mu-sweep (figures A, B, C, F)...")
    mu_results = run_mu_sweep()

    print("\nrunning F_qnd sweep (figure D right panel)...")
    fqnd_results = run_fqnd_sweep()

    print("\nrunning F_qnd isolated (figure D left panel)...")
    fqnd_iso_results = run_fqnd_isolated()

    print("\nrunning F_mem sweep (figure E)...")
    fmem_results = run_fmem_sweep()

    print("\ncomputing theta-efficiency curve (figure C, closed-form, instant)...")
    theta_results = run_theta_efficiency()

    print("\nrunning depolarization sweep (figure H)...")
    depol_results = run_depol_sweep()

    print("\ncomputing net advantage boundary (figures I, I2, closed-form, instant)...")
    boundary_results = run_boundary()

    # save all results
    with open('sweep_results.json', 'w') as f:
        json.dump({'mu_sweep': mu_results, 'fqnd_sweep': fqnd_results,
                   'fmem_sweep': fmem_results, 'theta_efficiency': theta_results}, f, indent=2)

    with open('depol_results.json',   'w') as f: json.dump(depol_results,   f, indent=2)
    with open('boundary_results.json','w') as f: json.dump(boundary_results, f, indent=2)
    with open('fqnd_isolated.json',   'w') as f: json.dump(fqnd_iso_results, f, indent=2)

    print("\nall results saved:")
    print("  sweep_results.json")
    print("  depol_results.json")
    print("  boundary_results.json")
    print("  fqnd_isolated.json")