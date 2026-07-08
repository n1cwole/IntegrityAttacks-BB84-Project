# generate_missing_jsons.py
#
# Creates the missing JSON files needed by the final make_figures.py.
# Run after:
#   python3 run_sweeps.py
#   python3 run_countermeasure_sweeps.py
#   python3 run_all_validations.py
#
# Outputs:
#   decomp_results.json
#   boundary_corrected.json
#   skr_results.json
#   dw_skr_results.json
#   ablation_results.json
#   lambda_sensitivity.json
#   convergence_results.json
# Also copies sweep_results.json -> sweep_results_v2.json for compatibility.

import json
import os
import math
import inspect
import shutil
import numpy as np

from hybrid_physics import expected_eve_info, expected_qber
from hybrid_adaptive_eve import HybridAdaptiveEve
from baselines import PNSOnlyEve, FixedCNOTEve

try:
    from evasive_hybrid_eve import EvasiveHybridEve
except Exception:
    EvasiveHybridEve = None

try:
    from depol_hybrid_eve import DepolHybridEve
except Exception:
    DepolHybridEve = None

try:
    from full_stack_eve import FullStackEve
except Exception:
    FullStackEve = None


MU_VALS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
MAIN_SEEDS = [0, 1, 2, 3]
ABLATION_SEEDS = list(range(7))
CONVERGENCE_SEEDS = list(range(8))
N_MAIN = 2000


def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved {filename}")


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def binary_entropy(p):
    if p is None or isinstance(p, float) and math.isnan(p):
        return float("nan")
    p = min(max(float(p), 0.0), 1.0)
    if p == 0.0 or p == 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def mean_std(values):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def metric_mean(entry, key):
    value = entry.get(key, float("nan"))
    if isinstance(value, dict):
        return float(value.get("mean", float("nan")))
    return float(value)


def metric_std(entry, key):
    value = entry.get(key, 0.0)
    if isinstance(value, dict):
        return float(value.get("std", 0.0))
    return 0.0


def summarize_runs(cls, runs_kwargs, n_rounds=N_MAIN, seeds=MAIN_SEEDS):
    summaries = []
    for seed in seeds:
        kwargs = dict(runs_kwargs)
        kwargs["seed"] = seed
        eve = flexible_init(cls, kwargs)
        eve.run_n_rounds(n_rounds)
        summaries.append(eve.summary())
    return summaries


def aggregate_summary(summaries):
    keys = [
        "n_sifted",
        "qber_total",
        "qber_z",
        "qber_x",
        "eve_acc_total",
        "eve_acc_pns",
        "eve_acc_probe",
        "coverage_pns",
        "coverage_probe",
        "mean_theta_used",
    ]
    out = {}
    for key in keys:
        vals = [s.get(key, float("nan")) for s in summaries]
        m, st = mean_std(vals)
        out[key] = {"mean": m, "std": st}
    return out


def flexible_init(cls, kwargs):
    """
    Instantiates a class using only the kwargs its __init__ accepts.
    This makes the script robust to slightly different class signatures.
    """
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return cls(**filtered)


def ensure_sweep_copy():
    if os.path.exists("sweep_results.json") and not os.path.exists("sweep_results_v2.json"):
        shutil.copyfile("sweep_results.json", "sweep_results_v2.json")
        print("copied sweep_results.json -> sweep_results_v2.json for compatibility")


def get_sweep():
    if not os.path.exists("sweep_results.json"):
        raise FileNotFoundError("Missing sweep_results.json. Run python3 run_sweeps.py first.")
    sweep = load_json("sweep_results.json")
    print("sweep_results.json top-level keys:", list(sweep.keys()))
    return sweep


def generate_decomp_results(sweep):
    """
    Figure A2.
    Prefer using sweep_results.json so decomposition matches Figures A/B.
    If the needed structure is missing, rerun the three strategies.
    """
    print("\n generating decomp_results.json ")

    if all(k in sweep for k in ["mu_vals", "pns_only", "fixed_cnot", "hybrid"]):
        mu_vals = sweep["mu_vals"]
        hybrid = sweep["hybrid"]
        fixed = sweep["fixed_cnot"]

        data = {
            "mu_vals": mu_vals,
            "pns_coverage": [metric_mean(e, "coverage_pns") for e in hybrid],
            "pns_accuracy": [metric_mean(e, "eve_acc_pns") for e in hybrid],
            "probe_coverage": [metric_mean(e, "coverage_probe") for e in hybrid],
            "probe_accuracy": [metric_mean(e, "eve_acc_probe") for e in hybrid],
            "hybrid_total": [metric_mean(e, "eve_acc_total") for e in hybrid],
            "fixed_cnot_total": [metric_mean(e, "eve_acc_total") for e in fixed],
            "pns_coverage_std": [metric_std(e, "coverage_pns") for e in hybrid],
            "probe_coverage_std": [metric_std(e, "coverage_probe") for e in hybrid],
            "pns_accuracy_std": [metric_std(e, "eve_acc_pns") for e in hybrid],
            "probe_accuracy_std": [metric_std(e, "eve_acc_probe") for e in hybrid],
            "hybrid_total_std": [metric_std(e, "eve_acc_total") for e in hybrid],
            "fixed_cnot_total_std": [metric_std(e, "eve_acc_total") for e in fixed],
        }
        save_json("decomp_results.json", data)
        return

    print("sweep_results.json did not have final structure, rerunning decomposition directly.")
    out = {
        "mu_vals": MU_VALS,
        "pns_coverage": [],
        "pns_accuracy": [],
        "probe_coverage": [],
        "probe_accuracy": [],
        "hybrid_total": [],
        "fixed_cnot_total": [],
        "pns_coverage_std": [],
        "probe_coverage_std": [],
        "pns_accuracy_std": [],
        "probe_accuracy_std": [],
        "hybrid_total_std": [],
        "fixed_cnot_total_std": [],
    }

    for mu in MU_VALS:
        print(f"  mu={mu}")
        h_sum = summarize_runs(
            HybridAdaptiveEve,
            {"mu": mu, "F_qnd": 1.0, "F_mem": 1.0, "qber_budget": 0.11},
            n_rounds=N_MAIN,
            seeds=MAIN_SEEDS,
        )
        c_sum = summarize_runs(
            FixedCNOTEve,
            {"mu": mu, "F_qnd": 1.0, "F_mem": 1.0},
            n_rounds=N_MAIN,
            seeds=MAIN_SEEDS,
        )

        def vals(summaries, key):
            return [s.get(key, float("nan")) for s in summaries]

        out["pns_coverage"].append(mean_std(vals(h_sum, "coverage_pns"))[0])
        out["pns_coverage_std"].append(mean_std(vals(h_sum, "coverage_pns"))[1])
        out["probe_coverage"].append(mean_std(vals(h_sum, "coverage_probe"))[0])
        out["probe_coverage_std"].append(mean_std(vals(h_sum, "coverage_probe"))[1])
        out["pns_accuracy"].append(mean_std(vals(h_sum, "eve_acc_pns"))[0])
        out["pns_accuracy_std"].append(mean_std(vals(h_sum, "eve_acc_pns"))[1])
        out["probe_accuracy"].append(mean_std(vals(h_sum, "eve_acc_probe"))[0])
        out["probe_accuracy_std"].append(mean_std(vals(h_sum, "eve_acc_probe"))[1])
        out["hybrid_total"].append(mean_std(vals(h_sum, "eve_acc_total"))[0])
        out["hybrid_total_std"].append(mean_std(vals(h_sum, "eve_acc_total"))[1])
        out["fixed_cnot_total"].append(mean_std(vals(c_sum, "eve_acc_total"))[0])
        out["fixed_cnot_total_std"].append(mean_std(vals(c_sum, "eve_acc_total"))[1])

    save_json("decomp_results.json", out)


def estimate_operational_theta():
    """
    Estimate representative theta from actual HybridAdaptiveEve runs at mu=0.6.
    """
    vals = []
    for seed in range(8):
        eve = HybridAdaptiveEve(mu=0.6, F_qnd=1.0, F_mem=1.0, qber_budget=0.11, seed=seed)
        eve.run_n_rounds(N_MAIN)
        s = eve.summary()
        theta = s.get("mean_theta_used", float("nan"))
        if not math.isnan(theta):
            vals.append(theta)
    m, st = mean_std(vals)
    print(f"estimated operational theta at mu=0.6: mean={m:.4f}, std={st:.4f}")
    return m, st, vals


def p_multi(mu):
    return 1.0 - math.exp(-mu) * (1.0 + mu)


def generate_boundary_corrected():
    """
    Figure I.
    Corrected closed-form grid using Acc_total = P_multi*Acc_PNS + (1-P_multi)*Acc_probe
    Net = Acc_total - 0.5
    """
    print("\n generating boundary_corrected.json ")
    theta_op, theta_std, theta_samples = estimate_operational_theta()

    mu_vals = np.linspace(0.1, 2.0, 50)
    fmem_vals = np.linspace(0.0, 1.0, 50)

    grid = np.zeros((len(mu_vals), len(fmem_vals)))
    for i, mu in enumerate(mu_vals):
        pm = p_multi(float(mu))
        pp = 1.0 - pm
        for j, fm in enumerate(fmem_vals):
            acc_pns = 0.5 + 0.5 * float(fm)
            acc_probe = 0.5 + expected_eve_info(theta_op)
            acc_total = pm * acc_pns + pp * acc_probe
            grid[i, j] = acc_total - 0.5

    # Bootstrap by resampling theta values from actual runs.
    boot_grids = []
    rng = np.random.default_rng(123)
    if len(theta_samples) > 1:
        for _ in range(200):
            theta_b = float(rng.choice(theta_samples))
            g = np.zeros_like(grid)
            for i, mu in enumerate(mu_vals):
                pm = p_multi(float(mu))
                pp = 1.0 - pm
                for j, fm in enumerate(fmem_vals):
                    acc_pns = 0.5 + 0.5 * float(fm)
                    acc_probe = 0.5 + expected_eve_info(theta_b)
                    g[i, j] = pm * acc_pns + pp * acc_probe - 0.5
            boot_grids.append(g)
        grid_std = np.std(np.stack(boot_grids), axis=0)
    else:
        grid_std = np.zeros_like(grid)

    data = {
        "mu_vals": mu_vals.tolist(),
        "fmem_vals": fmem_vals.tolist(),
        "grid_net": grid.tolist(),
        "grid_std": grid_std.tolist(),
        "note": (
            f"Corrected closed-form grid. theta_op estimated from actual "
            f"HybridAdaptiveEve runs at mu=0.6: mean={theta_op:.6f}, std={theta_std:.6f}. "
            "Net advantage = P_multi*(0.5+0.5F_mem) + (1-P_multi)*(0.5+E_info(theta)) - 0.5."
        ),
    }
    save_json("boundary_corrected.json", data)


def get_strategy_arrays_from_sweep(sweep):
    """
    Supports both sweep formats.

    New/flat format:
      {
        "mu_vals": [...],
        "pns_only": [...],
        "fixed_cnot": [...],
        "hybrid": [...]
      }

    Old/nested format:
      {
        "mu_sweep": {
          "mu_values": [...],
          "pns_only": [...],
          "fixed_cnot": [...],
          "hybrid": [...]
        },
        ...
      }
    """
    if all(k in sweep for k in ["mu_vals", "pns_only", "fixed_cnot", "hybrid"]):
        mu_vals = sweep["mu_vals"]
        strategies = {
            "pns": sweep["pns_only"],
            "cnot": sweep["fixed_cnot"],
            "hybrid": sweep["hybrid"],
        }
        return mu_vals, strategies

    if "mu_sweep" in sweep:
        ms = sweep["mu_sweep"]

        mu_vals = ms.get("mu_vals", ms.get("mu_values"))

        strategies = {
            "pns": ms["pns_only"],
            "cnot": ms["fixed_cnot"],
            "hybrid": ms["hybrid"],
        }
        return mu_vals, strategies

    raise ValueError(
        "sweep_results.json structure not recognized. "
        f"Top-level keys are: {list(sweep.keys())}"
    )


def generate_skr_and_dw(sweep):
    print("\n generating skr_results.json and dw_skr_results.json ")
    mu_vals, strategies = get_strategy_arrays_from_sweep(sweep)

    skr_out = {"mu_vals": mu_vals}
    dw_out = {"mu_vals": mu_vals}

    for name, rows in strategies.items():
        skr_out[name] = {"skr": [], "iae": [], "qber": []}
        dw_out[name] = []

        for row in rows:
            qber = metric_mean(row, "qber_total")
            acc = metric_mean(row, "eve_acc_total")

            sp = max(0.0, 1.0 - 2.0 * binary_entropy(qber))
            iae = max(0.0, 1.0 - binary_entropy(1.0 - acc))
            iab = max(0.0, 1.0 - binary_entropy(qber))
            dw_rate = iab - iae

            skr_out[name]["skr"].append(sp)
            skr_out[name]["iae"].append(iae)
            skr_out[name]["qber"].append(qber)
            dw_out[name].append(dw_rate)

    save_json("skr_results.json", skr_out)
    save_json("dw_skr_results.json", dw_out)


def run_variant(label, cls, kwargs, n_rounds=2000, seeds=ABLATION_SEEDS):
    qbers, accs, thetas = [], [], []
    for seed in seeds:
        run_kwargs = dict(kwargs)
        run_kwargs["seed"] = seed
        eve = flexible_init(cls, run_kwargs)
        eve.run_n_rounds(n_rounds)
        s = eve.summary()
        qbers.append(s.get("qber_total", float("nan")))
        accs.append(s.get("eve_acc_total", float("nan")))
        thetas.append(s.get("mean_theta_used", float("nan")))
    qm, qs = mean_std(qbers)
    am, astd = mean_std(accs)
    tm, _ = mean_std(thetas)
    return {
        "label": label,
        "qber_mean": qm,
        "qber_std": qs,
        "acc_mean": am,
        "acc_std": astd,
        "theta": tm,
    }


def generate_ablation():
    print("\n generating ablation_results.json ")

    # Choose best available full-stack class.
    # If FullStackEve exists, use it. Otherwise use DepolHybridEve or EvasiveHybridEve.
    if FullStackEve is not None:
        full_cls = FullStackEve
    elif DepolHybridEve is not None:
        full_cls = DepolHybridEve
    elif EvasiveHybridEve is not None:
        full_cls = EvasiveHybridEve
    else:
        full_cls = HybridAdaptiveEve

    base = {
        "mu": 0.6,
        "F_qnd": 0.85,
        "F_mem": 0.85,
        "qber_budget": 0.11,
        "lam": 50.0,
        "L": 75,
        "fiber_length": 75,
        "symmetrize": True,
    }

    rows = []

    rows.append(run_variant("Full-stack (all components)", full_cls, base))

    no_depol = dict(base)
    no_depol["L"] = 0
    no_depol["fiber_length"] = 0
    rows.append(run_variant("Remove depolarization (L=0)", full_cls, no_depol))

    no_mem = dict(base)
    no_mem["F_mem"] = 1.0
    rows.append(run_variant("Remove memory loss (F_mem=1)", full_cls, no_mem))

    no_qnd = dict(base)
    no_qnd["F_qnd"] = 1.0
    rows.append(run_variant("Remove QND noise (F_qnd=1)", full_cls, no_qnd))

    if EvasiveHybridEve is not None or FullStackEve is not None:
        no_sym = dict(base)
        no_sym["symmetrize"] = False
        rows.append(run_variant("Remove symmetrization", full_cls, no_sym))
    else:
        rows.append(run_variant("Remove symmetrization", HybridAdaptiveEve, {
            "mu": 0.6,
            "F_qnd": 0.85,
            "F_mem": 0.85,
            "qber_budget": 0.11,
            "lam": 50.0,
        }))

    rows.append(run_variant("Remove optimizer (Fixed-CNOT)", FixedCNOTEve, {
        "mu": 0.6,
        "F_qnd": 0.85,
        "F_mem": 0.85,
    }))

    rows.append(run_variant("Remove probe (PNS-only)", PNSOnlyEve, {
        "mu": 0.6,
        "F_qnd": 0.85,
        "F_mem": 0.85,
    }))

    save_json("ablation_results.json", rows)


def generate_lambda_sensitivity():
    print("\n generating lambda_sensitivity.json ")
    lam_values = [10, 20, 50, 100, 200]
    out = {
        "lam_values": lam_values,
        "qber_mean": [],
        "qber_std": [],
        "acc_mean": [],
        "acc_std": [],
        "theta_mean": [],
        "theta_std": [],
    }

    for lam in lam_values:
        print(f"  lambda={lam}")
        summaries = summarize_runs(
            HybridAdaptiveEve,
            {"mu": 0.6, "F_qnd": 1.0, "F_mem": 1.0, "qber_budget": 0.11, "lam": lam},
            n_rounds=2000,
            seeds=list(range(7)),
        )
        qbers = [s.get("qber_total", float("nan")) for s in summaries]
        accs = [s.get("eve_acc_total", float("nan")) for s in summaries]
        thetas = [s.get("mean_theta_used", float("nan")) for s in summaries]

        out["qber_mean"].append(mean_std(qbers)[0])
        out["qber_std"].append(mean_std(qbers)[1])
        out["acc_mean"].append(mean_std(accs)[0])
        out["acc_std"].append(mean_std(accs)[1])
        out["theta_mean"].append(mean_std(thetas)[0])
        out["theta_std"].append(mean_std(thetas)[1])

    save_json("lambda_sensitivity.json", out)


def generate_convergence():
    print("\n generating convergence_results.json ")
    checkpoints = [500, 1000, 2000, 4000]
    raw = {cp: {"qber": [], "acc": [], "qber_x": [], "qber_z": [], "theta": []} for cp in checkpoints}

    for seed in CONVERGENCE_SEEDS:
        print(f"  seed={seed}")
        eve = HybridAdaptiveEve(mu=0.6, F_qnd=1.0, F_mem=1.0, qber_budget=0.11, lam=50.0, seed=seed)
        prev = 0
        for cp in checkpoints:
            eve.run_n_rounds(cp - prev)
            prev = cp
            s = eve.summary()
            raw[cp]["qber"].append(s.get("qber_total", float("nan")))
            raw[cp]["acc"].append(s.get("eve_acc_total", float("nan")))
            raw[cp]["qber_x"].append(s.get("qber_x", float("nan")))
            raw[cp]["qber_z"].append(s.get("qber_z", float("nan")))
            raw[cp]["theta"].append(s.get("mean_theta_used", float("nan")))

    summary = {}
    for cp in checkpoints:
        summary[str(cp)] = {}
        for metric in ["qber", "acc", "qber_x", "qber_z", "theta"]:
            m, st = mean_std(raw[cp][metric])
            summary[str(cp)][metric] = {"mean": m, "std": st}

    save_json("convergence_results.json", {"checkpoints": checkpoints, "summary": summary})


def print_checklist():
    needed = [
        "sweep_results.json",
        "decomp_results.json",
        "fqnd_isolated.json",
        "depol_results.json",
        "boundary_corrected.json",
        "dw_skr_results.json",
        "skr_results.json",
        "countermeasure_results.json",
        "ablation_results.json",
        "lambda_sensitivity.json",
        "convergence_results.json",
    ]
    print("\n JSON availability checklist ")
    for f in needed:
        print(f"{f:32s} {'OK' if os.path.exists(f) else 'MISSING'}")


def main():
    print(" generating missing final-paper JSONs ")
    ensure_sweep_copy()
    sweep = get_sweep()

    generate_decomp_results(sweep)
    generate_boundary_corrected()
    generate_skr_and_dw(sweep)
    generate_ablation()
    generate_lambda_sensitivity()
    generate_convergence()

    ensure_sweep_copy()
    print_checklist()
    print("\nDone. Now run: python3 make_figures.py")


if __name__ == "__main__":
    main()