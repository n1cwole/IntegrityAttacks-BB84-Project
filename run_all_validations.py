# runs all validation checks for the simulation model.
# expected output: every line prints PASS.
# any FAIL indicates a logic bug or a formula mismatch that must be resolved before trusting any figure produced by make_figures.py.

# checks covered:
#   1. closed-form anchor points (machine precision, no simulation needed)
#   2. simulation vs closed-form across theta grid (monte carlo tolerance)
#   3. F_mem limits: acc_pns = 0.5 + 0.5*F_mem formula
#   4. F_qnd=1 gives zero qber for pns-only
#   5. qber stays under budget across sample sizes
#   6. lambda invariance: results identical for lam in {10, 50, 200}
#   7. evasive eve containment: symmetrization doesn't leak into pns path

import numpy as np
from hybrid_physics import (qber_z, qber_x, eve_acc_z, eve_acc_x,
                             expected_qber, expected_eve_info)
from hybrid_adaptive_eve import HybridAdaptiveEve
from baselines import PNSOnlyEve
from evasive_hybrid_eve import EvasiveHybridEve


def check(name, got, expect, tol=1e-9):
    ok = abs(got - expect) < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got={got:.6f}  expect={expect}")
    return ok


# closed-form anchor points (no simulation) 

print("1. closed-form anchor points")
check("qber_z(0)",        qber_z(0),           0.0)
check("qber_z(pi)",       qber_z(np.pi),        0.0)
check("qber_z(pi/2)",     qber_z(np.pi/2),      0.0)
check("qber_x(0)",        qber_x(0),            0.0)
check("qber_x(pi)",       qber_x(np.pi),        0.5)
check("eve_acc_z(0)",     eve_acc_z(0),          0.5)
check("eve_acc_z(pi)",    eve_acc_z(np.pi),      1.0)
check("eve_acc_x(any)",   eve_acc_x(1.0),        0.5)
check("E[qber](pi)",      expected_qber(np.pi),  0.25)
check("E[info](pi)",      expected_eve_info(np.pi), 0.25)
check("E[qber](0)",       expected_qber(0),      0.0)
check("E[info](0)",       expected_eve_info(0),  0.0)

# simulation vs closed-form across theta grid 

print("\n 2. simulation vs closed-form (3000 trials each) ")

import random

TRIALS = 3000
max_err = 0.0

for theta in np.linspace(0, np.pi, 7):
    # X-basis QBER validation: Alice and Bob must BOTH use X, otherwise this is not a sifted X-basis check.
    x_errors = 0
    for _ in range(TRIALS):
        alice_bit = random.randint(0, 1)
        eve = HybridAdaptiveEve(mu=0.6, F_qnd=1.0, F_mem=1.0)
        bob_bit, _ = eve._run_probe(theta, alice_bit, 'X', 'X')
        if bob_bit != alice_bit:
            x_errors += 1
    sim_qber_x = x_errors / TRIALS

    # Z-basis Eve accuracy validation: Alice and Bob both use Z; Eve's ancilla should increasingly encode the Z bit.
    z_correct = 0
    for _ in range(TRIALS):
        alice_bit = random.randint(0, 1)
        eve = HybridAdaptiveEve(mu=0.6, F_qnd=1.0, F_mem=1.0)
        _, eve_bit = eve._run_probe(theta, alice_bit, 'Z', 'Z')
        if eve_bit == alice_bit:
            z_correct += 1
    sim_acc_z = z_correct / TRIALS

    cf_qber_x = qber_x(theta)
    cf_acc_z = eve_acc_z(theta)

    err = max(abs(sim_qber_x - cf_qber_x), abs(sim_acc_z - cf_acc_z))
    max_err = max(max_err, err)

    status = "PASS" if err < 0.03 else "FAIL"
    print(
        f"  {status}  theta={theta:.3f}  "
        f"sim_qber_x={sim_qber_x:.4f} cf={cf_qber_x:.4f}  "
        f"sim_acc_z={sim_acc_z:.4f} cf={cf_acc_z:.4f}  "
        f"err={err:.4f}"
    )

print(
    f"  max deviation: {max_err:.4f}  "
    f"({'within' if max_err < 0.03 else 'EXCEEDS'} 0.03 tolerance)"
)

# F_mem limits 

print("\n 3. F_mem limits (pns-only, mu=5.0, n=3000) ")
for fm, expect in [(1.0, 1.0), (0.5, 0.75), (0.0, 0.5)]:
    e = PNSOnlyEve(mu=5.0, F_mem=fm, seed=1)
    e.run_n_rounds(3000)
    got = e.summary()['eve_acc_pns']
    ok  = abs(got - expect) < 0.03
    print(f"  {'PASS' if ok else 'FAIL'}  F_mem={fm}: acc_pns={got:.4f}  expect~{expect}")

# F_qnd=1 gives zero qber for pns 

print("\n 4. F_qnd=1.0 -> qber=0 for pns-only ")
for mu in [0.3, 0.7, 1.1]:
    e = PNSOnlyEve(mu=mu, F_qnd=1.0, F_mem=1.0, seed=1)
    e.run_n_rounds(2000)
    q = e.summary()['qber_total']
    print(f"  {'PASS' if q == 0.0 else 'FAIL'}  mu={mu}: qber={q:.6f}  expect=0.0")

# qber stays under budget across sample sizes 

print("\n 5. qber under 11% budget for all sample sizes ")
for n in [500, 1000, 2000, 4000]:
    e = HybridAdaptiveEve(mu=0.6, qber_budget=0.11, seed=42)
    e.run_n_rounds(n)
    q = e.summary()['qber_total']
    print(f"  {'PASS' if q < 0.11 else 'FAIL'}  n={n}: qber={q:.4f}  budget=0.11")

#  lambda invariance 

print("\n 6. lambda invariance ")
qbers = []
for lam in [10, 50, 200]:
    e = HybridAdaptiveEve(mu=0.6, lam=lam, seed=42)
    e.run_n_rounds(2000)
    qbers.append(e.summary()['qber_total'])
spread = max(qbers) - min(qbers)
ok = spread < 0.002
print(f"  {'PASS' if ok else 'FAIL'}  qber spread across lam=[10,50,200]: {spread:.6f}  (expect <0.002)")

# evasive eve containment 

print("\n 7. evasive eve containment ")
e = EvasiveHybridEve(mu=0.6, qber_budget=0.11, seed=7, symmetrize=True)
e.run_n_rounds(2000)
s = e.summary()

# pns path qber must be zero. symmetrization only touches probe-path z events
pns_mask = np.array(e.path_log) == 'PNS'
if pns_mask.sum() > 0:
    pns_qber = np.mean(np.array(e.alice_key)[pns_mask] != np.array(e.bob_key)[pns_mask])
else:
    pns_qber = 0.0

print(f"  {'PASS' if pns_qber == 0.0 else 'FAIL'}  pns-path qber unaffected: {pns_qber:.6f}  expect=0.0")
print(f"  {'PASS' if s['qber_total'] < 0.11 else 'FAIL'}  pooled qber within budget: {s['qber_total']:.4f}")
print(f"  {'PASS' if s['qber_z'] > 0.01 else 'FAIL'}  z-basis gap reduced (qber_z>0): {s['qber_z']:.4f}")

print("\n all checks complete ")