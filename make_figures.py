import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

with open('sweep_results.json') as f:
    data = json.load(f)

OUT = os.path.join(os.getcwd(), "outputs", "figures")
os.makedirs(OUT, exist_ok=True)


def safe(vals):
    return [v if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan for v in vals]


# FIGURE A: Eve Accuracy vs μ
mu_values = data['mu_sweep']['mu_values']

pns_acc = safe([s['eve_acc_total'] for s in data['mu_sweep']['pns_only']])
cnot_acc = safe([s['eve_acc_total'] for s in data['mu_sweep']['fixed_cnot']])
hybrid_acc = safe([s['eve_acc_total'] for s in data['mu_sweep']['hybrid']])

plt.figure(figsize=(7, 5))
plt.plot(mu_values, pns_acc, marker='o', label='PNS-only')
plt.plot(mu_values, cnot_acc, marker='s', label='Fixed-CNOT')
plt.plot(mu_values, hybrid_acc, marker='^', label='Hybrid-Adaptive')
plt.axhline(0.5, linestyle=':', color='gray', label='Random baseline')

plt.xlabel('Mean Photon Number (μ)')
plt.ylabel('Eve Accuracy')
plt.title('Eve Information Gain vs μ')
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(f'{OUT}/figA_accuracy.png', dpi=150)
plt.close()


# FIGURE B: QBER vs μ
pns_qber = safe([s['qber_total'] for s in data['mu_sweep']['pns_only']])
cnot_qber = safe([s['qber_total'] for s in data['mu_sweep']['fixed_cnot']])
hybrid_qber = safe([s['qber_total'] for s in data['mu_sweep']['hybrid']])

plt.figure(figsize=(7, 5))
plt.plot(mu_values, pns_qber, marker='o', label='PNS-only')
plt.plot(mu_values, cnot_qber, marker='s', label='Fixed-CNOT')
plt.plot(mu_values, hybrid_qber, marker='^', label='Hybrid-Adaptive')
plt.axhline(0.11, linestyle='--', color='black', label='11% threshold')

plt.xlabel('Mean Photon Number (μ)')
plt.ylabel('QBER')
plt.title('Detectability vs Attack Strategy')
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(f'{OUT}/figB_qber.png', dpi=150)
plt.close()


# FIGURE C: theta efficiency
theta = data['theta_efficiency']['thetas']
info = data['theta_efficiency']['info_gains']
qber = data['theta_efficiency']['qbers']
eff = data['theta_efficiency']['efficiency']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(theta, info, label='Info Gain')
plt.plot(theta, qber, label='QBER')
plt.axvline(np.pi, linestyle=':', color='gray')
plt.title('Info vs QBER')
plt.legend()
plt.grid(alpha=0.4)

plt.subplot(1, 2, 2)
plt.plot(theta, eff, color='purple')
plt.axvline(np.pi, linestyle=':', color='gray')
plt.title('Efficiency = Info/QBER')
plt.grid(alpha=0.4)

plt.tight_layout()
plt.savefig(f'{OUT}/figC_theta.png', dpi=150)
plt.close()


# FIGURE D: F_qnd sweep

fqnd_values = data['fqnd_sweep']['fqnd_values']
hybrid = data['fqnd_sweep']['hybrid']

qber_x_fqnd = [s['qber_x'] for s in hybrid]

# theoretical constraint (QBER_Z = 0 always)
qber_z_fqnd = np.zeros(len(fqnd_values))

theta_fqnd = [
    s['mean_theta_used'] if s['mean_theta_used'] is not None else np.nan
    for s in hybrid
]
theta_fqnd = np.array(theta_fqnd, dtype=float)

# isolated baseline (deterministic model assumption)
iso_qber_z = np.zeros(len(fqnd_values))
iso_qber_x = np.full(len(fqnd_values), 0.1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# left
ax1.plot(fqnd_values, iso_qber_z,
         marker='o', color='royalblue', label='QBER_Z')

ax1.plot(fqnd_values, iso_qber_x,
         marker='s', color='salmon', label='QBER_X')

ax1.invert_xaxis()
ax1.set_xlabel('QND Measurement Fidelity ($F_{qnd}$)')
ax1.set_ylabel('QBER')
ax1.set_title(
    'Isolated Effect (No Adaptive Control)\n'
    'Intrinsic basis asymmetry of channel model'
)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# right
ax2.plot(fqnd_values, qber_x_fqnd,
         marker='s', color='salmon', label='QBER_X')

ax2.plot(fqnd_values, qber_z_fqnd,
         marker='o', color='royalblue', label='QBER_Z (theory = 0)')

ax2.set_xlabel('QND Measurement Fidelity ($F_{qnd}$)')
ax2.set_ylabel('QBER')
ax2.invert_xaxis()

# θ is a separate control output 
ax2b = ax2.twinx()

ax2b.plot(fqnd_values, theta_fqnd,
          marker='^', color='purple', linestyle='--',
          label='Mean θ (adaptive policy)')

ax2b.set_ylabel('Probe Angle θ', color='purple')

# explicit decoupling annotation
ax2b.text(
    0.55, 1.395,
    "θ = optimizer response\n(not a QBER-dependent metric)",
    transform=ax2b.transAxes,
    color='purple',
    fontsize=8
)

ax2.set_title(
    'Hybrid Adaptive Eve\n'
    'QBER is channel physics; θ is constrained optimization output'
)

# combined legend fix
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

ax2.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(f'{OUT}/figD_fqnd_basis_asymmetry.png', dpi=150)
plt.close()


# FIGURE E: F_mem sweep
fmem_values = data['fmem_sweep']['fmem_values']
eve_pns = safe([s['eve_acc_pns'] for s in data['fmem_sweep']['hybrid']])

plt.figure(figsize=(7, 5))
plt.plot(fmem_values, eve_pns, marker='o')
plt.axhline(0.5, linestyle=':', color='gray')

plt.gca().invert_xaxis()
plt.xlabel('F_mem')
plt.ylabel('Eve PNS Accuracy')
plt.title('Memory Decay Effect on Stored Photon Readout')
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(f'{OUT}/figE_fmem.png', dpi=150)
plt.close()


# FIGURE F: coverage breakdown
coverage_pns = safe([s['coverage_pns'] for s in data['mu_sweep']['hybrid']])
coverage_probe = safe([s['coverage_probe'] for s in data['mu_sweep']['hybrid']])

x = np.arange(len(mu_values))

plt.figure(figsize=(7, 5))
plt.bar(x, coverage_pns, label='PNS')
plt.bar(x, coverage_probe, bottom=coverage_pns, label='Probe')

plt.xticks(x, mu_values)
plt.xlabel('μ')
plt.ylabel('Coverage')
plt.title('Hybrid Attack Coverage Split')
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(f'{OUT}/figF_coverage.png', dpi=150)
plt.close()


print("All figures saved to:", OUT)
print("theta range:", np.nanmin(theta_fqnd), np.nanmax(theta_fqnd))