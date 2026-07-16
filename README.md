## Project title: "Quantum Eavesdropper Detection: Evaluating the Threat Landscape of BB84 Attacks"
---
> # 📄[**Read the Full Research Paper (PDF)**](https://drive.google.com/file/d/1-41F5pUfaZcrjvg3Cv4dygVaavt6pIsn/view?usp=drive_link)
---

### Project Overview:
This project provides a comprehensive simulation and analysis of the BB84 Measurement-Based Quantum Key Distribution (QKD) protocol's vulnerability to various measurement-based eavesdropping strategies. By simulating different individual attacks, we quantify the Quantum Bit Error Rate (QBER) as well as other metrics to determine which methods most effectively compromise key security without detection.

The simulation evaluates the susceptibility of the BB84 protocol against four distinct individual measurement attacks:

Intercept-Resend: The most basic attack where Eve measures the qubits and sends a new state to Bob.

Photon Number Splitting (PNS): Exploiting multi-photon pulses to gain information without disturbing the signal.

CNOT Probe: Using an auxiliary qubit and a CNOT gate to entangle Eve's system with the quantum channel.

CZ Probe: Utilizing Controlled-Z gates to induce phase-shift errors for information extraction.

---

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/n1cwole/IntegrityAttacks-BB84-Project.git
cd IntegrityAttacks-BB84-Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### Project Structure

```
├── original_simulations/           # Reference implementations from the research paper
│   ├── CNOT_Probe_Attack.py        # CNOT/CZ probe attacks + Intercept-Resend
│   └── PNS_Attacks.py              # Photon Number Splitting attack
│
├── hybrid_physics.py               # Closed-form CRy(θ) physics model (analytically derived)
├── hybrid_adaptive_eve.py          # Hybrid adaptive Eve: PNS + CRy(θ) optimization
├── baselines.py                    # PNS-only and fixed-CNOT baseline Eve models
├── evasive_hybrid_eve.py           # Evasive Eve: basis-symmetrized noise injection
├── depolarization.py               # Fiber depolarization model: P(L) = 1 - exp(-γL)
├── depol_hybrid_eve.py             # Hybrid Eve extended with fiber depolarization
├── full_stack_eve.py               # Complete model: adaptive θ + evasion + depolarization
│
├── run_sweeps.py                   # Main sweep runner (produces sweep_results.json etc.)
├── run_countermeasure_sweeps.py    # Basis-split detection monitor sweep (figure G)
├── run_all_validations.py          # Validation suite — all checks should print PASS
├── generate_missing_jsons.py       # Generates remaining JSON files needed by make_figures.py
├── make_figures.py                 # Generates all paper figures (figures A–M)
│
├── figures/                        # Generated paper figures (PNG)
├── requirements.txt
└── README.md
```

---

### Running the Simulations

**Step 1 — Validate the model (optional but recommended):**
```bash
python3 run_all_validations.py
```
Every line should print `PASS`. Any `FAIL` indicates a logic bug that must be resolved before trusting any figures.

**Step 2 — Run the main parameter sweeps:**
```bash
python3 run_sweeps.py
```
Produces: `sweep_results.json`, `depol_results.json`, `boundary_results.json`, `fqnd_isolated.json`

Runtime: ~5–8 minutes depending on hardware.

**Step 3 — Run the countermeasure sweep:**
```bash
python3 run_countermeasure_sweeps.py
```
Produces: `countermeasure_results.json`

**Step 4 — Generate remaining JSON files:**
```bash
python3 generate_missing_jsons.py
```
Produces: `decomp_results.json`, `boundary_corrected.json`, `skr_results.json`, `dw_skr_results.json`, `ablation_results.json`, `lambda_sensitivity.json`, `convergence_results.json`

**Step 5 — Generate all paper figures:**
```bash
python3 make_figures.py
```
Outputs figures A–M to `figures/`.

**Original simulations (reference only):**
```bash
cd original_simulations
python3 CNOT_Probe_Attack.py
python3 PNS_Attacks.py
```

---

### Tech Stack

- **Language:** Python 3.11
- **Quantum simulation:** Cirq
- **Numerical computing:** NumPy, SciPy
- **Visualization:** Matplotlib
- **Documentation:** LaTeX

---

### Key Findings:
PNS Attack (Most Critical): Exploited multi-photon pulses to extract key bits with 0% QBER disturbance. This confirms that hardware-level vulnerabilities are the most significant threat to "unconditional" security.

CNOT/CZ Probes (Detectable): Entanglement-based attacks produced a QBER of approximately 25%. This validates BB84's dual-basis design, which transforms cloning attempts into measurable disturbances.

Intercept-Resend (Robust): Reaffirmed the strength of basis randomization, as simple measurement strategies introduced clear, detectable error rates.

### Conclusion:
The research highlights that BB84's security is implementation-dependent. While mathematics protects the protocol against measurement attacks, physical countermeasures (like decoy states) are required to defend against hardware exploits.

### Team Members: Nicole Igbozulike, Luke Fu, Harshitha Poludas, Naeem Hagans
