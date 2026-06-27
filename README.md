## Project title: "Quantum Eavesdropper Detection: Evaluating the Threat Landscape of BB84 Attacks"
---
> # 📄[**Read the Full Research Paper (PDF)**](https://drive.google.com/file/d/1-41F5pUfaZcrjvg3Cv4dygVaavt6pIsn/view?usp=drive_link)
---

### Project Overview:
This project provides a comprehensive simulation and analysis of the BB84 Measurment-Based Quantum Key Distribution (QKD) protocol's vulnerability to various measurement-based eavesdropping strategies. By simulating different individual attacks, we quantify the Quantum Bit Error Rate (QBER) as well as other metrics to determine which methods most effectively compromise key security without detection.

The simulation evaluates the susceptibility of the BB84 protocol against four distinct individual measurement attacks:

Intercept-Resend: The most basic attack where Eve measures the qubits and sends a new state to Bob.

Photon Number Splitting (PNS): Exploiting multi-photon pulses to gain information without disturbing the signal.

CNOT Probe: Using an auxiliary qubit and a CNOT gate to entangle Eve's system with the quantum channel.

CZ Probe: Utilizing Controlled-Z gates to induce phase-shift errors for information extraction.

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

### Project Structure

```
├── original_simulations/       # Reference implementations used in the research paper
│   ├── CNOT_Probe_Attack.py    # CNOT/CZ probe attacks + Intercept-Resend
│   └── PNS_Attacks.py          # Photon Number Splitting attack
├── src/                        # Modified hybrid attack models (active development)
│   ├── hybrid_physics.py       # Closed-form CRy physics model
│   ├── hybrid_adaptive_eve.py  # Hybrid adaptive Eve (PNS + CRy optimization)
│   ├── baselines.py            # PNS-only and fixed-CNOT baseline Eve models
│   ├── evasive_hybrid_eve.py   # Evasive Eve with basis-symmetrized noise injection
│   ├── run_sweeps.py           # Parameter sweep framework
│   └── make_figures.py         # Paper figure generation
├── requirements.txt
└── README.md
```

### Running the Simulations

**Original simulations (reference):**
```bash
cd original_simulations
python3 CNOT_Probe_Attack.py    # CNOT and CZ probe attacks + Intercept-Resend
python3 PNS_Attacks.py          # Photon Number Splitting attack
```

**Hybrid/adaptive Eve models:**
```bash
cd src
python3 hybrid_adaptive_eve.py  # Hybrid adaptive attack with CRy optimization
python3 evasive_hybrid_eve.py   # Evasive Eve with basis-symmetrized noise injection
```

**Parameter sweeps and figure generation:**
```bash
cd src
python3 run_sweeps.py           # Run experimental sweeps (outputs sweep_results.json)
python3 make_figures.py         # Generate paper figures (outputs to outputs/figures/)
```

### Tech Stack

- **Language:** Python 3.11
- **Quantum simulation:** Cirq
- **Numerical computing:** NumPy, SciPy
- **Visualization:** Matplotlib
- **Documentation:** LaTeX

### Key Findings:
PNS Attack (Most Critical): Exploited multi-photon pulses to extract key bits with $0\%$ QBER disturbance. This confirms that hardware-level vulnerabilities are the most significant threat to "unconditional" security.

CNOT/CZ Probes (Detectable): Entanglement-based attacks produced a QBER of approximately 25%. This validates BB84's dual-basis design, which transforms cloning attempts into measurable disturbances.

Intercept-Resend (Robust): Reaffirmed the strength of basis randomization, as simple measurement strategies introduced clear, detectable error rates.

### Conclusion:
The research highlights that BB84’s security is implementation-dependent. While math protects the protocol against measurement attacks, physical countermeasures (like decoy states) are required to defend against hardware exploits.

### Team Members: Nicole Igbozulike, Luke Fu , Harshitha Poludas , Naeem Hagans
