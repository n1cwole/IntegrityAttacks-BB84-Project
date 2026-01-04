# Project title: "Quantum Eavesdropper Detection: Evaluating the Threat Landscape of BB84 Attacks"
---
> ## 📄[**Read the Full Research Paper (PDF)**](https://drive.google.com/file/d/1-41F5pUfaZcrjvg3Cv4dygVaavt6pIsn/view?usp=drive_link)
---

### Project Overview:
This project provides a comprehensive simulation and analysis of the BB84 Measurment-Based Quantum Key Distribution (QKD) protocol's vulnerability to various measurement-based eavesdropping strategies. By simulating different individual attacks, we quantify the Quantum Bit Error Rate (QBER) as well as other metrics to determine which methods most effectively compromise key security without detection.

The simulation evaluates the susceptibility of the BB84 protocol against four distinct individual measurement attacks:

Intercept-Resend: The most basic attack where Eve measures the qubits and sends a new state to Bob.

Photon Number Splitting (PNS): Exploiting multi-photon pulses to gain information without disturbing the signal.

CNOT Probe: Using an auxiliary qubit and a CNOT gate to entangle Eve's system with the quantum channel.

CZ Probe: Utilizing Controlled-Z gates to induce phase-shift errors for information extraction.

### Tech Stack

#### Python 3.11

#### Libraries utilized: 

NumPy

Cirq

Matplotlib

#### Documentation: 

LaTeX

### Key Findings:
PNS Attack (Most Critical): Exploited multi-photon pulses to extract key bits with $0\%$ QBER disturbance. This confirms that hardware-level vulnerabilities are the most significant threat to "unconditional" security.

CNOT/CZ Probes (Detectable): Entanglement-based attacks produced a QBER of approximately 25%. This validates BB84's dual-basis design, which transforms cloning attempts into measurable disturbances.

Intercept-Resend (Robust): Reaffirmed the strength of basis randomization, as simple measurement strategies introduced clear, detectable error rates.

### Conclusion:
The research highlights that BB84’s security is implementation-dependent. While math protects the protocol against measurement attacks, physical countermeasures (like decoy states) are required to defend against hardware exploits.

### Team Members: Nicole Igbozulike, Luke Fu , Harshitha Poludas , Naeem Hagans
