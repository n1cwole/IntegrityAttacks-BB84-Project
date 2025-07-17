import cirq
import random
import numpy as np
import matplotlib.pyplot as plt

class BB84PNS:
    def __init__(self, mu):
        self.mu = mu  # mean photon number
        self.simulator = cirq.Simulator()
        self.qubit = cirq.NamedQubit("q0")

        # Keys
        self.alice_key = []
        self.bob_key = []
        self.eve_known_bits = []  # None if Eve doesn't know it

        # Prebuild circuits
        self.alice_send = {}
        self.bob_measure = {}
        self.build_circuits()

    def build_circuits(self):
        q = self.qubit
        # Alice: (bit, basis) → Circuit
        self.alice_send[(0, 'Z')] = cirq.Circuit(cirq.I(q))
        self.alice_send[(1, 'Z')] = cirq.Circuit(cirq.X(q))
        self.alice_send[(0, 'X')] = cirq.Circuit(cirq.H(q))
        self.alice_send[(1, 'X')] = cirq.Circuit(cirq.X(q), cirq.H(q))

        # Bob: basis → Circuit (no measurement key yet)
        self.bob_measure['Z'] = cirq.Circuit(cirq.measure(q))
        self.bob_measure['X'] = cirq.Circuit(cirq.H(q), cirq.measure(q))

    def simulate_round(self):
        alice_bit = random.choice([0, 1])
        alice_basis = random.choice(['Z', 'X'])
        bob_basis = random.choice(['Z', 'X'])

        photon_count = np.random.poisson(self.mu)
        eve_can_split = (photon_count > 1) and (alice_basis == 'X') and (bob_basis == 'X')

        circuit = cirq.Circuit()
        circuit += self.alice_send[(alice_bit, alice_basis)]

        meas_key = f"q0_{random.randint(0, 1e9)}"
        bob_circuit = self.bob_measure[bob_basis].copy()
        bob_circuit[-1] = cirq.Moment([cirq.measure(self.qubit, key=meas_key)])  # ✅ FIXED LINE
        circuit += bob_circuit

        result = self.simulator.run(circuit)
        bob_bit = result.measurements[meas_key][0][0]

        if alice_basis == bob_basis:
            self.alice_key.append(alice_bit)
            self.bob_key.append(int(bob_bit))
            if eve_can_split:
                self.eve_known_bits.append(alice_bit)
            else:
                self.eve_known_bits.append(None)


    def run_n_rounds(self, n):
        for _ in range(n):
            self.simulate_round()

    def calculate_stats(self):
        sifted_bits = len(self.alice_key)
        if sifted_bits == 0:
            return 0.0, 0.0, 0.0

        bob_errors = sum([a != b for a, b in zip(self.alice_key, self.bob_key)])
        bob_error_rate = bob_errors / sifted_bits

        known_pairs = [(a, e) for a, e in zip(self.alice_key, self.eve_known_bits) if e is not None]
        eve_coverage = len(known_pairs) / sifted_bits
        eve_correct = sum([a == e for a, e in known_pairs])
        eve_accuracy = (eve_correct / len(known_pairs)) if known_pairs else 0.0

        return bob_error_rate, eve_accuracy, eve_coverage

def run_simulation(mu_values, rounds=1000):
    bob_errors = []
    eve_accs = []
    eve_covs = []

    for mu in mu_values:
        protocol = BB84PNS(mu)
        protocol.run_n_rounds(rounds)
        err, acc, cov = protocol.calculate_stats()
        bob_errors.append(err)
        eve_accs.append(acc)
        eve_covs.append(cov)

    return mu_values, bob_errors, eve_accs, eve_covs

import matplotlib.pyplot as plt
import numpy as np

def plot_all_results(mu_vals, bob_errors, eve_accs, eve_covs):
    fig, axs = plt.subplots(3, 1, figsize=(12, 16))  # taller figure

    mu_labels = [f"{mu:.2f}" for mu in mu_vals]
    x = np.arange(len(mu_vals))

    # 1. Bob Error Rate
    axs[0].bar(x, bob_errors, color='salmon', edgecolor='salmon')
    axs[0].set_title("Bob Error Rate vs Mean Photon Number (μ)", fontsize=14)
    axs[0].set_ylabel("Bob Error Rate", fontsize=12)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(mu_labels, fontsize=10)

    # 2. Eve Accuracy
    axs[1].bar(x, eve_accs, color='royalblue', edgecolor='royalblue')
    axs[1].set_title("Eve Accuracy (on Known Bits) vs Mean Photon Number (μ)", fontsize=14)
    axs[1].set_ylabel("Eve Accuracy", fontsize=12)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(mu_labels, fontsize=10)

    # 3. Eve Coverage
    axs[2].bar(x, eve_covs, color='lightgreen', edgecolor='lightgreen')
    axs[2].set_title("Eve Coverage vs Mean Photon Number (μ)", fontsize=14)
    axs[2].set_ylabel("Eve Coverage (fraction known)", fontsize=12)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(mu_labels, fontsize=10)
    axs[2].set_xlabel("Mean Photon Number (μ)", fontsize=12)

    plt.tight_layout(pad=4.0) 
    plt.show()


if __name__ == "__main__":
    mu_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    mu_vals, bob_errors, eve_accs, eve_covs = run_simulation(mu_values, rounds=1000)
    plot_all_results(mu_vals, bob_errors, eve_accs, eve_covs)
