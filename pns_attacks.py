import random
from typing import Optional

class PNSAttackSimulator:
    """
    simulate photon number splitting (PNS) attacks on a simplified BB84-like QKD protocol
    
    class assumes:
    - Alice sends weak coherent pulses with a photon number distribution (simplified)
    - Eve can split off one photon from multi-photon pulses
    - Eve delays measurement until after basis reconciliation
    """
    def __init__(self, mean_photon_number: float = 0.1, eve_intercept: bool = True):
        """
        initialize the PNS attack simulator

        - mean_photon_number: avg photon number per pulse (default 0.1 typical for weak pulses).
        - eve_intercept: whether Eve attempts a PNS attack
        """
        self.mean_photon_number = mean_photon_number
        self.eve_intercept = eve_intercept

        self.alice_bits = []
        self.alice_bases = []

        self.bob_bits = []
        self.bob_bases = []

        self.eve_bits = []       # eve’s stored bits from split photons
        self.eve_bases = []      # bases Eve will use to measure after basis reconciliation
    
    def generate_photon_number(self) -> int:
        """
        simulates the number of photons in a pulse using a simplified discrete distribution.

        returns:
        - integer photon count, typically 0, 1, or 2.
        """
        # 0 photons (vacuum), 1 photon (single), or 2 photons (multi-photon)
        return random.choices([0, 1, 2], weights=[0.7, 0.25, 0.05])[0]
    
    def send_bit(self, alice_bit: int, alice_basis: str, bob_basis: str):
        """
        simulate sending one bit through the channel with possible PNS attack.

        Parameters:
        - alice_bit: 0 or 1 bit Alice wants to send.
        - alice_basis: 'Z' or 'X' basis Alice uses to encode the qubit.
        - bob_basis: 'Z' or 'X' basis Bob uses to measure the qubit.

    
        - generate photon number for the pulse.
        - if photon number > 1 and Eve intercepts, Eve splits one photon off and delays measurement.
        - Bob receives the rest of the photons and measures.
        - if photon number <= 1 or Eve doesn’t intercept, normal transmission.
        """

        photon_num = self.generate_photon_number()

        # store Alice's info for later basis reconciliation
        self.alice_bits.append(alice_bit)
        self.alice_bases.append(alice_basis)

        self.bob_bases.append(bob_basis)

        if photon_num == 0:
            # no photon sent; Bob receives nothing (loss)
            self.bob_bits.append(None)
            if self.eve_intercept:
                self.eve_bits.append(None)
            return

        if self.eve_intercept and photon_num > 1:
            # eve performs PNS attack: keeps one photon, sends remaining to Bob
            # eve stores bit but delays measurement (basis unknown yet)
            self.eve_bits.append(None)  # placeholder for delayed measurement
            self.eve_bases.append(None)  # will assign after basis reconciliation

            # Bob receives remaining photons and measures in his basis
            # assume Bob's measurement matches Alice's bit if bases agree (ideal channel)
            if alice_basis == bob_basis:
                self.bob_bits.append(alice_bit)  # perfect detection
            else:
                # different bases → Bob has 50% chance to get bit 0 or 1 randomly
                self.bob_bits.append(random.choice([0, 1]))
        else:
            # no PNS attack or single photon: Eve may do intercept-resend or no attack
            if self.eve_intercept:
                # simple intercept-resend: Eve measures randomly basis and resends
                eve_basis = random.choice(['Z', 'X'])
                self.eve_bases.append(eve_basis)
                if eve_basis == alice_basis:
                    eve_bit = alice_bit
                else:
                    eve_bit = random.choice([0, 1])
                self.eve_bits.append(eve_bit)

                # Bob measures in his basis (with error if Eve chose wrong basis)
                if bob_basis == eve_basis:
                    self.bob_bits.append(eve_bit)
                else:
                    self.bob_bits.append(random.choice([0, 1]))
            else:
                # No Eve, perfect channel: Bob gets bit if bases match
                self.eve_bits.append(None)
                self.eve_bases.append(None)
                if alice_basis == bob_basis:
                    self.bob_bits.append(alice_bit)
                else:
                    self.bob_bits.append(random.choice([0, 1]))

    def eve_delayed_measurement(self):
        """
        Eve measures her stored photons after basis reconciliation.

        for rounds where Eve delayed measurement (PNS attack), she now measures in the correct basis.

        updates self.eve_bits with the measurement results.
        """

        for i in range(len(self.eve_bits)):
            if self.eve_bits[i] is None and self.eve_bases[i] is None:
                # this was a delayed measurement due to PNS attack
                alice_basis = self.alice_bases[i]

                # Eve measures in the correct basis (same as Alice's)
                self.eve_bases[i] = alice_basis

                # Eve gets the correct bit perfectly in this idealized simulation
                self.eve_bits[i] = self.alice_bits[i]
