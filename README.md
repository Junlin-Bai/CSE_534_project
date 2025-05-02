# quantum-similation

Simulate Quantum Network verification protocols Using Net-Squid

# Goal

Compare two entanglement varification protocols

# Protocol Overview

# Protocols

## GenEntanglement

Generate entanglement between two nodes. `GenEntanglementCocurrent.py` is the implementation of this protocol.

The protocol is as follows:

0. Setup network structure (set up in experiment module).
1. Each node will constantly generate entanglement pairs until the specified number of pairs is reached.
2. For each pair of entanglement, the protocol will send the **memory position** and **initial fidelity** to the
   Purification protocol via `self.send_signal()`.
3. Once all paris are generated, the protocol will then wait for the Purification protocol to send a request for
   generating new entanglement pairs.
4. The protocol will then generate new entanglement based on the memory position sent by purification protocol.

## Verification

Once the Entanglement pairs are generated, we will run two varification protocols CHSH.py and GHZ.py to test 
the fidelity of them. 


## TODO

- [ ] Implement two node connection
    - [x] Implement preparation between two nodes
    - [x] Implement verification between two nodes
      - [x] Implement CHSH verification between two nodes
      - [x] Implement GHZ verification between two nodes
    - [ ] Implement entanglement swapping between two nodes
        - [x] Implement Teleportation between two nodes
