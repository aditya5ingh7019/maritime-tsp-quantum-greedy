# maritime-tsp-quantum-greedy
Hybrid quantum–classical greedy optimization for the Maritime Traveling Salesman Problem (TSP) using QAOA, with realistic sea-route distances and empirical evaluation of NISQ-era limitations.


# Maritime TSP — Quantum-Assisted Greedy Optimization

This repository contains the reference implementation for the paper:

**“An Evaluation of a Hybrid Quantum–Classical Approach for Maritime Route Optimization”**

## ⚠️ Important Disclaimer
This work **does not claim quantum advantage**.
The goal is to empirically evaluate how shallow QAOA circuits behave
when embedded into greedy routing heuristics under NISQ constraints.

## Installation
```bash
git clone https://github.com/adityaSingh709/maritime-tsp-quantum-greedy.git
cd maritime-tsp-quantum-greedy

# Recommended: use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

## Problem Description
- Traveling Salesman Problem (TSP) over real maritime routes
- Distances computed using the `searoute` library
- 15 major Indian Ocean and East Asian ports

## Methodology
- Classical baselines:
  - Nearest Neighbor
  - Hill-Climbing
  - LKH-3 (external solver)
- Hybrid approach:
  - Greedy route construction
  - QAOA-based decision-making on subproblems
  - Classical fallback when problem size exceeds qubit limits

## Key Design Choices
- `MAX_QUBITS = 8` to avoid exponential memory blow-up
- Mixed-state simulation (`default.mixed`)
- Explicit noise injection via depolarizing channels
- Early stopping in QAOA optimization

## Reproducing Paper Results
Example command for the hybrid QAOA-greedy run (3 layers, 150 optimization steps, starting from port index 0):

```bash
python maritime_tsp_quantum_greedy.py --start_idx 0 --layers 3 --steps 150
