#!/usr/bin/env python
# coding: utf-8

# In[38]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maritime TSP — Quantum-Assisted Greedy Optimization (Research Code)

This repository implements a hybrid quantum–classical greedy algorithm
for the Maritime Traveling Salesman Problem (TSP), using realistic sea
distances computed via the `searoute` library.

IMPORTANT CONTEXT:
------------------
• This code does NOT claim quantum advantage.
• Quantum optimization is applied only to reduced subproblems
  (≤ MAX_QUBITS) due to NISQ-era limitations.
• Classical greedy fallbacks are intentionally used for larger decision
  spaces to ensure scalability and reproducibility.

The implementation is designed to study:
• How QAOA behaves when embedded inside greedy decision pipelines
• Sensitivity to circuit depth, optimizer parameters, and random seeds
• Limitations of shallow variational circuits on structured routing tasks

This code accompanies the research paper:
"An Evaluation of a Hybrid Quantum–Classical Approach for Maritime Route Optimization"
(IEEE-style manuscript).

Dependencies:
-------------
numpy, random, searoute, pennylane, plotly

Execution:
----------
Terminal:
    python maritime_tsp_quantum_greedy.py --start_idx 0 --layers 3 --steps 150

Jupyter:
    Run directly (argparse is automatically bypassed).

Reproducibility:
----------------
• All randomness is controlled via explicit seeds.
• Distance matrices are precomputed.
• Results reported in the paper correspond to tagged releases.

Author: Aditya Singh
"""


import argparse
import time
import numpy as np
import random
import searoute as sr
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import NesterovMomentumOptimizer
import plotly.graph_objects as go


# =============================================================================
# PORT DATA AND DISTANCE MATRIX
# =============================================================================

PORTS = {
    "Mumbai": (18.94, 72.83),
    "Chennai": (13.08, 80.28),
    "Kolkata": (22.56, 88.34),
    "Kochi": (9.96, 76.27),
    "Visakhapatnam": (17.70, 83.30),
    "Paradip": (20.32, 86.61),
    "Goa": (15.40, 73.81),
    "Tuticorin": (8.77, 78.13),
    "Haldia": (22.03, 88.06),
    "Mormugao": (15.41, 73.81),
    "Singapore": (1.30, 103.77),
    "Colombo": (6.95, 79.85),
    "Jebel Ali": (25.00, 55.05),
    "Port Klang": (3.00, 101.40),
    "Shanghai": (31.23, 121.47)
}

PORT_LIST = list(PORTS.keys())
N_PORTS = len(PORT_LIST)


def maritime_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute real maritime distance in km using searoute."""
    if abs(lat1 - lat2) < 1e-6 and abs(lon1 - lon2) < 1e-6:
        return 0.0
    coord1 = (lon1, lat1)
    coord2 = (lon2, lat2)
    try:
        route = sr.searoute(coord1, coord2)
        return route['properties']['length']
    except Exception:
        return 999999.0


# Precompute distance matrix
D = np.zeros((N_PORTS, N_PORTS))
for i in range(N_PORTS):
    for j in range(N_PORTS):
        if i != j:
            lat1, lon1 = PORTS[PORT_LIST[i]]
            lat2, lon2 = PORTS[PORT_LIST[j]]
            D[i, j] = maritime_distance(lat1, lon1, lat2, lon2)


def route_cost(route: list[int]) -> float:
    """Total tour cost (closed cycle)."""
    total = 0.0
    for i in range(len(route) - 1):
        total += D[route[i], route[i + 1]]
    total += D[route[-1], route[0]]
    return total


# =============================================================================
# CLASSICAL BASELINES
# =============================================================================

def random_search(n_trials: int = 50000) -> tuple[list[int], float]:
    """Pure random search baseline."""
    best_cost = float('inf')
    best_route = None
    for _ in range(n_trials):
        route = list(range(N_PORTS))
        random.shuffle(route)
        cost = route_cost(route)
        if cost < best_cost:
            best_cost = cost
            best_route = route[:]
    return best_route, best_cost


def hill_climbing(start_route: list[int], n_steps: int = 20000) -> tuple[list[int], float]:
    """Swap-based hill-climbing."""
    current_route = start_route[:]
    current_cost = route_cost(current_route)
    for _ in range(n_steps):
        i, j = random.sample(range(N_PORTS), 2)
        new_route = current_route[:]
        new_route[i], new_route[j] = new_route[j], new_route[i]
        new_cost = route_cost(new_route)
        if new_cost < current_cost:
            current_route = new_route
            current_cost = new_cost
    return current_route, current_cost


def nearest_neighbor_tour(start_idx: int) -> tuple[list[int], float]:
    """Nearest-neighbor greedy tour."""
    unvisited = set(range(N_PORTS))
    unvisited.remove(start_idx)
    tour = [start_idx]
    current = start_idx
    while unvisited:
        next_port = min(unvisited, key=lambda x: D[current, x])
        tour.append(next_port)
        unvisited.remove(next_port)
        current = next_port
    tour.append(start_idx)
    return tour, route_cost(tour)

# =============================================================================
# LKH-3 INTEGRATION (call external solver)
# =============================================================================
import subprocess
import os
import tempfile

LKH_EXE = r".\LKH-3.exe"

def save_tsplib_file(dist_matrix, port_names, filename):
    """Save distance matrix in TSPLIB explicit format"""
    n = len(port_names)
    with open(filename, 'w') as f:
        f.write(f"NAME : MaritimeTSP\n")
        f.write(f"COMMENT : {n} Indian Ocean / Asia ports\n")
        f.write(f"TYPE : TSP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(n):
            row = " ".join(f"{int(round(dist_matrix[i,j]))}" for j in range(n))
            f.write(row + "\n")
        f.write("EOF\n")

def run_lkh(dist_matrix, port_names, max_trials=100, pop_size=1):
    """Call LKH-3 executable and return best tour + cost"""
    with tempfile.TemporaryDirectory() as tmpdir:
        problem_file = os.path.join(tmpdir, "problem.tsp")
        par_file     = os.path.join(tmpdir, "params.par")
        tour_file    = os.path.join(tmpdir, "best.tour")
        output_file  = os.path.join(tmpdir, "lkh.out")

        save_tsplib_file(dist_matrix, port_names, problem_file)

        with open(par_file, 'w') as f:
            f.write(f"PROBLEM_FILE = {problem_file}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tour_file}\n")
            f.write(f"RUNS = {max_trials}\n")
            f.write("TRACE_LEVEL = 1\n")
            f.write("MOVE_TYPE = 5\n")

        # Debug print
        print("\n=== Parameter file written to LKH ===")
        with open(par_file, 'r', encoding='utf-8') as debug:
            print(debug.read())
        print("=====================================\n")

        exe_name = LKH_EXE
        if not os.path.exists(exe_name):
            print(f"ERROR: LKH executable not found at: {exe_name}")
            print("Please download from: http://webhotel4.ruc.dk/~keld/research/LKH-3/")
            print("Current working dir:", os.getcwd())
            return None, float('inf')

        try:
            result = subprocess.run(
                [exe_name, par_file],
                stdout=open(output_file, "w"),
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
                shell=False
            )
        except FileNotFoundError:
            print(f"ERROR: Cannot find '{exe_name}'")
            print("Current dir:", os.getcwd())
            print("Files:", os.listdir('.'))
            return None, float('inf')
        except subprocess.CalledProcessError as e:
            print(f"LKH failed (code {e.returncode})")
            if os.path.exists(output_file):
                print("\nLKH output:\n" + open(output_file, encoding='utf-8', errors='ignore').read())
            return None, float('inf')
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None, float('inf')

        # Parse tour
        tour = []
        reading_tour = False
        try:
            with open(tour_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == "TOUR_SECTION":
                        reading_tour = True
                        continue
                    if reading_tour:
                        if line in ("-1", "EOF"):
                            break
                        try:
                            idx = int(line) - 1
                            if 0 <= idx < len(port_names):
                                tour.append(idx)
                        except ValueError:
                            pass
        except FileNotFoundError:
            print("Tour file not created")
            return None, float('inf')

        if not tour or len(tour) != len(port_names):
            print(f"Tour length mismatch or empty: got {len(tour)}, expected {len(port_names)}")
            if os.path.exists(output_file):
                print("\nLKH output:\n" + open(output_file, encoding='utf-8', errors='ignore').read())
            return None, float('inf')

        cost = route_cost(tour)
        return tour, cost


# =============================================================================
# QUANTUM-ASSISTED GREEDY (WITH PROGRESS)
# =============================================================================

def build_quantum_greedy_tour(
    start_idx: int = 0,
    layers: int = 3,
    steps_per_decision: int = 150,
    alpha: float = 0.5
) -> tuple[list[str], float, list[int]]:
    """
    Quantum-assisted greedy TSP solver with detailed progress output.
    """

    MAX_QUBITS = 8  # safe upper bound for mixed-state simulation

    unvisited = set(range(N_PORTS))
    unvisited.remove(start_idx)
    tour = [start_idx]
    current = start_idx

    penalty = 10.0 * np.max(D)

    print(f"\nStarting from {PORT_LIST[start_idx]}")

    while unvisited:
        k = len(unvisited)

        # --------------------------------------------------------------
        # SAFETY GUARD: avoid exponential memory blow-up
        # --------------------------------------------------------------
        if k > MAX_QUBITS:
            next_idx = min(unvisited, key=lambda x: D[current, x])
            print(
                f" {k:2d} ports left → chose {PORT_LIST[next_idx]} "
                f"(classical fallback: exceeds {MAX_QUBITS} qubits)"
            )
            tour.append(next_idx)
            unvisited.remove(next_idx)
            current = next_idx
            continue

        rem = list(unvisited)

        # Effective costs
        effective_costs = []
        for j in rem:
            future = min(D[j, x] for x in unvisited if x != j) if len(unvisited) > 1 else 0
            effective_costs.append(D[current, j] + alpha * future)

        # Cost Hamiltonian
        coeffs, ops = [], []
        for i, d in enumerate(effective_costs):
            coeffs += [d / 2, -d / 2]
            ops += [qml.Identity(i), qml.PauliZ(i)]
        for i in range(k):
            for j in range(i + 1, k):
                coeffs.append(penalty / 4)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        H_cost = qml.Hamiltonian(coeffs, ops)

        # XY mixer
        mixer_coeffs, mixer_ops = [], []
        for i in range(k):
            for j in range(i + 1, k):
                mixer_coeffs.append(1.0)
                mixer_ops.append(qml.PauliX(i) @ qml.PauliX(j))
                mixer_coeffs.append(1.0)
                mixer_ops.append(qml.PauliY(i) @ qml.PauliY(j))
        H_mixer = qml.Hamiltonian(mixer_coeffs, mixer_ops)

        dev = qml.device("default.mixed", wires=k)

        @qml.qnode(dev)
        def circuit(g, b):
            for i in range(k):
                qml.Hadamard(i)
            for gg, bb in zip(g, b):
                qml.qaoa.cost_layer(gg, H_cost)
                qml.qaoa.mixer_layer(bb, H_mixer)
                for wire in range(k):
                    qml.DepolarizingChannel(0.01, wires=wire)
            return qml.probs(wires=range(k))

        g = pnp.random.uniform(0.05, 0.2, layers, requires_grad=True)
        b = pnp.random.uniform(1.0, 1.5, layers, requires_grad=True)
        opt = NesterovMomentumOptimizer(0.03)

        prev_energy = float("inf")
        stable = 0

        print(
            f" {k:2d} ports left → optimizing "
            f"({layers} layers, max {steps_per_decision} steps)... ",
            end="",
            flush=True
        )

        for step in range(steps_per_decision):
            g, b = opt.step(lambda x, y: pnp.sum(circuit(x, y)), g, b)
            energy = pnp.sum(circuit(g, b))

            if abs(energy - prev_energy) < 0.1:
                stable += 1
                if stable > 20:
                    print(f"(early stop at step {step})", end=" ", flush=True)
                    break
            else:
                stable = 0
            prev_energy = energy

            if step % 30 == 0 and step > 0:
                print(f"{step} ", end="", flush=True)

        print("done", flush=True)

        probs = circuit(g, b)

        valid_states, weights = [], []
        for s in range(2**k):
            bs = format(s, f"0{k}b")
            if bs.count("1") == 1:
                valid_states.append(bs)
                weights.append(probs[s])

        if sum(weights) > 0:
            chosen = random.choices(valid_states, weights=weights)[0]
            next_idx = rem[chosen.index("1")]
            prob_str = f"quantum prob: {max(weights):.4f}"
        else:
            next_idx = min(unvisited, key=lambda x: D[current, x])
            prob_str = "fallback (nearest neighbor)"

        print(f" → chose {PORT_LIST[next_idx]} ({prob_str})", flush=True)

        tour.append(next_idx)
        unvisited.remove(next_idx)
        current = next_idx

    tour.append(start_idx)
    total_cost = route_cost(tour)

    tour_names = [PORT_LIST[i] for i in tour]
    path_indices = tour[:-1]

    return tour_names, total_cost, path_indices



# =============================================================================
# PLOTTING
# =============================================================================

def plot_routes(q_tour: list[str], q_cost: float, q_indices: list[int],
                hc_route: list[int], hc_cost: float):
    """Save interactive Plotly comparison map."""
    fig = go.Figure()

    port_lats = [PORTS[name][0] for name in PORT_LIST]
    port_lons = [PORTS[name][1] for name in PORT_LIST]
    fig.add_trace(go.Scattergeo(
        lon=port_lons, lat=port_lats,
        mode='markers+text', text=PORT_LIST,
        textposition="top center",
        marker=dict(size=10, color='black', symbol='circle'),
        name='Ports'
    ))

    # Quantum route
    q_lons = [PORTS[PORT_LIST[i]][1] for i in q_indices + [q_indices[0]]]
    q_lats = [PORTS[PORT_LIST[i]][0] for i in q_indices + [q_indices[0]]]
    fig.add_trace(go.Scattergeo(
        lon=q_lons, lat=q_lats,
        mode='lines', line=dict(width=3, color='green'),
        name=f'Quantum Route ({q_cost:.0f} km)'
    ))

    # Hill-climbing route
    hc_lons = [PORTS[PORT_LIST[i]][1] for i in hc_route + [hc_route[0]]]
    hc_lats = [PORTS[PORT_LIST[i]][0] for i in hc_route + [hc_route[0]]]
    fig.add_trace(go.Scattergeo(
        lon=hc_lons, lat=hc_lats,
        mode='lines', line=dict(width=2, color='blue', dash='dash'),
        name=f'Hill-climbing ({hc_cost:.0f} km)'
    ))

    fig.update_layout(
        title_text='Maritime TSP: Quantum vs Classical Routes',
        showlegend=True,
        geo=dict(
            resolution=50, showland=True, showlakes=True,
            landcolor='rgb(243,243,243)', countrycolor='rgb(204,204,204)',
            lakecolor='rgb(255,255,255)', projection_type='equirectangular',
            coastlinewidth=2,
            lataxis=dict(range=[-10, 40]),
            lonaxis=dict(range=[40, 130]),
        ),
        width=1000, height=600
    )

    fig.write_image("tsp_routes_plotly.png", scale=2)
    print("Interactive plot saved as 'tsp_routes_plotly.png'")


# =============================================================================
# MAIN + STATISTICAL RUNNER
# =============================================================================

def main(start_idx: int = 0, layers: int = 3, steps: int = 150):
    print("Running classical baselines...")
    _, best_cost_rand = random_search()
    print(f"Best random cost: {best_cost_rand:.0f} km")

    current_route, current_cost = hill_climbing(list(range(N_PORTS)))
    print(f"Hill-climbing final cost: {current_cost:.0f} km")
    print()

    print("Running LKH-3 ...")
    lkh_route, lkh_cost = run_lkh(D, PORT_LIST, max_trials=100, pop_size=5)
    if lkh_route is not None:
        print(f"LKH-3 tour cost: {lkh_cost:,.0f} km")
        lkh_names = [PORT_LIST[i] for i in lkh_route] + [PORT_LIST[lkh_route[0]]]
        print("LKH tour: " + " → ".join(lkh_names))
    else:
        print("LKH-3 failed — skipping")

    print("Running quantum-assisted greedy...")
    q_tour, q_cost, q_indices = build_quantum_greedy_tour(
        start_idx=start_idx, layers=layers, steps_per_decision=steps
    )

    print("\nQuantum greedy tour:")
    print(" → ".join(q_tour))
    print(f"Quantum tour cost: {q_cost:.0f} km")
    print(f"Approximation ratio vs hill-climbing: {q_cost / current_cost:.3f}")

    plot_routes(q_tour, q_cost, q_indices, current_route, current_cost)

    # =============================================================================
    # Ablation study: layers & alpha
    # =============================================================================
    print("\n=== Ablation study: QAOA layers & future cost weight (alpha) ===")
    ablation_results = []

    for layers in [1, 2, 3, 4]:
        for alpha_val in [0.0, 0.3, 0.5, 0.7]:
            print(f"\nAblation run → layers={layers}, alpha={alpha_val}")

            q_tour_ab, q_cost_ab, _ = build_quantum_greedy_tour(
                start_idx=start_idx,
                layers=layers,
                steps_per_decision=80,   # lower for ablation speed
                alpha=alpha_val
            )
            ablation_results.append((layers, alpha_val, q_cost_ab))
            print(f"  → cost: {q_cost_ab:,.0f} km")

    # Simple table-like output
    print("\nAblation summary:")
    print("layers | alpha | cost (km)")
    print("-"*35)
    for l, a, c in sorted(ablation_results, key=lambda x: (x[0], x[1])):
        print(f"{l:6} | {a:5} | {c:,.0f}")

    # Statistical ablation
    NUM_EXPERIMENTS = 4
    seeds = [42 + i * 13 for i in range(NUM_EXPERIMENTS)]
    stats = {
        'qa_greedy_mumbai': [],
        'nearest_neighbor_mumbai': [],
        'qa_greedy_singapore': [],
    }

    for exp_id, seed in enumerate(seeds):
        print(f"\n=== Experiment {exp_id+1}/{NUM_EXPERIMENTS} | Seed {seed} ===")
        random.seed(seed)
        np.random.seed(seed)

        start_time = time.time()

        _, q_cost_mum, _ = build_quantum_greedy_tour(
            start_idx=start_idx, layers=layers, steps_per_decision=steps
        )
        stats['qa_greedy_mumbai'].append(q_cost_mum)

        _, nn_cost = nearest_neighbor_tour(start_idx)
        stats['nearest_neighbor_mumbai'].append(nn_cost)

        #Variant: QA-greedy Singapore 
        print("Running QA-greedy Singapore...")
        q_tour_sin, q_cost_sin, _ = build_quantum_greedy_tour(start_idx=10, layers=6, steps_per_decision=120)
        stats['qa_greedy_singapore'].append(q_cost_sin)

        elapsed = time.time() - start_time
        print(f"Experiment {exp_id+1} finished in {elapsed/60:.1f} minutes")

    print("\n=== STATISTICAL SUMMARY (4 runs) ===")
    for variant, costs in stats.items():
        if costs:
            mean = np.mean(costs)
            std = np.std(costs)
            print(f"{variant:25} → Mean {mean:,.0f} ± {std:,.0f} km "
                  f"(values: {[f'{c:,.0f}' for c in costs]})")


# =============================================================================
# ENTRY POINT (Notebook + Script friendly)
# =============================================================================

import sys

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Maritime TSP - Quantum-assisted Greedy")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting port index")
    parser.add_argument("--layers", type=int, default=3, help="QAOA layers")
    parser.add_argument("--steps", type=int, default=150, help="Steps per decision")

    # Jupyter/IPython detection: skip argparse, use defaults
    if "ipykernel" in sys.modules or "IPython" in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    main(args.start_idx, args.layers, args.steps)


# In[ ]:




