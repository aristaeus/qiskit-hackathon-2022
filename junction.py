#! /usr/bin/env python3

from qiskit import QuantumCircuit, Aer, assemble, transpile
from qiskit.visualization import plot_histogram
import itertools
import math
import matplotlib.pyplot as plt


# mark each state with probability
def cu_with_prob(circuit, control, target, phase):
    circuit.cu(0, phase, 0, 0, control, target)

def compute_oracle(circuit, state_a, state_b):
    if state_a[0] == 0:
        circuit.x(0)
    if state_a[1] == 0:
        circuit.x(1)
    if state_b[0] == 0:
        circuit.x(2)
    if state_b[1] == 0:
        circuit.x(3)

def mark_state_with_phase(circuit, state_a, state_b, phase):
    compute_oracle(circuit, state_a, state_b)
    # extra ancilla qubits to make grover's work
    circuit.x(4)
    circuit.x(5)
    circuit.mcrz(phase, [0, 1, 2, 4, 5], 3)
    circuit.x(4)
    circuit.x(5)
    compute_oracle(circuit, state_a, state_b)


def calculate_junction(sky_weight, flower_weight, stem_weight, junction_weight):
    # sky, flower, stem, junction
    sky_state = [0, 0]
    flower_state = [0, 1]
    stem_state = [1, 0]
    junction_state = [1, 1]

    num_qubits = 6

    circuit = QuantumCircuit(num_qubits)
    # initialise in |+> state
    for q in range(num_qubits):
        circuit.h(q)

    phase_states = [
            (sky_state, sky_state),
            (flower_state, flower_state),
            (stem_state, stem_state),
            (junction_state, junction_state)
            ]

    # run oracle
    for ((p_1, a), (p_2, b)) in itertools.product(phase_states, phase_states):
        if a == [0, 0] and b == [0, 0]:
            # don't mark sky, sky
            continue
            # mark_state_with_phase(circuit, a, b, p_1 * p_2 * math.pi)
        else:
            mark_state_with_phase(circuit, a, b, p_1 * p_2 * math.pi)
            # continue

    # grover's algorithm 
    for q in range(num_qubits):
        circuit.h(q)
        circuit.x(q)

    # qiskit doesn't have a native mcz
    circuit.h(num_qubits - 1)
    circuit.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    circuit.h(num_qubits - 1)

    for q in range(num_qubits):
        circuit.x(q)
        circuit.h(q)

    # print(circuit.draw('text'))

    # run and parse result
    aer_sim = Aer.get_backend('aer_simulator')
    circuit.measure_all()
    qobj = assemble(circuit)
    result = aer_sim.run(qobj, shots=1).result()
    counts = result.get_counts()
    result = list(counts.keys())[0]
    if result[0:2] != '00' or result[2:] == '0000':
        return calculate_junction()
    # result = result[2:]
    # reverse result?
    result = result[-1:1:-1]
    return((int(result[:2], 2), int(result[2:], 2)))


print(calculate_junction(1, 1, 1, 1))
