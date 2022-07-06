import numpy as np
import qiskit
from qiskit.tools.visualization import circuit_drawer
from qiskit.circuit.library import ZGate
from qiskit.providers.aer.noise import NoiseModel
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import qiskit.providers.aer.noise as noise

import matplotlib.pyplot as plt
import random
import HackathonFuncs

from Charles import visualise


#this initialises the initial state
#also I will be appending more and more layers to as the system evolves
PictureWidth=10
BigPicture=np.zeros([1,PictureWidth])-np.ones([1,PictureWidth])
BigPicture[0,:]=0
BigPicture[0,1]=2
BigPicture[0,3]=2
BigPicture[0,8]=2


#BigPicture=HackathonFuncs.UntilStopClass(BigPicture)
#Max=10
#BigPicture=HackathonFuncs.FiniteIterationClass(BigPicture,Max)
#print(BigPicture)

sky=1
stem=0.7
fork=0.9
flower=0.7

#BigPicture=HackathonFuncs.UntilStopQuantum(BigPicture,sky,stem,fork,flower)
Max=10
BigPicture=HackathonFuncs.FiniteIterationQuantum(BigPicture,Max,sky,stem,fork,flower)
print(BigPicture)

test=visualise(BigPicture)

test.save("test.png")

"""
# Error probabilities
prob_1 = 0.1  # 1-qubit gate
prob_2 = 0.15   # 2-qubit gate

# Depolarizing quantum errors
error_1 = noise.depolarizing_error(prob_1, 1)
error_2 = noise.depolarizing_error(prob_2, 2)

# Add errors to noise model
noise_model = noise.NoiseModel()
noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

# Get basis gates from noise model
basis_gates = noise_model.basis_gates

# Make a circuit
circ = QuantumCircuit(3, 3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure([0, 1, 2], [0, 1, 2])

# Perform a noise simulation
result = execute(circ, Aer.get_backend('qasm_simulator'),
                 basis_gates=basis_gates,
                 noise_model=noise_model).result()
counts = result.get_counts(0)
plot_histogram(counts)
plt.show()
"""
"""
#this is a test code of I think it's Grover's Algorithm
#this initialises the circuit
circuit = qiskit.QuantumCircuit(5,5)

#this is the Hademar gates that gets every qbit into an even superposition with
#no phase
circuit.h([0,1,2,3,4])
circuit.barrier()

#this is the oracle
#these next 2 lines are the controlled z-gate: z-gate on qbit4 with control1 on
#all other qbits, this shifts the phase of |11111> state and then an xgate on
#other qbits will change the state such that an xgate on e.g. qbit3 will make
#the phase shifted term |11101> where the order here is |qbit0,1,2,3,4>
c4z = ZGate().control(4)
circuit.append(c4z, range(5))
circuit.x(0)
circuit.barrier()

#This is the inverter
circuit.h([0,1,2,3,4])
circuit.x([0,1,2,3,4])
c4z = ZGate().control(4)
circuit.append(c4z, range(5))
circuit.x([0,1,2,3,4])
circuit.h([0,1,2,3,4])
circuit.barrier()

#this measures all the qbits
circuit.measure(range(5),range(5))

circuit.draw(output='mpl')
plt.show()
print(circuit)

simulator = qiskit.Aer.get_backend('qasm_simulator')
#simulator = qiskit.Aer.get_backend('statevector_simulator')
job = qiskit.execute(circuit, simulator, shots=8000)
result = job.result()
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
qiskit.visualization.plot_histogram(counts)
plt.show()
"""
