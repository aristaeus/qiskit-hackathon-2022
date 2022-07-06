import numpy as np
import random
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, assemble
from numpy import pi
from qiskit.circuit.library import ZGate

import itertools
import math

def NonJunctionEvolutionClass(BigPicture, N):
    """
    This takes in the state of the system at some iteration number N and gives the
    state of the system at N+1 but does not do the junction evolution

    This is the classical evolution, just use randint(1,3) to determine the
    stem evolution and we have nothing for the junction evolution, that's a
    different function

    I think this needs to be applied after the junction operations

    Params:
    ------------------
    BigPicture: a PictureHeight x PictureWidth array of ints
        it's the states of the system at 0 to N inclusive
        state of the N+1 system is still being defined
        Junction operatations have already been performed to
        partially determine the N+1 state

    N: an int less than PictureHeight
        refers to the Nth iteraction of the system

    Outputs:
    -------------------
    NewPicture: a PictureHeight x PictureWidth array of ints
        it's the states of the system at 0 to N+1 inclusive
        The N+1 state should now be fully determined

    """

    PictureWidth=np.shape(BigPicture)[1]
    NewPicture=BigPicture


    for counter in range(PictureWidth):
        #if the state of the N+1 cell has not already been decided then do this code
        if BigPicture[N+1,counter] ==-1:
            #if a cell is sky then the state above it is sky
            if BigPicture[N,counter]==0:
                NewPicture[N+1,counter]=0
            #if a cell is a flower then the state above it is sky
            if BigPicture[N,counter]==1:
                NewPicture[N+1,counter]=0
            #if a cell is a stem then it could be a flower, a stem or a junction
            if BigPicture[N,counter]==2:
                #replace this with Grover's later to make it quantum
                #and also implement bias
                x=random.randint(1,3)
                NewPicture[N+1,counter]=x

        #this is to avoid the situation where, due to a junction, the cell above a
        #stem is inappropriately set to be sky
        if NewPicture[N+1,counter] == 0 and NewPicture[N,counter] == 2:
            #replace this with Grover's later to make it quantum
            #and also implement bias
            x=random.randint(1,3)
            NewPicture[N+1,counter]=x


    return NewPicture

def FiniteIterationClass(Seed,MAX):
    """
    This grows the plant for MAX number of steps. The plant might grow beyond
    that in which case this function will plot a bunch of empty sky

    this is the classical version, we call the classical function for state
    evolution

    -1 unassigned
    0 sky
    1 flower
    2 system
    3 junction

    Params:
    ------------------
    Seed: a 1 x PictureWidth array of ints
        the initial state the plant grows from
        the entries will be 0,1,2,3

    Output:
    ------------------
    BigPicture: a MAX+1 x PictureWidth array of ints
        A picture of the plant's growth
        It's MAX+1 because there is the initial state and then MAX iterations
    """

    PictureWidth=np.shape(Seed)[1]
    Output=np.zeros([1,PictureWidth])
    Output[0,:]=Seed[:]

    #this point of holding is the have Holding[0,:] be the Nth layer of the system and
    # use it to determine the N+1 layer which we store in Holding[1,:] and then we append
    #Holding[1,:] to BigPicture to build up the system layer by layer
    Holding=np.zeros([2,PictureWidth])-np.ones([2,PictureWidth])


    for counter in range(MAX):
        #this print statement is going to give you every interation except the
        #last one
        print(Output)
        print('--------')


        Holding[0,:]=Output[counter,:]
        Holding[1,:]=np.zeros(PictureWidth)-np.ones(PictureWidth)

        Holding=NonJunctionEvolutionClass(Holding,0)

        Output=np.vstack([Output,Holding[1,:]])

    return Output

def UntilStopClass(Seed):
    """
    This grows the plant until it naturally stops. If the plant grows forever
    then this function will never stop

    this is the classical version, we call the classical function for state
    evolution

    -1 unassigned
    0 sky
    1 flower
    2 system
    3 junction

    Params:
    ------------------
    Seed: a 1 x PictureWidth array of ints
        the initial state the plant grows from
        the entries will be 0,1,2,3

    Output:
    ------------------
    BigPicture: a ? x PictureWidth array of ints
        it will be however tall as it needs to be for the plant to finish growing
        the topmost layer will be sky (or unassigned)
    """

    PictureWidth=np.shape(Seed)[1]
    Output=np.zeros([1,PictureWidth])
    Output[0,:]=Seed[:]

    #this point of holding is the have Holding[0,:] be the Nth layer of the system and
    # use it to determine the N+1 layer which we store in Holding[1,:] and then we append
    #Holding[1,:] to BigPicture to build up the system layer by layer
    Holding=np.zeros([2,PictureWidth])-np.ones([2,PictureWidth])


    #I use STOP later to control when the code stops
    STOP=0

    counter=0
    while STOP == 0:

        #this print statement is going to give you every interation except the
        #last one
        print(Output)
        print('--------')


        Holding[0,:]=Output[counter,:]
        Holding[1,:]=np.zeros(PictureWidth)-np.ones(PictureWidth)

        Holding=NonJunctionEvolutionClass(Holding,0)

        Output=np.vstack([Output,Holding[1,:]])

        #This bit says to stop unless there is a flower, stem, or junction
        STOP=1
        for counter2 in range(PictureWidth):
            if Holding[1,counter2] >0:
                STOP=0
                break

        counter=counter+1

    return Output

###Ben wrote code below####

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
            (sky_weight, sky_state),
            (flower_weight, flower_state),
            (stem_weight, stem_state),
            (junction_weight, junction_state)
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

###Ben wrote code above####


def stemQC(stem,fork,flower):
    """
    This is the code that Blayney wrote, generates a random integer [1,2,3]
    via a modified Grover's algorithm

    """

    qreg_q = QuantumRegister(3, 'q')
    creg_c = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(qreg_q, creg_c)
    #ccz = ZGate().control(2)
    #initialize
    qc.h([0,1,2])
    #oracle 1
    qc.x([0])
    qc.mcrz(stem*pi, [0, 1],2)
    qc.x([0])
    #oracle 2
    qc.x([1])
    qc.mcrz(fork*pi, [0, 1],2)
    qc.x([1])
    #oracle 3
    qc.mcrz(flower*pi, [0, 1],2)
    #Inversion
    qc.h([0,1,2])
    qc.x([0,1,2])
    #qc.append(ccz, range(3))
    qc.mcrz(pi/2, [0, 1],2)
    qc.x([0,1,2])
    qc.h([0,1,2])
    #measurement
    qc.measure([0,1,2],[0,1,2])
    #Run sim
    sim = Aer.get_backend('aer_simulator')
    qc_sim = qc.copy()
    qc_sim.measure_all()
    qobj = assemble(qc_sim)
    result = sim.run(qobj, shots=1).result()
    counts = result.get_counts(experiment=None)
    a = max(counts)
    b = a[-3:]
    if b == '111':
        return(1)
    elif b == '101':
        return(3)
    elif b == '110':
        return(2)
    else:
        stemQC(stem,fork,flower)

def NonJunctionEvolutionQuantum(BigPicture,N):
    """
    This takes in the state of the system at some iteration number N and gives the
    state of the system at N+1 but does not do the junction evolution

    This is the quantum evolution version. Use a modified Grover's algorithm to
    determine stem evolution

    I think this needs to be applied after the junction operations

    Params:
    ------------------
    BigPicture: a PictureHeight x PictureWidth array of ints
        it's the states of the system at 0 to N inclusive
        state of the N+1 system is still being defined
        Junction operatations have already been performed to
        partially determine the N+1 state

    N: an int less than PictureHeight
        refers to the Nth iteraction of the system

    Outputs:
    -------------------
    NewPicture: a PictureHeight x PictureWidth array of ints
        it's the states of the system at 0 to N+1 inclusive
        The N+1 state should now be fully determined

    """

    PictureWidth=np.shape(BigPicture)[1]
    NewPicture=BigPicture


    for counter in range(PictureWidth):
        #if the state of the N+1 cell has not already been decided then do this code
        if BigPicture[N+1,counter] ==-1:
            #if a cell is sky then the state above it is sky
            if BigPicture[N,counter]==0:
                NewPicture[N+1,counter]=0
            #if a cell is a flower then the state above it is sky
            if BigPicture[N,counter]==1:
                NewPicture[N+1,counter]=0
            #if a cell is a stem then it could be a flower, a stem or a junction
            if BigPicture[N,counter]==2:
                #replace this with Grover's later to make it quantum
                #and also implement bias
                x=random.randint(1,3)
                NewPicture[N+1,counter]=x
            if BigPicture[N,counter]==3:
                NewPicture[N+1,counter]=0

        #this is to avoid the situation where, due to a junction, the cell above a
        #stem is inappropriately set to be sky
        if NewPicture[N+1,counter] == 0 and NewPicture[N,counter] == 2:
            #replace this with Grover's later to make it quantum
            #and also implement bias

            #BLAYNEY'S QUANTUM CODE GOES HERE REPLACING RANDINT
            x=stemQC(0.7,0.9,0.7)
            NewPicture[N+1,counter]=x


    return NewPicture

def FiniteIterationQuantum(Seed, MAX):
    """
    This grows the plant for MAX number of steps. The plant might grow beyond
    that in which case this function will plot a bunch of empty sky

    this is the quantum version, we call the quantum function for state evolution

    -1 unassigned
    0 sky
    1 flower
    2 system
    3 junction

    Params:
    ------------------
    Seed: a 1 x PictureWidth array of ints
        the initial state the plant grows from
        the entries will be 0,1,2,3

    Output:
    ------------------
    BigPicture: a MAX+1 x PictureWidth array of ints
        A picture of the plant's growth
        It's MAX+1 because there is the initial state and then MAX iterations
    """

    PictureWidth=np.shape(Seed)[1]
    Output=np.zeros([1,PictureWidth])
    Output[0,:]=Seed[:]

    #this point of holding is the have Holding[0,:] be the Nth layer of the system and
    # use it to determine the N+1 layer which we store in Holding[1,:] and then we append
    #Holding[1,:] to BigPicture to build up the system layer by layer
    Holding=np.zeros([2,PictureWidth])-np.ones([2,PictureWidth])


    for counter in range(MAX):
        #this print statement is going to give you every interation except the
        #last one
        print(Output)
        print('--------')


        Holding[0,:]=Output[counter,:]
        Holding[1,:]=np.zeros(PictureWidth)-np.ones(PictureWidth)



        #Holding=Ben's Junction Code
        for i in range(PictureWidth):
            if Holding[0,i]==3:
                left,right=calculate_junction(1,1,1,1)
                Holding[1,(i-1)%PictureWidth]=left
                Holding[1,(i+1)%PictureWidth]=right

        Holding=NonJunctionEvolutionQuantum(Holding,0)

        Output=np.vstack([Output,Holding[1,:]])

    return Output

def UntilStopQuantum(Seed):
    """
    This grows the plant until it naturally stops. If the plant grows forever
    then this function will never stop

    this is the classical version, we call the classical function for state
    evolution

    -1 unassigned
    0 sky
    1 flower
    2 system
    3 junction

    Params:
    ------------------
    Seed: a 1 x PictureWidth array of ints
        the initial state the plant grows from
        the entries will be 0,1,2,3

    Output:
    ------------------
    BigPicture: a ? x PictureWidth array of ints
        it will be however tall as it needs to be for the plant to finish growing
        the topmost layer will be sky (or unassigned)
    """

    PictureWidth=np.shape(Seed)[1]
    Output=np.zeros([1,PictureWidth])
    Output[0,:]=Seed[:]

    #this point of holding is the have Holding[0,:] be the Nth layer of the system and
    # use it to determine the N+1 layer which we store in Holding[1,:] and then we append
    #Holding[1,:] to BigPicture to build up the system layer by layer
    Holding=np.zeros([2,PictureWidth])-np.ones([2,PictureWidth])


    #I use STOP later to control when the code stops
    STOP=0

    counter=0
    while STOP == 0:

        #this print statement is going to give you every interation except the
        #last one
        print(Output)
        print('--------')


        Holding[0,:]=Output[counter,:]
        Holding[1,:]=np.zeros(PictureWidth)-np.ones(PictureWidth)

        #Ben's junction evolution code
        for i in range(PictureWidth):
            if Holding[0,i]==3:
                left,right=calculate_junction(1,1,1,1)
                Holding[1,(i-1)%PictureWidth]=left
                Holding[1,(i+1)%PictureWidth]=right

        Holding=NonJunctionEvolutionQuantum(Holding,0)

        Output=np.vstack([Output,Holding[1,:]])

        #This bit says to stop unless there is a flower, stem, or junction
        STOP=1
        for counter2 in range(PictureWidth):
            if Holding[1,counter2] >0:
                STOP=0
                break

        counter=counter+1

    return Output
