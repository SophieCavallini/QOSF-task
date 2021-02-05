# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:37:40 2021

@author: Sophie Cavallini
"""

import numpy as np
import math
import cmath
import numpy.random

# Qubit in |0> state (100% probability of measuring 0)
q0 = [1, 0]

# Qubit in |1> state (100% probability of measuring 1)
q1 = [0, 1] 

# Qubit |+> state (superposition: 50% probability of measuring 0 and 50% probability of measuring 1)
q2 = [0.7071067811865475, 0.7071067811865475]

# Qubit |-> state (superposition: 50% probability of measuring 0 and 50% probability of measuring 1) with phase pi
q3 = [0.7071067811865475, -0.7071067811865475]

# Qubit |i> state (superposition: 50% probability of measuring 0 and 50% probability of measuring 1) with phase pi/2
q3 = [0.7071067811865475, 0+0.7071067811865475j]

# Qubit |-i> state (superposition: 50% probability of measuring 0 and 50% probability of measuring 1) with phase -pi/2
q4 = [0.7071067811865475, 0-0.7071067811865475j]

#Define X (NOT) gate:
X = np.array([[0, 1], [1, 0]])

#Define Z gate
Z = np.array([[1, 0], [0, -1]])

# Define H (Hadamard) gate:
H = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]])

#Define Y gate:
Y = np.array([[0, -1j], [1j, 0]])

#Define S gate
S = np.array([[1, 0], [0, cmath.exp(1j*math.pi/2)]])

#Define T gate
T = np.array([[1, 0],[0, cmath.exp(1j*math.pi/4)]])

#Define CNOT gate
CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

#Define SWAP gate
SWAP = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]

# Define projection operator |0><0|
P0x0 = np.array([[1, 0],[0, 0]])

# Define projection operator |1><1|
P1x1 = np.array([[0, 0],[0, 1]])

#Define identity matrix
I = np.identity(2)



def get_ground_state(num_qubits):
    #return vector of size 2**num_qubits with all zeroes except first element which is 1
    vq = np.zeros_like(np.arange(2**num_qubits))
    vq[0] = 1
    return vq

def cnot(control, target, num_qubits):
    # return unitary operator of size 2**n x 2**n for cnot gate given the control and target qubits
    if(control>target):
        #if the control qubit is larger than the target one it will compute the unitary operator with target and control inverted and then
        #rotate and traspose the matrix to obtain the one required
        return np.rot90(cnot(target, control, num_qubits),2).T
    #P0x0 and P1x1 need to be in position: control (so if control = 0 then they will be the first factor of the tensor product)
    if (control == 0):
        O1 = P0x0
        O2 = P1x1
    else:
        O1 = I
        for i in range(1, control):
            O1 = np.kron(O1, I)
        O2 = np.kron(O1, P1x1)
        O1 = np.kron(O1, P0x0)
    #in position: target, for O2 there will be the X unitary, will for O1 there will be simply the identity matrix
    for i in range(1, np.abs(target-control)):
        O1 = np.kron(O1, I)
        O2 = np.kron(O2, I)
    O1 = np.kron(O1, I)
    O2 = np.kron(O2, X)
    for i in range(1, num_qubits-target):
        O1 = np.kron(O1, I)
        O2 = np.kron(O2, I)
    O = O1 + O2
    return O

def get_operator(total_qubits, gate_unitary, target_qubits):
    # return unitary operator of size 2**n x 2**n for given single-qubit gate and target qubits
    I = np.identity(2)
    Ik = I
    if target_qubits==0:
        O=gate_unitary
    else:
        for i in range(1, target_qubits-1):
            Ik = np.kron(Ik, I)
        O = np.kron(Ik, gate_unitary) 
    for i in range(1, total_qubits-target_qubits):
        O = np.kron(O, I)
    return O

def find_unitary(name_gate):
    # return the unitary matrix of single-qubit elementary gate, given the gate name
    if(name_gate == 'x'): return X
    if(name_gate == 'z'): return Z
    if(name_gate == 'h'): return H
    if(name_gate == 'y'): return Y
    if(name_gate == 's'): return S
    if(name_gate == 't'): return T
    else:
        print('The gate: ', name_gate, 'does not exists, yet, the program will go to the next one' )
        return I

def u3_unitary(param, gparam):
    # return the unitary of a parametric gate given the parameters
    
    #if there are global parameters it will used them, else it will you use the parameter given in the circuit
    if(param["theta"]=="global_1"): theta = gparam["global_1"]
    else: theta = param["theta"]
    if(param["phi"]=="global_2"): phi = param["global_2"]
    else: phi = param["phi"]
    lam = param["lambda"]
    U = [[math.cos(theta/2), -cmath.exp(1j * lam) * math.sin(theta / 2)], 
         [cmath.exp(1j * phi) * math.sin(theta / 2), cmath.exp(1j * lam + 1j * phi) * math.cos(theta / 2)]]
    return U

def run_program(initial_state, program, gparams):
    size = len(initial_state)
    num_qubits = int(math.log2(size))
    final_state = initial_state
    for i in range(0, len(program)):
        if(len((program[i])["target"]) == 1):
            target = ((program[i])["target"])[0]
            if((program[i])["gate"]=="u3"):
                #for parametric gates
                unitary = u3_unitary((program[i])["params"], gparams)
            else:
                #for elementary gates
                unitary = find_unitary((program[i])["gate"])
            operator = get_operator(num_qubits, unitary, target)
        elif((program[i])["gate"] == 'cx'):
            #if the gate is cnot it will compute the operator via the specific cnot function (it will not use get_operator)
            operator = cnot(((program[i])["target"])[0], ((program[i])["target"])[1], num_qubits)
        else:
            print('The gate: ', (program[i])["gate"], 'does not exists, yet, the program will go to the next one' )
        final_state = np.dot(final_state, operator)
    
    return final_state

def measure_all(state_vector):
    # choose element from state_vector using weighted random and return it's index
    pstate_vector = np.abs(state_vector)**2
    a = numpy.random.rand(1)[0]
    i = 0
    prec = 0
    while a > pstate_vector[i]+prec:
        prec = prec + pstate_vector[i]
        i = i+1
    return i

def get_counts(state_vector, num_shots):
    # simply execute measure_all in a loop num_shots times and return object with statistics
    occurences = np.zeros_like(np.arange(len(state_vector)))
    for i in range(0, num_shots):
        place = measure_all(state_vector)
        occurences[place] = occurences[place]+1
    # convert the array in a dictionary, so it can be easier to understand the output    
    odict = {}
    type(odict)
    for i in range(0, len(occurences)):
        if occurences[i]>0: odict["{0:b}".format(i)] = occurences[i]
    return odict
