import torch
from torch.autograd import Function
import torch.nn as nn
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Statevector

from qiskit import QuantumCircuit,QuantumRegister, Aer, execute
import numpy as np
import pandas as pd
import qiskit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.visualization import dag_drawer
from collections import OrderedDict

#%% YplusZ_gate definition (for use in y-measurement)
yz_mtx = np.zeros((2,2), dtype=complex)
yz_mtx[0][0] = 1
yz_mtx[0][1] = -1j
yz_mtx[1][0] = 1j
yz_mtx[1][1] = -1
yz_mtx = yz_mtx*(1/np.sqrt(2)) #making sure the matrix is unitary
YplusZ_gate = UnitaryGate(yz_mtx, label="Y+Z") #casts the matrix as a qiskit circuit gate


class QKTCallback:
    """Callback wrapper class."""
    def __init__(self) -> None:
        self._data = [[] for i in range(5)]

    def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
        """
        x[0]: number of function evaluations
        x[1]: the parameters
        x[2]: the function value
        x[3]: the stepsize
        x[4]: whether the step was accepted
        """
        self._data[0].append(x0)
        self._data[1].append(x1)
        self._data[2].append(x2)
        self._data[3].append(x3)
        self._data[4].append(x4)

    def get_callback_data(self):
        return self._data

    def clear_callback_data(self):
        self._data = [[] for i in range(5)]



def ising_interaction(qc,x_params,theta,n_inputs,n_layers,n_external_inputs,n_extra_qubits):
    typ=isinstance(theta,list)
    param_index=0
    for layer in range(n_layers):
        for i in range(n_external_inputs):
            qc.rx(x_params[i], i)
        for j in range(n_extra_qubits):
            qc.h(n_external_inputs+j)
        for q1 in range(n_inputs):
            for q2 in range(q1,n_inputs):
                if q1!=q2:
                    qc.rzz(theta[param_index],q1,q2)
                    param_index+=1
        #for q1 in range(n_inputs):
        #    for q2 in range(q1,n_inputs):
        #        if q1!=q2:
        #            qc.rzx(theta[param_index],q1,q2)
        #            param_index+=1

        for q_tmp in range(n_inputs):
            qc.ry(theta[param_index],q_tmp)
            param_index+=1
 
    for i in range(n_external_inputs):
        qc.rx(x_params[i], i)
    for j in range(n_extra_qubits):
        qc.h(n_external_inputs+j)

    return qc



def rx_kernel(qc,x_params,n_external_inputs,n_extra_qubits):
    for i in range(n_external_inputs):
        qc.rx(x_params[i], i)
    for j in range(n_extra_qubits):
        qc.h(n_external_inputs+j)


def rx_circuit(x_params,n_inputs,n_extra_qubits,n_output):
    qc=QuantumCircuit(n_inputs,n_output)
    state_vector=Statevector(qc)
    qc.initialize(state_vector,list(range(0,n_inputs)))
    qc.barrier()
    rx_kernel(qc,x_params,n_inputs,n_extra_qubits)
    qc.barrier()
    qc.measure(list(range(n_output)),list(range(n_output)))
    return qc


def ising_quantum_circuit(n_inputs,x_params,theta_emb_params,n_layers_emb,n_output):
    n_output=n_inputs

    qc=QuantumCircuit(n_inputs,n_output)
    state_vector=Statevector(qc)
    qc.initialize(state_vector,list(range(0,n_inputs)))
    qc.barrier()
    n_dummy=0
    n_feature=n_inputs-n_dummy
    ising_interaction(qc,x_params,theta_emb_params,n_inputs,n_layers_emb,n_feature,0) #set n_dummy=0 in drc
    qc.barrier()
    qc.measure(list(range(n_output)),list(range(n_output)))

    return qc

