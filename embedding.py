import torch
from torch.autograd import Function
import torch.nn as nn

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.extensions import UnitaryGate
import numpy as np
import pandas as pd

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

def ising_interaction(qc,x_params,theta,n_inputs,n_layers,n_extra_qubits):
    param_index=0
    for layer in range(n_layers):
        for i in range(n_inputs):
            qc.rx(x_params[i], i)
        for j in range(n_extra_qubits):
            qc.h(n_inputs+j)
            
        for q1 in range(n_inputs):
            for q2 in range(q1,n_inputs):
                if q1!=q2:
                    qc.rzz(theta[param_index],q1,q2)
                    param_index+=1
        for q_tmp in range(n_inputs):
            qc.ry(theta[param_index],q_tmp)
            param_index+=1

    for i in range(n_inputs):
        qc.rx(x_params[i], i)
    for j in range(n_extra_qubits):
        qc.h(n_inputs+j)
    return qc

def rx_kernel(qc,x_params,n_external_inputs,n_extra_qubits):
    for i in range(n_external_inputs):
        qc.rx(x_params[i], i)
    for j in range(n_extra_qubits):
        qc.h(n_external_inputs+j)