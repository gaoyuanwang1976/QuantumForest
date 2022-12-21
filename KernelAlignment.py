from qiskit_machine_learning.utils.loss_functions import KernelLoss
from qiskit_machine_learning.kernels import TrainableKernel
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator

from qiskit import assemble,Aer,execute
from typing import Sequence
import numpy as np
from sklearn.svm import SVC
import embedding

# KernelAlignment Loss
class KALoss(KernelLoss):

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Arbitrary keyword arguments to pass within
                      KALoss evaluation.
        """
        self.kwargs = kwargs

    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: TrainableKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Args:
            parameter_values: an array of values to assign to the user params
            quantum_kernel: A ``QuantumKernel`` object to evaluate
            data: An ``(N, M)`` matrix containing the data
                    ``N = # samples, M = dimension of data``
            labels: A length-N array containing the truth labels
        """
        # Bind training parameters
        quantum_kernel.assign_training_parameters(parameter_values)

        # Get kernel matrix of the input features, inner product is calculated
        kmatrix_o=np.zeros((len(data),len(data)))
        for index_a,a in enumerate(data):
            for index_b,b in enumerate(data):
                kmatrix_o[index_a][index_b]=np.dot(a, b)
        kmatrix_o=normalize_kernel(kmatrix_o)
        # Get estimated kernel matrix after applying feature mapping
        #If y_vec is None, self inner product is calculated. If using statevector_simulator, only build circuits for Ψ(x)|0⟩, then perform inner product classically.

        mapped_data=evaluate_map(data,quantum_kernel,self.kwargs['n_output'])
        #mapped_data=data
        kmatrix=np.zeros((len(mapped_data),len(mapped_data)))
        for index_a,a in enumerate(mapped_data):
            for index_b,b in enumerate(mapped_data):
                kmatrix[index_a][index_b]=np.dot(a, b)

        kmatrix=normalize_kernel(kmatrix)
    
        # Calculate loss
        loss = -1.*frobenius_alignment(kmatrix,kmatrix_o)
 
        return loss

def normalize_kernel(kernel):
    maxData=(max(kernel.flatten())).round(3)
    minData=(min(kernel.flatten())).round(3)
    kernel=(kernel-minData)/(maxData-minData)
    return kernel

def frobenius_alignment(k1,k2):
    k1_conj=k1.conjugate()
    k1_k2_P=np.multiply(k1_conj,k2) # elementwise-product
    k1_k2_F=sum([sum(i) for i in k1_k2_P])
    k1_F=np.linalg.norm(k1, 'fro')
    k2_F=np.linalg.norm(k2, 'fro')
    alignment=k1_k2_F/k1_F/k2_F
    return alignment


def evaluate_map(X,kernel,n_output):
    shots=512
    n=2**n_output
    theta_params_optimized=list(kernel.training_parameter_binds.values())
    n_layers_emb=1
    n_inputs=len(X[0])
    
    mapped_X=[]
    backend=AerSimulator(method='statevector')
    for x_params in X:
        qc_optimized=embedding.ising_quantum_circuit(n_inputs,x_params,theta_params_optimized,n_layers_emb,n_output)
        job=execute(qc_optimized, backend,shots=shots)
        result = job.result()
        keys=result.get_counts().keys()
        new_data=[0]*n
        for k_bin in keys:
            k_int=int(k_bin, 2)
            #print(result.get_counts()[k_bin])
            new_data[k_int]=result.get_counts()[k_bin]*1./shots
        mapped_X.append(new_data[:-1])
    return mapped_X

