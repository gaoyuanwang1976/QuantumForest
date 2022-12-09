from qiskit_machine_learning.utils.loss_functions import KernelLoss
from qiskit_machine_learning.kernels import TrainableKernel
from typing import Sequence
import numpy as np
from sklearn.svm import SVC

# KernelAlignment Loss
class KALoss(KernelLoss):

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Arbitrary keyword arguments to pass to SVC constructor within
                      SVCLoss evaluation.
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
        kmatrix = quantum_kernel.evaluate(np.array(data)) 
        kmatrix=normalize_kernel(kmatrix)
    
        # Calculate loss
        loss = -1.*frobenius_alignment(kmatrix,kmatrix_o)

        return loss

def normalize_kernel(kernel):
    maxData=(max(kernel.flatten())).round(3)
    minData=(min(kernel.flatten())).round(3)
    kernel=(kernel-minData)/(maxData-minData)*np.pi
    return kernel

def frobenius_alignment(k1,k2):
    k1_conj=k1.conjugate()
    k1_k2_P=np.multiply(k1_conj,k2) # elementwise-product
    k1_k2_F=sum([sum(i) for i in k1_k2_P])
    k1_F=np.linalg.norm(k1, 'fro')
    k2_F=np.linalg.norm(k2, 'fro')
    alignment=k1_k2_F/k1_F/k2_F
    return alignment
