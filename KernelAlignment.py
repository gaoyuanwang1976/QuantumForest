from qiskit_machine_learning.utils.loss_functions import KernelLoss
from qiskit_machine_learning.kernels import TrainableKernel
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector

from qiskit import assemble,Aer,execute
from typing import Sequence
import numpy as np
from sklearn.svm import SVC
import embedding
import time
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
        dis=np.zeros((len(data),len(data)))

        dis=data[:,None]-data[None,:]
        dis=np.linalg.norm(dis,axis=2)
        dis2=np.multiply(dis,dis)
        kmatrix_o=np.exp((-1./data.shape[1])*dis2)
        #kmatrix_o=normalize_kernel(kmatrix_o)

        # Get estimated kernel matrix after applying feature mapping
        mapped_data=evaluate_map(data,quantum_kernel,self.kwargs['n_output'])

        kmatrix=np.zeros((len(mapped_data),len(mapped_data)))
        dis_mapped=mapped_data[:,None]-mapped_data[None,:]
        dis_mapped_norm=np.linalg.norm(dis_mapped,axis=2)
        dis2_mapped=np.multiply(dis_mapped_norm,dis_mapped_norm)
        kmatrix=np.exp((-1./data.shape[1])*dis2_mapped)
        #print('data',data[:5],mapped_data[:5])
        #print('dis2_mapped',dis2_mapped[0])
        #print('norm',dis_mapped_norm[0])
        #print('kernel',kmatrix[0])
        #kmatrix=normalize_kernel(kmatrix)

        # Calculate loss
        #loss = -1.*frobenius_alignment(kmatrix,kmatrix_o)
        loss=L1Loss_matrix(kmatrix,kmatrix_o)
        
        print('loss',loss,'\n')
        return loss

#def normalize_kernel(kernel):
#    maxData=(max(kernel.flatten())).round(3)
#    minData=(min(kernel.flatten())).round(3)
#    kernel=(kernel-minData)/(maxData-minData)
#    return kernel

def frobenius_alignment(k1,k2):
    k1_conj=k1.conjugate()
    k1_k2_P=np.multiply(k1_conj,k2) # elementwise-product
    k1_k2_F=sum([sum(i) for i in k1_k2_P])
    k1_F=np.linalg.norm(k1, 'fro')
    k2_F=np.linalg.norm(k2, 'fro')
    alignment=k1_k2_F/k1_F/k2_F
    return alignment

def L1Loss_matrix(k1,k2):
    summe=0
    for sub1, sub2 in zip(k1, k2):
        for ele1, ele2 in zip(sub1, sub2):
            summe=summe+abs(ele2 - ele1)*1./len(sub1)/len(sub1)
    return summe

def evaluate_map(X,kernel,n_output):
    shots=1000
    ##n=2**n_output
    theta_params_optimized=list(kernel.training_parameter_binds.values())
    #print('theta',theta_params_optimized)
    n_layers_emb=1
    n_inputs=len(X[0])
    n=2**n_inputs
    mapped_X=[]
    backend=AerSimulator(method='statevector')
    for x_params in X:
        qc_optimized=embedding.ising_quantum_circuit(n_inputs,x_params,theta_params_optimized,n_layers_emb,n_output) # n_dummy is set 0 for drc
        job=execute(qc_optimized, backend,shots=shots)

        result = job.result()
        keys=result.get_counts().keys()

        new_data=[0]*n
        for k_bin in keys:
            k_int=int(k_bin, 2)
            new_data[k_int]=result.get_counts()[k_bin]*1./shots*np.pi
        ##mapped_X.append(new_data[:-1])
        mapped_X.append(new_data[:n_output])


    return np.array(mapped_X)

