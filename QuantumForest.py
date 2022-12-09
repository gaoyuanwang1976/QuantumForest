import numpy as np
import random 
import os

from qiskit import QuantumCircuit,assemble,Aer
from qiskit.circuit import ParameterVector
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM, TNC
from qiskit.providers.aer import AerSimulator

from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer

from math import comb
import os
abspath = os.path.abspath('__file__')
dname = os.path.dirname(abspath)
os.chdir(dname)

import argparse
#for performance metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc

import preprocessing
import embedding
import KernelAlignment
# %% parsing

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simulate a QNN with the appropriate hyperparameters.")
    parser.add_argument('-n','--num_layers_emb', required = False, type=int, help='determines the number of layers of embedding', default=1)
    parser.add_argument('-g', '--genomics_dataset', required=False, help='determines which number genomics dataset to run. defaults to dataset 1', default=1)
    parser.add_argument('--partition_size', required=False, help='sets partition size for splitting data into train, test, and validation sets (scales the partition_ratio arg)', default='max')
    parser.add_argument('--partition_ratio', required=False, type=str, help="governs the ration of partition sizes in the training, validation, and test sets. a list of the form [train, val, test]", default="0.4:0.3:0.3")
    parser.add_argument('--shuffle', required=False, type=bool, help='determines whether to shuffle data before alternating', default=False)
    parser.add_argument('--shuffleseed', required=False, type=int, help='a seed for use in shuffling the dataset, if left False and --shuffle=True, will be completely random', default=False)

    args = parser.parse_args()
    import_path='dataset/genomics_datasets_tpm_log'
    shuffle=args.shuffle
    shuffleseed = args.shuffleseed
    n_layers_emb = args.num_layers_emb

    dataset = "data_"+str(args.genomics_dataset)
    partition_size=args.partition_size
    if partition_size != 'max':
        parition_size = int(partition_size)
    ratio = args.partition_ratio.split(":")
    ratio = [float(entry) for entry in ratio]

    dataset = preprocessing.import_dataset(import_path,dataset,shuffle, shuffleseed)
    dataset = preprocessing.normalize_dataset(dataset)

    print(f"using dataset of length {len(dataset)}")
    if partition_size != 'max':
        partition_split = int(partition_size)
    else:
        partition_split=len(dataset)
    print(f'using partition size of {partition_split}')
    train_set, val_set, test_set = preprocessing.train_val_test(dataset, partition_split, ratio)
    print("for training:")
    preprocessing.get_info_g(train_set, True)
    print("for testing:")
    preprocessing.get_info_g(test_set, True)

    Xtrain, ytrain = preprocessing.convert_for_qiskit(train_set)
    Xval, yval = preprocessing.convert_for_qiskit(val_set)
    Xtest, ytest = preprocessing.convert_for_qiskit(test_set)

#######################
##### circuit def #####
#######################

    n_inputs = len(Xtrain[0])
    n_embedding_gates=comb(n_inputs,2)     #number of gates for ising_interaction (pair-wise) embedding, this number may change for another embedding
    n_gates_emb = (n_inputs+n_embedding_gates)*n_layers_emb
    x_params = ParameterVector('x',n_inputs)
    theta_emb_params = ParameterVector('theta_emb', n_gates_emb)
    qc=QuantumCircuit(n_inputs)
    
#### initialization ####
    from qiskit.quantum_info import Statevector
    state_vector=Statevector(qc)
    qc.initialize(state_vector,list(range(0,n_inputs)))
    qc.barrier()

#### ansatz ####
    n_extra_qubits=0
    embedding.ising_interaction(qc,x_params,theta_emb_params,n_inputs,n_layers_emb,n_extra_qubits) 
    qc.barrier()

############################
###### Quantum Kernel ######
############################


    backend=AerSimulator(method='statevector')
    quant_kernel = QuantumKernel(feature_map=qc,training_parameters=theta_emb_params,quantum_instance=backend)
    cb_qkt = embedding.QKTCallback()
    opt = SPSA(maxiter=10, callback=cb_qkt.callback)
    loss_func = KernelAlignment.KALoss()

    for epoch in range(1):
        qk_trainer = QuantumKernelTrainer(quantum_kernel=quant_kernel,loss=loss_func,optimizer=opt)
        qkt_results = qk_trainer.fit(Xtrain, ytrain)
        optimized_kernel = qkt_results.quantum_kernel
            
    sim = Aer.get_backend('qasm_simulator')
    theta_params_optimized=[]
    for index in range(n_gates_emb):
        theta_params_optimized.append(quant_kernel.training_parameter_binds[theta_emb_params[index]])
    for x_params in Xtest:
        qc_optimized=QuantumCircuit(n_inputs)
        state_vector=Statevector(qc_optimized)
        qc_optimized.initialize(state_vector,list(range(0,n_inputs)))
        qc_optimized.barrier()

        n_extra_qubits=0
        embedding.ising_interaction(qc_optimized,x_params,theta_params_optimized,n_inputs,n_layers_emb,n_extra_qubits) 
        qc_optimized.save_statevector()   # Tell simulator to save statevector
        qobj = assemble(qc_optimized)     # Create a Qobj from the circuit for the simulator to run
        result = sim.run(qobj).result()
        out_state = result.get_statevector()
        amplitude=np.absolute(out_state)
        print(amplitude)
    #for index in range(10):
    #    print(quant_kernel.user_param_binds[theta_emb_params[index]])