import numpy as np
import math
import random 
import os
import qiskit
from qiskit import QuantumCircuit,assemble,Aer,execute
from qiskit.circuit import ParameterVector,QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM, TNC
from qiskit.providers.aer import AerSimulator

from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer

from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils.loss_functions import SVCLoss


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
    parser.add_argument('-di','--dimension_in', required=False, type=int, help='input dimension of the dimension reduction', default=3)
    parser.add_argument('-do','--dimension_out', required=False, type=int, help='output dimension of the dimension reduction', default=1)

    args = parser.parse_args()
    import_path='dataset/genomics_datasets_tpm_log'
    shuffle=args.shuffle
    shuffleseed = args.shuffleseed
    n_layers_emb = args.num_layers_emb
    n_in_dim = args.dimension_in
    n_out_dim = args.dimension_out

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
##### divide input ####
#######################



#######################
##### circuit def #####
#######################
    n_etage=2
    for etage in range(n_etage):

        n_total_feature=len(Xtrain[0])
        n_groups=math.ceil(n_total_feature/n_in_dim)
        excess_group=n_groups*n_in_dim-n_total_feature

        Xtrain_mapped=[[] for _ in range(len(Xtrain))]#np.zeros((len(Xtrain),n_groups*n_out_dim))
        Xtest_mapped=[[] for _ in range(len(Xtest))]#np.zeros((len(Xtest),n_groups*n_out_dim))
        Xval_mapped=[[] for _ in range(len(Xval))]#np.zeros((len(Xval),n_groups*n_out_dim))

        Xtrain_group=preprocessing.divide_input(Xtrain,excess_group,n_groups,n_in_dim)
        Xtest_group=preprocessing.divide_input(Xtest,excess_group,n_groups,n_in_dim)
        Xval_group=preprocessing.divide_input(Xval,excess_group,n_groups,n_in_dim)

        for g_index in range(n_groups):
            n_inputs = len(Xtrain_group[g_index][0])
            n_embedding_gates=comb(n_inputs,2)     #number of gates for ising_interaction (pair-wise) embedding, this number may change for another embedding
            n_gates_emb = (n_inputs+n_embedding_gates)*n_layers_emb
            x_params = ParameterVector('x',n_inputs)
            theta_emb_params = ParameterVector('theta_emb', n_gates_emb)
            qc=embedding.ising_quantum_circuit(n_inputs,x_params,theta_emb_params,n_layers_emb,n_out_dim)

            backend=AerSimulator(method='statevector')
            quant_kernel = QuantumKernel(feature_map=qc,training_parameters=theta_emb_params,quantum_instance=backend)
            cb_qkt = embedding.QKTCallback()
            opt = SPSA(maxiter=2, callback=cb_qkt.callback)
            loss_func = KernelAlignment.KALoss(n_output=n_out_dim)

            for epoch in range(1):
                qk_trainer = QuantumKernelTrainer(quantum_kernel=quant_kernel,loss=loss_func,optimizer=opt)
                print('start fitting')
                qkt_results = qk_trainer.fit(Xtrain_group[g_index], ytrain)
                optimized_kernel = qkt_results.quantum_kernel
            
            print('start producing output')

            tmp_tr=KernelAlignment.evaluate_map(Xtrain_group[g_index],optimized_kernel,n_out_dim)
            Xtrain_mapped=preprocessing.append_mapped(Xtrain_mapped,tmp_tr)
            tmp_te=KernelAlignment.evaluate_map(Xtest_group[g_index],optimized_kernel,n_out_dim)
            Xtest_mapped=preprocessing.append_mapped(Xtest_mapped,tmp_te)
            tmp_va=KernelAlignment.evaluate_map(Xval_group[g_index],optimized_kernel,n_out_dim)
            Xval_mapped=preprocessing.append_mapped(Xval_mapped,tmp_va)
            print("hi",tmp_va)


        print("Xval",Xval)
        Xtrain=np.array(Xtrain_mapped, dtype="object")
        Xtest=np.array(Xtest_mapped, dtype="object")
        Xval=np.array(Xval_mapped, dtype="object")
        print("Mapped",Xval_mapped)


