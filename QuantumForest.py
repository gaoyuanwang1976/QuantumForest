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
    parser.add_argument('-nd','--num_layers_drc', required = False, type=int, help='determines the number of layers of dimension reduction kernel', default=1)
    parser.add_argument('-nc','--num_layers_emb', required = False, type=int, help='determines the number of layers of classification kernel', default=3)
    parser.add_argument('-g', '--genomics_dataset', required=False, help='determines which number genomics dataset to run. defaults to dataset 1', default=1)
    parser.add_argument('--partition_size', required=False, help='sets partition size for splitting data into train, test, and validation sets (scales the partition_ratio arg)', default='max')
    parser.add_argument('--partition_ratio', required=False, type=str, help="governs the ration of partition sizes in the training, validation, and test sets. a list of the form [train, val, test]", default="0.4:0.3:0.3")
    parser.add_argument('--shuffle', required=False, type=bool, help='determines whether to shuffle data before alternating', default=False)
    parser.add_argument('--shuffleseed', required=False, type=int, help='a seed for use in shuffling the dataset, if left False and --shuffle=True, will be completely random', default=False)
    parser.add_argument('-di','--dimension_in', required=False, type=int, help='input dimension of the dimension reduction', default=3)
    parser.add_argument('-do','--dimension_out', required=False, type=int, help='output dimension of the dimension reduction', default=1)
    parser.add_argument('-n','--num_auxiliary_qubits', required=False,help='number of auxiliary qubits',default=0)

    args = parser.parse_args()
    print("hyper parameters: ",args)
    import_path = 'dataset/QC_all_datasets/genomics/'
    shuffle=args.shuffle
    shuffleseed = args.shuffleseed
    n_layers_drc=args.num_layers_drc
    n_layers_emb = args.num_layers_emb
    n_in_dim = args.dimension_in
    n_out_dim = args.dimension_out
    n_extra_qubits=int(args.num_auxiliary_qubits)

    dataset = "data_"+str(args.genomics_dataset)
    partition_size=args.partition_size
    if partition_size != 'max':
        parition_size = int(partition_size)
    ratio = args.partition_ratio.split(":")
    ratio = [float(entry) for entry in ratio]

    dataset = preprocessing.import_dataset(import_path,dataset,shuffle, shuffleseed)
    dataset = preprocessing.alternate_g(dataset)
    #dataset = preprocessing.get_uq_g(dataset)
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

    n_total_feature=len(Xtrain[0])
    n_groups=math.ceil(n_total_feature/n_in_dim)


############################
###### Quantum Kernel dimension reduction
############################

    while n_groups > 1:
        excess_group=n_groups*n_in_dim-n_total_feature   ### the number of (empty) dimensions in the last group that need to be filled with zeros.

        Xtrain_mapped=[[] for _ in range(len(Xtrain))]
        Xtest_mapped=[[] for _ in range(len(Xtest))]
        Xval_mapped=[[] for _ in range(len(Xval))]

        ### add another dimension for different feature groups
        Xtrain_group=preprocessing.divide_input(Xtrain,excess_group,n_groups,n_in_dim)
        Xtest_group=preprocessing.divide_input(Xtest,excess_group,n_groups,n_in_dim)
        Xval_group=preprocessing.divide_input(Xval,excess_group,n_groups,n_in_dim)

        ### map the features in each group separately 
        for g_index in range(n_groups):
            print('group: ',g_index,'\n\n\n\n')
            n_inputs = len(Xtrain_group[g_index][0])
            n_drc_gates=comb(n_inputs,2)     #number of gates for ising_interaction (pair-wise) embedding, this number may change for another embedding
            n_gates_drc = (n_inputs+n_drc_gates)*n_layers_drc
            
            x_params_drc = ParameterVector('x_drc',n_inputs)
            theta_params_drc = ParameterVector('theta_drc', n_gates_drc)
            qc=embedding.ising_quantum_circuit(n_inputs,x_params_drc,theta_params_drc,n_layers_drc,n_out_dim)
            ##qc=embedding.rx_circuit(x_params_drc,n_inputs,n_extra_qubits,n_out_dim)
            backend=AerSimulator(method='statevector')
            ##quant_kernel = QuantumKernel(feature_map=qc,quantum_instance=backend)
            quant_kernel = QuantumKernel(feature_map=qc,training_parameters=theta_params_drc,quantum_instance=backend)
            cb_qkt_drc = embedding.QKTCallback()
            opt_drc = COBYLA(maxiter=100, rhobeg= 3)
            #SPSA(callback=cb_qkt_drc.callback)
            loss_func_drc = KernelAlignment.KALoss(n_output=n_out_dim)
            
            initial_params = [0]*len(theta_params_drc)
            #initial_params = np.random.rand(len(theta_params_drc))*np.pi*2
            for epoch in range(1):
                qk_trainer = QuantumKernelTrainer(quantum_kernel=quant_kernel,loss=loss_func_drc,optimizer=opt_drc, initial_point=initial_params)
                print('start fitting')
                qkt_results = qk_trainer.fit(Xtrain_group[g_index], ytrain)
                optimized_kernel = qkt_results.quantum_kernel
            ##optimized_kernel=quant_kernel
            print('start producing output')

            tmp_tr=KernelAlignment.evaluate_map(Xtrain_group[g_index],optimized_kernel,n_out_dim)
            Xtrain_mapped=preprocessing.append_mapped(Xtrain_mapped,tmp_tr)
            tmp_te=KernelAlignment.evaluate_map(Xtest_group[g_index],optimized_kernel,n_out_dim)
            Xtest_mapped=preprocessing.append_mapped(Xtest_mapped,tmp_te)
            tmp_va=KernelAlignment.evaluate_map(Xval_group[g_index],optimized_kernel,n_out_dim)
            Xval_mapped=preprocessing.append_mapped(Xval_mapped,tmp_va)
            print('origin: ',Xtrain_group[g_index][:5])
            print('mapped: ',Xtrain_mapped[:5])
        Xtrain=preprocessing.normalize_feature(np.array(Xtrain_mapped, dtype="object"))
        Xtest=preprocessing.normalize_feature(np.array(Xtest_mapped, dtype="object"))
        Xval=preprocessing.normalize_feature(np.array(Xval_mapped, dtype="object"))

        n_total_feature=len(Xtrain[0])
        n_groups=math.ceil(n_total_feature/n_in_dim)



##################
#### final classification, standard kernel/QNN classification
##################

    n_inputs = len(Xtrain[0])+n_extra_qubits
    n_external_inputs=len(Xtrain[0])
    n_embedding_gates=comb(n_inputs,2)     #number of gates for ising_interaction (pair-wise) embedding, this number may change for another embedding
    n_gates_emb = (n_inputs+n_embedding_gates)*n_layers_emb
    x_params = ParameterVector('x',n_external_inputs)
    theta_emb_params = ParameterVector('theta_emb', n_gates_emb)
    qc_classifier=QuantumCircuit(n_inputs)
    
    from qiskit.quantum_info import Statevector
    state_vector=Statevector(qc_classifier)
    qc_classifier.initialize(state_vector,list(range(0,n_inputs)))
    embedding.ising_interaction(qc_classifier,x_params,theta_emb_params,n_inputs,n_layers_emb,n_external_inputs,n_extra_qubits)
############################
###### Quantum Kernel classification
############################

    backend=AerSimulator(method='statevector')
    quant_kernel_classifier = QuantumKernel(feature_map=qc_classifier,training_parameters=theta_emb_params,quantum_instance=backend)
    cb_qkt = embedding.QKTCallback()
    opt = SPSA(maxiter=100, callback=cb_qkt.callback)
    loss_func = SVCLoss(C=1.0)

    # validation
    patience_counter = 0
    best_val_score = 0
    best_params = [random.uniform(0, np.pi)]*len(theta_emb_params)
    best_val_epoch = 0
    best_model=None
    test_loss_list=[]
    for epoch in range(1):
        qk_trainer = QuantumKernelTrainer(quantum_kernel=quant_kernel_classifier,loss=loss_func,optimizer=opt, initial_point=best_params)
        qkt_results = qk_trainer.fit(Xtrain, ytrain)
        optimized_kernel_classifier = qkt_results.quantum_kernel

        kernel_params=cb_qkt.get_callback_data()[1][-1]
        train_loss=loss_func(parameter_values=kernel_params,quantum_kernel=quant_kernel_classifier,data=Xtrain,labels=ytrain)
        test_loss=loss_func(parameter_values=kernel_params,quantum_kernel=quant_kernel_classifier,data=Xtest,labels=ytest)
        print('Hyper parameters: ',parser.parse_args())
        print('train: ',train_loss, 'test: ',test_loss)
        qsvc = QSVC(quantum_kernel=optimized_kernel_classifier,probability=True)
        qsvc.fit(Xtrain,ytrain)

        train_score=qsvc.score(Xtrain,ytrain)
        test_score=qsvc.score(Xtest,ytest)
        yscore = qsvc.predict_proba(Xtest)
        yscore=yscore.T[1].T
        fpr1, tpr1, thresholds = roc_curve(ytest, yscore)
        tn, fp, fn, tp = confusion_matrix(ytest, preprocessing.round_yscore(yscore)).ravel()
        AUROC = auc(fpr1, tpr1)
        print(f'  AUC: {AUROC}')
        print(f"  confusion matrix: [tn {tn}, fp {fp}, fn {fn}, tp {tp}]")
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        print(f'  precision = {precision}')
        print(f'  recall = {recall}')
        print("train score: ",train_score,"test score: ",test_score)