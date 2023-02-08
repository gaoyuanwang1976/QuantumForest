
#%% Imports

import numpy as np
import random
import torch

def array_to_dataset(array):
    """Converts data arrays into appropriate form for use in QNN. Assumes each entry in array is a seperate datapoint w the last column corresponding to label."""
    dataset = []
    for entry in array:
        input = torch.tensor(entry[:len(entry)-1])
        label = entry[len(entry)-1]
        dataset.append((input,label))
    return dataset


def import_dataset(dirname, filename, shuffle=False, shuffleseed=False):
    """
    Imports appropriately-formatted text matrix, converting to array then to dataset.
    Includes options to shuffle randomly or according to a given seed.
    """
    array = np.loadtxt(dirname+"/"+filename)

    if shuffle:
        if shuffleseed==False:
            np.random.shuffle(array)
        else:
            np.random.seed(shuffleseed)
            np.random.shuffle(array)
    return array_to_dataset(array)

def train_val_test(dataset, scale=100, ratio=[0.33,0.33,0.33]):
    """splits dataset into training, validation, and test partitions according to list 'ratio' scaled by 'scale'"""
    scale = int(scale) #no idea why this needs to be here because scale should already be an int, but for some reason it's a string
    if sum(ratio)*scale>len(dataset):
        return print("error with partition size")
    else:
        part1 = int(scale*ratio[0])
        part2 = part1 + int(scale*ratio[1])
        part3 = part2 + int(scale*ratio[2])
        train = dataset[:part1]
        val = dataset[part1:part2]
        test = dataset[part1:part3]
        return (train, val, test)

def get_info_g(dataset, verbose=False):
    """Determines the number of inputs labeled one and zero in a dataset."""
    zeros = 0
    ones = 0
    for data in dataset:
        input=data[:-1]
        label = data[-1]
        if label == 0:
            zeros+=1
        elif label ==1:
            ones+=1
    if verbose:
        print(f'In this dataset, there are {zeros} inputs labeled "0" and {ones} inputs labeled "1".')
    return (ones, zeros)

#genomics data comes pre-balanced, but i do have a balance_g alt function that's written in the genodock secion of qnn
# note that this is overruled by the alternate_g function, which is used by default
def balance_g(dataset):
    ones, zeros = get_info_g(dataset)
    if ones<zeros:
        length = ones
    else:
        length = zeros

    balanced = []
    one_n=0
    zero_n=0
    for item in dataset:
        label = item[-1]
        if label == 0 and zero_n<length:
            balanced.append(item)
            zero_n+=1
        if label == 1 and one_n<length:
            balanced.append(item)
            one_n+=1
    return balanced

def alternate_g(dataset):
    ones, zeros = sort_dataset(dataset)
    return coallated_dataset(ones, zeros)

def sort_dataset(dataset, a_label=1):
    labeled_a = []
    labeled_b = []
    for data in dataset:
        input, label = data
        if label == a_label:
            labeled_a.append(data)
        else:
            labeled_b.append(data)
    return (labeled_a, labeled_b)

def coallated_dataset(set1, set2):
    dataset = []
    if len(set1)<len(set2):
        length = len(set1)
    else:
        length = len(set2)
    for i in range(length):
        dataset.append(set1[i])
        dataset.append(set2[i])
    return dataset

def get_uq_g(dataset):
    """Removes all duplicate and conflicting inputs from a dataset."""
    uq = []
    num_0 = 0
    num_1 = 0
    num_overlap = 0
    dict = collections.defaultdict(set)
    for data, label in dataset:
        key = tuple(data.flatten().tolist())
        dict[key].add(label)
    for item in dataset:
        data, label = item
        key = tuple(data.flatten().tolist())
        if dict[key] == {0}:
            num_0 +=1
            uq.append(item)
        elif dict[key] == {1}:
            num_1 +=1
            uq.append(item)
        elif dict[key] == {0,1}:
            num_overlap +=1
        else:
            print("Error with item", item)
            break
    return uq



def round_yscore(yscore):
    rounded = []
    for item in yscore:
        rounded.append(round(item))
    return rounded

def normalize_dataset(dataset):
    X=[]
    y=[]
    for data,label in dataset:
        X.append(data.numpy())
        y.append(label)

    #X=np.around(X,1)
    X_t=np.array(X).T
    for i,feature in enumerate(X_t):
        ##maxData=(max(feature)).round(1)
        ##minData=(min(feature)).round(1)
        maxData=max(feature)
        minData=min(feature)
        #feature=(feature-minData)/(maxData-minData)*np.pi
        X_t[i]=(feature-minData)/(maxData-minData)*2*np.pi
    #X=(X-minData)/(maxData-minData)*np.pi
    X=X_t.T
    dataset=np.append(X.T,np.array([y]),axis=0).T

    return dataset

def normalize_feature(X):
    #X=np.around(X,1)
    X_t=np.array(X).T
    for i,feature in enumerate(X_t):
        #maxData=(max(feature)).round(1)
        #minData=(min(feature)).round(1)
        maxData=max(feature)
        minData=min(feature)
        X_t[i]=(feature-minData)/(maxData-minData)*2*np.pi
    X=X_t.T

def convert_for_qiskit(dataset):
    X = []
    y = []
    for data in dataset:
        input=data[:-1]
        #input = input.round(1)
        X.append(input)
        y.append(data[-1])
    X = np.array(X)
    y = np.array(y)
    return(X,y)

def divide_input(X,excess_group,n_groups,n_in_dim):
    n_data=len(X)
    while excess_group>0:
        X=np.vstack([X.T, [0]*n_data]).T
        excess_group=excess_group-1

    X_group=np.zeros((n_groups,n_data,n_in_dim))

    for g in range(n_groups):
        X_group[g]=X.T[g*n_in_dim:(g+1)*n_in_dim].T

    return X_group


def append_mapped(X,X_mapped):
    for index in range(len(X_mapped)):
        X[index].extend(X_mapped[index])
    return X
