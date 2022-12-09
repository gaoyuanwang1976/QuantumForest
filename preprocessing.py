
#%% Imports

import numpy as np
import random

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
    return array

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
    ratio = (zeros-ones)/(zeros)
    balanced = []
    for item in dataset:
        data, label = item
        if label == 0:
            if random.random()>ratio:
                balanced.append(item)
        else:
            balanced.append(item)
    return balanced



def round_yscore(yscore):
    rounded = []
    for item in yscore:
        rounded.append(round(item))
    return rounded

def normalize_dataset(dataset):
    X = (dataset.T[:-1]).T
    y=dataset.T[-1]
    maxData=(max(X.flatten())).round(3)
    minData=(min(X.flatten())).round(3)
    X=(X-minData)/(maxData-minData)*np.pi
    dataset=np.append(X.T,np.array([y]),axis=0).T

    return dataset

def convert_for_qiskit(dataset):
    X = []
    y = []
    for data in dataset:
        input=data[:-1]
        input = input.round(3)
        X.append(input)
        y.append(data[-1])
    X = np.array(X)
    y = np.array(y)
    return(X,y)
