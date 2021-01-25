import os
from numpy import exp, array, random, dot

os.chdir('..')
SIZE = 8
WEIGHTS_PATH = os.getcwd() + r"\weights.txt"


# activation function
def sigmoid(arg):
    return 1 / (1 + exp(-arg))


# reading weights
def read_weights():
    with open(WEIGHTS_PATH, 'r', encoding='UTF-8') as file:
        weights = [[]]
        for line in file:
            weights[0].append([float(line[:-2])])
    return weights


# writing weights to a file for further use
def write_weights(weights):
    with open(WEIGHTS_PATH, 'w', encoding='UTF-8') as file:
        for weight in weights:
            file.write(str(*weight) + "\n")
