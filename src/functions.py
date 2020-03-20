from math import exp


def fill_weights():
    file = open(r"PredictionNumber\weights.txt", "r")
    weights = []
    for line in file:
        weights.append([float(line[:-2])])
    return [weights]


def sigmoid(arg):
    return 1 / (1 + exp(-arg))
