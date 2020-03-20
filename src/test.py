from numpy import array, dot, exp



def sigmoid(arg):
    return 1 / (1 + exp(-arg))


def binGEN(n, s=''):
    global DATA_INPUTS

    if n == 0:
        DATA_INPUTS.append(array([int(char) for char in s]))
        return
    
    binGEN(n - 1, s + '0')
    binGEN(n - 1, s + '1')


FILE = open(r"D:\PredictionNumber\weights.txt", "r")
WEIGHTS = [[]]
for line in FILE:
    WEIGHTS[0].append([float(line[:-2])])
SIZE = len(WEIGHTS[0])

DATA_INPUTS = []
binGEN(SIZE)
OUTPUTS = []
for i in DATA_INPUTS:
    output = sigmoid(dot(i, WEIGHTS))
    OUTPUTS.append(int(*output[0].round()))

# testing
for i in range(SIZE):
    if DATA_INPUTS[i][0] != OUTPUTS[i]:
        print("ERROR in", i)
else:
    print("SUCCESS")
