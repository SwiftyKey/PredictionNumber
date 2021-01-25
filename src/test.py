from global_params import *


def binGEN(n, s=''):
    global DATA_INPUTS

    if n == 0:
        DATA_INPUTS.append(array([int(char) for char in s]))
        return

    binGEN(n - 1, s + '0')
    binGEN(n - 1, s + '1')


# reading weights
WEIGHTS = read_weights()

DATA_INPUTS = []
binGEN(SIZE)
OUTPUTS = []
for i in DATA_INPUTS:
    OUTPUTS.append(int(*sigmoid(dot(i, WEIGHTS))[0].round()))

# testing
for i in range(2 ** SIZE):
    if DATA_INPUTS[i][0] != OUTPUTS[i]:
        print("ERROR in", i)
else:
    print("SUCCESS")
