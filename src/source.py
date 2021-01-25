from global_params import *

DATA_TRAIN_INPUTS = array([[0, 0, 1, 0, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 1, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0],
                           [0, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0, 0],
                           [1, 0, 1, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0]])
DATA_TRAIN_OUTPUTS = array([[0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1]]).T

random.seed(1)
SYNAPSES_WEIGHTS = 2 * random.random((SIZE, 1)) - 1
OUTPUTS = [[]]

# neural network training
for _ in range(1000000):
    input_layer = DATA_TRAIN_INPUTS
    OUTPUTS = sigmoid(dot(input_layer, SYNAPSES_WEIGHTS))
    # weight adjustment
    SYNAPSES_WEIGHTS += dot(input_layer.T,
                            (DATA_TRAIN_OUTPUTS - OUTPUTS) * (OUTPUTS * (1 - OUTPUTS)))

# writing weights to a file for further use
write_weights(SYNAPSES_WEIGHTS)
