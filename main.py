import neuralnetwork_class as N
import numpy as np
import time

Z = 1
d = 10


def main():

    np.set_printoptions( precision=4 )

    arr = np.array( [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]] )
    O_array = np.array( [0, 0, 0, 1] )

    NeuralNetwork_1 = N.NeuralNetwork( Z, d, arr, O_array )

    NeuralNetwork_1.add_neuron()
    NeuralNetwork_1.neurons[0].add_dendrite('W13')
    NeuralNetwork_1.neurons[0].add_dendrite('W23')
    NeuralNetwork_1.neurons[0].connect_z_with_neuron('W03')

    NeuralNetwork_1.add_neuron()
    NeuralNetwork_1.neurons[1].add_dendrite('W14')
    NeuralNetwork_1.neurons[1].add_dendrite('W24')
    NeuralNetwork_1.neurons[1].connect_z_with_neuron('W04')


    NeuralNetwork_1.sigmoeidis_function()





if __name__ == "__main__":

    time_started = time.perf_counter()
    main()
    time_finished = time.perf_counter()

    print(f"The program has taken {time_finished - time_started} second(s) to find a solution!")