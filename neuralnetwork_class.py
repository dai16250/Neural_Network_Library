import random
import time
import numpy as np
import math

random.seed( time.time() )


def sumOfList(list, size):
    if size == 0:
        return 0
    else:
        return list[size - 1] + sumOfList( list, size - 1 )


def vimatoeidis(sum_of_weights):
    return 1 if sum_of_weights > 0 else 0


def sigmoeidis(total):
    return 1 / (1 + math.exp( -total ))


def dispatch_dict(method, sum_of_weights):
    return {

        'vimatoeidis': lambda: vimatoeidis( sum_of_weights ),
        'sigmoeidis':  lambda: sigmoeidis( sum_of_weights ),

    }.get( method, lambda: None )()


class NeuralNetwork:

    def __init__(self, Z, d, table, O_array):
        self.neurons = []
        self.table = np.array( table )
        self.o_array = np.array( O_array )
        self.Z = Z
        self.d = d
        self.array_of_weights = np.array( [] )
        self.array_of_DW = np.array( [[]] )
        self.last_seasons_weight_array = None
        self.new_seasons_weight_array = None
        self.arr = None
        self.Tmp_Array = None
        self.array_with_z = None
        self.a = None
        self.b = None
        self.method = None

    def initialize_array_of_weights(self):
        for neuron in self.neurons:
            for dendrites in neuron.get_dendrites():
                self.array_of_weights = np.concatenate( (self.array_of_weights, dendrites.get_value()), axis=None )


        for _ in self.array_of_weights:
            self.array_of_DW = np.concatenate( (self.array_of_DW, 0), axis=None )

    def clean_history_of_season_weight_arrays(self):
        self.last_seasons_weight_array = None
        self.new_seasons_weight_array = None

    def append_previous_weight_array(self):
        tmp = np.array( [[]] )

        for items in self.array_of_weights:
            tmp = np.concatenate( (tmp, [items]), axis=None )

        if self.last_seasons_weight_array is None:
            self.last_seasons_weight_array = np.array( [tmp] )
        else:
            self.last_seasons_weight_array = np.insert( self.last_seasons_weight_array,
                                                        len( self.last_seasons_weight_array ) - 1, tmp, axis=0 )

    def create_header(self):

        header = []
        for i, _ in enumerate(self.array_of_weights):
            header.append(f'---X{i}---|')

        for i, _ in enumerate( self.array_of_weights ):
            header.append(f'---W{i}---|')

        header.append(f'---S---|')
        header.append(f'---a---|')
        header.append(f'---o---|')

        for i, _ in enumerate( self.array_of_weights ):
            header.append(f'---DW{i}---|')

        if self.method == "sigmoeidis":
            header.append(f'---(a-o)^2)---|')
            header.append(f'---sum((a-o)^2))---|')
            header.append(f'---S|w|---|')


        return [' '.join(head) for head in header]


    def add_neuron(self):
        self.neurons.append( NeuralNetwork.Neuron( self.table.shape[1] ) )

    def sum(self, array_with_z, ):
        S = 0
        for i, weigh in enumerate( self.array_of_weights ):
            S += weigh * array_with_z[i]
        self.Tmp_Array = np.concatenate( (self.Tmp_Array, [S]), axis=None )
        return S

    def initialize_a_b(self):
        self.a = np.array( [[]] )
        self.b = np.array( [[]] )

    def sigmoeidis_function(self):
        global sum_of_squared
        self.initialize_array_of_weights()
        Queue = list()
        self.method = "sigmoeidis"

        times = 0
        while True:

            self.initialize_a_b()
            self.clean_history_of_season_weight_arrays()

            for counter in range( self.table.shape[0] ):
                times += 1

                tmp_items = np.array( [[self.table[counter]]] )
                self.array_with_z = np.concatenate( (tmp_items, [self.Z]), axis=None )
                self.Tmp_Array = np.array( self.array_with_z )
                self.Tmp_Array = np.concatenate( (self.Tmp_Array, self.array_of_weights), axis=None )

                S = self.sum( self.array_with_z )

                x = dispatch_dict( self.method, S )

                self.a = np.concatenate( [self.a, x], axis=None )
                self.b = np.concatenate( [self.b, self.o_array[counter]], axis=None )

                self.Tmp_Array = np.concatenate( (self.Tmp_Array, [x]), axis=None )
                self.Tmp_Array = np.concatenate( (self.Tmp_Array, [self.o_array[counter]]), axis=None )

                self.delta_rule( x,counter)
                self.append_previous_weight_array()

                self.Tmp_Array = np.concatenate( (self.Tmp_Array, self.array_of_DW), axis=None )

                error_squared = math.pow( x - self.o_array[counter], 2 )
                Queue.append( error_squared )

                self.Tmp_Array = np.concatenate( (self.Tmp_Array, [error_squared]), axis=None )

                sum_of_squared = sumOfList( Queue, len( Queue ) )

                self.Tmp_Array = np.concatenate( (self.Tmp_Array, sum_of_squared), axis=None )

                SW = [math.fabs(weight) for weight in self.array_of_weights]
                self.Tmp_Array = np.concatenate( (self.Tmp_Array, [sum(SW)]), axis=None )

                if len(Queue) > 3:
                    Queue.pop(0)

                if self.arr is None:
                    self.arr = np.array( [self.Tmp_Array] )
                else:
                    self.arr = np.insert( self.arr, len( self.arr ), [self.Tmp_Array], axis=0 )

            if sum_of_squared <= 0.001:
                break


        print( f"A solution has been found in {int( len( self.arr ) / 4 )} seasons and the final weights are:" )
        print( self.array_of_weights )


        with open( 'results.txt', 'w' ) as f:
            np.savetxt( f, self.arr, header=str( self.create_header()), fmt="%f", comments='' )


    def vimatoeidis_function(self):

        self.initialize_array_of_weights()

        self.method = "vimatoeidis"

        while True:

            self.initialize_a_b()
            self.clean_history_of_season_weight_arrays()

            for counter in range( self.table.shape[0] ):

                tmp_items = np.array( [[self.table[counter]]] )
                self.array_with_z = np.concatenate( (tmp_items, [self.Z]), axis=None )
                self.Tmp_Array = np.array( self.array_with_z )
                self.Tmp_Array = np.concatenate( (self.Tmp_Array, self.array_of_weights), axis=None )

                S = self.sum( self.array_with_z )

                x = dispatch_dict( self.method, S )

                self.a = np.concatenate( [self.a, x], axis=None )
                self.b = np.concatenate( [self.b, self.o_array[counter]], axis=None )

                self.Tmp_Array = np.concatenate( (self.Tmp_Array, [x]), axis=None )
                self.Tmp_Array = np.concatenate( (self.Tmp_Array, [self.o_array[counter]]), axis=None )

                self.delta_rule( x, counter)
                self.append_previous_weight_array()

                self.Tmp_Array = np.concatenate( (self.Tmp_Array, self.array_of_DW), axis=None )

                if self.arr is None:
                    self.arr = np.array( [self.Tmp_Array] )
                else:
                    self.arr = np.insert( self.arr, len( self.arr ), [self.Tmp_Array], axis=0 )


            if np.array_equal( self.a, self.b ) or np.array_equal( self.new_seasons_weight_array,
                                                                   self.last_seasons_weight_array ):
                break

        print( f"A solution has been found in {int( len( self.arr ) / 4 )} seasons and the final weights are:" )
        print( self.array_of_weights )



        with open( 'results.txt', 'w' ) as f:
            np.savetxt( f, self.arr, header=str( self.create_header()), fmt="%f", comments='' )

    def delta_rule(self, A, x):

        tmp = np.array( [[]] )

        for i, z in enumerate( self.array_with_z):

            if self.method == "vimatoeidis":
                self.array_of_DW[i] = self.d * (A - self.o_array[x]) * z * (-1)
            elif self.method == "sigmoeidis":
                self.array_of_DW[i] = self.d * (A - self.o_array[x]) * z * A * (1-A) * (-1)

        for i, (weight, DW) in enumerate( zip( self.array_of_weights, self.array_of_DW ) ):
            tmp = np.concatenate( (tmp, [weight]), axis=None )
            self.array_of_weights[i] = weight + DW

        if self.new_seasons_weight_array is None:
            self.new_seasons_weight_array = np.array( [tmp] )
        else:
            self.new_seasons_weight_array = np.insert( self.new_seasons_weight_array,
                                                       len( self.new_seasons_weight_array ) - 1, tmp, axis=0 )

    def print_neuron_info(self):
        print( f"This neural network has in total {len( self.neurons )} with the values: " )
        [neuron.print_neuron_info() for neuron in self.neurons]

    class Neuron:
        def __init__(self, dendrite_number):
            self.dendrites = []
            self.add_dendrite()
            [self.add_dendrite() for _ in range( dendrite_number )]

        def add_dendrite(self):
            self.dendrites.append( NeuralNetwork.Neuron.Dendrites() )

        def print_neuron_info(self):
            [print( f"{dendrite.value}" ) for dendrite in self.dendrites]

        def get_dendrites(self):
            return self.dendrites

        class Dendrites:
            def __init__(self):
                self.value = random.random()

            def get_value(self):
                return self.value
