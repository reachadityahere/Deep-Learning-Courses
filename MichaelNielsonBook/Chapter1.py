'''
    Python3 program to classify MNIST digits based on Michael Nielson's book Chapter 1.
'''
import numpy as np
import random
import pickle
import gzip
####################################################################################################################################
def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('./mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
    ####################################################################################################################################

class Network(object):
    def __init__(self, sizes):#sizes is list of layer sizes
            self.num_layers = len(sizes)
            self.sizes = sizes
            self.biases =  [np.random.randn(y,1) for y in sizes[1:]]#input layer has no bias
            self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
            #weights go from previous to next layer
            #weights[1] stores weights from first to second layer

            def sigmoid(z):
                return (1 / (1.0 + np.exp(-z)))

            def sigmoid_derivative(z):
                return sigmoid(z)*(1-sigmoid(z))

            def feedforward(self, a):
                #n - 1 matrices for n layers because input layer has no weights and biases
                for current_layer in range(self.num_layers - 1):
                    z = self.weights[current_layer].dot(a) + self.biases[current_layer]
                    z = sigmoid(z)
                return z

            def update_mini_batch(self, mini_batch, alpha):
                delta_w = np.zeros_like(self.weights)
                delta_b = np.zeros_like(self.biases)

                for x, y in mini_batch:
                    #generate w, b delta matrices for entire network
                    change_in_b, change_in_w = self.backprop(x, y)
                    #adds the matrices for all network to obtain net change for current mini_batch
                    for i in range(self.num_layers - 1):
                        delta_b[i] += change_in_b[i]
                        delta_w[i] += change_in_w[i]
                
                for i in range(self.num_layers - 1):
                    self.weights[i] += ((-1*alpha) / len(mini_batch)) * delta_w[i]
                    self.biases[i]  += ((-1*alpha) / len(mini_batch)) * delta_b[i]


            def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None):
                '''
                training_data is rows of tuples(x,y)
                '''
                if test_data:
                    number_tests = len(test_data)

                n = len(training_data)
                for e in range(epochs):
                    random.shuffle(training_data)
                    mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                    
                    for mini_batch in mini_batches:
                        self.update_mini_batch(mini_batch, alpha)

                    if(test_data):
                        print("Epoch {0}: {1} / {2}".format(e, self.evaluate(test_data), number_tests))
                    else:
                        print("Epoch {0} finished".format(e))

####################################################################################################################################
            def backprop(self, x, y):
                """Return a tuple ``(nabla_b, nabla_w)`` representing the
                gradient for the cost function C_x.  ``nabla_b`` and
                ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
                to ``self.biases`` and ``self.weights``."""
                nabla_b = np.zeros_like(self.biases)
                nabla_w = np.zeros_like(self.weights)
                # feedforward
                activation = x
                activations = [x] # list to store all the activations, layer by layer
                zs = [] # list to store all the z vectors, layer by layer
                for b, w in zip(self.biases, self.weights):
                    z = np.dot(w, activation)+b
                    zs.append(z)
                    activation = sigmoid(z)
                    activations.append(activation)
                # backward pass
                delta = self.cost_derivative(activations[-1], y) * \
                    sigmoid_derivative(zs[-1])
                nabla_b[-1] = delta
                nabla_w[-1] = np.dot(delta, activations[-2].transpose())
                # Note that the variable l in the loop below is used a little
                # differently to the notation in Chapter 2 of the book.  Here,
                # l = 1 means the last layer of neurons, l = 2 is the
                # second-last layer, and so on.  It's a renumbering of the
                # scheme in the book, used here to take advantage of the fact
                # that Python can use negative indices in lists.
                for l in xrange(2, self.num_layers):
                    z = zs[-l]
                    sp = sigmoid_derivative(z)
                    delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                    nabla_b[-l] = delta
                    nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
                return (nabla_b, nabla_w)

            def evaluate(self, test_data):
                """Return the number of test inputs for which the neural
                network outputs the correct result. Note that the neural
                network's output is assumed to be the index of whichever
                neuron in the final layer has the highest activation."""
                test_results = [(np.argmax(self.feedforward(x)), y)
                                for (x, y) in test_data]
                return sum(int(x == y) for (x, y) in test_results)

            def cost_derivative(self, output_activations, y):
                """Return the vector of partial derivatives partial C_x /
                    partial a for the output activations."""
                return (output_activations-y)

####################################################################################################################################

training_data, validation_data, test_data = load_data_wrapper()
net = Network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data = test_data)
print("test")
            #TODO: Change initialization to random.random
            #TODO: Play with mini_batch_size, set as 1