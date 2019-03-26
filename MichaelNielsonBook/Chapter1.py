'''
    Python3 program to classify MNIST digits based on Michael Nielson's book Chapter 1.
'''
import numpy as np
import random
import pickle
import gzip

def load_mnist_data():
    f = gzip.open('./mnist.pkl.gz', 'rb')#open file
    training_data, validation_data, test_data = pickle.load(f, encoding = 'latin1')
    f.close()
    
    #reshape data and convert to one hot
    training_x = [np.reshape(x, (784,1)) for x in training_data[0]]
    training_y = onehot(training_data[1], 10)
    training_data = zip(training_x, training_y)

    validation_x = [np.reshape(x, (784,1)) for x in validation_data[0]]
    validation_y = onehot(validation_data[1], 10)
    validation_data = zip(validation_x, validation_y)

    test_x = [np.reshape(x, (784,1)) for x in test_data[0]]
    test_y = onehot(test_data[1], 10)
    test_data = zip(test_x, test_y)

    return (training_data, validation_data, test_data)

def onehot(x, nb_classes):
    targets = x.reshape(-1)
    return np.eye(nb_classes)[targets].flatten()

def sigmoid(z):
        return (1 / (1.0 + np.exp(-z)))

def sigmoid_derivative(z):
        return sigmoid(z)*(1-sigmoid(z))

class Network(object):
    def __init__(self, sizes):#sizes is list of layer sizes
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases =  [np.random.randn(y,1) for y in sizes[1:]]#input layer has no bias
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
        #weights go from previous to next layer
        #weights[1] stores weights from first to second layer

    def feedforward(self, a):
        #n - 1 matrices for n layers because input layer has no weights and biases

        #TODO BUGGY Implementation
        # for current_layer in range(self.num_layers - 1):
        #     print(a.shape)
        #     z = self.weights[current_layer].dot(a) + self.biases[current_layer]
        #     z = sigmoid(z)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a)+b
            a = sigmoid(z)
        return (a, z)

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
        if test_data:
            number_tests = len(test_data)

        n = len(training_data)
        for e in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if(test_data):
                print("Epoch {0}: {1} / {2}".format(e, self.evaluate(test_data), number_tests))
            else:
                print("Epoch {0} finished".format(e))

    def backprop(self, x, y):
    	temp_b = np.zeros_like(self.biases)
    	temp_w = np.zeros_like(self.weights)

    	activation = x
    	activation_storage = [x]
    	z_storage = []

    	for w, b in zip(self.weights, self.biases):
    		z = w.dot(activation) + b
    		z_storage.append(z)
    		activation = sigmoid(z)
    		activation_storage.append(activation)

    	#Computes error at output layer
    	delta = self.cost_derivative(activation_storage[-1], y) * sigmoid_derivative(z_storage[-1])
    	temp_b[-1] = delta
    	temp_w[-1] = delta.dot(activation_storage[-2].transpose())

    	#back-propagates error to earlier layers
    	for layer in range(2, self.num_layers):
    		z = z_storage[-layer]
    		sp = sigmoid_derivative(z)
    		delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
    		temp_b[-layer] = delta
    		temp_w[-layer] = delta.dot(activation_storage[-layer-1].transpose())

    	return (temp_b, temp_w)

    def evaluate(self, test_data):
    	test_results = [(np.argmax(self.feedforward(x)[0]), y) for (x, y) in test_data]
    	return sum(int(x==y) for (x, y) in test_results)

    def cost_derivative(self, output_activation_storage, y):
    	return (output_activation_storage-y)


training_data, validation_data, test_data = load_mnist_data()
test_data = list(test_data)
training_data = list(training_data)
validation_data = list(validation_data)
net = Network([784, 100, 10])
# net.SGD(training_data, 30, 10, 3.0)
net.SGD(training_data, 10, 10, 2, test_data = test_data)
print("test")
            #TODO: Change initialization to random.random
            #TODO: Play with mini_batch_size, set as 1
            #TODO: Change net = Network.Network([784, 10])