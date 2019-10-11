import numpy as np
import msvcrt as m
from mlxtend.data import loadlocal_mnist
import json as j

guesses = {
    0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}

def returnNum(guess):
    m = np.zeros_like(guess)
    index = np.nonzero(guess == np.max(guess))[0][0]
    m[index] = 1
    for k,v in guesses.items():
        if np.array_equal(m,v):
            return k



def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(y):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  return y * (1 - y)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()


def reLu(x):
    return max(0.0,x)

def reLuDeriv(y):
    if(y> 0.0):
        return 1.0
    else:
        return 0.0


def wait():
    print("Press Enter")
    m.getch()


class Neuron:
    def __init__(self,weights,biases):
        self.weights = np.array(weights)
        self.biases = biases
        self.netOut = 0
        self.errror = 0

    def feedForward(self,input):
        result = np.dot(self.weights,input)+ self.biases
        self.netOut = result
        return reLu(result)


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons
        self.n = len(neurons)
        weights = list()
        biases = list()
        for neuron in neurons:
            weights.append(neuron.weights)
            biases.append(neuron.biases)
        self.weights = np.array(weights)
        self.biases = np.array(biases)
        self.deltas = list()

    def feedForward(self,inputs):
        outs = []
        for neuron in self.neurons:
            outs.append(neuron.feedForward(inputs))

        return outs



class Network:
    def __init__(self):
        neurons= list()
        for count in range (2):
            weights = np.random.rand(2)
            neurons.append(Neuron(weights,np.random.rand(1)[0]))
        first = Layer(neurons)
        neurons= list()
        for count in range (1):
            weights = np.random.rand(2)
            neurons.append(Neuron(weights,np.random.rand(1)[0]))
        second = Layer(neurons)
        neurons= list()
        self.layers = [first,second]

        self.learningRate = 0.1
        self.niterations = 1000
        self.batchSize = 128

    def feedForward(self,input):
        result = input
        for i in range(len(self.layers)):
            result = self.layers[i].feedForward(result)

        return result

    def learn(self,data,hot):
        answers = np.array(hot)
        for epoch in range(self.niterations):
            loop = 0
            for dataset, answer in zip(data,answers):

                deltas = list()
                deltasBiases = list()
                layerOuts = []
                layerOuts.append(dataset)
                input = dataset
                for layer in self.layers:
                    input = layer.feedForward(input)
                    layerOuts.append(input)

                outputs = np.array(layerOuts[-1])
                gradient =np.array(list( map(reLuDeriv,outputs)))
                errorOutput = outputs - np.array(answer)
                hiddenOutput = np.array(layerOuts[-2])

                gradient = np.asmatrix(errorOutput.transpose()* gradient * self.learningRate)
                deltaWeightsOutput =  gradient.transpose() * hiddenOutput
                deltas.append(deltaWeightsOutput)

                deltasBiases.append(gradient)

                for i in range( len(self.layers) - 2, -1, -1):

                    weightsToMultiply = self.layers[i+1].weights
                    errorOutput = np.dot(weightsToMultiply.transpose(),errorOutput)
                    layerOutput = np.array(layerOuts[i+1])
                    gradient = np.array(list(map(deriv_sigmoid, layerOutput)))
                    hidden_gradient = errorOutput.transpose()*gradient * self.learningRate
                    deltasBiases.append(hidden_gradient)
                    hidden_gradient = np.matrix(hidden_gradient)
                    hiddenInputs =np.array(layerOuts[i])
                    deltaHiddenWeights = hidden_gradient.transpose() * hiddenInputs
                    deltas.append(deltaHiddenWeights)

                for i in range(len(deltas)):

                    self.layers[i].weights -= deltas[-1*i -1]
                    if i == len(deltas)-1 :
                        deltasBiases[-1*i -1] = deltasBiases[-1*i -1].getA1()
                    self.layers[i].biases -= deltasBiases[-1*i -1]

                if(loop % self.batchSize == 0):
                    print( "Epoch: ", epoch," For: ", answer, "Guessed: ", outputs, "Number : ", returnNum(outputs))
                loop+=1
        print("DONE!!!!")





data, answers = loadlocal_mnist(
        images_path='C:/Users/Mateusz/Desktop/train-images.idx3-ubyte',
        labels_path='C:/Users/Mateusz/Desktop/train-labels.idx1-ubyte')

print('Dimensions: %s x %s' % (data.shape[0], data.shape[1]))


oneHot = list()
for answer in answers:
    oneHot.append(guesses.get(answer))

normalizedData = list()
for dataset in data:
    normalizedData.append(dataset / 255)

xorData = [[0,0],[0,1],[1,0],[1,1]]
xorAnswers = [0,1,1,0]
print(normalizedData[0])
network = Network()