import numpy as np
import sys
import matplotlib.pyplot as plt
import parameters
"""
Matthew Walters - Steven Koprowicz
2/13/23

For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""

# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):
        self.act = activation
        #for now, we're going to have self.numinps be the number of inputs, disregarding the bias input, 
        # which means the length of self.weights must be the length of self.numinps + 1
        self.numinps = input_num
        self.lr = lr
        self.weights = []
        #since I wrote the FullyConnectedLayer to randomly generate weights if not given, 
        # the neuron will be given weights, random or not
        for i in weights:
            self.weights.append(i)
        
    #This method returns the activation of the net
    def activate(self,net):
        if self.act == 0:
            return net
        elif self.act == 1:
            f = 1 / (1 + np.exp(-net))
            return f
        else:
            raise Exception("Activation Function not Supported")
        
    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,inputs):
        self.input = inputs
        x = 0
        #   here, I put range(len(input)) + 1, so the bias would be included, 
        #       and when it gets to the bias, the input is always 1, hence the if statement
        for i in range(len(inputs) + 1):
            if i >= len(inputs):
                x += 1*self.weights[i]
            else:
                x += self.input[i]*self.weights[i]
        self.output = self.activate(x)
        return self.output

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        #derivative of log is output(1-output)
        #derivative of linear is the slope of the linear function, which I imagine just returns 1 in this case, 
        # since activate() just returns it's input
        if self.act == 1:
            return self.output * (1 - self.output)
        else:
            return 1
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        self.delta = wtimesdelta*self.activationderivative()
        self.deltaw = []
        for i in range(len(self.weights)):
            self.deltaw.append(self.weights[i]*self.delta)
        return self.deltaw
    
    #Simply update the weights using the partial derivatives and the learning rate
    def updateweight(self):
        newweights = []
        for i in range(len(self.weights)):
            if i >= self.numinps:
                newweights.append(self.weights[i] - self.delta*1*self.lr)
            else:
                newweights.append(self.weights[i] - self.delta*self.input[i]*self.lr)
        self.weights = []
        for i in range(len(newweights)):
            self.weights.append(newweights[i])
        return self.weights

#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation, the input size, 
    # the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.numN = numOfNeurons
        self.activation = activation
        self.numinps = input_num
        self.lr = lr
        self.neurons = []
        self.weights = []
        # the length of each list of weights must be the number of inputs + 1 or it won't use them
        # it also won't accept the weights if there isn't a of list of weights for each neuron, no more no less
        if weights is None or len(weights) != self.numN or len(weights[0]) != self.numinps + 1:
            # print("random weights")
            for i in range(self.numN):
                w = []
                for j in range(self.numinps + 1):
                    w.append(np.random.uniform(0.1, 0.9))
                self.weights.append(w)
        else:
            for i in weights:
                w = []
                for j in i:
                    w.append(j)
                self.weights.append(w)
        for i in range(self.numN):
            x = Neuron(self.activation, self.numinps, self.lr, self.weights[i])
            self.neurons.append(x)
        
    #calculate the output of all the neurons in the layer and return a vector with those values 
    # (go through the neurons and call the calcualte() method)      
    def calculate(self, inputs):
        self.outputs = []
        for i in self.neurons:
            self.outputs.append(i.calculate(inputs))
        return self.outputs
        
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() 
    # for each (with the correct value), sum up its own w*delta, and then update the weights (using the updateweight() method). 
    # I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        self.wdeltas = []
        for i in range(len(self.neurons)):
            x = self.neurons[i].calcpartialderivative(wtimesdelta[i])
            self.wdeltas.append(x)
        self.wtimesdeltas = []
        for i in range(len(self.weights[i])):
            y = 0
            for j in range(len(self.wdeltas)):
                y += self.wdeltas[j][i]
            self.wtimesdeltas.append(y)
        # wtimesdelta done
        # update weights
        for i in self.neurons:
            i.updateweight()
        return self.wtimesdeltas           

class ConvolutionalLayer:
    #always assume stride 1, padding 'valid' for this lab
    def __init__(self, numKernels, kernSize, activation, numInputs, inputSize, lr, weights=None):
        self.numKernels=numKernels
        self.kernSize=kernSize
        self.activation=activation
        self.numInputs=numInputs
        self.inputSize=inputSize
        self.lr=lr
        self.numOutputs=numKernels
        self.numWeightsPerKernel=kernSize*kernSize*numInputs # not including bias
        self.numLearnableParameters=(self.numWeightsPerKernel+1)*numKernels
        self.numNeuronsPerKernel=(inputSize-kernSize+1)**2
        self.outputSize=(inputSize-kernSize+1)
        self.weights = []
        self.neurons = []
        self.outputs = []
        if weights is None or \
            not (len(weights) == self.numKernels and len(weights[0]) == self.numWeightsPerKernel + 1) or \
            not (len(weights) == (self.numWeightsPerKernel+1)*self.numKernels):
            for i in range(self.numKernels):
                w = []
                for j in range(self.numWeightsPerKernel + 1):
                    w.append(np.random.uniform(0.1, 0.9))
                self.weights.append(w)
        elif len(weights) == (self.numWeightsPerKernel+1)*self.numKernels:
            for i in range(self.numKernels):
                w = []
                for j in range(self.numWeightsPerKernel+1):
                    w.append(weights[i*(self.numWeightsPerKernel+1)+j])
                self.weights.append(w)
        else:
            for i in weights:
                w = []
                for j in i:
                    w.append(j)
                self.weights.append(w)
        for i in range(self.numKernels):
            for j in range(self.numNeuronsPerKernel):
                xs = []
                x = Neuron(self.activation, self.numWeightsPerKernel+1, self.lr, self.weights[i])
                xs.append(x)
            self.neurons.append(xs)

    def calculate(self, inputs):
        self.outputs = []
        for k in range(self.numKernels):
            fea_map = []
            for neu in self.neurons[k]:
                #to make this work, we need to split up the inputs among the neurons, each neuron doesn't get all of them
                # plus, it's probably easier to divide them up here if we can, rather than change neuron.calculate
                myinputs = []
                # we need some kind of if statement in this loop to make each neuron only see the inputs within it's kernel
                for i in inputs:
                    myinputs.append(i)
                print("length of myinputs = ", len(myinputs))
                fea_map.append(neu.calculate(myinputs))
            self.outputs.append(fea_map)
        return self.outputs
        
    def calculatewdeltas(self, wtimesdelta):
        self.wdeltas = []
        for k in range(self.numOutputs):
            perKer=[]
            for i in range(self.numNeuronsPerKernel):
                x = self.neurons[k][i].calcpartialderivative(wtimesdelta[k][i])
                perKer.append(x)
            self.wdeltas.append(perKer)
        self.wtimesdeltas = []
        for k in range(self.numKernels):
            y = []
            for i in range(self.inputSize):
                y.append(0)
            self.wtimesdeltas.append(y)
        for k in range(self.numKernels):
            for drow in range(self.outputSize):
                for dcol in range(self.outputSize):
                    for wrow in range(self.kernSize):
                        for wcol in range(self.kernSize):
                            self.wtimesdeltas[k][ 
                                (dcol+wcol)*self.inputSize + (drow+wrow)
                                ] += self.wdeltas[k][ \
                                dcol*self.outputSize + drow][ \
                                wcol*self.kernSize + wrow]
        for i in self.neurons:
            i.updateweight()
        return self.wtimesdeltas
        # self.wdeltas = []
        # for k in range(self.numKernels):
        #     perKer=[]
        #     for i in range(self.numNeuronsPerKernel):
        #         x = self.neurons[k][i].calcpartialderivative(wtimesdelta[k][i])
        #         perKer.append(x)
        #     self.wdeltas.append(perKer)
        # self.wtimesdeltas = []
        # for k in range(self.numKernels):
        #     perKer = []
        #     for row in range(self.outputDim):
        #         for column in range(self.outputDim):
        #             y = 0
        #             for i in range(self.kernSize):
        #                 for j in range(self.kernSize):
        #                     y += self.wdeltas[k][column*self.outputDim+row][j*self.kernSize+i]
        #             perKer.append(y)
        #     y = 0
        #     for row in range(self.kernSize):
        #         for column in range(self.kernSize):
        #             y += self.wdeltas[k][self.kernSize**2][column*self.kernSize+row]
        #     perKer.append(y)
        #     self.wtimesdeltas.append(perKer)
        # for i in self.neurons:
        #     i.updateweight()
        # return self.wtimesdeltas

class maxPoolingLayer:
    #assume stride is the same as filter size
    def __init__(self, kernSize, numInputs, inputSize):
        self.kernSize=kernSize
        self.numInputs=numInputs
        self.inputSize=inputSize
        self.outputSize=np.ceil(inputSize/kernSize)
        self.maxInputLocations = []
        self.maxOutputLocations = []

    def calculate(self, inputs):
        self.maxInputLocations = []
        self.maxOutputLocations = []
        outputs = []
        for channel in range(self.inputSize):
            self.maxInputLocations.append([])
            self.maxOutputLocations.append([])
            output_table = []
            for ocol in range(self.outputSize):
                for orow in range(self.outputSize):
                    for kcol in range(self.kernSize):
                        for krow in range(self.kernSize):
                            outputIndex = ocol*self.outputSize+orow
                            inputIndex = (ocol*self.kernSize+kcol)*self.inputSize + (orow*self.kernSize+krow)
                            if len(inputs[channel]) <= inputIndex: continue
                            if(krow==0 & kcol==0):
                                self.maxInputLocations[channel].append(inputIndex)
                                self.maxOutputLocations[channel].append(outputIndex)
                                output_table[outputIndex] = inputs[channel][inputIndex]
                            else:
                                oldMax = output_table[outputIndex]
                                newValue = inputs[channel][inputIndex]
                                if newValue > oldMax:
                                    self.maxInputLocations[channel][ocol*self.outputSize+orow] = inputIndex
                                output_table[outputIndex] = newValue
            outputs.append(channel)        

    def calculatewdeltas(self):
        pass

class flattenLayer:
    def __init__(self, inputSize):
        pass
    #there will be no neurons here, it just resizes the output of the previous layer from 2d to 1d
    def calculate(self):
        pass
    def calculatewdeltas(self):
        pass

#An entire neural network 
class NeuralNetwork:
    #now, when you don't provide numOfLayers, numOfNeurons, and activation, you can add to them later
    # I kept the other init function in here if we need to compare or test, but for the forseeable future we'll be able to add layers
    def __init__(self, inputSize, loss, lr):
        self.numL = 0
        self.numN = []
        self.numinps = inputSize
        self.numouts = inputSize
        self.activation = []
        self.loss = loss
        self.lr = lr
        self.weights = []
        self.layers = []
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), 
    # the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    # def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
    #     self.numL = numOfLayers
    #     self.numN = numOfNeurons.copy()
    #     self.numinps = inputSize
    #     self.activation = activation.copy()
    #     self.loss = loss
    #     self.lr = lr
    #     self.layers = []
    #     isize = self.numinps
    #     if weights is not None and len(weights) == self.numL:
    #         for i in range(self.numL):
    #             x = FullyConnected(self.numN[i], self.activation[i], isize, self.lr, weights[i])
    #             isize = self.numN[i]
    #             self.layers.append(x)
    #     else:
    #         for i in range(self.numL):
    #             x = FullyConnected(self.numN[i], self.activation[i], isize, self.lr)
    #             isize = self.numN[i]
    #             self.layers.append(x)
    
    #addLayer will use self.numouts as it's inputsize and reset self.numouts to be the output size of the new layer
    # once the other layers are implemented, we need to add to this
    # layerType = 0: FullyConnected
    #             1: ConvolutionalLayer
    #             2: MaxPoolingLayer
    #             3: FlattenLayer
    def addLayer(self, numOfNeurons, activation, numKernels=0, kernSize=0, numChannels=0, numFilters=0, layerType=0, weights=None):
        act = activation
        numins = self.numouts
        self.numouts = numOfNeurons
        self.numN.append(numOfNeurons)
        self.numL += 1
        if layerType == 0:
            if weights is not None:
                layer = FullyConnected(numOfNeurons, act, numins, self.lr, weights)
                self.layers.append(layer)
            else:
                layer = FullyConnected(numOfNeurons, act, numins, self.lr)
                self.layers.append(layer)
        elif layerType == 1:
            # add numInputs, inputSize from above
            if self.numL == 1:
                self.outputDim = int(np.sqrt(self.numinps))
            if weights is not None and len(weights) == kernSize**2 * (numChannels + 1) * (numFilters):
                layer = ConvolutionalLayer(numKernels, kernSize, act, self.outputDim, lr, weights)
                self.outputDim = layer.outputDim
                self.layers.append(layer)
            else:
                layer = ConvolutionalLayer(numKernels, kernSize, activation, self.numinps, self.outputDim, self.lr)
                self.outputDim = layer.outputSize
                self.layers.append(layer)

    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,inputs):
        self.input = inputs
        output = inputs
        for i in self.layers:
            output = i.calculate(output)
        return output
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        # self.loss == 0: sum of squared errors
        # self.loss == 1: binary cross entropy
        #the binary cross entropy makes the assumption you only have 1 output, sum squared errors assumes you have a list of outputs
        if self.loss == 0:
            loss = 0
            for i in range(len(yp)):
                loss += 1/2*((yp[i] - y[i])**2)
        elif self.loss == 1:
            loss = -(y * np.log(yp)) + ((1 - y) * np.log(1 - yp))
        else:
            raise Exception("Loss Function Not Supported")
        return loss
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        if self.loss == 0:
            #derivative of sum squared error
            ld = (yp - y)
        elif self.loss == 1:
            #derivative of binary cross entropy
            ld = -(y/yp) + ((1-y)/(1-yp))
        else:
            raise Exception("Loss Function Not Supported")
        return ld
    
    #Given a single input and desired output preform one step of backpropagation 
    # (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values)
    def train(self,x,y):
        #PSEUDOCODE
        output = self.calculate(x)
        ld = []
        for i in range(self.layers[self.numL - 1].numN):
            ld.append(self.lossderiv(output[i], y[i]))
        for i in reversed(range(len(self.layers))):
           ld = self.layers[i].calcwdeltas(ld)

if __name__=="__main__":
    if (len(sys.argv)<2):
        x = np.array([[1,0,1],[1,1,0],[0,1,1]])
        network = NeuralNetwork(9,0,.5)
        network.addLayer(4,1,4,2,1,1,1)
        print("weights[0]:",network.layers[0].weights)
        print("calculate:",network.calculate(x))

    elif (sys.argv[1] == 'example'):
        print('making the lab1 example from class but with addLayer')
        w1 = np.array([[.15,.2,.35],[.25,.3,.35]])
        w2 = np.array([[.4,.45,.6],[.5,.55,.6]])
        x = np.array([0.05,0.1])
        y = np.array([0.01,0.99])
        network = NeuralNetwork(2, 0, .5)
        network.addLayer(2, 1, weights=w1)
        network.addLayer(2, 1, weights=w2)
        print(network.calculate(x))
        network.train(x,y)
        print(network.calculate(x))

    elif (sys.argv[1] == 'param'):
        #Generate data and weights for "example2"
        l1k1,l1k2,l1b1,l1b2,l2k1,l2b,l3,l3b,x,y = parameters.generateExample2()
        newl = []
        for i in x:
            for j in i:
                newl.append(j)
        x = newl.copy()
        network = NeuralNetwork(7, 0, .5)
        l1k1 = list(l1k1)
        newl = []
        #flatten the l1k1, add the bias on the end
        for i in l1k1:
            for j in i:
                newl.append(j)
        l1k1 = newl.copy()
        l1k1.append(l1b1[0])
        print(l1k1)
        l1k2 = list(l1k2)
        newl = []
        for i in l1k2:
            for j in i:
                newl.append(j)
        l1k2 = newl.copy()
        l1k2.append(l1b2[0])
        #not sure how to make 2 kernels work, but this is the network, but calculate doesn't currently work
        network.addLayer([9,9],1,2,2,1,2,1,[[l1k1],[l1k2]])
        l2k1 = list(l2k1)
        newl = []
        for i in l2k1:
            for j in i:
                newl.append(j)
        l2k1 = newl.copy()
        l2k1.append(l2b[0])
        network.addLayer(18, 1, 1, 3, 2, 1, 1, [[l2k1]])
        l3 = list(l3)
        l3.append(l3b)
        network.addLayer(9, 1, weights=[l3])
        print(network.calculate(x))