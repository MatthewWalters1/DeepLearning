import numpy as np
import sys
import matplotlib.pyplot as plt
import parameters
"""
Matthew Walters - Steven Koprowicz
3/6/23

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
        #for now, we're going to have self.numInputs be the number of inputs, disregarding the bias input, 
        # which means the length of self.weights must be the length of self.numInputs + 1
        self.numInputs = input_num
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
        #print(f"self.numInputs {self.numInputs}")
        #print(f"self.lr {self.lr}")
        #print(f"self.weights {self.weights}")
        #print(f"len {len(self.input)}")
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
    def updateweight(self, conv=False, updatedWeights=None):
        if conv == False:
            newweights = []
            for i in range(len(self.weights)):
                if i >= self.numInputs:
                    newweights.append(self.weights[i] - self.delta*1*self.lr)
                else:
                    newweights.append(self.weights[i] - self.delta*self.input[i]*self.lr)
            self.weights = []
            for i in range(len(newweights)):
                self.weights.append(newweights[i])
        if conv == True:
            self.weights = updatedWeights.copy()
        return self.weights

#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation, the input size, 
    # the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.numNeurons = numOfNeurons
        self.numOutputs = numOfNeurons
        self.outputSize = 1
        self.inputSize = 1
        self.activation = activation
        self.numInputs = input_num
        self.lr = lr
        self.neurons = []
        self.weights = []
        # the length of each list of weights must be the number of inputs + 1 or it won't use them
        # it also won't accept the weights if there isn't a of list of weights for each neuron, no more no less
        if weights is None or len(weights) != self.numNeurons or len(weights[0]) != self.numInputs + 1:
            # print("random weights")
            for i in range(self.numNeurons):
                w = []
                for j in range(self.numInputs + 1):
                    w.append(np.random.uniform(0.1, 0.9))
                self.weights.append(w)
        else:
            for i in weights:
                w = []
                for j in i:
                    w.append(j)
                self.weights.append(w)
        for i in range(self.numNeurons):
            x = Neuron(self.activation, self.numInputs, self.lr, self.weights[i])
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
    def calculatewdeltas(self, wtimesdelta):
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
        self.numLayersearnableParameters=(self.numWeightsPerKernel+1)*numKernels
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
        for k in range(self.numKernels):
            xs = []
            for j in range(self.numNeuronsPerKernel):
                x = Neuron(self.activation, self.numWeightsPerKernel+1, self.lr, self.weights[k])
                xs.append(x)
            self.neurons.append(xs)
        # print(f"line196(self.numKernels):{self.numKernels}")
        # print(f"line196(self.kernSize):{self.kernSize}")
        # print(f"line196(self.numInputs):{self.numInputs}")
        # print(f"line196(self.inputSize):{self.inputSize}")
        # print(f"line196(self.numWeightsPerKernel):{self.numWeightsPerKernel}")
        # print(f"line196(self.numLayersearnableParameters):{self.numLayersearnableParameters}")
        # print(f"line196(self.numNeuronsPerKernel):{self.numNeuronsPerKernel}")
        # print(f"line196(self.outputSize):{self.outputSize}")
        # print(f"line196(self.weights):{self.weights}")
        # print(f"line196(self.neurons):{self.neurons}")
        # print(f"line196(self.outputs):{self.outputs}")

    def calculate(self, inputs):
        self.inputs = inputs
        self.outputs = []
        # print("length of inputs = ", len(inputs))
        # print("length of inputs[0] = ", len(inputs[0]))
        for k in range(self.numKernels):
            fea_map = []
            for orow in range(self.outputSize):
                for ocol in range(self.outputSize):
                    neuron = self.neurons[k][ocol*self.outputSize+orow]
                    # print(f"len(self.neurons):{len(self.neurons)}")
                    # print(f"len(self.neurons[0]):{len(self.neurons[0])}")

                    #to make this work, we need to split up the inputs among the neurons, each neuron doesn't get all of them
                    # plus, it's probably easier to divide them up here if we can, rather than change neuron.calculate
                    neuInputs = []
                    # we need some kind of if statement in this loop to make each neuron only see the inputs within it's kernel
                    for krow in range(self.kernSize):
                        for kcol in range(self.kernSize):
                            neuInputs.append(inputs[k][(ocol+kcol)*self.inputSize + (orow+krow)])
                    # print("length of myinputs = ", len(neuInputs))
                    fea_map.append(neuron.calculate(neuInputs))     
            self.outputs.append(fea_map)
        return self.outputs
        
    def calculatewdeltas(self, wtimesdelta):
        #print(f"wtimesdelta:{wtimesdelta}")
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
            for i in range(self.inputSize**2):
                y.append(0)
            self.wtimesdeltas.append(y)
        for k in range(self.numKernels):
            for dcol in range(self.outputSize):
                for drow in range(self.outputSize):
                    for wcol in range(self.kernSize):
                        for wrow in range(self.kernSize):
                            # print(f"k{k}, drow{drow}, dcol{dcol}, wrow{wrow}, wcol{wcol}, self.inputSize{self.inputSize}")
                            # #print(f"self.wtimesdeltas--{self.wtimesdeltas}")
                            # #print(f"self.wtimesdeltas[0]--{self.wtimesdeltas[0]}")
                            # #print(f"self.wdeltas--{self.wdeltas}")
                            # #print(f"self.wdeltas[0]--{self.wdeltas[0]}")
                            # #print(self.wdeltas[k])
                            # #print((dcol+wcol)*self.inputSize + (drow+wrow))
                            # print(self.wdeltas[k][ \
                            #     dcol*self.outputSize + drow][ \
                            #     wcol*self.kernSize + wrow])
                            # print(self.wtimesdeltas[k][ 
                            #     (dcol+wcol)*self.inputSize + (drow+wrow)
                            #     ])
                            self.wtimesdeltas[k][ 
                                (dcol+wcol)*self.inputSize + (drow+wrow)
                                ] += self.wdeltas[k][ \
                                dcol*self.outputSize + drow][ \
                                wcol*self.kernSize + wrow]
        updatedWeights = []
        #print(f"wtimesdelta{wtimesdelta[0]}")
        #print(f"self.outputs{self.outputs[0]}")
        #print(f"self.numWeightsPerKernel{self.numWeightsPerKernel+1}")
        for k in range(self.numKernels):
            wsPDeriv = []
            for w in range(self.numWeightsPerKernel+1) :
                wPDeriv = 0
                for neur in range(self.numNeuronsPerKernel):
                    neuron = self.neurons[k][neur]
                    #print(f"285<<<<  k{k} w{w} neur{neur} ")
                    if w==self.numWeightsPerKernel:
                        wPDeriv += neuron.delta*1
                    else:
                        wPDeriv += neuron.delta*neuron.input[w]
                wsPDeriv.append(wPDeriv)
            updatedWeights.append(wsPDeriv)

        for k in range(self.numKernels):
            for i in self.neurons[k]:
                i.updateweight(conv=True, updatedWeights=updatedWeights[k])
        self.weights = updatedWeights            
        return self.wtimesdeltas

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
        self.outputs = []
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
            self.outputs.append(channel)  
        return self.outputs 

    def calculatewdeltas(self, wtimesdeltas):
        #given wtimesdeltas, return a list filled with zeroes except for the maxInputLocations, where the wtimesdeltas will go
        alldeltas = []
        for channel in range(self.inputSize):
            wdelta = np.zeros(self.inputSize**2)
            for location in range(self.maxInputLocations):
                wdelta[self.maxInputLocations[location]] = wtimesdeltas[channel][location]
            alldeltas.append(wdelta)
        return alldeltas    

class FlattenLayer:
    def __init__(self, numInputs, inputSize):
        self.inputSize = inputSize
        self.numInputs = numInputs

    #there will be no neurons here, it just resizes the output of the previous layer from 2d to 1d
    def calculate(self, inputs):
        self.outputs = []
        if len(inputs) != self.inputs:
            print("incorrect number of inputs, fully connected layers won't work right")
        if len(inputs[0]) != self.inputSize**2:
            print("input of incorrect size, fully connected layers won't work right")
        #because we index our convolutions in 1d (even though their treated as if they're 2d), this won't really do anything, 
        # it's just a buffer so the next layer can be fully connected
        for i in range(len(inputs)):
            for j in range(len(inputs[i])):
                self.outputs.append(inputs[i][j])
        return self.outputs

    def calculatewdeltas(self, wtimesdelta):
        #here, given the wdeltas from the next layer, it gives them to the appropriate neurons in the previous one
        #but, because calculate is identical to the previous layer, you just take wdeltas and send it back a layer
        allDeltaW = []
        for i in range(self.numInputs):
            deltaw = []
            for j in range(self.inputSize):  
                deltaw.append(wtimesdelta[i*self.inputSize+j])
            allDeltaW.append(deltaw)
        return allDeltaW

#An entire neural network 
class NeuralNetwork:
    #now, when you don't provide numOfLayers, numOfNeurons, and activation, you can add to them later
    # I kept the other init function in here if we need to compare or test, but for the forseeable future we'll be able to add layers
    def __init__(self, numInputs, inputSize, loss, lr):
        self.numLayers = 0
        self.numNeurons = []
        self.numInputs = numInputs
        self.numOutputs = numInputs
        self.inputSize = inputSize
        self.outputSize = inputSize
        self.activation = []
        self.loss = loss
        self.lr = lr
        self.weights = []
        self.layers = []

    def addLayer(self, activation, numOfNeurons=0, numKernels=0, kernSize=0, layerType=0, weights=None):
        curNumOutputs = self.numOutputs
        curOutputSize = self.outputSize
        self.numLayers += 1
        if layerType == 0:
            layer = FullyConnected(numOfNeurons=numOfNeurons, activation=activation, \
                input_num=curNumOutputs, lr=self.lr, weights=weights)
            self.outputSize = 1
            self.numOutputs = numOfNeurons
            self.layers.append(layer)
            self.numNeurons.append(numOfNeurons)
        elif layerType == 1:
            layer = ConvolutionalLayer(numKernels=numKernels, kernSize=kernSize, activation=activation, \
                numInputs=curNumOutputs, inputSize=curOutputSize, lr=self.lr, weights=weights)
            self.outputSize = layer.outputSize
            self.numOutputs = numKernels
            self.layers.append(layer)
            self.numNeurons.append(numOfNeurons)
        elif layerType == 2:
            layer = maxPoolingLayer(kernSize, curNumOutputs, curOutputSize)
            self.outputSize = layer.outputSize
            self.numOutputs = self.numInputs
            self.layers.append(layer)
            self.numNeurons.append(0)
        elif layerType == 3:
            layer = FlattenLayer(numInputs=curNumOutputs, inputSize=curOutputSize)
            self.outputSize = 1
            self.numOutputs = curNumOutputs*curOutputSize
            self.layers.append(layer)
            self.numNeurons.append(0)
            

    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,inputs):
        #print(f"len(inputs {len(inputs)}")
        #print(f"len(inputs {len(inputs[0])}")
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
    # (including a forward pass, getting the derivative of the loss, and then calling calculatewdeltas for layers with the right values)
    def train(self,x,y):
        #PSEUDOCODE
        output = self.calculate(x)
        # print(f"output:{output}")
        allLD = []
        lastLayer = self.layers[self.numLayers - 1]
        for i in range(lastLayer.numOutputs):
            ld=[]
            for j in range(lastLayer.outputSize**2):
                # print(f"i{i},j{j}")
                # print(output[i])
                # print(y[i])
                ld.append(self.lossderiv(output[i][j], y[i][j]))
            allLD.append(ld)
        # print(f"ld:{ld}")
        for i in reversed(range(len(self.layers))):
           allLD = self.layers[i].calculatewdeltas(allLD)

def flat(x):
    newl = []
    for i in x:
        for j in i:
            newl.append(j)
    return newl.copy()

if __name__=="__main__":
    if (len(sys.argv)<2):
        print('try using the arguments example1, example2, or example3')

    elif (sys.argv[1] == 'example'):
        print('making the lab1 example from class but with addLayer')
        w1 = np.array([[.15,.2,.35],[.25,.3,.35]])
        w2 = np.array([[.4,.45,.6],[.5,.55,.6]])
        x = np.array([[0.05],[0.1]])
        y = np.array([[0.01],[0.99]])
        network = NeuralNetwork(2, 1, 0, .5)
        network.addLayer(1, numOfNeurons=2, weights=w1)
        network.addLayer(1, numOfNeurons=2, weights=w2)
        print(network.calculate(x))
        for i in range(1000):
            network.train(x,y)
        print(network.calculate(x))

    elif (sys.argv[1] == 'example1'):
        np.random.seed(10)
        network = NeuralNetwork(1, 5, 0, .5)
        input = [np.random.rand(5*5)]
        weights1 = np.random.rand(3*3+1)
        testIn = [np.random.rand(5*5)]
        testOut = [np.random.rand(3*3)]
        #3x3 conv, 1 kernel (didn't say what the size of the kernel should be)
        network.addLayer(1, kernSize=3, numKernels=1, layerType=1, weights=weights1)
        #flatten layer
        network.addLayer(1, layerType=3)
        print(network.calculate(input))
        for i in range(10):
            network.train(testIn,testOut)
        print(network.calculate(input))
        print(testOut)
        # #1 neuron
        # network.addLayer(1, 1)
        
    elif (sys.argv[1] == 'example2'):
        #Generate data and weights for "example2"
        l1k1,l1k2,l1b1,l1b2,l2k1,l2b,l3,l3b,x,y = parameters.generateExample2()
        #flatten x
        x = flat(x)
        network = NeuralNetwork(1, 7, 0, 100)
        #flatten l1k1 and append the bias
        l1k1 = flat(l1k1)
        l1k1.append(l1b1[0])
        print(l1k1)
        l1k2 = flat(l1k2)
        l1k2.append(l1b2[0])
        network.addLayer(1, (3*3)+(3*3), 2, 3, layerType=1, weights=[l1k1,l1k2])
        #flatten l2k1 (add bias), it has 2 channels
        l2k1 = flat(l2k1)
        l2k1.append(l2b[0])
        network.addLayer(2, (3*3)+(3*3), 1, 3, layerType=1, weights=[l2k1])
        #flatten layer in between l2k1 and l3 
        # (I think flatten layer is redundant if you don't make your convolution lists 2d, even if you treat them as 2d internally)
        network.addLayer(1, layerType=3)
        l3 = list(l3)
        l3.append(l3b)
        #fully connected layer
        network.addLayer(1, 9, 1, weights=[l3])
        print(network.calculate(x))
    

    elif (sys.argv[1] == 'example3'):
        np.random.seed(10)
        #8x8 input, 3x3 conv w/ 2kernels, 2x2 max pooling, flatten layer, 1 neuron output
        np.random.seed(10)
        x = np.random.rand(8,8)
        y = np.random.rand(1)
        l1k1 = np.random.rand(3,3)
        l1b1 = np.random.rand(1)
        l1k2 = np.random.rand(3,3)
        l1b2 = np.random.rand(1)
        l3 = np.random.rand(1,18)
        l3b = np.random.rand(1)
        l1k1 = flat(l1k1)
        l1k1.append(l1b1)
        l1k2 = flat(l1k2)
        l1k2.append(l1b2)
        l3 = list(l3)
        l3.append(l3b)
        network = NeuralNetwork(1, 8, 1, 100)
        #3x3 conv
        network.addLayer(1, 3*3, 2, 3, 1, weights=[l1k1,l1k2])
        #2x2 pool
        network.addLayer(1, 0, 1, 2, 2)
        #flat
        network.addLayer(1, layerType=3)
        #fully connected
        network.addLayer(1, 1, weights=[l3])
        print(network.calculate(x))
        