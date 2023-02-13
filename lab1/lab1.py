import numpy as np
import sys
"""
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
        for i in self.numinps + 1:
            if i > self.numinps:
                x += 1*self.weights[i]
            else:
                x += self.input[i]*self.weights[i]
        self.output = activate(x)
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
        self.delta = 0
        print("This function is hard to write")
    
    #Simply update the weights using the partial derivatives and the learning rate
    def updateweight(self):
        newweights = []
        for i in range(len(self.weights)):
            if i > self.numinps:
                newweights.append(self.weights[i] - self.delta*1*self.lr)
            else:
                newweights.append(self.weights[i] - self.delta*self.input[i]*self.lr)
        self.weights = []
        for i in range(len(newweights)):
            self.weights.append(newweights[i])

        
#A fully connected layer 
#for Matthew       
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
        if len(weights) != self.numN or len(weights[0]) != self.numinps + 1:
            print("random weights")
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
        print('calcwdeltas') 
           
        
#An entire neural network 
#for Steven       
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), 
    # the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
        self.numL = numOfLayers
        self.numN = numOfNeurons.copy()
        self.numinps = inputSize
        self.activation = activation.copy()
        self.loss = loss
        self.lr = lr
        self.layers = []
        isize = self.numinps
        if len(weights) == self.numL:
            for i in range(len(self.numL)):
                x = FullyConnected(self.numN[i], self.activation[i], isize, self.lr, weights[i])
                isize = self.numN[i]
                self.layers.append(x)
        else:
            for i in range(len(self.numL)):
                x = FullyConnected(self.numN[i], self.activation[i], isize, self.lr)
                isize = self.numN
                self.layers.append(x)
    
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
                loss += ((yp[i] - y[i]) * (yp[i] - y[i]))
        elif self.loss == 1:
            loss = -(y * np.log(yp)) + ((1 - y) * np.log(1 - yp))
        else:
            raise Exception("Loss Function Not Supported")
        return loss
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        loss = calculateloss(yp, y)
        #find derivative of said loss
        if self.loss == 0:
            #derivative of sum squared error
            ld = 0
            pass
        elif self.loss == 1:
            #derivative of binary cross entropy
            ld = 0
            pass
        else:
            raise Exception("Loss Function Not Supported")
        return ld
    
    #Given a single input and desired output preform one step of backpropagation 
    # (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values)
    def train(self,x,y):
        print('train')
        #PSEUDOCODE
        # output = calculate(x)
        # ld = lossderiv(output, y)
        # for i in reversed(self.layers):
        #   i.calcwdeltas

if __name__=="__main__":
    if (len(sys.argv)<2):
        print('a good place to test different parts of your code')
        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1])
        y=np.array([0.01,0.99])
        
    elif(sys.argv[1]=='and'):
        print('learn and')
        
    elif(sys.argv[1]=='xor'):
        print('learn xor')