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
        #for now, I'm going to assume we have the bias included in the number of inputs, and weights
        # ^^^ if that's not how we do it, change this code and update this comment ^^^ 
        self.numinps = input_num
        self.lr = lr
        self.weights = []
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
    def calculate(self,input):
        self.input = input
        x = 0
        #   here, I put range(len(input)) + 1, so the bias would be included, 
        #       and when it gets to the bias, the input is always 1, hence the if statement
        for i in range(len(input)) + 1:
            if i > range(len(input)):
                x += 1*self.weights[i]
            else:
                x += input[i]*self.weights[i]
        self.output = activate(x)
        return self.output

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        #derivative of log is output(1-output)
        #derivative of linear is ?
        if self.act == 1:
            return self.output * (1 - self.output)
        else:
            return
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        deltaw = 0
        self.partials = []
        # given wtimesdelta, deltaw is the sum of wtimesdelta * each input
        # given wtimesdelta, partials is a list of wtimesdelta * each weight
        for i in range(len(self.weights)):
            ding = wtimesdelta * self.weights[i] * self.inps[i]
            deltaw += ding
            self.partials.append(ding * activationderivative() * self.weights[i])
        return deltaw
    
    #Simply update the weights using the partial derivatives and the learning rate
    def updateweight(self):
        newweights = []
        for i in range(len(self.weights)):
            newweights.append(self.weights[i] - self.partials[i]*self.lr)
        self.weights = []
        for i in range(len(newweights)):
            self.weights.append(newweights[i])

        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation, the input size, 
    # the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        print('constructor') 
        
        
    #calcualte the output of all the neurons in the layer and return a vector with those values 
    # (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        print('calculate') 
        
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() 
    # for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). 
    # I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        print('calcwdeltas') 
           
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), 
    # the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
        print('constructor') 
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        print('constructor')
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        print('calculate')
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        print('lossderiv')
    
    #Given a single input and desired output preform one step of backpropagation 
    # (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values)
    def train(self,x,y):
        print('train')

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