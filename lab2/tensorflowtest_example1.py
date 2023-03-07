import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from parameters import generateExample2

#Create a feed forward network
model=Sequential()

# Add convolutional layers, flatten, and fully connected layer
model.add(layers.Conv2D(1,3,input_shape=(5,5,1),activation='sigmoid')) 
model.add(layers.Flatten())
model.add(layers.Dense(1,activation='sigmoid'))

# Call weight/data generating function
# l1k1,l1k2,l1b1,l1b2,l2k1,l2b,l3,l3b,input, output = generateExample2()
np.random.seed(10)
input = np.random.rand(5,5)
output = np.random.rand(1)
l1k1 = np.random.rand(3,3)
l1b = np.random.rand(1)
l3 = np.random.rand(1,9)
l3b = np.random.rand(1)
#Set weights to desired values 

#setting weights and bias of first layer.
l1k1=l1k1.reshape(3,3,1,1)

model.layers[0].set_weights([l1k1,np.array([l1b[0]])]) #Shape of weight matrix is (w,h,input_channels,kernels)

#setting weights and bias of fully connected layer.
model.layers[2].set_weights([np.transpose(l3),l3b])

#Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
img=np.expand_dims(input,axis=(0,3))


#print needed values.
np.set_printoptions(precision=5)
print('model output before:')
print(model.predict(img))
sgd = optimizers.SGD(lr=100)
model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
history=model.train_on_batch(img,output)
print('model output after:')
print(model.predict(img))

print('1st convolutional layer, 1st kernel weights:')
print(np.squeeze(model.get_weights()[0][:,:,0,0]))
print('1st convolutional layer, 1st kernel bias:')
print(np.squeeze(model.get_weights()[1][0]))

print('fully connected layer weights:')
print(np.squeeze(model.get_weights()[2]))
print('fully connected layer bias:')
print(np.squeeze(model.get_weights()[3]))


