"""
This script is for the fully-connected feed-forward NN
"""
#   imports
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import keras
from keras.callbacks import EarlyStopping
from keras import optimizers
import matplotlib.pyplot as plt
import math

#   read training data
print("Reading data...")
trainingData_RAW = pd.read_csv("optdigits/optdigits.tra",dtype=np.int32,header=None)
trainingData_RAW = np.asarray(trainingData_RAW)
#print(trainingData_RAW)
print("Done.\n Processing data...")
### Each data point is an 8x8 matrix of already normalized handwritten characters
### + 1 class attribute that is val 0-9

#print(trainingData_RAW.shape[1])
def splitData(data, percent:float):
    ### cool thing I found out, if you put the dtype after a semicolon in the def input
    ### when you hover a call of the def, it will show what dtype is expected
    """
    returns: verifyingData, trainingData
    data: the data as an array that you want split up 
    percent: the fraction of the data you want to be verification data
    """
    trainingData_split = np.empty(shape=(0,data.shape[1]), dtype=int)
    verifyData_split = np.empty(shape=(0,data.shape[1]), dtype=int)
    for i in data:  
        ### somehow i is already the whole row, 
        ### found out after an hour of debugging...
        ### the more you know /star
        if random.random() <= percent:
            verifyData_split = np.vstack((verifyData_split, i))
        else:
            trainingData_split = np.vstack((verifyData_split, i))
    del data
    return verifyData_split, trainingData_split

verData_RAW, traData_RAW = splitData(trainingData_RAW, 0.2)

#   split the data into inputs and labels
### I'm running out of different ways to say that this data is not ready
verData = verData_RAW[:,:-1]    # this is fine
verY_unprocessed = verData_RAW[:,-1]    # this is unreadable to our NN
traData = traData_RAW[:,:-1]    # this is fine
traY_unprocessed = traData_RAW[:,-1]    # this is unreadable to our NN

#   process those labels into 1x10 arrays
def processLabel(labelArray):
    """
    labelArray: the whole list of labels you want converted
    """
    outputArray = np.empty(shape=(0,10), dtype=int)
    for i in labelArray:
        tempArray = np.array([0,0,0,0,0,0,0,0,0,0])
        tempArray[i] = 1
        outputArray = np.vstack((outputArray, tempArray))
        del tempArray
    
    return outputArray

verY = processLabel(verY_unprocessed)
traY = processLabel(traY_unprocessed)   # oh yeah, it's all coming together
print("Done.\n Tidying up...")

#   do a little bit of tidying
### I know that I have 32GB of RAM but I still wanna clean up bc clean code is good code
del trainingData_RAW
del verData_RAW
del traData_RAW
del traY_unprocessed
del verY_unprocessed
print("Done.")
#print(verData.shape[1])
#print(verData.shape[0])
#print(verY)

filename = "FC_CCE_tanh_HL"

#   define some funky activation functions to try
def swish(x):
    return x / (1 + math.e**(-x))

#learningRate = 0.05
#sgd = optimizers.SGD(lr=learningRate)

#   NN time
layerSize = [32, 32]

#   define our model
print("Making model...")
model = keras.Sequential()

#   build the input layer
model.add(keras.layers.Dense(64, input_dim=64))

#   build the hidden layers
for i in layerSize:
    model.add(keras.layers.Dense(i, activation='tanh'))

#   build our output layer
model.add(keras.layers.Dense(10, activation='softmax'))
print("Layers are defined.")

#   throw it in the oven
print("Compiling model...")
model.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer = 'sgd', metrics = ['accuracy'])

#   for 100 epochs or when the dough is fully risen
print("Training...")
stoppingEpoch = EarlyStopping(monitor='val_accuracy', mode = 'max', verbose = 1, patience = 10)
history = model.fit(traData, traY, epochs=1000, batch_size=traData.shape[1], validation_data=(verData, verY), validation_batch_size=verData.shape[1],callbacks=[stoppingEpoch])
print("Done.\n Saving and graphing....")

model.save('FC_CCE/' + filename + '.h5')
history_df = pd.DataFrame(history.history)
history_df.to_csv(('FC_CCE/' + filename + '.csv'), index=False)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.grid()
plt.savefig('FC_CCE/accuracy_' + filename + '.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'val'], loc='upper left')
plt.grid()
plt.savefig('FC_CCE/losses_' + filename + '.png')
plt.show()
