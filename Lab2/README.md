Jacob Trzcinski
CS 614
Lab 2
Dr. Adam Blumenthal

So my code definitely is not user friendly, not even to myself so I will do my best to explain it to you, 
but no promises that anything will work. If I had more time and experience, I would have made the variables
I changed into input variables so you could just write in what they are but I did not have the foresight
nor the motivation to overhaul my code after the fact. So, I will write out the parameters I used for each 
model and you will have to go through my code and find where to change the variables.

Fully-Connected Feed-Forward Neural Network Models:
- MEAN SQUARED ERROR   loss = tf.keras.losses.MSE:
    - Base:
        activation_function = 'tanh'
        layerSize = [64, 32]
        learning_rate -- commented out

    - Hidden Layer:
        activation_function = 'tanh'
        layerSize = [32, 32]
        learning_rate -- commented out
    
    - Learning Rate:
        activation_function = 'tanh'
        layerSize = [64, 32]
        learning_rate = 0.05 -- uncomment the code under the file name

    - ReLU:
        activation_function = 'relu'
        layerSize = [64, 32]
        learning_rate -- commented out

    - Swish:
        activation_function = swish  <-- literally just swish, we are calling my method, no quotation marks
        layerSize = [64, 32]
        learning_rate -- commented out

- CATEGORICAL CROSS-ENTROPY     loss = tf.keras.losses.categorical_crossentropy
    - see models from MSE

- CONVOLUTIONAL NETWORK MODEL:
    - Base:
        layer 1: 16 4 by 4      keras.layers.Conv2D(16, kernel_size = 4, activation = 'relu', input_shape = (8, 8, 1))
        layer 2: 8  2 by 2      keras.layers.Conv2D(8, kernel_size = 2, activation = 'relu', input_shape = (4, 4, 1))
        learning_rate -- commented out
        activation = 'relu'

    - TanH:
        same as base, but in the layers, change 'relu' to 'tanh'

    - learning Rate:
        see Base but uncomment out learning rate stuff under the filename

    - filter size:
        layer 1: 9 6 by 6   keras.layers.Conv2D(9, kernel_size = 6, activation = 'relu', input_shape = (8, 8, 1))
        layer 2:16 3 by 3   keras.layers.Conv2D(16, kernel_size = 3, activation = 'relu', input_shape = (4, 4, 1))
                Yes, I did mess up the input of the second layer but I'm in too deep now to wanna fix it.
        learning_rate -- commented out
        
