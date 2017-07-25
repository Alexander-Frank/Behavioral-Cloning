'''
This sccript is used to load my model weights into the architecture and 
save the entire model.

This was necessary because I was experiencing issues pulling the entire model 
from the AWS server I was using. On the server I just saved the model weights.
'''

# make all imports for the keras model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Conv2D, Dropout, Cropping2D
from keras.layers.advanced_activations import ELU

# define input shape
ch, row, col = 3, 66, 200

# create model in keras
# HAS TO BE THE SAME AS USED DURING TRAINING

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
model.add(Conv2D(24, (5,5), strides=(2,2), padding='valid',
                        kernel_initializer='he_normal'))
model.add(ELU())
model.add(Conv2D(36, (5,5), strides=(2,2), padding='valid',
                        kernel_initializer='he_normal'))
model.add(ELU())
model.add(Conv2D(48, (5,5), strides=(2,2), padding='valid',
                        kernel_initializer='he_normal'))
model.add(ELU())
model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid',
                        kernel_initializer='he_normal'))
model.add(ELU())
model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid',
                        kernel_initializer='he_normal'))
model.add(ELU())
model.add(Flatten())
model.add(Dense(1164, kernel_initializer='he_normal'))
model.add(ELU())
model.add(Dense(100, kernel_initializer='he_normal'))
model.add(ELU())
model.add(Dense(50, kernel_initializer='he_normal'))
model.add(ELU())
model.add(Dense(10, kernel_initializer='he_normal'))
model.add(ELU())
model.add(Dense(1, kernel_initializer='he_normal'))

model.compile(loss='mse', optimizer='adam')

# load weights and save entire model
model.load_weights('model_weights.h5')
model.save('model.h5')
