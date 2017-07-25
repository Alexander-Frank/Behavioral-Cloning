# imports for reading the data
import os
import csv
from sklearn.model_selection import train_test_split

# path to data Folder
path = './PATH/'

# read in the csv
samples = []
with open(path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# split data into train and validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# make all necessary imports for working with the data
import cv2
import numpy as np
import sklearn
from scipy.misc import imresize
import random

# define a generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                                
                # read in all 3 images
                name = path+'IMG/'+batch_sample[0].split('\\')[-1]
                name_left = path+'IMG/'+batch_sample[1].split('\\')[-1]
                name_right = path+'IMG/'+batch_sample[2].split('\\')[-1]

                # read in steering angle
                center_angle = float(batch_sample[3])

                # make random number to omit certian samples with a probaility
                random_num = random.random()

                # correction parameter for images left and right
                correction = 0.25
				
				# randomly omit 70% of driving "straight" (angle between -0.15 and 0.15)                
                if center_angle < 0.15 and center_angle > -0.15 and random_num > 0.3:
                    pass

                else:
                    # create adjusted steering measurements for the side camera images
                    steering_left = center_angle + correction
                    steering_right = center_angle - correction
                
                    center_image = cv2.imread(name)
                    left_image = cv2.imread(name_left)
                    right_image = cv2.imread(name_right)

                    # trim image to only see section with road and resize for nvidea model (66 x 200)
                    center_image = center_image[68:136, 0:320]
                    center_image = cv2.resize(center_image, (200, 66))

                    left_image = left_image[68:136, 0:320]
                    left_image = cv2.resize(left_image, (200, 66))

                    right_image = right_image[68:136, 0:320]
                    right_image = cv2.resize(right_image, (200, 66))
					
					# randomly flip images with a probability of 50%
                    if random_num > 0.5:
                        images.extend([center_image, left_image, right_image])
                        angles.extend([center_angle, steering_left, steering_right])

                    else:
                        # flip images
                        image_flipped_c = np.fliplr(center_image)
                        measurement_flipped_c = -center_angle

                        image_flipped_l = np.fliplr(left_image)
                        measurement_flipped_l = -steering_left

                        image_flipped_r = np.fliplr(right_image)
                        measurement_flipped_r = -steering_right

                        images.extend([image_flipped_c, image_flipped_l, image_flipped_r])
                        angles.extend([measurement_flipped_c, measurement_flipped_l, measurement_flipped_r])
                        
            X_train = np.array(images)          
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

# define needed variables
ch, row, col = 3, 66, 200 # Trimmed image format 

# make all imports for the keras model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Conv2D, Dropout, Cropping2D
from keras.layers.advanced_activations import ELU

# define keras model based on the nvidea model architecture
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
		
# add convolution layers
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

# add dropout
model.add(Dropout(0.5))

# flatten and add fully connected layers
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

# comiple model
model.compile(loss='mse', optimizer='adam')

# train model and create history object for visualization
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/64,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)/64, epochs=50)

# save model and model weights
'''
This script was run on the AWS server.
I only used the model weights and then loaded these and saved the model
using load_and_save.py
'''
model.save('model_before.h5')
model.save_weights('model_weights_before.h5')

# make imports for visualizing the data
import matplotlib as mpl
mpl.use('Agg') # for running it on the server
import matplotlib.pyplot as plt

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(history_object.history['loss'])
ax.plot(history_object.history['val_loss'])
ax.set_title('model mean squared error loss')
ax.set_ylabel('mean squared error loss')
ax.set_xlabel('epoch')
ax.legend(['training set', 'validation set'], loc='upper right')
fig.savefig('./history_before.png')

print('ALL DONE')
