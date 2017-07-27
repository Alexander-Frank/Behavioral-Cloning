# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[mb_sample1]: ./images/10_samples_model_before0.png "Model Before sample 1"
[mb_sample2]: ./images/10_samples_model_before1.png "Model Before sample 2"
[mb_sample3]: ./images/10_samples_model_before2.png "Model Before sample 3"
[mb_sample4]: ./images/10_samples_model_before3.png "Model Before sample 4"
[mb_sample5]: ./images/10_samples_model_before4.png "Model Before sample 5"
[mb_sample6]: ./images/10_samples_model_before5.png "Model Before sample 6"
[mb_sample7]: ./images/10_samples_model_before6.png "Model Before sample 7"
[mb_sample8]: ./images/10_samples_model_before7.png "Model Before sample 8"
[mb_sample9]: ./images/10_samples_model_before8.png "Model Before sample 9"
[mb_sample10]: ./images/10_samples_model_before9.png "Model Before sample 10"

[m_sample1]: ./images/10_samples_model0.png "Model sample 1"
[m_sample2]: ./images/10_samples_model1.png "Model sample 2"
[m_sample3]: ./images/10_samples_model2.png "Model sample 3"
[m_sample4]: ./images/10_samples_model3.png "Model sample 4"
[m_sample5]: ./images/10_samples_model4.png "Model sample 5"
[m_sample6]: ./images/10_samples_model5.png "Model sample 6"
[m_sample7]: ./images/10_samples_model6.png "Model sample 7"
[m_sample8]: ./images/10_samples_model7.png "Model sample 8"
[m_sample9]: ./images/10_samples_model8.png "Model sample 9"
[m_sample10]: ./images/10_samples_model9.png "Model sample 10"

[histogram]: ./images/histogram.png "Histogram"
[histogram_2]: ./imageshistogram_2.png "Histogram 2nd training data set"
[histogram_gen_model_before]: ./images/histogram_gen_model_before.png "Generator data model ebfore"
[histogram_gen_model]: ./images/histogram_gen_model.png "Generator data model"
[history_before]: ./images/history_before.png "Training visualization model before"
[history_model]: ./images/history_model.png "Training visualization model"

[model_viz]: ./images/model_viz.png "Model architecture Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the final model (a two step training process was used)
* model_before.py containing the script to create and train the initial model
* drive.py for driving the car in autonomous mode
* model.h5 containing the final trained convolution neural network 
* model_before.h5 containing the initial trained convolution neural network
* load_and_save.py containing the script to load the weights and save my model
* model_weights.h5 containing the final model weights
* writeup_report.md summarizing the results
* Explore.ipynb summarizing the data exploration
* images containing all images
* videos containing all videos

---

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

I employed a two-step training aproach. The model used in both training steps remained the same. However, the data and augmentation techniques were changed.

The model_before.py file contains the code for the initial training and saving the convolutional neural network and its weights. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. I ran this code on an AWS server with GPU support. I had issues using the model.save() generated model when downloading it to my local machine. I worked around the problem by saving and downloading the model weights and using the load_and_save.py script to load the saved weights into the model on my local amchine and then saving it.

The model.py file contains the code for training the final model and saving the convolutional neural network and its weights. Procedure is the same as before. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the Nvidea Network architecture, which can be found [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

The model consists of a normalization layer using a Keras lambda layer, followed by 5 convolutional layers, followed by 3 fully connected layers. I added dropout in between the colvolutional and fully connected layers, which was not used in the original model.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 228). 

The model was trained and validated on different data sets using a two step training approach. First, I used the model_before.py code which does not iclude any augmentation. It only omits a some of the data of going straight and randomly flips images. The model weights were saved and loaded into the model.py code architecture, which included different data and 3 augmentation techniques, adapted from [this blog post](https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee). After looking at the blog post I decided to try out the 'he_normal' kernel initializer and ended up keeping it for my model. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track for both models.

After training the initial model the car made it around the track at 9mph (see videos/video_before.mp4) but failed at higher speeds (see videos/video_before_20mph.mp4).

After continuing training the model with my new data the car was going around the track smoother at 9mph (see videos/video.mp4) and also made it at 20mph (see videos/video_20mph.mp4). At higher speeds the car tends to ping pong between the lanes. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 243).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, smooth curve driving and going around the track the other way.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the Nvidea model architecture and tune that if needed.

My first step was to use the original Nvidea architecture. I thought this model might be appropriate because the task for which it was designed fits this task.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that all of my attemps yielded decent results on the mean squared error loss for both the validation and traing set. However, some models were keeping the car "hugging" the left lane, others ended up beeing trained too long and just driving straight (very smooth, but unfortunately it couldn't take the corners).

To combat the overfitting, I modified the model and added a droupout layer.

Then I adjusted the number of epochs, batch_size of the generator and augmentation techniques used.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, usually around the corners and the dirt curb. To improve the driving behavior in these cases, I recorded additional data and used the two-step training approach.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Below are the model training visualizations for both model, the model_before and model in this order).

![alt text][history_before]
![alt text][History_model]

#### 2. Final Model Architecture

My final model (model.py lines 206-243) uses the Nvidea model as a starting point:
* Input (66x200x3 color image)
* Conv Layer (24 filters, 5x5 kernel, strides 2x2 to 31x98x24)
* ELU
* Conv Layer (36 filters, 5x5 kernel, strides 2x2 to 14x47x36)
* ELU
* Conv Layer (48 filters, 5x5 kernel, strides 2x2 to 5x22x48)
* ELU
* Conv Layer (64 filters, 3x3 kernel, strides 1x1 to 3x20x64)
* ELU
* Conv Layer (64 filters, 3x3 kernel, strides 1x1 to 1x18x64)
* ELU
* Dropout 
* Flatten (1152)
* Fully connected layer (1152 -> 1164)
* ELU
* Fully connected layer (1164 -> 100)
* ELU
* Fully connected layer (100 -> 50)
* ELU
* Fully connected layer (50 -> 10)
* ELU
* Output (10 -> 1)

Below you can find a graphical representation of the model architecture.

![alt text][model_viz]

#### 3. Creation of the Training Set & Training Process
*for further details please see Explore.ipynb*

To capture good driving behavior for my first model training step, I recorded 2 laps driven in each dircetion on track one and special rounds with just the curvy section in each direction.

The following depicts the data distribution for the first training data set:

![alt text][histogram]

It includes 6355 data points (each data point includes 3 pictures: left, center, right).
Total number of images: 19065

I omitted some data of going straigt (model_before.py line 53-54) and randomly flipped images and angles.

Here is a sample histogram after calling the generator function:

![alt text][histogram_gen_model_before]

Here are 10 sample images:

![alt text][mb_sample1]
![alt text][mb_sample2]
![alt text][mb_sample3]
![alt text][mb_sample4]
![alt text][mb_sample5]
![alt text][mb_sample6]
![alt text][mb_sample7]
![alt text][mb_sample8]
![alt text][mb_sample9]
![alt text][mb_sample10]

The final model was trained with data augmentation used (brighness, flipping, shadows, shifts). 1st track driven 2 laps. Recovery lap driven with recordings from the side driving back towards the center of the road (from the left and right side).

The following depicts the data distribution for the second training data set:

![alt text][histogram_2]

It includes 3785 data points (each data point includes 3 pictures: left, center, right).
Total number of images: 11355

I omitted some of the data with negative steering angles (going left, model.py line 100-101)
I omitted some data of going straight (model.py line 105) and  flipped all images and angles.
I augmented all data by applying shadows, brightness adjustions and shifts.

Here is a sample histogram after calling the generator function:

![alt text][histogram_gen_model]

Here are 10 sample images:

![alt text][m_sample1]
![alt text][m_sample2]
![alt text][m_sample3]
![alt text][m_sample4]
![alt text][m_sample5]
![alt text][m_sample6]
![alt text][m_sample7]
![alt text][m_sample8]
![alt text][m_sample9]
![alt text][m_sample10]

For both trainings:

20% of the data are put into the validation set.
The generator shuffles the data.
The optimum number of epochs was determined by trial and error.