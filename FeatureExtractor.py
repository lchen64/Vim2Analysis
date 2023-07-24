#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tables
import skimage.transform as st

chunk_number = 400
chunk_length = 16
C3D_input_size = 112
sliding_window_size = 16
sliding_window_stride = 1
number_of_windows = 107985
num_windows_test = 8100 - 15
original_size = 128
number_of_frame_to_load = number_of_windows -  sliding_window_size + 1

resized_stimulus_path = "/resized/resized_stimulus.npy"
extracted_features_path = "resized/extracted_features_train.npy"
extracted_features_path_test = "resized/extracted_features_test.npy"


# In[2]:


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = 'x'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
        
'''
printProgressBar(0, 10, prefix = 'Prefix:', suffix = 'Complete', length = 20)
for i in range(10):
    # Do something
    printProgressBar(i + 1, 10, prefix = 'Prefix:', suffix = 'Complete', length = 20)
'''


# In[5]:


######################
#    Load Dataset    #
######################

def roi_index(region, file):
	roi = file.get_node('/roi/v1lh')[:].flatten()
	return np.nonzero(roi==region)[0] 

def load_train_stimulus():
	stimuli = tables.open_file('Stimuli.mat')
	return stimuli.get_node('/st')[:]

def load_train_response(subject, roi):
	path = "VoxelResponses_subject" + subject + ".mat"
	response = tables.open_file(path) 
	data = response.get_node('/rt')[:]
	return data[roi_index(roi, response)]

def load_train_response_all(subject):
	path = "VoxelResponses_subject" + subject + ".mat"
	response = tables.open_file(path) 
	data = response.get_node('/rt')[:]
	return data

def load_validation_stimulus():
	stimuli = tables.open_file('Stimuli.mat')
	return stimuli.get_node('/sv')[:]

def load_validation_response(subject, roi):
	path = "VoxelResponses_subject" + subject + ".mat"
	response = tables.open_file(path) 
	data = response.get_node('/rv')[:]
	return data[roi_index(roi, response)]

def load_validation_response_all(subject):
	path = "VoxelResponses_subject" + subject + ".mat"
	response = tables.open_file(path) 
	data = response.get_node('/rv')[:]
	return data

stimulus_train = load_train_stimulus()


# In[8]:


np.save("stimulus_train.npy", np.asarray(stimulus_train))
print("Stimulus Loaded. Shape:" + str(stimulus_train.shape))


# In[9]:


######################
#     Load Model     #
######################

import h5py
import tensorflow as tf

from keras.models import Model
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D

def create_model():
    """ Creates model object with the sequential API:
    https://keras.io/models/sequential/
    """

    model = Sequential()
    input_shape = (16, 112, 112, 3)

    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                     padding='same', name='conv1',
                     input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                     padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3a'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    return model

def intermediate_layer_model(layer, model):

 	return Model(inputs=model.input, outputs=model.get_layer(layer).output)

def feature_hook(layer, model, data):

	model = intermediate_layer_model(layer, model)
	return model(data)

def create_features_extractor(model, layer_name):
    extractor = Model(inputs= model.input,
                      outputs= model.get_layer(layer_name).output)
    return extractor

model = create_model()
model.summary()


# In[10]:


model.load_weights('c3d-sports1M_weights.h5' , by_name = True) 
print("Weights Loaded")


# In[11]:


#####################################
# Pick a Layer to Create the Extractor #
#####################################
output_layer_name = 'flatten'
extractor = create_features_extractor(model,output_layer_name)
extractor.summary()


# In[7]:


################################
#   Extract TrainSet Features  #
################################

#extracted_feature = np.zeros((number_of_windows,extractor.output.shape[1]))
extracted_feature = np.zeros((number_of_windows,8192))

print("Starting to extract features. Expected output:" + str(extracted_feature.shape))
printProgressBar(0, number_of_windows, prefix = 'Progress:', suffix = '', length = 100)
for i in range(number_of_windows):
    chunk = stimulus_train[i:i+sliding_window_size, :,:,:]
    chunk_transposed = np.transpose(chunk,(0,2,3,1))
    chunk_resized = st.resize(chunk_transposed, (sliding_window_size,C3D_input_size, C3D_input_size,3))
    to_be_fed = np.zeros((1,sliding_window_size, C3D_input_size, C3D_input_size, 3))
    to_be_fed[0,:,:,:,:] = chunk_resized
    extracted_feature[i,:] = extractor.predict(to_be_fed)
    printProgressBar(i + 1, number_of_windows, prefix = 'Progress:', suffix = '', length = 100)

#save to file
np.save(extracted_features_path, extracted_feature)


# In[ ]:


checking = np.load(extracted_features_path)
print(str(checking.shape))
print(str(checking.mean()))


# In[ ]:


stimulus_test = load_validation_stimulus()


# In[ ]:


#############################
#   Extract Test Features  #
#############################

#extracted_feature = np.zeros((number_of_windows,extractor.output.shape[1]))
extracted_feature = np.zeros((num_windows_test, 8192))

print("Starting to extract features. Expected output:" + str(extracted_feature.shape))
printProgressBar(0, num_windows_test, prefix = 'Progress:', suffix = '', length = 100)
for i in range(num_windows_test):
    chunk = stimulus_test[i:i+sliding_window_size, :,:,:]
    chunk_transposed = np.transpose(chunk,(0,2,3,1))
    chunk_resized = st.resize(chunk_transposed, (sliding_window_size, C3D_input_size, C3D_input_size,3))
    to_be_fed = np.zeros((1,sliding_window_size, C3D_input_size, C3D_input_size, 3))
    to_be_fed[0,:,:,:,:] = chunk_resized
    extracted_feature[i,:] = extractor.predict(to_be_fed)
    printProgressBar(i + 1, num_windows_test, prefix = 'Progress:', suffix = '', length = 100)


# In[ ]:


#save to file
np.save(extracted_features_path_test, extracted_feature)


# In[ ]:


print(str(extracted_feature.shape))

