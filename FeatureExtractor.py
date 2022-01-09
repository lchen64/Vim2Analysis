import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers

def create_model(summary=False, backend='tf'):
	    """ Return the Keras model of the network
	    """
	    model = Sequential()
	  
	    input_shape=(16, 112, 112, 3) # l, h, w, c
	   
	    model.add(Convolution3D(64, 3, 3, 3, activation='relu',
	                            border_mode='same', name='conv1',
	                            input_shape=input_shape,kernel_regularizer = regularizers.l1(0.01)))
	    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
	                           border_mode='valid', name='pool1'))
	    # 2nd layer group
	    model.add(Convolution3D(128, 3, 3, 3, activation='relu',
	                            border_mode='same', name='conv2',kernel_regularizer=regularizers.l1(0.01)))
	    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
	                           border_mode='valid', name='pool2'))
	    # 3rd layer group
	    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
	                            border_mode='same', name='conv3a',kernel_regularizer=regularizers.l1(0.01)))
	    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
	                            border_mode='same', name='conv3b',kernel_regularizer=regularizers.l1(0.01)))
	    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
	                           border_mode='valid', name='pool3'))
	    # 4th layer group
	    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
	                            border_mode='same', name='conv4a',kernel_regularizer=regularizers.l1(0.01)))
	    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
	                            border_mode='same', name='conv4b',kernel_regularizer=regularizers.l1(0.01)))
	    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
	                           border_mode='valid', name='pool4'))
	    # 5th layer group
	    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
	                            border_mode='same', name='conv5a',kernel_regularizer=regularizers.l1(0.01)))
	    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
	                            border_mode='same', name='conv5b',kernel_regularizer=regularizers.l1(0.01)))
	    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
	    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
	                           border_mode='valid', name='pool5'))
	    model.add(Flatten())
	    
	    # FC layers group
	    model.add(Dense(4096, activation='relu', name='fc6', kernel_regularizer=regularizers.l1(0.01)))
	    model.add(Dropout(.5))
	    model.add(Dense(4096, activation='relu', name='fc7', kernel_regularizer=regularizers.l1(0.01)))
	    model.add(Dropout(.5))
	    model.add(Dense(487, activation='softmax', name='fc8'))
	    
	    if summary:
	        print(model.summary())

	    return model

def intermediate_layer_model(layer, model):

 	return Model(inputs=model.input, outputs=model.get_layer(layer).output)

def feature_hook(layer, model, data):

	model = intermediate_layer_model(layer, model)
	return model(data)

def create_alexnet_model(summary=False, backend='tf'):
	
	model = Sequential()

	# 1st Convolutional Layer
	model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding=’valid’))
	model.add(Activation(‘relu’))
	# Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’))

	# 2nd Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding=’valid’))
	model.add(Activation(‘relu’))
	# Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’))

	# 3rd Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=’valid’))
	model.add(Activation(‘relu’))

	# 4th Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=’valid’))
	model.add(Activation(‘relu’))

	# 5th Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=’valid’))
	model.add(Activation(‘relu’))
	# Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’))

	# Passing it to a Fully Connected layer
	model.add(Flatten())
	# 1st Fully Connected Layer
	model.add(Dense(4096, input_shape=(224*224*3,)))
	model.add(Activation(‘relu’))
	# Add Dropout to prevent overfitting
	model.add(Dropout(0.4))

	# 2nd Fully Connected Layer
	model.add(Dense(4096))
	model.add(Activation(‘relu’))
	# Add Dropout
	model.add(Dropout(0.4))

	# 3rd Fully Connected Layer
	model.add(Dense(1000))
	model.add(Activation(‘relu’))
	# Add Dropout
	model.add(Dropout(0.4))

	# Output Layer
	model.add(Dense(17))
	model.add(Activation(‘softmax’))

	if summary:
	    print(model.summary())

	return model







