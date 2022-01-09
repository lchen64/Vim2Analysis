import tensorflow as tf
from tensorflow.keras.models import Model
from keras.models import model_from_json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers

from tikreg import models
from tikreg import utils as tikutils


def regression_model(output_size, summary=False, backend='tf'):
    """ Return the Keras model of the multi-output ridge regression network
    """
    model = Sequential()
    model.add(Dense(8192, input_dim= 8192, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(.5))
    model.add(Dense(output_size, activation='softmax'))

    if summary:
        print(model.summary())

    return models

def train_regressor(regressor,features,responses):

    regressor.compile(loss='mean_squared_error', optimizer='adam')
    regressor.fit(features, responses)

def tikreg_model((Xtrain, Xtest), (Ytrain, Ytest)):
	# Specify fit
	options = dict(ridges=np.logspace(0,3,11), weights=True, metric='rsquared')
	fit = models.cvridge(Xtrain, Ytrain, Xtest, Ytest, **options)

	# Evaluate results
	weights_estimate = fit['weights']
	
	return (fit['cvresults'].shape) # (5, 1, 11, 2): (nfolds, 1, nridges, nresponses)

#def banded_ridge():

#Extract Features from Architecture
class FeatureHook():
    def __init__(self, module):
        def hook_fn(module, input, output):
            self.features = output.clone().detach()

        self.hook = module.register_forward_hook(hook_fn)

    def close(self):
        self.hook.remove()

def optimizer(self, lr, momentum):
    l2_reg = 0 if self.reg == "L1" else self.alpha
    return optim.SGD(self.regression_layer.parameters(), lr=lr, momentum=momentum, weight_decay=l2_reg)

