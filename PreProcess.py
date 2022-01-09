import skimage.transform as st
import numpy
from tempfile import TemporaryFile
from scipy import signal
import pandas as pd

def resize(stimulus):
	resized = numpy.empty((108785, 3, 112, 112))
	i = 0
	for image in stimulus:
		resized[i] = st.resize(image, (3, 112, 112))
		i+=1
	outfile = TemporaryFile()
	np.save(outfile, resized)

def upsample(response, timepoints):
	upsampled = numpy.empty((73728, timepoints))
	return signal.resample(response, timepoints)

def dropna(response):
	response = pd.DataFrame(response)
	response = response.dropna()
	return response.to_numpy()