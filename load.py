import tables, numpy

def roi_index(region, file):

	roi = file.get_node('/roi/' + region)[:].flatten()
	return numpy.nonzero(roi==1)[0]

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

def resize(images, dim):
	return st.resize(images, dim)

