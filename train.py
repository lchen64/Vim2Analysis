from load import load_train_stimulus, load_train_response, load_validation_stimulus, load_validation_response, resize
from FeatureExtractor import create_model, intermediate_layer_model, feature_hook
from MultiOutputRegression import regression_model

from keras.optimizers import SGD
from keras import regularizers
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_history(self, epochs, train_losses, val_losses):
        train_epochs = np.linspace(0, epochs, len(train_losses))
        val_epochs = np.linspace(0, epochs, len(val_losses))
        plt.clf()
        plt.plot(train_epochs, train_losses)
        plt.plot(val_epochs, val_losses)
        plt.xlabel("Epochs")
        plt.legend(["Training Loss", "Validation Loss"])
        plt.savefig('results/{}/history.png'.format(self.output_name))

def compute_correlations(model, dataloader):
    model.evaluate()
    all_preds, all_targets = None, None
    for data, targets in dataloader:
        preds = model(data).detach().numpy()
        targets = targets
        if all_preds is None:
            all_preds = preds
            all_targets = targets
        else:
            all_preds = np.vstack((all_preds, preds))
            all_targets = np.vstack((all_targets, targets))
    corrs = []
    for i in range(all_preds.shape[1]):
        corr, _ = pearsonr(all_preds[:, i], all_targets[:, i])
        corrs.append(corr)
    return corrs

def write_correlations(cell_ids, corrs, output_name):
    rows = sorted(zip(cell_ids, corrs), key=lambda x: x[1], reverse=True)
    with open(output_name, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Cell Id', 'Correlation'])
        for row in rows:
            writer.writerow(row)

def plot_corr_histogram(corrs, output_name):
    plt.clf()
    plt.hist(corrs)
    plt.xlabel("Correlation")
    plt.ylabel("Number of Cells")
    plt.savefig(output_name)

def average_r2(voxels):
    rsquare = np.empty(voxels)
    for i in range(y_pred.shape[0]):
        rsquare[i] = r2_score(upsampled_test[i], y_pred[i])
    return np.average(rsquare) 


if __name__ == '__main__':

    #save model to json for future use
	json_string = model.to_json()
	with open('sports1M_model.json', 'w') as f:
	    f.write(json_string)

	train_regressor(features, responses)
	




