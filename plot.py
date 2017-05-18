import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from GloVe import *

def execute_several():

	GloVe('3')
	GloVe('4')
	GloVe('5')
	GloVe('6')
	GloVe('7')

def plot_epochs():

	path = 'Data/history/struct1param1'

	val_acc = np.zeros((7, 8))
	train_acc = np.zeros((7, 8))

	#dummy = np.load(path+'_vall_accFOLD1.npy')
	#print(dummy.shape)
	#exit()

	val_acc[0] = np.load(path+'_vall_accFOLD1.npy')
	val_acc[1] = np.load(path+'_vall_accFOLD2.npy')
	val_acc[2] = np.load(path+'_vall_accFOLD3.npy')
	val_acc[3] = np.load(path+'_vall_accFOLD4.npy')
	val_acc[4] = np.load(path+'_vall_accFOLD5.npy')
	val_acc[5] = np.load(path+'_vall_accFOLD6.npy')
	val_acc[6] = np.load(path+'_vall_accFOLD7.npy')
	#val_acc[7,:] = np.load(path+'_vall_accFOLD8.npy')
	train_acc[0] = np.load(path+'_train_accFOLD1.npy')
	train_acc[1] = np.load(path+'_train_accFOLD2.npy')
	train_acc[2] = np.load(path+'_train_accFOLD3.npy')
	train_acc[3] = np.load(path+'_train_accFOLD4.npy')
	train_acc[4] = np.load(path+'_train_accFOLD5.npy')
	train_acc[5] = np.load(path+'_train_accFOLD6.npy')
	train_acc[6] = np.load(path+'_train_accFOLD7.npy')
	#train_acc[7,:] = np.load(path+'_train_accFOLD8.npy')

	#print(val_acc)
	val_acc_mean = np.mean(val_acc, axis=0)
	#print("mean:")
	#print(val_acc_mean)
	val_acc_std = np.std(val_acc, axis=0)
	train_acc_mean = np.mean(train_acc, axis=0)
	train_acc_std = np.std(train_acc, axis=0)

	plt.figure(1)
	plt.title("Categorical accuracy over epochs of training", fontsize=30)
	valid = plt.errorbar(np.arange(1,9), val_acc_mean, val_acc_std, fmt='-s', label = 'Test Accuracy')
	train = plt.errorbar(np.arange(1,9), train_acc_mean, train_acc_std, fmt='-s',label = 'Train Accuracy')
	plt.xlabel("Epoch", fontsize = 20)
	plt.ylabel("Categorical accuracy", fontsize = 20)
	plt.legend(loc=4, fontsize=20)
	plt.rc('xtick', labelsize=20) 
	plt.rc('ytick', labelsize=20)
	plt.xlim([0.9,8.1])
	plt.grid()
	plt.show()

	# **** FILTERS AND HIDDEN DIMS *********

	filt_acc_mean = np.load('Data/structure/filters_acc_mean.npy')
	filt_acc_std = np.load('Data/structure/filters_acc_std.npy')

	print(filt_acc_std.shape)

	filters_list = [1, 10, 50, 100, 200, 250, 300, 350, 400, 500]

	plt.figure(2)
	plt.title("Influence of the filter size on accuracy", fontsize=30)
	plt.errorbar(filters_list, filt_acc_mean[:,1], filt_acc_std[:,1], fmt='-s', label = 'Test Accuracy')
	#plt.errorbar(filters_list, filt_acc_mean[:,0], filt_acc_std[:,0], fmt='-s', label = 'Train Accuracy')
	plt.xlabel("Number of convolutional filters", fontsize = 20)
	plt.ylabel("Categorical accuracy", fontsize = 20)
	plt.legend(loc=4, fontsize=20)
	plt.rc('xtick', labelsize=20) 
	plt.rc('ytick', labelsize=20)
	plt.xlim([5,505])
	plt.ylim([0.6,0.65])
	plt.grid()
	plt.show()

	dims_acc_mean = np.load('Data/structure/hidden_dims_acc_mean.npy')
	dims_acc_std = np.load('Data/structure/hidden_dims_acc_std.npy')

	hidden_list = [50, 100, 200, 300, 400, 500]

	plt.figure(3)
	plt.title("Influence of the hidden dimension on accuracy", fontsize=30)
	plt.errorbar(hidden_list, dims_acc_mean[:,1], dims_acc_std[:,1], fmt='-s', label = 'Test Accuracy')
	#plt.errorbar(hidden_list, dims_acc_mean[:,0], dims_acc_std[:,0], fmt='-s', label = 'Train Accuracy')
	plt.xlabel("Number of nodes of the hidden layer", fontsize = 20)
	plt.ylabel("Categorical accuracy", fontsize = 20)
	plt.legend(loc=4, fontsize=20)
	plt.rc('xtick', labelsize=20) 
	plt.rc('ytick', labelsize=20)
	plt.xlim([45,505])
	plt.ylim([0.6,0.65])
	plt.grid()
	plt.show()

