import math
import numpy as np
import csv
import os
from os import listdir
from os.path import isfile, join
from numpy import array
from numpy.linalg import norm
from numpy import sqrt
from numpy import loadtxt
import pandas as pd


##################################### Data structure ############################################

#data structure
# 0: OptoForce 1 x
# 1: OptoForce 1 y
# 2: OptoForce 1 z
# 3: Euclidean norm OptoForce1 x, y, z
# 4: OptoForce 2 x
# 5: OptoForce 2 y
# 6: OptoForce 2 z
# 7: Euclidean norm OptoForce1 x, y, z
# 8: OptoForce 3 x
# 9: OptoForce 3 y
# 10: OptoForce 3 z
# 11: Euclidean norm OptoForce1 x, y, z
# 12: finger angle 1
# 13: finger angle 2
# 14: finger angle 3



######################################### Functions #####################################



# load data from datafile
def load_data(NN_type):
	# load data
	global number_samples
	data = []
	data_buf = []
	labels = []
	#os.chdir('/')   # set working directory
	print('Working directory: ' + os.getcwd())
	filename =  '000_PhilipMaus_data.csv'
	print("Load data from " + filename)
	data_buf = np.array(pd.read_csv(filename, header=None))

	number_samples = int(data_buf[0][6])					# number of samples in data files
	nMeas = int(((data_buf.shape[0]-1)/number_samples)-2)	# number of measurements per sample
	header = data_buf[0]
	data_with_header = np.reshape(data_buf[1:],(number_samples, int((data_buf.shape[0]-1)/number_samples), data_buf.shape[1]))
	data = np.array(data_with_header[:,2:,1:],dtype=float)
	for i in range(0,data_with_header.shape[0]):
		labels = np.append(labels, int(data_with_header[i,0,10]))

	# only do weight compensation and normalization for original dataset
	# augmented dataset have that already done
	if filename == '000_PhilipMaus_data.csv':
		# object weight compensation
		print("Object weight compensation")
		for i in range(0,data.shape[0]):
			# compensate object weight effect on finger 2
			weight_2 = 0
			if np.amax(data[i,0:3,6]) > 0.05:	# only compensate if initial measurement over limit value
				weight_2 = np.mean(data[i,0:3,6])	# get weight as average of first 3 force measurments
				for j in range(0,data.shape[1]):	# for each timestep
					data[i,j,6] -= weight_2 * math.cos((data[i,j,13]-20)/360*2*math.pi)	# normal force compensation
					data[i,j,5] += weight_2 * math.sin((data[i,j,13]-20)/360*2*math.pi)	# tangential force compensation
			# compensate object weight effect on finger 2
			weight_3 = 0
			if np.amax(data[i,0:3,10]) > 0.05:
				weight_3 = np.mean(data[i,0:3,10])
				for j in range(0,data.shape[1]):
					data[i,j,10] -= weight_3 * math.cos((data[i,j,13]-20)/360*2*math.pi)
					data[i,j,9] += weight_3 * math.sin((data[i,j,13]-20)/360*2*math.pi)

		# normalize each feature over all samples to mean value 0 and std 1
		print("Data normalization")
		for j in range(0,data.shape[2]):
			data[:,:,j] = data[:,:,j] - np.mean(data[:,:,j])
			data[:,:,j] = data[:,:,j] / np.std(data[:,:,j])

	# one finger only
	#data = np.delete(data,[4,5,6,7,8,9,10,11],axis=2)
	#print(data.shape)

	# adapt data to NN type needed input
	merged_data = np.zeros((data.shape[0], data.shape[1], data.shape[2]+1))
	for j in range(0,data.shape[0]):
		label_buf = np.zeros((data.shape[1])) + labels[j]
		label_buf.shape = (50,1)
		merged_data[j,:,:] = np.hstack((data[j,:,:], label_buf))
		
	#print("merged_data", merged_data)
	if NN_type == "MLP":		# get last sample only
		data_MLP = []
		for i in range(0,len(merged_data)):
			data_MLP.append(merged_data[i][-1])
		data_MLP = np.array(data_MLP)
		return data_MLP
	elif NN_type == "LSTM":
		return merged_data
	elif NN_type == "CNN" or NN_type == "CNNLSTM":	# [sample,timesteps,features,channel]
		data_CNN2D = merged_data.reshape(merged_data.shape[0],merged_data.shape[1], merged_data.shape[2],1)
		return data_CNN2D
	else:
		print("Wrong NN type input")
		exit()


NN_type = "CNN" 	# options: MLP, LSTM, CNN, CNNLSTM, ConvLSTM
data_augm_factor = '3'
data_augm_type = 'jitter'
data = load_data(NN_type)
print(data.shape)