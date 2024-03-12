import pickle
import numpy as np
import tensorflow as tf
import os

def unpickle(file):
	"""
	CIFAR data contains the files data_batch_1, data_batch_2, ..., 
	as well as test_batch. 

	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def get_data(file_path, first_class, second_class):
	"""
	Given a file path and two target classes, returns an array of 
	normalized inputs (images) and an array of labels. 
	:param file_path: file path for inputs and labels, something 
	like 'CIFAR_data_compressed/train'
	:param first_class:  an integer (0-9) representing the first target
	class in the CIFAR10 dataset, for a cat, this would be a 3
	:param first_class:  an integer (0-9) representing the second target
	class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where 
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
	has size (num_examples, num_classes)
	"""
	unpickled_file = unpickle(file_path)
	inputs = unpickled_file[b'data']
	labels = unpickled_file[b'labels']
	
	#filter for 2 classes
	inputs = np.asarray(inputs)
	labels = np.asarray(labels)
	indices = np.nonzero((labels == first_class) | (labels == second_class))[0]

	filtered_inputs = inputs[indices]
	filtered_labels = labels[indices]

	filtered_inputs = tf.reshape(filtered_inputs, (-1, 3, 32 ,32))
	filtered_inputs = tf.transpose(filtered_inputs, perm=[0,2,3,1])

	filtered_labels = np.where(filtered_labels == 3, 0, 1)
	one_hot_labels = tf.one_hot(filtered_labels, 2)

	normalized_inputs = filtered_inputs / 255
	return normalized_inputs, one_hot_labels
