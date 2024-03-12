from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
	"""
	Performs 2D convolution given 4D inputs and filter Tensors.
	:param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
	:param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
	:param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
	:param padding: either "SAME" or "VALID", capitalization matters
	:return: outputs, Tensor with shape [num_examples, output_height, output_width, output_channels]
	"""
	num_examples = inputs.shape[0]
	in_height = inputs.shape[1]
	in_width = inputs.shape[2]
	input_in_channels = inputs.shape[3]

	filter_height = filters.shape[0]
	filter_width = filters.shape[1]
	filter_in_channels = filters.shape[2]
	filter_out_channels = filters.shape[3]

	num_examples_stride = strides[0]
	strideY = strides[1]
	strideX = strides[2]
	channels_stride = strides[3]

	# Check if input channels equivalent to filter channels
	assert input_in_channels == filter_in_channels

	# Cleaning padding input

	pad_height = 0
	pad_width = 0
	if padding == "SAME":
		# pad_height = math.floor((filter_height - 1) / 2)
		pad_height = (filter_height - 1) / 2
		# pad_width = math.floor((filter_width - 1) / 2)
		pad_width = (filter_width - 1) / 2

	# pad input
	if filter_height % 2 == 0:
		padded_inputs = np.pad(inputs, pad_width=[(0, 0), (math.floor(pad_height), math.floor(pad_height) + 1), (math.floor(pad_width), math.floor(pad_width) + 1), (0, 0)], mode = 'constant', constant_values = 0)

	else:
		padded_inputs = np.pad(inputs, pad_width=[(0, 0), (math.floor(pad_height), math.floor(pad_height)), (math.floor(pad_width), math.floor(pad_width)), (0, 0)], mode = 'constant', constant_values = 0)

	# Calculate output dimensions
	# Your output dimension height is equal to (in_height + 2*padY - filter_height) / strideY + 1 and your output dimension width is equal to (in_width + 2*padX - filter_width) / strideX + 1. Again, strideX and strideY will always be 1 for this assignment. Refer to the CNN slides if you'd like to understand this derivation.
	output_height = int((in_height + 2 * pad_height - filter_height) / strideY + 1) # might change 
	output_width = int((in_width + 2 * pad_width - filter_width) / strideX + 1)

	#Initialize output array
	output = np.zeros((num_examples, output_height, output_width, filter_out_channels)) # out channels is number of filters
	
	# You will want to iterate the entire height and width including padding, stopping when you cannot fit a filter over the rest of the padding input. For convolution with many input channels, you will want to perform the convolution per input channel and sum those dot products together.	
	for num in range(num_examples):
		for height in range(output_height):
			for width in range(output_width):
				for channel in range(filter_out_channels): # number of channels 
					filtered_slice = padded_inputs[num, height * strideY:height * strideY + filter_height,  width * strideX:width * strideX + filter_width, :]
					output[num, height, width, channel] = np.sum(filtered_slice * filters[:, :, :, channel])
					
	return tf.convert_to_tensor(output, dtype = tf.float32)
def same_test_0():
	'''
	Simple test using SAME padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
	print("SAME_TEST_0:", "my conv2d:", my_conv[0][0][0], "tf conv2d:", tf_conv[0][0][0].numpy())

def valid_test_0():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_0:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_1():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[3,5,3,3],[5,1,4,5],[2,5,0,1],[3,3,2,1]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = tf.Variable(tf.random.truncated_normal([3, 3, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_2():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[1,3,2,1],[1,3,3,1],[2,1,1,3],[3,2,3,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = np.array([[1,2,3],[0,1,0],[2,1,2]]).reshape((3,3,1,1)).astype(np.float32)
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def main():
	return

if __name__ == '__main__':
	main()
