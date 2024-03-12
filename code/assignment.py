from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 250
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # Initialize all hyperparameters
        self.dropout_rate = 0.5
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

        # Initialize all trainable parameters
        self.filter1 = tf.Variable(tf.random.truncated_normal([5,5,3,16], stddev=0.1))
        self.bias1 = tf.Variable(tf.random.truncated_normal([16], stddev=0.1))
        self.filter2 = tf.Variable(tf.random.truncated_normal([5,5,16,20], stddev=0.1))
        self.bias2 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1))
        self.filter3 = tf.Variable(tf.random.truncated_normal([3,3,20,20], stddev=0.1))
        self.bias3 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1))
        self.W1      = tf.Variable(tf.random.truncated_normal([20, 256], stddev=0.1))
        self.b1      = tf.Variable(tf.random.truncated_normal([256], stddev=0.1))
        self.W2      = tf.Variable(tf.random.truncated_normal([256, 64], stddev=0.1))
        self.b2      = tf.Variable(tf.random.truncated_normal([64], stddev=0.1))
        self.W3      = tf.Variable(tf.random.truncated_normal([64, self.num_classes], stddev=0.1))
        self.b3      = tf.Variable(tf.random.truncated_normal([self.num_classes], stddev=0.1))

    def conv_step(self, x, filter, conv_args, act_fn, pool_fn):
        out = tf.nn.convolution(x, filter, **conv_args)
        out = act_fn(out)
        out = pool_fn(out)
        return out
    
    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        conv_fns_1 = {      
            'act_fn'  : tf.nn.relu,
            'pool_fn' : lambda l: tf.nn.max_pool(l, 3, 2, padding='SAME')
        }
        conv_fns_2 = {      
            'act_fn'  : tf.nn.relu,
            'pool_fn' : lambda l: tf.nn.max_pool(l, 2, 2, padding='SAME')
        }

        l1_out = self.conv_step(inputs, self.filter1, {'strides' : 2, 'padding' : 'SAME'}, conv_fns_1['act_fn'], conv_fns_1['pool_fn'])
        
        l1_out = tf.nn.bias_add(l1_out, self.bias1)

        l2_out = self.conv_step(l1_out, self.filter2, {'strides' : 2, 'padding' : 'SAME'}, **conv_fns_2)
        
        l2_out = tf.nn.bias_add(l2_out, self.bias2)

        if is_testing:
            l3_out = tf.nn.relu(conv2d(l2_out, self.filter3, [1, 1, 1, 1], 'SAME'))
            l3_out = tf.nn.max_pool(l3_out, 2, 2, padding='SAME')
            l3_out = tf.cast(l3_out, tf.float32)
        else:
            l3_out = self.conv_step(l2_out, self.filter3, {'strides' : 1, 'padding' : 'SAME'}, **conv_fns_2)
        
        l3_out = tf.nn.bias_add(l3_out, self.bias3)

        l3_out = tf.reshape(l3_out, [l3_out.shape[0], -1]) 

        dense1_out = tf.matmul(l3_out, self.W1) + self.b1

        dense2_out = tf.matmul(dense1_out, self.W2) + self.b2

        logits = tf.matmul(dense2_out, self.W3) + self.b3

        return logits

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        prbs = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        loss = tf.reduce_mean(prbs)
        self.loss_list.append(loss)
        return loss

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. 
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Return the average accuracy across batches of the train inputs/labels
    '''
    indices = tf.range(start=0, limit=len(train_inputs))
    shuff_indices = tf.random.shuffle(indices)
    train_inputs = tf.gather(train_inputs, shuff_indices)
    train_labels = tf.gather(train_labels, shuff_indices)
    train_inputs = tf.image.random_flip_left_right(train_inputs)

    accuracies = []
    for b, b1 in enumerate(range(model.batch_size, train_inputs.shape[0] + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        train_inputs_batches = train_inputs[b0:b1]
        train_labels_batches = train_labels[b0:b1]
        with tf.GradientTape() as tape:
            pred = model.call(train_inputs_batches)
            loss = model.loss(pred, train_labels_batches) # watching the loss and prediction gradient calculations
        accuracy = model.accuracy(pred, train_labels_batches)
        accuracies.append(accuracy)
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return np.mean(np.array(accuracies))

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """   
    accuracies = []
    pred = model.call(test_inputs, is_testing = True)
    loss = model.loss(pred, test_labels) # watching the loss and prediction gradient calculations
    accuracy = model.accuracy(pred, test_labels)
    accuracies.append(accuracy)
    print(" Loss: ", loss.numpy(), " Accuracy: ", accuracy.numpy())
    return np.mean(np.array(accuracies))


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. 
    
    :return: None
    '''
    from preprocess import get_data
    AUTOGRADER_TRAIN_FILE = '../data/train'
    AUTOGRADER_TEST_FILE = '../data/test'

    LOCAL_TRAIN_FILE = '../data/train'
    LOCAL_TEST_FILE = '../data/test'

    # Read in CIFAR2 data 
    train_inputs, train_labels = get_data(LOCAL_TRAIN_FILE, 3, 5)
    test_inputs, test_labels = get_data(LOCAL_TEST_FILE, 3, 5)

    # initialize model
    model = Model()
    for e in range(model.num_epochs):
        print("Epoch: ", e)
        train(model, train_inputs, train_labels)
    print("NOW IN TEST")
    test(model, test_inputs, test_labels)

    return


if __name__ == '__main__':
    main()
