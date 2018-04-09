# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import multiprocessing as mp
import itertools
import dill

from tensorflow.examples.tutorials.mnist import input_data
from pathos.multiprocessing import ProcessingPool as Pool

import tensorflow as tf
import matplotlib as plt
import numpy as np
import math

FLAGS = None


hidden_sizes=[20]

input_size=784

output_size=10

batch_size = 50

data_dir = '/jmain01/home/JAD017/sjr01/mxw35-sjr01/mnist/input_data'

save_dir = '/jmain01/home/JAD017/sjr01/mxw35-sjr01/Projects/NIPS/output/MNIST/hessians'


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def makeMLP(batch_size, n_input, n_hidden, n_output):
  # First create placeholders for inputs and targets: x_input, y_target
  x_input = tf.placeholder(tf.float32, shape=[batch_size, n_input])
  #
  # Start constructing a computational graph for multilayer perceptron
  ###  Since we want to store parameters as one long vector, we first define our parameters as below and then
  ### reshape it later according to each layer specification.
  parameters = tf.Variable(tf.concat([tf.truncated_normal([n_input * n_hidden, 1]), tf.zeros([n_hidden, 1]), tf.truncated_normal([n_hidden * n_output,1]), tf.zeros([n_output, 1])],0))
  with tf.name_scope("hidden") as scope:
    idx_from = 0 
    weights = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_input*n_hidden, 1]), [n_input, n_hidden])
    idx_from = idx_from + n_input*n_hidden
    biases = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_hidden, 1]), [n_hidden]) # tf.Variable(tf.truncated_normal([n_hidden]))
    hidden = tf.nn.relu(tf.matmul(x_input, weights) + biases)
  with tf.name_scope("linear") as scope:
    idx_from = idx_from + n_hidden
    weights = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_hidden*n_output, 1]), [n_hidden, n_output])
    idx_from = idx_from + n_hidden*n_output
    biases = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_output, 1]), [n_output]) 
    output = tf.matmul(hidden, weights) + biases
  #
  return x_input, output, parameters


def jacobian(y_flat, x):
    n = y_flat.shape[0]
    #
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    #
    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, jacobian_piece(y_flat, x, j))),
        loop_vars,swap_memory=True)
    #
    return jacobian.stack()

def hessian_via_jacobian(loss, parameters):
  #
  #
  ### Note: We can call tf.trainable_variables to get GraphKeys.TRAINABLE_VARIABLES 
  ### because we are using g as our default graph inside the "with" scope. 
  # Get trainable variables
  # Get gradients of loss with repect to parameters
  dloss_dw = tf.gradients(loss, parameters)[0]
  #
  hess = jacobian(dloss_dw, parameters)
  return hess


def jacobian_piece(vect, parameters, index):
  dfx_i = tf.slice(vect, begin=[index,0] , size=[1,1])
  ddfx_i = tf.gradients(dfx_i, parameters)[0] 
  return ddfx_i


mnist = input_data.read_data_sets(data_dir, one_hot=True)

y_ = tf.placeholder(tf.float32, [None, 10])


x, y, parameters = makeMLP(batch_size,input_size,hidden_sizes[0],output_size)


# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Get gradients of loss with repect to parameters
dloss_dw = tf.gradients(cross_entropy, parameters)[0]


hess = hessian_via_jacobian(cross_entropy, parameters)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_accuracy = []
# Train

# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(batch_size)
  if i % 100 == 0:
    np.save(save_dir+'/hess'+str(i)+'.npy', hess.eval(feed_dict={x: batch[0], y_: batch[1]}))
    train_accuracy.append(accuracy.eval(feed_dict={x: batch[0], y_: batch[1]}))
    print('step %d, training accuracy %g' % (i, train_accuracy[-1]))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})


