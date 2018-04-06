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

import tensorflow as tf

FLAGS = None


hidden_sizes=[20]

input_size=784

output_size=10

batch_size = 50

data_dir = '/jmain01/home/JAD017/sjr01/mxw35-sjr01/mnist/input_data'

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



import tensorflow as tf
import matplotlib as plt
import numpy as np
import math

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

def hessian(loss, parameters):
  #
  #
  ### Note: We can call tf.trainable_variables to get GraphKeys.TRAINABLE_VARIABLES 
  ### because we are using g as our default graph inside the "with" scope. 
  # Get trainable variables
  tvars = tf.trainable_variables()
  # Get gradients of loss with repect to parameters
  dloss_dw = tf.gradients(loss, tvars)[0]
  dim, _ = dloss_dw.get_shape()
  for i in range(dim):
    # tf.slice: https://www.tensorflow.org/versions/0.6.0/api_docs/python/array_ops.html#slice
    dfx_i = tf.slice(dloss_dw, begin=[i,0] , size=[1,1])
    ddfx_i = tf.gradients(dfx_i, parameters)[0] # whenever we use tf.gradients, make sure you get the actual tensors by putting [0] at the end
    hess.append(ddfx_i)
  hess = tf.squeeze(hess) 
  return hess


def slice_grad(args):
    i,dloss_dw = args
    dfx_i = tf.slice(dloss_dw, begin=[i,0] , size=[1,1])
    ddfx_i = tf.gradients(dfx_i, parameters)[0] 
    return ddfx_i



def apply_packed_function_for_map((dumped_function, item, args, kwargs),):
    """
    Unpack dumped function as target function and call it with arguments.

    :param (dumped_function, item, args, kwargs):
        a tuple of dumped function and its arguments
    :return:
        result of target function
    """
    target_function = dill.loads(dumped_function)
    res = target_function(item, *args, **kwargs)
    return res


def pack_function_for_map(target_function, items, *args, **kwargs):
    """
    Pack function and arguments to object that can be sent from one
    multiprocessing.Process to another. The main problem is:
        «multiprocessing.Pool.map*» or «apply*»
        cannot use class methods or closures.
    It solves this problem with «dill».
    It works with target function as argument, dumps it («with dill»)
    and returns dumped function with arguments of target function.
    For more performance we dump only target function itself
    and don't dump its arguments.
    How to use (pseudo-code):

        ~>>> import multiprocessing
        ~>>> images = [...]
        ~>>> pool = multiprocessing.Pool(100500)
        ~>>> features = pool.map(
        ~...     *pack_function_for_map(
        ~...         super(Extractor, self).extract_features,
        ~...         images,
        ~...         type='png'
        ~...         **options,
        ~...     )
        ~... )
        ~>>>

    :param target_function:
        function, that you want to execute like  target_function(item, *args, **kwargs).
    :param items:
        list of items for map
    :param args:
        positional arguments for target_function(item, *args, **kwargs)
    :param kwargs:
        named arguments for target_function(item, *args, **kwargs)
    :return: tuple(function_wrapper, dumped_items)
        It returs a tuple with
            * function wrapper, that unpack and call target function;
            * list of packed target function and its' arguments.
    """
    dumped_function = dill.dumps(target_function)
    dumped_items = [(dumped_function, item, args, kwargs) for item in items]
    return apply_packed_function_for_map, dumped_items


def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)

def apply_map(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.map(run_dill_encoded, (payload,))




mnist = input_data.read_data_sets(data_dir, one_hot=True)

y_ = tf.placeholder(tf.float32, [None, 10])


x, y, p = makeMLP(batch_size,input_size,hidden_sizes[0],output_size)


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

hess = hessian(cross_entropy, p)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

hessian_list = []
# Train


# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
for i in range(20000):
  #batch = mnist.train.next_batch(batch_size)
  batch = {x: np.random.random([batch_size, n_input]), y_: np.random.random([batch_size, n_output])
  if i % 100 == 0:
    hessian_list.append(hess.eval(feed_dict=batch))
    train_accuracy = accuracy.eval(feed_dict=batch)
    print('step %d, training accuracy %g' % (i, train_accuracy))
  train_step.run(feed_dict=batch)



# print('test accuracy %g' % accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels}))

# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
#                       help='Directory for storing input data')
#   FLAGS, unparsed = parser.parse_known_args()
#   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)