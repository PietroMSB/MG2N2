# coding: utf-8

import tensorflow as tf
import numpy as np

#defines a random weight initialization method for every weight matrix to be instantiated
#	shape:	defines the matrix dimensions
#	nm:		name assigned in the tensorflow graph to the object that corresponds to this weight matrix
def weight_variable(shape, nm):
	glorot_initializer = tf.initializers.glorot_normal()
	initial = glorot_initializer(shape, tf.float32)
	#initial = tf.truncated_normal(shape, stddev=0.1)
	tf.summary.histogram(nm, initial, collections=['always'])
	return tf.Variable(initial, name=nm, trainable = True)

#function that applies a Gumbel softmax to the input logits, by sampling Gumbel distributed white noise G(0,1)
def GumbelSoftmax(logits, temperature, namespace):
	#draw a tensor of i.i.d. gumbel noise samples of the same shape of logits
	gumbel_noise = tf.numpy_function(np.random.gumbel, [0, 1, tf.shape(logits)], tf.float64, name=namespace+"gumbel_noise_sampler")
	#apply softmax
	return tf.nn.softmax(tf.divide(tf.add(logits, tf.cast(gumbel_noise, tf.float32)), temperature))

#class that defines the architecture of state and output networks, the loss function and the evaluation metric
class StandardNet:

	#constructor: intializes all the parameters for state and output networks
	def __init__(self, input_dim, state_dim, output_dim, hidden_units_state, hidden_units_output, namespace=""):

		#define temperature to allow the usage of gumbel softmax (defaults to 1)
		self.gumbel_softmax_temperature = 1 
			
		#acquire namespace string
		self.namespace = namespace

		self.input_dim = input_dim  # source and destination ids + arc_label
		self.state_dim = state_dim
		self.output_dim = output_dim

		self.state_input = self.input_dim - 2 + 2 * state_dim  # input dimension - 2*node_ids + neighbour's state + node's state
		self.arc_input = self.input_dim - 2 + 2 * state_dim # input dimension - 2*node_ids + source state + destination state

		#number of units for each network layer
		self.state_l1 = hidden_units_state
		self.state_l2 = self.state_dim
		self.output_l1 = hidden_units_output
		self.output_l2 = self.output_dim

		#define a weight matrix for each layer in each network
		self.weights = {'State_L1': weight_variable([self.state_input, self.state_l1], self.namespace+"WEIGHT_STATE_L1"),
						'State_L2': weight_variable([self.state_l1, self.state_l2], self.namespace+"WEIGHT_STATE_L2"),

						'Output_L1': weight_variable([self.arc_input, self.output_l1], self.namespace+"WEIGHT_OUTPUT_L1"),
						'Output_L2': weight_variable([self.output_l1, self.output_l2], self.namespace+"WEIGHT_OUTPUT_L2")
						}

		#defines a vector of biases for each layer in each network
		self.biases = {'State_L1': weight_variable([self.state_l1], self.namespace+"BIAS_STATE_L1"),
						'State_L2': weight_variable([self.state_l2], self.namespace+"BIAS_STATE_L2"),

						'Output_L1': weight_variable([self.output_l1], self.namespace+"BIAS_OUTPUT_L1"),
						'Output_L2': weight_variable([self.output_l2], self.namespace+"BIAS_OUTPUT_L2")
						}

	#defines the state network
	def netSt(self, inp):
		with tf.variable_scope('State_net'):
			layer1 = tf.nn.tanh(tf.add(tf.matmul(inp,self.weights["State_L1"]),self.biases["State_L1"]))
			layer2 = tf.nn.tanh(tf.add(tf.matmul(layer1, self.weights["State_L2"]), self.biases["State_L2"]))
			return layer2

	#defines the output network
	def netOut(self, inp):
		with tf.variable_scope('Out_net'):
			layer3 = tf.nn.tanh(tf.add(tf.matmul(inp, self.weights["Output_L1"]), self.biases["Output_L1"]))
			layer4 = tf.add(tf.matmul(layer3, self.weights["Output_L2"]), self.biases["Output_L2"])
			return layer4

	#defines the final output activation (to be applied to the raw output tensor)
	def outputActivation(self, out_tensor):
		return GumbelSoftmax(out_tensor, self.gumbel_softmax_temperature, self.namespace)
		#return tf.nn.softmax(out_tensor)
		#return out_tensor

	#defines the loss function
	def Loss(self, output, target, output_weight=None):
		#weights = tf.matmul( target, np.array([[0.1], [1], [2], [2]], dtype=np.float32) )
		#weights = tf.reshape( weights, (tf.shape(weights)[0],) )
		return tf.losses.softmax_cross_entropy(target,output)
		#return tf.losses.mean_squared_error(target, output)

	#defines the evaluation metric
	def Metric(self, target, output, output_weight=None):
		correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
		metric = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return metric

