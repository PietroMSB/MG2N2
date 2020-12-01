import tensorflow as tf
import numpy as np
import pickle

#Graph Neural Network class
class GNN:

	#GNN constructor
	#	net: 			defines the architecture of state and output networks, the loss function and the evaluation metric
	#	max_it: 		maximum number of iterations for state convergence to be reached
	#	input_dim:		input tensor width
	#	output_dim:		output tensor width
	#	sate_dim:		state tensor width
	#	optimizer:		optimizer object to be used
	#	learning_rate:	learning rate value
	#	threshold:		sate convergence is reached when the relative distance between current and past state is smaller than "threshold"
	def __init__(self, net, max_it, input_dim, output_dim, state_dim, num_train_batches, optimizer, learning_rate, threshold, param, namespace=""):
		#transfer parameters		
		with tf.variable_scope(namespace):
			self.namespace = namespace
			self.max_iter = max_it
			self.net=net
			self.optimizer = optimizer(learning_rate, name="optim")
			self.state_threshold = threshold
			self.input_dim=input_dim
			self.output_dim=output_dim
			self.state_dim=state_dim
		#build architecture with hyperparameters
		self.build()
		#build other variables
		with tf.variable_scope(namespace):
			#configure session options
			self.config = tf.ConfigProto()
			self.config.gpu_options.allow_growth = True
			#build session
			self.session = tf.Session(config = self.config)
			#initialize variables
			self.session.run(tf.global_variables_initializer())
			self.init_l = tf.local_variables_initializer()    
			#define summaries   
			self.merged_all = tf.summary.merge_all(key='always') 
			self.merged_train = tf.summary.merge_all(key='train')
			self.merged_val = tf.summary.merge_all(key='val')
			self.writer = tf.summary.FileWriter('tmp/'+param,self.session.graph)
			self.writers_train = self.SpawnWriters('tmp/'+param+"/train_", num_train_batches)
			#define saver object
			self.saver = tf.train.Saver()		
			self.save_path = "Temp/Models/"+namespace+"/model.ckpt"
			#initialize validation parameters
			self.best_valloss = 10000	#initial value for the "best validation loss"
			self.best_step = 0
			#define operation run options
			self.run_options = tf.RunOptions()
		
	#defines tensor placeholders
	def VariableState(self):
		#input tensor placeholder
		self.a = tf.placeholder(tf.float32, shape=(None, self.input_dim), name="input")
		#target tensor placeholder
		self.y = tf.placeholder(tf.float32, shape=(None, self.output_dim), name="target")
		#current state tensor placeholder
		self.state = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="state")
		#past state tensor placeholder
		self.state_old = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="old_state")
		#output tensor placeholder for prediction
		self.out_tensor = tf.placeholder(tf.float32, shape=(None, self.output_dim), name="out_tensor")

		#arc-node conversion matrix
		self.ArcNode = tf.sparse_placeholder(tf.float32, name="ArcNode")
		#node mask placeholder
		self.Mask = tf.placeholder(tf.bool, shape=[None], name="Mask")
		#output mask placeholder
		self.output_mask = tf.placeholder(tf.bool, shape=[None], name="OutputMask")

	#builds the architecture, defining variables and operations
	def build(self):
		with tf.variable_scope(self.namespace):
			#initialize tensors with placeholders
			self.VariableState()
			#define output activation operation (used in prediction, not during training)
			self.output_activation_op = self.net.outputActivation(self.out_tensor) 
			#define forward step operation
			self.loss_op = self.Loop()
		#define the scope for <loss_op>
		with tf.variable_scope(self.namespace+'loss'):
			#define loss calculation operation
			self.loss=self.net.Loss(self.loss_op[1],self.y)
			self.summ_loss = tf.summary.scalar('train_loss', self.loss, collections=['train'])
			#define validation loss calculation operation
			self.val_loss = self.net.Loss(self.loss_op[1],self.y)
			self.summ_val_loss = tf.summary.scalar('val_loss', self.val_loss, collections=['val'])
		#define gradient calculation and training operations
		with tf.variable_scope(self.namespace+'train'):
			self.grads = self.optimizer.compute_gradients(self.loss)
			self.train_op = self.optimizer.apply_gradients(self.grads, name='train_op')
			#for index, grad in enumerate(self.grads):
			#	tf.summary.histogram("{}-grad".format(self.grads[index][1].name), self.grads[index], collections=['train'])

		#define the evaluation metrics
		with tf.variable_scope(self.namespace+'metrics'):
			#define test metrics
			self.metrics = self.net.Metric(self.y, self.loss_op[1])
			#define training metrics
			self.train_met = self.net.Metric(self.y, self.loss_op[1])
			self.summ_train_met = tf.summary.scalar('train_metric', self.train_met, collections=['train'])
			#define validation metrics
			self.val_met = self.net.Metric(self.y, self.loss_op[1])
			self.summ_val_met = tf.summary.scalar('val_metric', self.val_met, collections=['val'])
			
	#creates one writer for each batch in a set
	def SpawnWriters(self, path, num_batches):
		with tf.variable_scope(self.namespace):
			writers = list()
			for i in range(num_batches):
				writers.append(tf.summary.FileWriter(path+str(i), self.session.graph))
		return writers

	#calculates one step of state convergence
	def convergence(self, a, state, state_old, k):
		with tf.variable_scope(self.namespace+'convergence'):
			#assign prevoius "current state" to "old state"
			state_old = state
			#gather the state of the source node for each arc
			gat = tf.cast(tf.gather(state_old, tf.cast(a[:, 0], tf.int32)), tf.float32)
			#slice the input tensor, cutting out the source and destination node indices from each row
			sl = tf.slice(a, [0, 2], [tf.shape(a)[0], tf.shape(a)[1]-2])
			#concatenate the gathered source node states with the corresponding arc labels
			message = tf.concat([sl, gat], axis=1)
			#multiply by ArcNode matrix to get the average of the incoming messages on each node
			avg_message = tf.sparse_tensor_dense_matmul(self.ArcNode, message)
			#concatenate the destination node state to the average incoming message for each node
			inp = tf.concat([state_old, avg_message], axis=1)
			#compute the next state
			state = self.net.netSt(inp)
			#update the step counter
			k = k + 1
		return a, state, state_old, k

	#loop prosecution condition for the state convergence procedure
	def condition(self, a, state, state_old, k):
		with tf.variable_scope(self.namespace+'condition'):
			#evaluate the relative distance between current state and past state
			outDistance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(state, state_old)), 1))
			state_norm = tf.sqrt(tf.reduce_sum(tf.square(state_old), 1))
			#boolean vector that stores the "convergence reached" flag for each node
			checkDistanceVec = tf.greater(outDistance, tf.scalar_mul(self.state_threshold,state_norm))
			#check whether global convergence and/or the maximum number of iterations have been reached
			c1 = tf.reduce_any(checkDistanceVec)
			c2 = tf.less(k, self.max_iter)
		return tf.logical_and(c1, c2)

	#iterative method for the state convergence procedure
	def Loop(self):
		#calculate the next state
		with tf.variable_scope(self.namespace+'loop'):
			#initialize the state convergence iteration counter at 0
			k = tf.constant(0)
			#loop until convergence is reached
			res, st, old_st, num = tf.while_loop(self.condition, self.convergence, [self.a, self.state, self.state_old, k])
			self.summ_iter = tf.summary.scalar('iteration', num, collections=['always'])
			#gather the states of the nodes which belong to the current set			
			set_st = tf.boolean_mask(st, self.Mask, name="boolean_mask_set")
			#gather the state of output nodes by applying <output_mask> to <st>
			out_st = tf.boolean_mask(set_st, self.output_mask, name="boolean_mask_output")
			#calculate the output on the output nodes
			out = self.net.netOut(out_st)
		return st, out, num

	#direct on-memory training method
	def TrainOnMemory(self, batches, step):
		#draw a random permutation of the batches for this iteration
		permutation_index = list(range(len(batches)))
		np.random.shuffle(permutation_index)
		#training step
		for j in range(len(batches)):
			#retrieve the index according to the permutation
			jj = permutation_index[j]
			#select jj-th training batch
			current_batch = batches[jj]
			#collect data structures from the current batch
			inputs = current_batch.getInputTensor()
			ArcNode = current_batch.getArcNode().T
			targets = current_batch.getTargets()
			mask = current_batch.getSetMask()
			output_mask = current_batch.getOutputMask()
			#initialize the tensor of node states
			state_init = current_batch.initState()
			#define the dictionary of arguments for the training operations
			fd = {self.Mask:mask, self.output_mask:output_mask, self.a: inputs, self.state: state_init, self.state_old: np.ones((ArcNode.shape[0],self.state_dim)), self.ArcNode: tf.SparseTensorValue(indices=np.array([ArcNode.row, ArcNode.col]).T,values=ArcNode.data,dense_shape=ArcNode.shape), self.y: targets}
			#run the training operations
			_, loss, loop, merge_all, merge_tr = self.session.run([self.train_op, self.loss, self.loss_op, self.merged_all, self.merged_train], feed_dict=fd, options = self.run_options)
			#write summaries
			self.writer.add_summary(merge_all, step)
			self.writers_train[jj].add_summary(merge_tr, step)
		#return the loss value over the training set, and the number of iterations which were necessary to reach state convergence
		return loss, loop[2]

	#training method
	def Train(self, batch_paths, step):
		#draw a random permutation of the batches for this iteration
		permutation_index = list(range(len(batch_paths)))
		np.random.shuffle(permutation_index)
		#training step
		for j in range(len(batch_paths)):
			#retrieve the index according to the permutation
			jj = permutation_index[j]
			#load jj-th training batch
			temp_file = open(batch_paths[jj], 'rb')
			current_batch = pickle.load(temp_file)
			temp_file.close()
			#collect data structures from the current batch
			inputs = current_batch.getInputTensor()
			ArcNode = current_batch.getArcNode().T
			targets = current_batch.getTargets()
			mask = current_batch.getSetMask()
			output_mask = current_batch.getOutputMask()
			#initialize the tensor of node states
			state_init = current_batch.initState()
			#define the dictionary of arguments for the training operations
			fd = {self.Mask:mask, self.output_mask:output_mask, self.a: inputs, self.state: state_init, self.state_old: np.ones((ArcNode.shape[0],self.state_dim)), self.ArcNode: tf.SparseTensorValue(indices=np.array([ArcNode.row, ArcNode.col]).T,values=ArcNode.data,dense_shape=ArcNode.shape), self.y: targets}	
			#run the training operations
			_, loss, loop, merge_all, merge_tr = self.session.run([self.train_op, self.loss, self.loss_op, self.merged_all, self.merged_train], feed_dict=fd, options = self.run_options)
			#write summaries
			self.writer.add_summary(merge_all, step)
			self.writers_train[jj].add_summary(merge_tr, step)
			#delete the batch		
			del current_batch
		#return the loss value over the training set, and the number of iterations which were necessary to reach state convergence
		return loss, loop[2]

	#direct on-memory validation method
	def ValidateOnMemory(self, validation_batch, step):
		#collect data structures from the validation batch
		input_val = validation_batch.getInputTensor()
		ArcNode = validation_batch.getArcNode().T
		targets = validation_batch.getTargets()
		mask = validation_batch.getSetMask()
		output_mask = validation_batch.getOutputMask()
		#initialize the tensor of node states
		state_init = validation_batch.initState()
		#define the dictionary of arguments for the validation operations
		fd_val = {self.Mask:mask, self.output_mask:output_mask, self.a: input_val, self.state: state_init, self.state_old: np.ones((ArcNode.shape[0], self.state_dim)), self.ArcNode: tf.SparseTensorValue(indices=np.array([ArcNode.row, ArcNode.col]).T, values=ArcNode.data, dense_shape=ArcNode.shape), self.y: targets}
		#run the validation operations
		loss_val, loop, merge_all, merge_val = self.session.run([self.val_loss, self.loss_op, self.merged_all, self.merged_val], feed_dict=fd_val, options = self.run_options)
		self.writer.add_summary(merge_val, step)
		if loss_val<self.best_valloss:
			self.best_valloss=loss_val
			self.saver.save(self.session,self.save_path)
			self.best_step=step
		#print the id of the current best model
		print("epoch: "+str(step)+"\tbest_model: "+str(self.best_step))
		#return the loss value over the validation set
		return loss_val

	#validation method
	def Validate(self, validation_batch_path, step):
		#load validation batch
		temp_file = open(validation_batch_path, 'rb')
		validation_batch = pickle.load(temp_file)
		temp_file.close()
		#collect data structures from the validation batch
		input_val = validation_batch.getInputTensor()
		ArcNode = validation_batch.getArcNode().T
		targets = validation_batch.getTargets()
		mask = validation_batch.getSetMask()
		output_mask = validation_batch.getOutputMask()
		#initialize the tensor of node states
		state_init = validation_batch.initState()
		#define the dictionary of arguments for the validation operations
		fd_val = {self.Mask:mask, self.output_mask:output_mask, self.a: input_val, self.state: state_init, self.state_old: np.ones((ArcNode.shape[0], self.state_dim)), self.ArcNode: tf.SparseTensorValue(indices=np.array([ArcNode.row, ArcNode.col]).T, values=ArcNode.data, dense_shape=ArcNode.shape), self.y: targets}
		#run the validation operations
		loss_val, loop, merge_all, merge_val = self.session.run([self.val_loss, self.loss_op, self.merged_all, self.merged_val], feed_dict=fd_val, options = self.run_options)
		self.writer.add_summary(merge_val, step)
		if loss_val<self.best_valloss:
			self.best_valloss=loss_val
			self.saver.save(self.session,self.save_path)
			self.best_step=step
		#delete the batch		
		del validation_batch
		#print the id of the current best model
		print("epoch: "+str(step)+"\tbest_model: "+str(self.best_step))
		#return the loss value over the validation set
		return loss_val
	
	#evaluation method
	def Evaluate(self, inputs, ArcNode, target, mask, output_mask, state_init):
		self.saver.restore(self.session, self.save_path)
		fd = {self.Mask:mask, self.output_mask:output_mask, self.a: inputs, self.state: state_init, self.state_old: np.ones((ArcNode.shape[0],self.state_dim)), self.ArcNode: tf.SparseTensorValue(indices=np.array([ArcNode.row, ArcNode.col]).T,values=ArcNode.data,dense_shape=ArcNode.shape), self.y: target}
		_ = self.session.run([self.init_l], options = self.run_options)
		met = self.session.run([self.metrics], feed_dict=fd, options = self.run_options)
		return met

	#prediction method
	def Predict(self, inputs, ArcNode, target, mask, output_mask, state_init):
		#restore the best model
		self.saver.restore(self.session, self.save_path)
		fd = {self.Mask:mask, self.output_mask:output_mask, self.a: inputs, self.state: state_init, self.state_old: np.ones((ArcNode.shape[0],self.state_dim)), self.ArcNode: tf.SparseTensorValue(indices=np.array([ArcNode.row, ArcNode.col]).T,values=ArcNode.data,dense_shape=ArcNode.shape), self.y: target}
		loss, loop = self.session.run([self.loss, self.loss_op], feed_dict=fd, options = self.run_options)
		output = self.session.run([self.output_activation_op], feed_dict = {self.out_tensor: loop[1]})
		return output, loss, loop[0]

