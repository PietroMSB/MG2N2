#coding=utf-8

import sys
import os
import math
import pickle
import numpy as np
import tensorflow as tf
import scipy

from GNN_EdgeBased.graph_class import GraphObject
from GNN_EdgeBased import GNN
from GNN_EdgeBased import standard_net
from graph_decomposition import *

#parameters
NODE_AGGREGATION = "average"#modality of node aggregation implemented in the ArcNode tensors
INPUT_DIM = 6				#input tensor width (source id + destination id + arc_label)
STATE_DIM = 5				#node label width
OUTPUT_DIM = 4				#target tensor width
EPOCHS = 500				#training epochs
LR = 0.002					#learning rate
THRESHOLD = 0.001			#state convergence threshold, in terms of relative state difference
MAX_ITER = 6				#maximum number of state convergence iterations
VALIDATION_INTERVAL = 10	#interval between two validation checks, in training epochs
TRAINING_BATCHES = 20       #number of batches in which the training set should be split
HIDDEN_UNITS_STATE = 20     #number of units in the hidden layer of the state network
HIDDEN_UNITS_OUTPUT = 50    #number of units in the hidden layer of the output network
RUN_SUFFIX = sys.argv[1]    #string to be appended at the end of the output file's name, in order to identify the run which produced it
RUNNING_MODE = sys.argv[2]  #string that defines the running mode (determines whether to rebuild the batches or not)
EXAMPLES = 133582			#number of examples in the dataset
SPLITTING_SEED = 19920305   #seed for drawing examples at random for each set
VALIDATION_SIZE = 3582      #number of graphs in the validation set
TEST_SIZE = 10000			#number of graphs in the test set
GUMBEL_SOFTMAX_TEMPERATURE_MAX = 5.0
GUMBEL_SOFTMAX_TEMPERATURE_MIN = 1.0
debug_state_filename = "Temp/Debug/Linker/debug_state.txt"
debug_output_filename = "Temp/Debug/Linker/debug_output.txt"
output_filename = "Temp/Debug/Linker/output.txt"
path_model = "Temp/Models/Linker/Linker"
path_data = "Data/Graphs/"
path_batches = "Temp/Batches/Linker/"
path_results = "Temp/Results/Linker/res_linker_"+RUN_SUFFIX+".txt"
edge_type_distribution = [0.885, 0.102, 0.013] #distribution measured on the training set


#gpu parameters
use_gpu = True
target_gpu = "0"

#define parameter line
param_string = "Linker_"+RUN_SUFFIX

#set target gpu as the only visible device
if use_gpu:
	os.environ["CUDA_VISIBLE_DEVICES"]=target_gpu

#determine running mode
BUILD_BATCHES = None
if RUNNING_MODE == "full":
	BUILD_BATCHES = True
elif RUNNING_MODE == "short":
	BUILD_BATCHES = False
else:
	sys.exit("Unknown running mode |"+RUNNING_MODE+"|. Should be either |full| or |short|.")

#build directories
if not os.path.exists("Temp/Debug/Linker"):
	os.makedirs("Temp/Debug/Linker")
if not os.path.exists("Temp/Batches/Linker"):
	os.makedirs("Temp/Batches/Linker")
if not os.path.exists("Temp/Results/Linker"):
	os.makedirs("Temp/Results/Linker")

#determine execution mode
if BUILD_BATCHES:

	#delete temp batch files from previous runs
	fnames = os.listdir(path_batches)
	for fn in fnames:
		os.remove(path_batches+fn)

	#load training, validation and test set

	#draw training, validation and test sets at random
	ids_list = np.arange(EXAMPLES)
	np.random.seed(SPLITTING_SEED)
	np.random.shuffle(ids_list)
	#extract the sets from the shuffled list
	ids_test = ids_list[:TEST_SIZE]
	ids_validation = ids_list[TEST_SIZE:TEST_SIZE+VALIDATION_SIZE]
	ids_training = ids_list[TEST_SIZE+VALIDATION_SIZE:]
	#build sets
	graph_ids_training = list()
	graph_ids_validation = list()
	graph_ids_test = list()
	#retrieve graph id for each organism id
	for i in range(EXAMPLES):
		if i in ids_training:
			graph_ids_training.append("G_"+str(i))
		if i in ids_validation:
			graph_ids_validation.append("G_"+str(i))
		if i in ids_test:
			graph_ids_test.append("G_"+str(i))

	#build training set graphs
	i = 0
	graphs_training = dict()
	for g_id in graph_ids_training:
		i+=1
		print("Loading graph "+str(i)+" of "+str(len(graph_ids_training)+len(graph_ids_validation)+len(graph_ids_test)), end='\r')
		#load graph and add it to dictionary
		graphs_training[g_id] = GraphObject.load(path_data+g_id+"/", node_aggregation = NODE_AGGREGATION)
	graphs_validation = dict()
	#build validation set graphs
	for g_id in graph_ids_validation:
		i+=1
		print("Loading graph "+str(i)+" of "+str(len(graph_ids_training)+len(graph_ids_validation)+len(graph_ids_test)), end='\r')
		#load graph and add it to dictionary
		graphs_validation[g_id] = GraphObject.load(path_data+g_id+"/", node_aggregation = NODE_AGGREGATION)
	#build test set graphs
	graphs_test = dict()
	for g_id in graph_ids_test:
		i+=1
		print("Loading graph "+str(i)+" of "+str(len(graph_ids_training)+len(graph_ids_validation)+len(graph_ids_test)), end='\r')
		#load graph and add it to dictionary
		graphs_test[g_id] = GraphObject.load(path_data+g_id+"/", node_aggregation = NODE_AGGREGATION)
	print("")

	#uncomment below for measuring the probability distribution of edge types over the training set
	### COMMENT START ###
	##measure probability distribution of edge types
	#single_bonds = 0
	#double_bonds = 0
	#triple_bonds = 0
	#for g_id in graph_ids_training:
	#	edges = graphs_training[g_id].getArcs()
	#	for i in range(edges.shape[0]):
	#		if edges[i][2] == 1:
	#			single_bonds += 1
	#		elif edges[i][3] == 1:
	#			double_bonds += 1
	#		elif edges[i][4] == 1:
	#			triple_bonds += 1
	#print("Single bonds = "+str(single_bonds))
	#print("Double bonds = "+str(double_bonds))
	#print("Triple bonds = "+str(triple_bonds))
	#sys.exit()
	### COMMENT STOP ###

	#obtain the generation sequence for each graph through graph decomposition
	i=0
	#training set
	for g_id in graph_ids_training:
		i+=1
		print("Decomposing graph "+str(i)+" of "+str(len(graph_ids_training)+len(graph_ids_validation)+len(graph_ids_test)), end='\r')
		#translate the GraphObject to a networkx.DiGraph
		G = GraphObjectToNetworkxDiGraph(graphs_training[g_id])
		#decompose the graph into its generation sequence
		graph_sequence, supervision_sequence, output_node_indices = GraphDecomposition(G)
		#convert the supervision sequence to a numerical vector
		supervision_sequence = np.argmax(supervision_sequence, axis=1)
		#drop all the graphs (but the last) in which a "STOP" prediction is expected, as the linker will not run after "STOP" predictions
		drop_indices = list()
		for j in range(len(supervision_sequence)-1):
			if supervision_sequence[j] == translate_atom_direct['STOP']-1:
				drop_indices.append(j)
		graph_sequence = np.delete(graph_sequence, drop_indices, axis=0)
		supervision_sequence = np.delete(supervision_sequence, drop_indices, axis=0)
		output_node_indices = np.delete(output_node_indices, drop_indices, axis=0)
		#build a list of arc supervion matrices
		arc_supervision_list = list()
		#add the node to be generated in each example
		for j in range(len(graph_sequence)-1):
			#build an arc supervision list
			arc_targets = list()
			#determine arc targets by difference with the next graph
			arc_target_dict = dict()
			for e in graph_sequence[j+1].edges():
				if e not in graph_sequence[j].edges():
					arc_target_dict[e[0], e[1]] = graph_sequence[j+1].edges[e[0], e[1]]['info']
			#add new node
			new_index = len(graph_sequence[j].nodes())
			graph_sequence[j].add_node(new_index)
			#add node label
			graph_sequence[j].nodes[new_index]['info'] = translate_atom_inverse[supervision_sequence[j]+1]
			#add the set of arcs to be discriminated, connecting the new node to each other node
			for k in range(len(graph_sequence[j].nodes())):
				#skip the edge between the new node and the output node (which is processed by the classifier)
				if k == output_node_indices[j]:
					continue 
				#also avoid connecting the new node to itself
				if k!=new_index:
					graph_sequence[j].add_edge(k,new_index)
					graph_sequence[j].edges[k,new_index]['info'] = translate_bond_direct['candidate']
					#check if this edge has an entry in the arc target dictionary
					if (new_index, k) in arc_target_dict.keys():
						arc_targets.append(arc_target_dict[new_index, k])
					#otherwise add a "cut" target
					else:
						arc_targets.append(4)
			#translate the arc targets to one hot vectors
			arc_supervisions = np.zeros((len(arc_targets),4), dtype=int)
			for k in range(len(arc_targets)):
				arc_supervisions[k][arc_targets[k]-1] = 1
			arc_supervision_list.append(arc_supervisions)
		#drop the last graph as no prediction will be run on the complete graph
		graph_sequence = np.delete(graph_sequence, -1)
		#translate each networkx.DiGraph in the sequence to a GraphObject
		for j in range(len(graph_sequence)):
			graph_sequence[j] = NetworkxDiGraphToEBGraphObject(graph_sequence[j], arc_supervision_list[j], node_aggregation = NODE_AGGREGATION)
		#fuse the graphs together
		graphs_training[g_id] = GraphObject.fuse(graph_sequence)
	#validation set
	for g_id in graph_ids_validation:
		i+=1
		print("Decomposing graph "+str(i)+" of "+str(len(graph_ids_training)+len(graph_ids_validation)+len(graph_ids_test)), end='\r')
		#translate the GraphObject to a networkx.DiGraph
		G = GraphObjectToNetworkxDiGraph(graphs_validation[g_id])
		#decompose the graph into its generation sequence
		graph_sequence, supervision_sequence, output_node_indices = GraphDecomposition(G)
		#convert the supervision sequence to a numerical vector
		supervision_sequence = np.argmax(supervision_sequence, axis=1)
		#drop all the graphs (but the last) in which a "STOP" prediction is expected, as the linker will not run after "STOP" predictions
		drop_indices = list()
		for j in range(len(supervision_sequence)-1):
			if supervision_sequence[j] == translate_atom_direct['STOP']-1:
				drop_indices.append(j)
		graph_sequence = np.delete(graph_sequence, drop_indices, axis=0)
		supervision_sequence = np.delete(supervision_sequence, drop_indices, axis=0)
		output_node_indices = np.delete(output_node_indices, drop_indices, axis=0)
		#build a list of arc supervion matrices
		arc_supervision_list = list()
		#add the node to be generated in each example
		for j in range(len(graph_sequence)-1):
			#build an arc supervision list
			arc_targets = list()
			#determine arc targets by difference with the next graph
			arc_target_dict = dict()
			for e in graph_sequence[j+1].edges():
				if e not in graph_sequence[j].edges():
					arc_target_dict[e[0], e[1]] = graph_sequence[j+1].edges[e[0], e[1]]['info']
			#add new node
			new_index = len(graph_sequence[j].nodes())
			graph_sequence[j].add_node(new_index)
			#add node label
			graph_sequence[j].nodes[new_index]['info'] = translate_atom_inverse[supervision_sequence[j]+1]
			#add the set of arcs to be discriminated, connecting the new node to each other node
			for k in range(len(graph_sequence[j].nodes())):
				#skip the edge between the new node and the output node (which is processed by the classifier)
				if k == output_node_indices[j]:
					continue 
				#also avoid connecting the new node to itself
				if k!=new_index:
					graph_sequence[j].add_edge(k,new_index)
					graph_sequence[j].edges[k,new_index]['info'] = translate_bond_direct['candidate']
					#check if this edge has an entry in the arc target dictionary
					if (new_index, k) in arc_target_dict.keys():
						arc_targets.append(arc_target_dict[new_index, k])
					#otherwise add a "cut" target
					else:
						arc_targets.append(4)
			#translate the arc targets to one hot vectors
			arc_supervisions = np.zeros((len(arc_targets),4), dtype=int)
			for k in range(len(arc_targets)):
				arc_supervisions[k][arc_targets[k]-1] = 1
			arc_supervision_list.append(arc_supervisions)
		#drop the last graph as no prediction will be run on the complete graph
		graph_sequence = np.delete(graph_sequence, -1)
		#translate each networkx.DiGraph in the sequence to a GraphObject
		for j in range(len(graph_sequence)):
			graph_sequence[j] = NetworkxDiGraphToEBGraphObject(graph_sequence[j], arc_supervision_list[j], node_aggregation = NODE_AGGREGATION)
		#fuse the graphs together
		graphs_validation[g_id] = GraphObject.fuse(graph_sequence)
	#test set
	for g_id in graph_ids_test:
		i+=1
		print("Decomposing graph "+str(i)+" of "+str(len(graph_ids_training)+len(graph_ids_validation)+len(graph_ids_test)), end='\r')
		#translate the GraphObject to a networkx.DiGraph
		G = GraphObjectToNetworkxDiGraph(graphs_test[g_id])
		#decompose the graph into its generation sequence
		graph_sequence, supervision_sequence, output_node_indices = GraphDecomposition(G)
		#convert the supervision sequence to a numerical vector
		supervision_sequence = np.argmax(supervision_sequence, axis=1)
		#drop all the graphs (but the last) in which a "STOP" prediction is expected, as the linker will not run after "STOP" predictions
		drop_indices = list()
		for j in range(len(supervision_sequence)-1):
			if supervision_sequence[j] == translate_atom_direct['STOP']-1:
				drop_indices.append(j)
		graph_sequence = np.delete(graph_sequence, drop_indices, axis=0)
		supervision_sequence = np.delete(supervision_sequence, drop_indices, axis=0)
		output_node_indices = np.delete(output_node_indices, drop_indices, axis=0)
		#build a list of arc supervion matrices
		arc_supervision_list = list()
		#add the node to be generated in each example
		for j in range(len(graph_sequence)-1):
			#build an arc supervision list
			arc_targets = list()
			#determine arc targets by difference with the next graph
			arc_target_dict = dict()
			for e in graph_sequence[j+1].edges():
				if e not in graph_sequence[j].edges():
					arc_target_dict[e[0], e[1]] = graph_sequence[j+1].edges[e[0], e[1]]['info']
			#add new node
			new_index = len(graph_sequence[j].nodes())
			graph_sequence[j].add_node(new_index)
			#add node label
			graph_sequence[j].nodes[new_index]['info'] = translate_atom_inverse[supervision_sequence[j]+1]
			#add the set of arcs to be discriminated, connecting the new node to each other node
			for k in range(len(graph_sequence[j].nodes())):
				#skip the edge between the new node and the output node (which is processed by the classifier)
				if k == output_node_indices[j]:
					continue 
				#also avoid connecting the new node to itself
				if k!=new_index:
					graph_sequence[j].add_edge(k,new_index)
					graph_sequence[j].edges[k,new_index]['info'] = translate_bond_direct['candidate']
					#check if this edge has an entry in the arc target dictionary
					if (new_index, k) in arc_target_dict.keys():
						arc_targets.append(arc_target_dict[new_index, k])
					#otherwise add a "cut" target
					else:
						arc_targets.append(4)
			#translate the arc targets to one hot vectors
			arc_supervisions = np.zeros((len(arc_targets),4), dtype=int)
			for k in range(len(arc_targets)):
				arc_supervisions[k][arc_targets[k]-1] = 1
			arc_supervision_list.append(arc_supervisions)
		#drop the last graph as no prediction will be run on the complete graph
		graph_sequence = np.delete(graph_sequence, -1)
		#translate each networkx.DiGraph in the sequence to a GraphObject
		for j in range(len(graph_sequence)):
			graph_sequence[j] = NetworkxDiGraphToEBGraphObject(graph_sequence[j], arc_supervision_list[j], node_aggregation = NODE_AGGREGATION)
		#fuse the graphs together
		graphs_test[g_id] = GraphObject.fuse(graph_sequence)
	print("")

	#build batches by fusing graphs together
	#build test batch
	print("Fusing test batch")
	batch_test = GraphObject.fuse(list(graphs_test.values()))
	#build validation batch
	print("Fusing validation batch")
	batch_validation = GraphObject.fuse(list(graphs_validation.values()))
	#build training batches
	graph_list = list(graphs_training.values())
	batches_training = list()
	#determine size of each batch
	batch_sizes = list()
	#start from all batches of equal size
	for i in range(TRAINING_BATCHES):
		batch_sizes.append( int(len(graph_list)/TRAINING_BATCHES) )
	#if the sum of batch sizes exceeds the number of elements, take one element out of each batch, starting from the first, until the numbers match
	if sum(batch_sizes)>len(graph_list):
		for i in range(sum(batch_sizes)-len(graph_list)):
			batch_sizes[i] = batch_sizes[i] - 1
	#if the sum of batch sizes is smaller than the number of elements, add one element to each batch, starting from the first, until the numbers match
	if sum(batch_sizes)<len(graph_list):
		for i in range(len(graph_list)-sum(batch_sizes)):
			batch_sizes[i] = batch_sizes[i] + 1
	#build batches
	start_index = 0
	for i in range(TRAINING_BATCHES):
		print("Fusing training batch "+str(i+1)+" of "+str(TRAINING_BATCHES), end='\r')
		stop_index = start_index+batch_sizes[i]
		new_batch = GraphObject.fuse(graph_list[start_index:stop_index])
		batches_training.append(new_batch)
		start_index = stop_index
	print("")

	#save batches to file
	#save test batch
	temp_file = open(path_batches+"batch_test.pkl", 'wb')
	pickle.dump(batch_test, temp_file)
	temp_file.close()
	#save validation batch
	temp_file = open(path_batches+"batch_validation.pkl", 'wb')
	pickle.dump(batch_validation, temp_file)
	temp_file.close()
	#save training batches
	for i in range(TRAINING_BATCHES):
		temp_file = open(path_batches+"batch_training_"+str(i)+".pkl", 'wb')
		pickle.dump(batches_training[i], temp_file)
		temp_file.close()

	#delete all the variables that are no longer necessary
	del graph_ids_training
	del graph_ids_validation
	del graph_ids_test
	del graphs_training
	del graphs_validation
	del graphs_test
	del graph_list
	del start_index
	del batch_validation
	del batch_test
	del batches_training

#initialize model
print("Initializing Graph Neural Network")
net = standard_net.StandardNet(input_dim = INPUT_DIM, state_dim = STATE_DIM, output_dim = OUTPUT_DIM, hidden_units_state = HIDDEN_UNITS_STATE, hidden_units_output = HIDDEN_UNITS_OUTPUT, namespace="LINKER_")
model = GNN.GNN(net, max_it=MAX_ITER, input_dim=INPUT_DIM, output_dim = OUTPUT_DIM, state_dim=STATE_DIM, num_train_batches = TRAINING_BATCHES, optimizer=tf.train.AdamOptimizer, learning_rate=LR, threshold=THRESHOLD, param=param_string, namespace="LINKER_")
validation_best_loss = None

#build list of training batch paths
batch_paths = list()
for j in range(TRAINING_BATCHES):
	batch_paths.append(path_batches+"batch_training_"+str(j)+".pkl")

#build gumbel softmax annealing path
gumbel_softmax_temperatures = np.ones(EPOCHS)
for i in range(EPOCHS):
	gumbel_softmax_temperatures[i] = (GUMBEL_SOFTMAX_TEMPERATURE_MAX - GUMBEL_SOFTMAX_TEMPERATURE_MIN) * float((EPOCHS - i)) + GUMBEL_SOFTMAX_TEMPERATURE_MIN

#start training
for i in range(EPOCHS):
	#set gumbel softmax temperature
	model.net.gumbel_softmax_temperature = gumbel_softmax_temperatures[i]
	#perform the current training iteration on all the batches
	model.Train(batch_paths, i)
	#validation check
	if i % VALIDATION_INTERVAL == 0:
		#calculate validation loss
		val_loss = model.Validate(path_batches+"batch_validation.pkl", i)
		#initialize best loss with first recorded value
		if validation_best_loss is None:
			validation_best_loss = val_loss
		#check validation loss
		else:
			#in case of improvement, reset checks failed to 0 and continue training
			if val_loss < validation_best_loss:
				validation_best_loss = val_loss

#load test batch
temp_file = open(path_batches+"batch_test.pkl", 'rb')
test_batch = pickle.load(temp_file)
temp_file.close()
#check loss after training
out_wrap, loss, st = model.Predict(test_batch.getInputTensor(), test_batch.getArcNode().T, test_batch.getTargets(), test_batch.getSetMask(), test_batch.getOutputMask(), test_batch.initState())
out_tensor = out_wrap[0]
targets_test = test_batch.getTargets()
#delete the batch
del test_batch
#save output and state tensors for debugging
np.savetxt(debug_state_filename, st, delimiter = ",")
np.savetxt(debug_output_filename, out_tensor, delimiter = ",")

#apply hard-max filter
refined_output = np.zeros(out_tensor.shape)
argmax_vector = np.argmax(out_tensor, axis = 1)
for i in range(len(argmax_vector)):
	k = argmax_vector[i]
	refined_output[i][k] = 1 

#evaluate learning
TP = np.zeros(OUTPUT_DIM)
TN = np.zeros(OUTPUT_DIM)
FP = np.zeros(OUTPUT_DIM)
FN = np.zeros(OUTPUT_DIM)
RIGHT = 0
WRONG = 0
#iterate over all the classes
for k in range(OUTPUT_DIM):
	#iterate over all the targets
	for i in range(len(targets_test)):
		#check if the current class is the expected class of this example
		if targets_test[i][k] == 1:
			#check if the the example was correctly predicted (and it is a true positive for the class) or not (and it is a false negative for the class)
			if refined_output[i][k] == 1:
				RIGHT += 1
				TP[k] += 1
			else:
				WRONG += 1
				FN[k] += 1
		#otherwise just check for a false positive / true negative for this class
		else:
			if refined_output[i][k] == 1:
				FP[k] += 1
			else:
				TN[k] += 1

#calculate accuracy for each class and global accuracy
accuracy = float(RIGHT)/float(RIGHT+WRONG)
class_accuracy = np.zeros(OUTPUT_DIM)
class_precision = np.zeros(OUTPUT_DIM)
class_recall = np.zeros(OUTPUT_DIM)
for k in range(OUTPUT_DIM):
	class_accuracy[k] = float(TP[k] + TN[k]) / float(TP[k] + TN[k] + FP[k] + FN[k])
	if TP[k] + FP[k] > 0:
		class_precision[k] = float(TP[k]) / float(TP[k] + FP[k])
	if TP[k] + FN[k] > 0:
		class_recall[k] = float(TP[k]) / float(TP[k] + FN[k])
#print results
for k in range(OUTPUT_DIM):
	print("Accuracy for class "+str(k)+" = "+str(class_accuracy[k]))
print("Accuracy = "+str(accuracy))

#print results to file
out_file = open(path_results, 'w')
out_file.write("Model: Standard Graph Neural Network \n")
out_file.write("Dataset: QM9 \n\n")
out_file.write("Hidden Units State: "+str(HIDDEN_UNITS_STATE)+"\n")
out_file.write("Hidden Units Output: "+str(HIDDEN_UNITS_OUTPUT)+"\n")
out_file.write("State Dimension: "+str(STATE_DIM)+"\n")
out_file.write("State Convergence Threshold: "+str(THRESHOLD)+"\n")
out_file.write("Max Convergence Iterations:"+str(MAX_ITER)+"\n")
out_file.write("Iinitial Learning Rate:"+str(LR)+"\n")
out_file.write("Max Training Epochs:"+str(EPOCHS)+"\n\n")
out_file.write("Training Batches:"+str(TRAINING_BATCHES)+"\n")
out_file.write("Validation Interval:"+str(VALIDATION_INTERVAL)+"\n\n")
for k in range(OUTPUT_DIM):
	out_file.write("************************************************************\n")
	out_file.write("Results for class "+str(k)+" : \n")
	out_file.write("TP = "+str(TP[k])+", TN = "+str(TN[k])+"\n")
	out_file.write("FP = "+str(FP[k])+", FN = "+str(FN[k])+"\n")
	out_file.write("Precision for class "+str(k)+" = "+str(class_precision[k])+"\n")
	out_file.write("Recall for class "+str(k)+" = "+str(class_recall[k])+"\n")
	out_file.write("Accuracy for class "+str(k)+" = "+str(class_accuracy[k])+"\n")
out_file.write("************************************************************\n")
out_file.write("Accuracy = "+str(accuracy)+"\n")
out_file.flush()
out_file.close()

