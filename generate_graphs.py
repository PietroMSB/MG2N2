#coding=utf-8

import sys
import os
import math
import pickle
import numpy as np
import tensorflow as tf
import scipy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import networkx as nx

from GNN_EdgeBased.graph_class import GraphObject as EBGraphObject
from GNN_EdgeBased import GNN as EBGNN
from GNN_EdgeBased import standard_net as EBNet
from GNN_NodeBased.graph_class import GraphObject as NBGraphObject
from GNN_NodeBased import GNN as NBGNN 
from GNN_NodeBased import standard_net as NBNet
from graph_decomposition import *
from molecule_drawer import MoleculeDrawer

'''
This variant (A1) generates node labels through a 6-way softmax. The node ordering is a breadth-first scheme with atom-type sub-ordering (H, F, O, N, C) based on the average betweenness centrality of each atom type. A graph with N nodes requires N generation steps (N-1 nodes + 1 stop). Edge trimming is performed at every step on the bonds of the newly generated node.
'''

#generator parameters
G_INPUT_DIM = 6				#input tensor width (source id + destination id + arc_label)
G_STATE_DIM = 5				#node label width
G_OUTPUT_DIM = 6			#target tensor width
G_LR = 0.002				#learning rate
G_THRESHOLD = 0.001			#state convergence threshold, in terms of relative state difference
G_MAX_ITER = 6				#maximum number of state convergence iterations
G_TRAINING_BATCHES = 20     #number of training batches
G_HIDDEN_UNITS_STATE = 100 	#number of units in the hidden layer of the supernode state network
G_HIDDEN_UNITS_OUTPUT = 60  #number of units in the hidden layer of the output network
G_NODE_AGGREGATION = "sum"	#node aggregation mode
G_GUMBEL_SOFTMAX_TEMPERATURE = 1.0

#classifier parameters
C_INPUT_DIM = 6				#input tensor width (source id + destination id + arc_label)
C_STATE_DIM = 5				#node label width
C_OUTPUT_DIM = 3			#target tensor width
C_LR = 0.001				#learning rate
C_THRESHOLD = 0.001			#state convergence threshold, in terms of relative state difference
C_MAX_ITER = 4				#maximum number of state convergence iterations
C_TRAINING_BATCHES = 20     #number of batches in which the training set should be split
C_HIDDEN_UNITS_STATE = 40   #number of units in the hidden layer of the state network
C_HIDDEN_UNITS_OUTPUT = 60  #number of units in the hidden layer of the output network
C_NODE_AGGREGATION = "average"#node aggregation mode
C_GUMBEL_SOFTMAX_TEMPERATURE = 1.0

#linker parameters
L_INPUT_DIM = 6				#input tensor width (source id + destination id + arc_label)
L_STATE_DIM = 5				#node label width
L_OUTPUT_DIM = 4			#target tensor width
L_LR = 0.001				#learning rate
L_THRESHOLD = 0.001			#state convergence threshold, in terms of relative state difference
L_MAX_ITER = 6				#maximum number of state convergence iterations
L_TRAINING_BATCHES = 20     #number of batches in which the training set should be split
L_HIDDEN_UNITS_STATE = 50   #number of units in the hidden layer of the state network
L_HIDDEN_UNITS_OUTPUT = 50  #number of units in the hidden layer of the output network
L_NODE_AGGREGATION = "average"#node aggregation mode
L_GUMBEL_SOFTMAX_TEMPERATURE = 1.0

#parameters
RUN_SUFFIX = sys.argv[1]    #string to be appended at the end of the output file's name, in order to identify the run which produced it
EXAMPLES = 10000			#number of examples in the dataset
MAX_GRAPH_SIZE = 80			#maximum number of nodes in a graph
starting_node_distribution = [0.0, 0.675607, 0.096757, 0.223234, 0.004402]
path_model_generator = "Temp/Models/GENERATOR_/model.ckpt"
path_model_classifier = "Temp/Models/CLASSIFIER_/model.ckpt"
path_model_linker = "Temp/Models/LINKER_/model.ckpt"
path_data = "Generated/"
path_results = "Temp/Results/Construction/res_construction_"+RUN_SUFFIX+".txt"


#gpu parameters
use_gpu = True
target_gpu = "1"

#define parameter line
param_string = "Construction_"+RUN_SUFFIX

#set target gpu as the only visible device
if use_gpu:
	os.environ["CUDA_VISIBLE_DEVICES"]=target_gpu

#build directories
if not os.path.exists("Generated/"+RUN_SUFFIX+"/"):
	os.makedirs("Generated/"+RUN_SUFFIX+"/")
if not os.path.exists("Generated/"+RUN_SUFFIX+"/Graphs/"):
	os.makedirs("Generated/"+RUN_SUFFIX+"/Graphs/")
if not os.path.exists("Generated/"+RUN_SUFFIX+"/Images/"):
	os.makedirs("Generated/"+RUN_SUFFIX+"/Images/")

#initialize bond classifier
print("Initializing Bond Classifier")
#define a tensorflow graph for the bond classifier
classifier_graph = tf.Graph()
with classifier_graph.as_default():
	#define the network
	net_classifier = EBNet.StandardNet(input_dim = C_INPUT_DIM, state_dim = C_STATE_DIM, output_dim = C_OUTPUT_DIM, hidden_units_state = C_HIDDEN_UNITS_STATE, hidden_units_output = C_HIDDEN_UNITS_OUTPUT, namespace="CLASSIFIER_")
	#set the gumbel softmax temperature
	net_classifier.gumbel_softmax_temperature = C_GUMBEL_SOFTMAX_TEMPERATURE
	#define the classifier GNN
	classifier = EBGNN.GNN(net_classifier, max_it=C_MAX_ITER, input_dim=C_INPUT_DIM, output_dim = C_OUTPUT_DIM, state_dim=C_STATE_DIM, num_train_batches = C_TRAINING_BATCHES, optimizer=tf.train.AdamOptimizer, learning_rate=C_LR, threshold=C_THRESHOLD, param=param_string, namespace="CLASSIFIER_")
tf.reset_default_graph()

#initialize linker
print("Initializing Linker")
#define a tensorflow graph for the linker
linker_graph = tf.Graph()
with linker_graph.as_default():
	#define the network
	net_linker = EBNet.StandardNet(input_dim = L_INPUT_DIM, state_dim = L_STATE_DIM, output_dim = L_OUTPUT_DIM, hidden_units_state = L_HIDDEN_UNITS_STATE, hidden_units_output = L_HIDDEN_UNITS_OUTPUT, namespace="LINKER_")
	#set the gumbel softmax temperature
	net_linker.gumbel_softmax_temperature = L_GUMBEL_SOFTMAX_TEMPERATURE
	#define the linker GNN
	linker = EBGNN.GNN(net_linker, max_it=L_MAX_ITER, input_dim=L_INPUT_DIM, output_dim = L_OUTPUT_DIM, state_dim=L_STATE_DIM, num_train_batches = L_TRAINING_BATCHES, optimizer=tf.train.AdamOptimizer, learning_rate=L_LR, threshold=L_THRESHOLD, param=param_string, namespace="LINKER_")
tf.reset_default_graph()

#initialize generator
print("Initializing Generator")
#define a tensorflow graph for the generator
generator_graph = tf.Graph()
with generator_graph.as_default():
	#define the network
	net_generator = NBNet.StandardNet(input_dim = G_INPUT_DIM, state_dim = G_STATE_DIM, output_dim = G_OUTPUT_DIM, hidden_units_state = G_HIDDEN_UNITS_STATE, hidden_units_output = G_HIDDEN_UNITS_OUTPUT, namespace="GENERATOR_")
	#set the gumbel softmax temperature
	net_generator.gumbel_softmax_temperature = G_GUMBEL_SOFTMAX_TEMPERATURE
	#define the generator GNN
	generator = NBGNN.GNN(net_generator, max_it=G_MAX_ITER, input_dim=G_INPUT_DIM, output_dim = G_OUTPUT_DIM, state_dim=G_STATE_DIM, num_train_batches = G_TRAINING_BATCHES, optimizer=tf.train.AdamOptimizer, learning_rate=G_LR, threshold=G_THRESHOLD, param=param_string, namespace="GENERATOR_")
tf.reset_default_graph()

#create a fake target for the generation of GraphObjects
fake_node_target = np.zeros(G_OUTPUT_DIM)
fake_edge_target = np.zeros(C_OUTPUT_DIM)

#generate graphs
starting_node_floats = np.random.rand(EXAMPLES)
for i in range(EXAMPLES):
	print("Generating graph "+str(i+1)+" of "+str(EXAMPLES), end = '\r')
	#initialize graph
	G = nx.DiGraph()
	#spawn random first node, according to the starting node probability distribution measured on the training set
	snf = starting_node_floats[i]
	dist_sum = 0
	j = 0
	while snf > dist_sum: 
		dist_sum = dist_sum + starting_node_distribution[j]
		j += 1
		if j > len(starting_node_distribution):
			print("DEBUG: Starting node seed: "+str(snf))
			sys.exit("ERROR: Error encountered while sampling from starting node distribution")
	starting_type = translate_atom_inverse[j]
	#create first node
	G.add_node(0)
	G.nodes[0]['info'] = starting_type
	#call the generator to create the following nodes
	stop = False
	j = 0
	node_expansion_queue = [0]
	while node_expansion_queue and j < MAX_GRAPH_SIZE:
		#make a copy of the graph
		G_copy = G.copy()
		#insert an output mask value in each node of G_copy
		node_to_expand = node_expansion_queue[0]
		#the nodes are numbered according to Breadth-First-Search, so the next node to expand is the one with the lowest index
		output_node_indices = [node_to_expand]
		#use the output_node_indices list as the output mask
		InsertOutputMask(G_copy, output_node_indices)
		#translate the graph into a NodeBased GraphObject
		nb_graph = NetworkxDiGraphToNBGraphObject(G_copy, fake_node_target, node_aggregation = G_NODE_AGGREGATION)
		#ask the generator to create a new node
		with generator_graph.as_default():
			out_wrap, loss, st  = generator.Predict(nb_graph.getInputTensor(), nb_graph.getArcNode().T, nb_graph.getTargets(), nb_graph.getSetMask(), nb_graph.getOutputMask(), nb_graph.initState())
			next_node_probabilities = out_wrap[0]
		tf.reset_default_graph()
		next_node_index = np.argmax(next_node_probabilities, axis=1)
		next_node_literal = translate_atom_inverse[next_node_index[0]+1]
		#if a 'STOP' is predicted, just jump to the next node in the expansion queue, without calling the classifier nor the linker
		if next_node_literal == 'STOP':
			node_expansion_queue.pop(0)
			continue
		#otherwise add the new node to the graph
		j+=1
		G.add_node(j)
		G.nodes[j]['info'] = next_node_literal
		#add the new node to the expansion queue
		node_expansion_queue.append(j)
		#generate the link between the new node and its "parent" node
		G.add_edge(node_to_expand, j)
		G.edges[node_to_expand, j]['info'] = translate_bond_direct['candidate']
		#make a copy of G
		G_copy = G.copy()
		eb_graph = NetworkxDiGraphToEBGraphObject(G_copy, fake_edge_target, node_aggregation = C_NODE_AGGREGATION)
		#ask the classifier to predict the class of the link between the new node and its "parent" node
		with classifier_graph.as_default():
			out_wrap, loss, st = classifier.Predict(eb_graph.getInputTensor(), eb_graph.getArcNode().T, eb_graph.getTargets(), eb_graph.getSetMask(), eb_graph.getOutputMask(), eb_graph.initState())
			edge_probabilities = out_wrap[0]
		tf.reset_default_graph()
		edge_decision = np.argmax(edge_probabilities, axis=1)+1
		#update the edge label with the selected class
		G.edges[node_to_expand, j]['info'] = edge_decision
		#generate also the inverse arc
		G.add_edge(j, node_to_expand)
		G.edges[j, node_to_expand]['info'] = edge_decision
		#make a copy of G
		G_copy = G.copy()		
		#link the new node to each node in the graph other than its parent and itself
		num_arcs = len(G_copy.nodes)-2
		for k in range(len(G_copy.nodes)):
			if k != j and k != node_to_expand:
				G_copy.add_edge(k,j)
				G_copy.edges[k,j]['info'] = translate_bond_direct['candidate']
		#build a fake target tensor for the linker
		fake_arc_targets = np.zeros((num_arcs,L_OUTPUT_DIM))
		#translate the graph into a GraphObject
		l_graph = NetworkxDiGraphToEBGraphObject(G_copy, fake_arc_targets, node_aggregation = L_NODE_AGGREGATION) 		
		#ask the linker which additional arcs should be generated and their classes
		with linker_graph.as_default():
			out_wrap, loss, st = linker.Predict(l_graph.getInputTensor(), l_graph.getArcNode().T, l_graph.getTargets(), l_graph.getSetMask(), l_graph.getOutputMask(), l_graph.initState())
			arc_probabilities = out_wrap[0]
		tf.reset_default_graph()
		#extract decisions
		arc_decisions = np.argmax(arc_probabilities, axis=1)
		#add the suggested arcs to the original graph (k iterates on nodes, l on arc decisions)
		l = 0
		for k in range(len(G.nodes)):
			if k != j and k != node_to_expand:
				#check integrity
				if arc_decisions[l] not in [0,1,2,3]:
					sys.exit("ERROR: linker output is out of the expected range")
				#retrieve decision
				if not arc_decisions[l] == 3: #3 stands for a "DO NOT GENERATE" decision
					G.add_edge(k,j)
					G.add_edge(j,k)
					G.edges[k,j]['info'] = arc_decisions[l]+1
					G.edges[j,k]['info'] = arc_decisions[l]+1
				#update arc decision iterator
				l += 1
	#translate G to an undirected graph for drawing purposes
	G = G.to_undirected()
	#save the generated graph
	graph_dir = path_data+RUN_SUFFIX+"/Graphs/G_"+str(i)
	if not os.path.exists(graph_dir):
		os.makedirs(graph_dir)
	#pickle graph
	out_file = open(graph_dir+"/graph.pkl", "wb")
	pickle.dump(G, out_file)
	out_file.close()
	#print the generated graph to image
	image_path = path_data+RUN_SUFFIX+"/Images/G_"+str(i)+".png"
	### IMAGE DRAWING STARTS HERE ###
	'''
	#paint graph to image
	md = MoleculeDrawer([0.05, 0.05, 1.0, 1.0], print_operations = False)
	#get array of node colors
	node_list = list(G.nodes())
	node_colours = MoleculeDrawer.getNodeColours(G)
	edge_widths = MoleculeDrawer.getEdgeWidths(G)
	#sketch the molecule in 2D space
	coordinates_dict = md.translateGraphToStructuralFormula(G.to_undirected())
	#plot the image
	figure = plt.figure()
	plot_axes = figure.add_subplot(111)
	plot_axes.autoscale(enable = False)
	nx.draw_networkx(G, coordinates_dict, ax = plot_axes, nodelist=node_list, edgelist=G.edges(), arrows=False, with_labels=True, node_size = 300, node_color=node_colours, edge_color='k', linewidths = 1.0, width=edge_widths, font_size=12, font_color='k')
	figure.savefig(image_path)
	plt.close(figure)
	'''
	### IMAGE DRAWING ENDS HERE ###
print("")

print("Execution terminated successfully")
