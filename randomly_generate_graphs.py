#coding=utf-8

import sys
import os
import math
import pickle
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import networkx as nx


from graph_decomposition import *
from molecule_drawer import MoleculeDrawer

'''
This variant (A1) generates node labels through a 6-way softmax. The node ordering is a breadth-first scheme with atom-type sub-ordering (H, F, O, N, C) based on the average betweenness centrality of each atom type. A graph with N nodes requires N generation steps (N-1 nodes + 1 stop). Edge trimming is performed at every step on the bonds of the newly generated node.
'''

#parameters
RUN_SUFFIX = sys.argv[1]    #string to be appended at the end of the output file's name, in order to identify the run which produced it
EXAMPLES = 10000			#number of examples in the dataset
MAX_GRAPH_SIZE = 29			#maximum number of nodes in a graph
starting_node_distribution = [0.0, 0.675607, 0.096757, 0.223234, 0.004402]
all_node_distribution = [0.510986, 0.351657, 0.057948, 0.078030, 0.001379]
edge_distribution = [0.885141, 0.102039, 0.012820]
linker_distribution = [0.026916 , 0.003341 , 0.000778 , 0.968965]
path_data = "Generated/"
path_results = "Temp/Results/Construction/res_construction_"+RUN_SUFFIX+".txt"

#build directories
if not os.path.exists("Generated/"+RUN_SUFFIX+"/"):
	os.makedirs("Generated/"+RUN_SUFFIX+"/")
if not os.path.exists("Generated/"+RUN_SUFFIX+"/Graphs/"):
	os.makedirs("Generated/"+RUN_SUFFIX+"/Graphs/")
if not os.path.exists("Generated/"+RUN_SUFFIX+"/Images/"):
	os.makedirs("Generated/"+RUN_SUFFIX+"/Images/")

#sampling function
def RandomSample(distribution):
	random_value = np.random.rand()
	mapper = 0.0
	for i in range(len(distribution)):
		mapper = mapper + distribution[i]
		if random_value <= mapper:
			return i
	return None

#generator decision function
def GeneratorDecision():
	random_value = np.random.rand()
	if random_value > 0.5:
		return 5
	else:
		return RandomSample(all_node_distribution) 
		
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
		#ask the generator to create a new node
		next_node_index = GeneratorDecision()
		next_node_literal = translate_atom_inverse[next_node_index+1]
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
		#ask the classifier to predict the class of the link between the new node and its "parent" node
		edge_decision = RandomSample(edge_distribution)+1
		#update the edge label with the selected class
		G.edges[node_to_expand, j]['info'] = edge_decision
		#generate also the inverse arc
		G.add_edge(j, node_to_expand)
		G.edges[j, node_to_expand]['info'] = edge_decision
		#make a copy of G
		G_copy = G.copy()		
		#link the new node to each node in the graph other than its parent and itself
		num_arcs = len(G_copy.nodes)-2
		arc_decisions = np.zeros(num_arcs)
		for k in range(num_arcs):
			arc_decisions[k] = RandomSample(linker_distribution)
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
