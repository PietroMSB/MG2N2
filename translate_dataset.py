#coding=utf-8
import sys
import os
import re
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import networkx as nx
from molecule_drawer import MoleculeDrawer

#parameters
path_dataset = "Data/Graphs/"
path_images = "Data/Images/"
path_target = "targets.txt"
path_nodes = "nodes.txt"
path_arcs = "arcs.txt"
label_super_node = 4
	
#original dataset files
original_arcs_file = "Data/Raw/graph.txt"
original_targets_file = "Data/Raw/label.txt"
original_nodes_file = "Data/Raw/node.txt"

#dictionaries that map literal node labels to integer node labels
translate_atom_direct = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'STOP':6}
translate_atom_inverse = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F', 6: 'STOP'}
#dictionaries that map literal edge labels to integer edge labels
translate_bond_direct = {'single': 1, 'double': 2, 'triple': 3, 'supernode': 4}
translate_bond_inverse = {1: 'single', 2: 'double', 3: 'triple', 4: 'supernode'}

#translates integer node labels into one-hot labels
def translateNodeLabel(in_label):
	out_label = np.zeros(6, dtype=int) #last code is reserved for stop labels
	#obtain integer label given the literal label
	in_label = translate_atom_direct[in_label]
	#convert the input label (literal) into a 6-bit one-hot label
	out_label[in_label-1] = 1
	return out_label

#translates integer edge labels into one-hot labels
def translateEdgeLabel(in_label):
	out_label = np.zeros(4, dtype=int) #last code is reserved for the label of supernode edges
	#convert the input label (literal) into a 4-bit one-hot label
	out_label[in_label-1] = 1
	return out_label

#creates a dictionary of atom positions for the molecule graph in input
def createPosDict(G):
	pos_dict = {}
	#count backbone positions
	count_backbone = 0
	for i in len(G.nodes):
		if G.nodes[i]['info'] in ['C', 'N']:
			count_backbone += 1
	

#execution starts here


#build directories
if not os.path.exists("Data"):
	os.makedirs("Data")
if not os.path.exists("Data/Graphs"):
	os.makedirs("Data/Graphs")
if not os.path.exists("Data/Images"):
	os.makedirs("Data/Images")
if not os.path.exists("Temp"):
	os.makedirs("Temp")
if not os.path.exists("Temp/Debug"):
	os.makedirs("Temp/Debug")
if not os.path.exists("Temp/Models"):
	os.makedirs("Temp/Models")
if not os.path.exists("Temp/Models/TRIMMER_"):
	os.makedirs("Temp/Models/TRIMMER_")
if not os.path.exists("Temp/Models/GENERATOR_"):
	os.makedirs("Temp/Models/GENERATOR_")
if not os.path.exists("Temp/Results"):
	os.makedirs("Temp/Results")
if not os.path.exists("Temp/Batches"):
	os.makedirs("Temp/Batches")
if not os.path.exists(path_dataset):
	os.makedirs(path_dataset)
if not os.path.exists(path_images):
	os.makedirs(path_images)

#load connectivity matrices
print("Loading connectivity matrices")
original_arcs = list()
in_file = open(original_arcs_file,'r')
in_text = in_file.read()
file_lines = in_text.splitlines(in_text.count("\n"))
del in_text
in_file.close()
k = 0
j = 0
connect_matrix = None
for l in file_lines:
	#initialize next matrix
	if j==k:
		k = int(l)
		j = 0
		connect_matrix = np.zeros((k,k), dtype=int)
	#copy next row of the matrix
	else:
		c = re.split("\s",l)
		for i in range(len(c)):
			try:
				connect_matrix[j][i] = int(c[i])
			except:
				continue
		#if this is the last row, save the matrix
		if j==k-1:
			if connect_matrix is not None:
				original_arcs.append(connect_matrix.copy())
		#update j
		j += 1
del file_lines

#load graph targets
original_graph_classes = np.loadtxt(original_targets_file, dtype=float, delimiter=',')

#load node labels
print("Loading node information")
original_node_labels = list()
in_file = open(original_nodes_file, 'r')
in_text = in_file.read()
file_lines = in_text.splitlines(in_text.count("\n"))
del in_text
in_file.close()
nodes_in_graph = -1
for j in range(len(file_lines)):
	c = re.split("\s",file_lines[j])
	#skip lines reporting node counts
	if len(c)<=2:
		nodes_in_graph = int(c[0])
		continue
	#acquire node labels
	label_list = list()
	for k in range(len(c)):
		cell = None
		#skip unreadable cells
		try:
			cell = c[k]
		except:
			continue
		#skip endlines and other non-atom charachters
		if cell not in translate_atom_direct.keys():
			continue
		label_list.append(cell)
	#check the length of the list of labels
	if len(label_list) != nodes_in_graph:
		sys.exit("ERROR: error while parsing node labels!")
	nodes_in_graph = -1		
	original_node_labels.append(label_list)
del file_lines

#generate each graph
how_many = 133582
stop = False
arc_counter = 0
for i in range(how_many):
	print("Translating graph "+str(i+1)+" of "+str(how_many), end='\r')
	G = nx.DiGraph()
	#build nodes (node id, node label)
	for j in range(len(original_node_labels[i])):
		G.add_node(j)
		G.nodes[j]['info'] = original_node_labels[i][j]
	#build arcs (node id 1, node id 2, 1-hot label)
	for j in range(original_arcs[i].shape[0]):
		for k in range(original_arcs[i].shape[1]):
			if original_arcs[i][j][k] in [1, 2, 3]:
				#build edge			
				G.add_edge(j,k)
				G.edges[j, k]['info'] = original_arcs[i][j][k]
	#print graph data to files
	graph_dir = path_dataset+"G_"+str(i)+"/"
	if not os.path.exists(graph_dir):
		os.mkdir(graph_dir)
	#pickle graph
	out_file = open(graph_dir+"graph.pkl", "wb")
	pickle.dump(G, out_file)
	out_file.close()
	#write the node list on a txt file
	out_file = open(graph_dir+path_nodes, "w")
	for j in range(len(G.nodes)):
		#translate the node label into the corresponding embedding
		node_label = translateNodeLabel(G.nodes[j]['info'])
		#write node id (index) and node label
		out_file.write(str(j))
		for k in range(len(node_label)):
			out_file.write(", "+str(node_label[k]))
		out_file.write("\n")
	out_file.close()
	#write the arc list on a txt file
	out_file = open(graph_dir+path_arcs, "w")
	edge_list = G.edges(None, True, True)
	for arc in edge_list:
		#translate the edge label into the corresponding embedding
		edge_label = translateEdgeLabel(G.edges[arc[0],arc[1]]['info'])
		#write node ids and arc label (type)
		out_file.write(str(arc[0])+", "+str(arc[1]))
		for k in range(len(edge_label)):
			out_file.write(", "+str(edge_label[k]))
		out_file.write("\n")
	out_file.close()
	#write the target in a separate text file
	out_file = open(graph_dir+path_target, "w")
	class_one_hot = [1, 0, 0]
	out_file.write(str(class_one_hot[0]))
	for j in range(1,len(class_one_hot)):				
		out_file.write(", "+str(class_one_hot[j]))
	out_file.close()
	### IMAGE DRAWING STARTS HERE ###
	##paint graph to image
	#md = MoleculeDrawer([0.05, 0.05, 1.0, 1.0], print_operations = False)
	##get array of node colors
	#node_list = list(G.nodes())
	#node_colours = MoleculeDrawer.getNodeColours(G)
	#edge_widths = MoleculeDrawer.getEdgeWidths(G)
	##sketch the molecule in 2D space
	#coordinates_dict = md.translateGraphToStructuralFormula(G.to_undirected())
	##plot the image
	#figure = plt.figure()
	#plot_axes = figure.add_subplot(111)
	#plot_axes.autoscale(enable = False)
	#nx.draw_networkx(G, coordinates_dict, ax = plot_axes, nodelist=node_list, edgelist=G.edges(), arrows=False, with_labels=True, node_size = 300, node_color=node_colours, edge_color='k', linewidths = 1.0, width=edge_widths, font_size=12, font_color='k')
	#figure.savefig(path_images+"G_"+str(i)+".png")
	#plt.close(figure)
	### IMAGE DRAWING ENDS HERE ###

#terminate execution
print("\nTranslation terminated succesfully")


#execution ends here
	


