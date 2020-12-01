#coding=utf-8
import sys
import os
import re
import numpy as np
import networkx as nx

from GNN_NodeBased.graph_class import GraphObject as NBGraphObject
from GNN_EdgeBased.graph_class import GraphObject as EBGraphObject

#dictionaries that map literal node labels to integer node labels
translate_atom_direct = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'STOP':6}
translate_atom_inverse = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F', 6: 'STOP'}
#dictionaries that map literal edge labels to integer edge labels
translate_bond_direct = {'single': 1, 'double': 2, 'triple': 3, 'candidate': 4}
translate_bond_inverse = {1: 'single', 2: 'double', 3: 'triple', 4: 'candidate'}

#function that transforms a CompositeGraphObject into a networkx.DiGraph
def GraphObjectToNetworkxDiGraph(graph):
	#transform the graph to networkx.DiGraph
	G = nx.DiGraph()
	for n in graph.getNodes():
		G.add_node(int(n[0]))
		one_hot_label = n[1:]
		numerical_label = -1
		for z in range(len(one_hot_label)):
			if one_hot_label[z] == 1:
				numerical_label = z+1
		if numerical_label == -1:
			sys.exit("ERROR: Unknown node label")
		G.nodes[n[0]]['info'] = translate_atom_inverse[numerical_label]
	for e in graph.getArcs():
		G.add_edge(int(e[0]), int(e[1]))
		one_hot_label = e[2:]
		numerical_label = -1
		for z in range(len(one_hot_label)):
			if one_hot_label[z] == 1:
				numerical_label = z+1
		if numerical_label == -1:
			sys.exit("ERROR: Unknown edge label")
		G.edges[e[0],e[1]]['info'] = numerical_label
	return G

#function that transforms a networkx.DiGraph into a CompositeGraphObject
def NetworkxDiGraphToNBGraphObject(G, targets, node_aggregation = "standard"):
	nodes = list()
	arcs = list()
	node_size = len(translate_atom_direct.keys())+1 #node id + label -stop + output mask
	arc_size = len(translate_bond_direct.keys())+2 #source id + destination id + label
	nodes = np.zeros((len(G.nodes), node_size))
	arcs = np.zeros((len(G.edges), arc_size))
	#build nodes tensor
	for i in range(len(G.nodes)):
		numerical_label = translate_atom_direct[G.nodes[i]['info']]
		nodes[i][0] = i
		nodes[i][numerical_label] = 1
		#output mask
		if G.nodes[i]['output_mask'] == 1:
			nodes[i][-1] = 1
	#build arcs tensor
	i = 0
	for e in G.edges:
		numerical_label = G.edges[e[0], e[1]]['info']
		arcs[i][0] = e[0]
		arcs[i][1] = e[1]
		arcs[i][1+numerical_label] = 1
		i += 1
	return NBGraphObject(arcs, nodes, np.array(targets, ndmin=2), node_aggregation = node_aggregation)

#function that transforms a networkx.DiGraph into a GraphObject
def NetworkxDiGraphToEBGraphObject(G, targets, node_aggregation="standard"):
	nodes = list()
	arcs = list()
	node_size = len(translate_atom_direct.keys()) #node id + label - stop
	arc_size = len(translate_bond_direct.keys())+2 #source id + destination id + label
	nodes = np.zeros((len(G.nodes), node_size))
	arcs = np.zeros((len(G.edges), arc_size))
	#build nodes tensor
	for i in range(len(G.nodes)):
		numerical_label = translate_atom_direct[G.nodes[i]['info']]
		nodes[i][0] = i
		nodes[i][numerical_label] = 1
	#build arcs tensor
	i = 0
	for e in G.edges:
		numerical_label = G.edges[e[0], e[1]]['info']
		arcs[i][0] = e[0]
		arcs[i][1] = e[1]
		arcs[i][1+numerical_label] = 1
		i += 1
	return EBGraphObject(arcs, nodes, np.array(targets, ndmin=2), node_aggregation = node_aggregation)

#function that produces the decomposition of the graph in input, returning a vector of graphs which represent the intermediate steps of its generation
def GraphDecomposition(G):
	#initialize list of resulting graphs
	graph_list = list()
	#initialize list of supervisions
	supervisions = list()
	#initialize list of output node indices (each graph has got one, and only one, output node)
	output_node_indices = list()
	#define node ordering for BFS search following the H, F, O, N, C atom type sub-priority
	node_ordering = list()
	#cycle over all nodes, starting from node 0
	node_queue = list()
	node_queue.append(0)
	for i in range(len(G.nodes)):
		#fetch the next node from the queue
		current_node = node_queue.pop(0)
		#add node id to ordering list
		node_ordering.append(current_node)
		#fetch neighborhood of current node
		neighbors = list()
		neighbor_iterators = G.neighbors(current_node)
		for n in neighbor_iterators:
			if int(n) not in node_ordering and int(n) not in node_queue:
				neighbors.append(int(n))
		#hydrogen scan
		for n in neighbors:
			if G.nodes[n]['info'] == 'H':
				node_queue.append(n)
				supervisions.append([1, 0, 0, 0, 0, 0])
				output_node_indices.append(i)
		#fluorine scan
		for n in neighbors:
			if G.nodes[n]['info'] == 'F':
				node_queue.append(n)
				supervisions.append([0, 0, 0, 0, 1, 0])
				output_node_indices.append(i)
		#oxygen scan
		for n in neighbors:
			if G.nodes[n]['info'] == 'O':
				node_queue.append(n)
				supervisions.append([0, 0, 0, 1, 0, 0])
				output_node_indices.append(i)
		#nitrogen scan
		for n in neighbors:
			if G.nodes[n]['info'] == 'N':
				node_queue.append(n)
				supervisions.append([0, 0, 1, 0, 0, 0])
				output_node_indices.append(i)
		#carbon scan
		for n in neighbors:
			if G.nodes[n]['info'] == 'C':
				node_queue.append(n)	
				supervisions.append([0, 1, 0, 0, 0, 0])
				output_node_indices.append(i)
		#add a "stop" supervision for the last step on this node
		supervisions.append([0, 0, 0, 0, 0, 1])
		output_node_indices.append(i)
	#report an error if the queue is not empty or the ordering has not the correct length
	if node_queue:
		sys.exit("ERROR: node queue not empty after node ordering")
	if len(node_ordering) != len(G.nodes):
		sys.exit("ERROR: node ordering defined incorrectly")
	#initialize new graph
	new_graph = nx.DiGraph()
	#declare node index for new nodes
	i = -1
	#copy nodes and edges following the node ordering
	for h in range(len(supervisions)):
		#add the graph as it is if the supervision at the previous step was 'stop'
		if h > 0:
			if supervisions[h-1][5] == 1:
				graph_list.append(new_graph.copy())
				continue
		#otherwise add the new node and its edges to the graph
		i += 1	
		#retrieve i-th node of the ordering
		ii = node_ordering[i]
		#add the i-th node of the ordering to the new graph
		new_graph.add_node(i)
		#copy the i-th node's label
		new_graph.nodes[i]['info'] = G.nodes[ii]['info']
		#copy all the edges involving the new node and another node which has already been copied in the new graph
		for e in G.edges:
			jj = e[0]
			kk = e[1]
			j = node_ordering.index(jj)
			k = node_ordering.index(kk)
			if (j==i and k<i) or (j<i and k==i):
				new_graph.add_edge(j,k)
				new_graph.edges[j,k]['info'] = G.edges[jj,kk]['info']
		#add the new graph to the list
		graph_list.append(new_graph.copy())
	#return the list of graphs
	return graph_list, supervisions, output_node_indices
	
#function that adds an output mask value to each node of graph G. The value is 1 for each node whose index in the output_node_indices list, 0 for the other nodes.
def InsertOutputMask(G, output_node_indices):
	for i in range(len(G.nodes)):
		if i in output_node_indices:
			G.nodes[i]['output_mask'] = 1
		else:
			G.nodes[i]['output_mask'] = 0

