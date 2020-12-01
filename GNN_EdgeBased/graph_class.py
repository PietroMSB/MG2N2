#coding=utf-8

import sys
import os
import math
import scipy
import numpy as np

#graph object class
class GraphObject:
	
	#constructor
	def __init__(self, arcs, nodes, targets, ArcNode=None, node_aggregation = "average"):
		#transfer type of node aggregation
		self.aggregation = node_aggregation
		#store arcs, nodes and target
		self.arcs = arcs
		self.nodes = nodes
		self.targets = targets
		#copy dimensions
		self.DIM_NODE_LABEL = nodes.shape[1]-1 #first column contains node indices
		self.DIM_ARC_LABEL = arcs.shape[1]-2 #first two columns contain node indices
		self.DIM_INPUT = 2 + self.DIM_ARC_LABEL #source id, destination id, and arc label
 		#build masks and tensors
		self.buildSetMask()
		self.buildOutputMask()
		self.buildInputTensor()
		#build ArcNode tensor or acquire it from input
		if ArcNode is None:
			self.buildArcNode()
		else:
			self.ArcNode = ArcNode

	#build set mask (since each graph is fed separately to the network, this is a vector of ones for each graph)
	def buildSetMask(self):
		self.mask_set = np.ones(len(self.arcs), dtype=bool)

	#build output mask (a one in the last column of the arc label identifies output arcs)
	def buildOutputMask(self):
		self.output_mask = np.zeros(len(self.arcs), dtype=bool)
		for i in range(len(self.arcs)):
			if self.arcs[i][-1]==1:
				self.output_mask[i] = True

	#build the input tensor
	def buildInputTensor(self):
		self.input_tensor = np.zeros((len(self.arcs), self.DIM_INPUT))
		for i in range(len(self.arcs)):
			#acquire id of source node
			id1 = int(self.arcs[i][0])
			#acquire id of target node
			id2 = int(self.arcs[i][1])
			#transfer source node id to input tensor
			self.input_tensor[i][0] = id1
			#transfer target node id to input tensor
			self.input_tensor[i][1] = id2
			#transfer arc label to input tensor
			start_index = 2
			stop_index = 2 + self.DIM_ARC_LABEL
			for j in range(start_index,stop_index):
				self.input_tensor[i][j] = self.arcs[i][j]

	#build arc-node conversion matrix
	def buildArcNode(self):
		if self.aggregation=="average":
			col = np.array(self.arcs[:,1], dtype=int) #column indices are located in the second column of the arcs tensor
			row = np.arange(0, len(col)) #arc id (from 0 to #arcs)
			values_vector = np.ones(len(col)) #vector of ones used to create the sparse matrix (1 if arc i goes into node j, 0 otherwise)
			means, inverse, counts = np.unique(col, return_inverse=True, return_counts=True) #count the neighbours of each node
			values_vector = values_vector/counts[inverse] #normalize by dividing for the number of nodes in the neighbourhood of each node
			self.ArcNode = scipy.sparse.coo_matrix((values_vector, (row, col)), shape=(len(self.arcs), len(self.nodes)))
		elif self.aggregation=="normalized":
			col = self.arcs[:,1] #column indices are located in the second column of the arcs tensor
			row = np.arange(0, len(col)) #arc id (from 0 to #arcs)
			values_vector = np.ones(len(col)) #vector of ones used to create the sparse matrix (1 if arc i goes into node j, 0 otherwise)
			values_vector = values_vector*float(1/float(len(col))) #normalize by dividing for the number of nodes in the graph (average op)
			self.ArcNode = scipy.sparse.coo_matrix((values_vector, (row, col)), shape=(len(self.arcs), len(self.nodes)))
		elif self.aggregation=="sum":
			col = self.arcs[:,1] #column indices are located in the second column of the arcs tensor
			row = np.arange(0, len(col)) #arc id (from 0 to #arcs)
			values_vector = np.ones(len(col)) #vector of ones used to create the sparse matrix (1 if arc i goes into node j, 0 otherwise)
			self.ArcNode = scipy.sparse.coo_matrix((values_vector, (row, col)), shape=(len(self.arcs), len(self.nodes)))
		else:
			sys.exit("ERROR: Unknown aggregation mode")

	#get node labels for state initialization
	def initState(self):
		return self.nodes[:,1:]

	#get arcs
	def getArcs(self):
		return self.arcs
	
	#get nodes
	def getNodes(self):
		return self.nodes

	#get target
	def getTargets(self):
		return self.targets
	
	#get set mask
	def getSetMask(self):
		return self.mask_set

	#get output mask
	def getOutputMask(self):
		return self.output_mask

	#get input tensor
	def getInputTensor(self):
		return self.input_tensor

	#get arc-node conversion matrix
	def getArcNode(self):
		return self.ArcNode	

	#load a graph from a directory which contains: a "node.txt" file, an "arcs.txt" file and a "target.txt" file
	@staticmethod
	def load(graph_dir_path, node_aggregation="average"):
		nodes = np.loadtxt(graph_dir_path+"nodes.txt", delimiter=',', ndmin=2)
		arcs = np.loadtxt(graph_dir_path+"arcs.txt", delimiter=',', ndmin=2)
		targets = np.loadtxt(graph_dir_path+"targets.txt", delimiter=',', ndmin=2)
		return GraphObject(arcs, nodes, targets, node_aggregation=node_aggregation)

	#method to join <ArcNode> tensors when fusing graphs
	@staticmethod
	def fuseArcNodeTensors(graph_list, arc_array, node_array):
		col = arc_array[:,1] #column indices are located in the second column of the arcs tensor
		row = np.arange(0, len(col)) #arc id (from 0 to #arcs)
		#concatenate all the values vectors from the ArcNode tensors of the graphs in input
		values_vector = np.zeros(len(col))
		i = 0
		for g in graph_list:
			g_values = g.ArcNode.data
			for j in range(len(g_values)):
				values_vector[i] = g_values[j]
				i = i+1
		#build the new ArcNode tensor
		arc_node = scipy.sparse.coo_matrix((values_vector, (row, col)), shape=(arc_array.shape[0], node_array.shape[0]))
		#return the new ArcNode tensor
		return arc_node

	#method to join graphs: it takes in input a list of graphs and returns them as a single graph
	@staticmethod
	def fuse(graph_list):
		#declare global lists for nodes, arcs and targets
		arc_list = list()
		node_list = list()
		target_list = list()
		#declare an incremental identifier for nodes
		base_node_id = 0
		#for any graph in the list
		for g in graph_list:
			#transfer targets to global list
			for i in range(g.targets.shape[0]):
				target_list.append(g.targets[i,:])
			#transfer arcs to global list (adding <base_node_id> to each node id)
			for i in range(g.arcs.shape[0]):
				#source and target node ids
				new_arc = [ base_node_id+g.arcs[i,0] , base_node_id+g.arcs[i,1] ]
				#arc label
				for j in range(g.DIM_ARC_LABEL):
					new_arc.append( g.arcs[ i , j+2 ] )
				#append arc to list
				arc_list.append( new_arc )
			#transfer nodes to global list (adding <base_node_id> to each node id)
			for i in range(g.nodes.shape[0]):
				#node id
				new_node = [ base_node_id+g.nodes[i,0] ]
				#node label
				for j in range(g.DIM_NODE_LABEL):
					new_node.append( g.nodes[ i , j+1 ] )
				#append node to list
				node_list.append( new_node )
			#update the incremental identifier for nodes
			base_node_id = base_node_id + g.nodes.shape[0]
		#transform lists into numpy arrays
		arc_array = np.array(arc_list)
		node_array = np.array(node_list)
		target_array = np.array(target_list)
		#fuse ArcNode tensors
		arc_node = GraphObject.fuseArcNodeTensors(graph_list, arc_array, node_array)
		#build new graph
		return GraphObject(arc_array, node_array, target_array, arc_node)
		

