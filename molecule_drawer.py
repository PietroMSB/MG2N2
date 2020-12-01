#coding=utf-8

import sys
import os
import math
import scipy
import numpy as np

#graph object class
class MoleculeDrawer:
	
	#constructor
	def __init__(self, boundaries, print_operations=False):
		#<boundaries> defines the minimum and maximum coordinates in which to draw the molecule
		self.coords_x_min = boundaries[0]
		self.coords_y_min = boundaries[1]
		self.coords_x_max = boundaries[2]
		self.coords_y_max = boundaries[3]
		self.span_x = float(self.coords_x_max - self.coords_x_min )
		self.span_y = float(self.coords_y_max - self.coords_y_min )	
		#set verbose True/False
		self.verbose = print_operations
		#prepare data structures
		self.closed_nodes = list()
		self.closed_edges = list()
		self.explore_queue = list()
		self.directions_for_queue = list()
		self.coordinates_for_queue = list()
		self.explored_directions = dict()
		self.archived_cycles = list()
	
	#method that returns an array of node colours for the graph in input
	@staticmethod
	def getNodeColours(G):
		colours = list()
		for i in range(len(G.nodes)):
			atom_type = G.nodes[i]["info"]
			if atom_type == "H":
				colours.append('m')
			elif atom_type == "C":
				colours.append('0.7')
			elif atom_type == "N":
				colours.append('y')
			elif atom_type == "O":
				colours.append('b')
			elif atom_type == "F":
				colours.append('g')
		return colours
	
	#method that returns an array of node colours for the graph in input
	@staticmethod
	def getEdgeWidths(G):
		widths = list()
		for e in G.edges():
			bond_type = G.edges[e[0], e[1]]["info"]
			if bond_type == 1:
				widths.append(0.8)
			elif bond_type == 2:
				widths.append(1.6)
			elif bond_type == 3:
				widths.append(3.0)
		return widths

	#method that returns coordinates after one pace in the given direction
	@staticmethod
	def stride(coords, direction, atom_type_1, atom_type_2):
		#determine stride pace according to atom types
		pace = None
		if atom_type_1 == "H":
			if atom_type_2 == "H":
				pace = 0.4
			elif atom_type_2 in ["C", "N"]:
				pace = 0.4
			elif atom_type_2 in ["F", "O"]:
				pace = 0.2
		elif atom_type_1 in ["C", "N"]:
			if atom_type_2 == "H":
				pace = 0.4
			elif atom_type_2 in ["C", "N"]:
				pace = 1
			elif atom_type_2 in ["F", "O"]:
				pace = 0.6
		elif atom_type_1 in ["F", "O"]:
			if atom_type_2 == "H":
				pace = 0.2
			elif atom_type_2 in ["C", "N"]:
				pace = 0.6
			elif atom_type_2 in ["F", "O"]:
				pace = 0.6
		#calculate new coords according to stride direction
		new_coords = [-1, -1]
		if direction == "up":
			new_coords[0] = coords[0]
			new_coords[1] = coords[1]+1*pace
		elif direction == "down":
			new_coords[0] = coords[0]
			new_coords[1] = coords[1]-1*pace
		elif direction == "left":
			new_coords[0] = coords[0]-1*pace
			new_coords[1] = coords[1]
		elif direction == "right":
			new_coords[0] = coords[0]+1*pace
			new_coords[1] = coords[1]
		elif direction == "top-right":
			new_coords[0] = coords[0]+0.7*pace
			new_coords[1] = coords[1]+0.7*pace
		elif direction == "top-left":
			new_coords[0] = coords[0]-0.7*pace
			new_coords[1] = coords[1]+0.7*pace
		elif direction == "bottom-right":
			new_coords[0] = coords[0]+0.7*pace
			new_coords[1] = coords[1]-0.7*pace
		elif direction == "bottom-left":
			new_coords[0] = coords[0]-0.7*pace
			new_coords[1] = coords[1]-0.7*pace
		return new_coords

	#method that reorders the given edge tuple, so that the lower node id always comes first
	@staticmethod
	def edgeInvariant(edge):
		if edge[0] > edge[1]:
			return (edge[1], edge[0])
		else:
			return edge

	#method that reorders a list of edge tuples
	@staticmethod
	def edgeListInvariant(edges):
		new_list = list()
		for e in edges:
			new_list.append(edgeInvariant(e))

	#method that returns the list of the next directions, given the current direction and the number of outcoming bonds
	@staticmethod
	def getNextDirections(current_dir, num_bonds):
		if num_bonds == 1:
			if current_dir == "none":
				return ["right"]
			else:
				return []
		if num_bonds == 2:
			if current_dir == "none":
				return ["left", "right"]
			else:
				return [current_dir]
		elif num_bonds == 3:
			if current_dir == "none":
				return ["left", "top-right", "bottom-right"]
			elif current_dir == "up":
				return ["top-left", "top-right"]
			elif current_dir == "top-left":
				return ["left", "up"]
			elif current_dir == "left":
				return ["bottom-left", "top-left"]
			elif current_dir == "bottom-left":
				return ["down", "left"]
			elif current_dir == "down":
				return ["bottom-right", "bottom-left"]
			elif current_dir == "bottom-right":
				return ["right", "down"]
			elif current_dir == "right":
				return ["top-right", "bottom-right"]
			elif current_dir == "top-right":
				return ["up", "right"]	
		elif num_bonds == 4:
			if current_dir == "none":
				return ["left", "right", "down", "up"]
			elif current_dir == "up":
				return ["up", "left", "right"]
			elif current_dir == "top-left":
				return [ "top-left", "bottom-left", "top-right"]
			elif current_dir == "left":
				return ["left", "down", "up"]
			elif current_dir == "bottom-left":
				return [ "bottom-left", "bottom-right", "top-left"]
			elif current_dir == "down":
				return [ "down", "right", "left"]
			elif current_dir == "bottom-right":
				return ["bottom-right", "top-right", "bottom-left"]
			elif current_dir == "right":
				return ["right", "up", "down"]
			elif current_dir == "top-right":
				return ["top-right", "top-left", "bottom-right"]
		else:
			return []

	#builds structural formula from graph data, returning the dictionary (node_id: coordinates)
	def translateGraphToStructuralFormula(self, G):
		#prepare coordinate tensor
		raw_coordinates = np.zeros((len(G.nodes), 2), dtype=float)
		#prepare support dictionary for explored directions from every atom		
		for n in G.nodes:
			self.explored_directions[n] = list()
		#launch explorer
		next_coords = [0, 0]
		self.explorer(G, raw_coordinates, 0, next_coords, "none")
		#explore the graph until every node has been resolved
		while self.explore_queue:
			next_node = self.explore_queue[0]
			self.explorer(G, raw_coordinates, next_node, self.coordinates_for_queue[0], self.directions_for_queue[0])
			self.explore_queue.pop(0)
			if self.coordinates_for_queue:
				self.coordinates_for_queue.pop(0)
			if self.directions_for_queue:
				self.directions_for_queue.pop(0)
		#prepare final coordinate tensor
		coordinates = np.zeros((len(G.nodes), 2), dtype=float)
		#translate coordinates according to minimum and maximum values specified
		rc_x_min = min(raw_coordinates[:,0])
		rc_x_max = max(raw_coordinates[:,0])
		rc_y_min = min(raw_coordinates[:,1])
		rc_y_max = max(raw_coordinates[:,1])
		rc_x_span = float(rc_x_max - rc_x_min)
		rc_y_span = float(rc_y_max - rc_y_min)
		#to avoid divisions by zero, set spans to 1 for mono-atomic graphs (if the graph is composed of just one atom, rc_x_span and rc_y_span are equal to zero)
		if len(G.nodes) == 1:
			rc_x_span = 1
			rc_y_span = 1
		#prepare coordinate normalizer
		divisor = 1
		#check which direction has the highest ratio of raw coordinate span over boundary coordinate span
		x_centering = 0.0
		y_centering = 0.0
		if rc_x_span / self.span_x >= rc_y_span / self.span_y:
			divisor = rc_x_span / self.span_x
			y_centering = float(self.span_y)/2 - float(self.span_y*rc_y_span/rc_x_span)/2
		else:
			divisor = rc_y_span / self.span_y
			x_centering = float(self.span_x)/2 - float(self.span_x*rc_x_span/rc_y_span)/2
		#calculate new coordinates
		for i in range(raw_coordinates.shape[0]):
			coordinates[i][0] = float(raw_coordinates[i][0]-rc_x_min)/divisor
			coordinates[i][1] = float(raw_coordinates[i][1]-rc_y_min)/divisor
			#add minimum values
			coordinates[i][0] = coordinates[i][0]*self.span_x + self.coords_x_min + x_centering
			coordinates[i][1] = coordinates[i][1]*self.span_y + self.coords_y_min + y_centering
		#translate coordinate tensor to dictionary
		coord_dict = dict()
		for i in range(coordinates.shape[0]):
			coord_dict[i] = coordinates[i,:]
		#reset exploration lists
		self.closed_nodes = list()
		self.closed_edges = list()
		self.explore_queue = list()
		self.directions_for_queue = list()
		self.coordinates_for_queue = list()
		self.explored_directions = dict()
		self.archived_cycles = list()
		#return dictionary of coordinates
		return coord_dict
		

	#recursive method to explore graphs
	def explorer(self, G, raw_coordinates, node, next_coords, current_direction):
		if self.verbose: print("Exploring Node "+str(node))
		#if the node is already fully resolved, terminate this branch
		if node in self.closed_nodes:
			if self.verbose: print("Closed Node: return")
			return
		#otherwise process the node and all of its edges
		raw_coordinates[node, :] = next_coords
		#close node, preventing other exploration branches from setting its coordinates 
		self.closed_nodes.append(node)
		if self.verbose: print("Set atom coordinates to "+str(next_coords))
		#check number of bonds to determine how many directions will be explored
		num_bonds = len(G.edges(node))
		if self.verbose: print("Atom has "+str(num_bonds)+" bonds")
		next_directions = MoleculeDrawer.getNextDirections(current_direction, num_bonds)
		#eliminate directions which were already explored
		for nd in next_directions:
			if nd in self.explored_directions[node]:
				next_directions.remove(nd)
		if self.verbose: print("Directions to explore: "+str(next_directions))
		#otherwise retrieve list of edges to be explored
		edge_list = list(G.edges(node))
		if self.verbose: print("Edge list: "+str(edge_list))
		for e in edge_list:
			if MoleculeDrawer.edgeInvariant(e) in self.closed_edges:
				edge_list.remove(e)
				if self.verbose: print("Edge "+str(e)+" was explored previously")
		#if all the edges have been explored, terminate branch
		if not edge_list:
			if self.verbose: print("All edges explored: return")
			return
		#otherwise check if the node is located inside any aromatic cycle
		if self.verbose: print("Launching cycle seeker")
		res = MoleculeDrawer.cycleSeeker(G, node, list(), list(), self.verbose)
		#if cycles are found, follow them before proceeding to other nodes
		if res:
			if self.verbose: print("Cycles before minimality check: "+str(res))
			#check that every cycle is minimal (no shorter cycles exists between a subset of its nodes)
			for nodes_in_cycle_0 in res:
				#check minimality
				is_minimal = True
				for nodes_in_cycle_1 in res:
					#compare the list lengths, keep only lists which are shorter than the list 1
					if len(nodes_in_cycle_1) >= len(nodes_in_cycle_0):
						continue 
					#check if every node that belongs to list 1 also belongs to list 0
					is_a_subset = True
					for n1 in nodes_in_cycle_1:
						if n1 not in nodes_in_cycle_0:
							is_a_subset = False
					#if any other cycle is a subset of cycle 0, than the latter is not minimal
					if is_a_subset:
						is_minimal = False
				#if the cycle is not minimal, it should be eliminated
				if not is_minimal:
					res.remove(nodes_in_cycle_0)
			if self.verbose: print("Cycles after minimality check: "+str(res))
			#retrieve list of nodes in the cycle (for each cycle)
			for nodes_in_cycle in res:
				if self.verbose: print("Cycle found, with nodes: "+str(nodes_in_cycle))
				#check if the cycle was already explored
				sorted_node_list = nodes_in_cycle.copy()
				sorted_node_list.sort()
				#skip cycles that were already explored
				if sorted_node_list in self.archived_cycles:
					if self.verbose: print("This cycle was already resolved")
					continue
				#otherwise proceed with the exploration, and archive the cycle as explored
				self.archived_cycles.append(sorted_node_list)
				#obtain cycle layout
				if len(nodes_in_cycle) == 3:
					if self.verbose: print("Drawing triangle")
					cycle_coordinates, cycle_directions = MoleculeDrawer.layoutTriangle(G, current_direction, next_coords)
				elif len(nodes_in_cycle) == 4:
					if self.verbose: print("Drawing square")
					cycle_coordinates, cycle_directions = MoleculeDrawer.layoutSquare(G, current_direction, next_coords)
				elif len(nodes_in_cycle) == 5:
					if self.verbose: print("Drawing pentagon")
					cycle_coordinates, cycle_directions = MoleculeDrawer.layoutPentagon(G, current_direction, next_coords)
				elif len(nodes_in_cycle) == 6:
					if self.verbose: print("Drawing hexagon")
					cycle_coordinates, cycle_directions = MoleculeDrawer.layoutHexagon(G, current_direction, next_coords)
				#remove edges from edge_list
				for e in edge_list:
					if e[1] in nodes_in_cycle:
						edge_list.remove(e)
				#queue each node in the cycle (except 0, which is the current node) for exploration, closing the edges which connect them
				if self.verbose: print("Closing cycle edges")
				#close edges
				self.closed_edges.append(MoleculeDrawer.edgeInvariant((nodes_in_cycle[0], nodes_in_cycle[-1])))
				for i in range(1,len(nodes_in_cycle)):
					self.closed_edges.append(MoleculeDrawer.edgeInvariant((nodes_in_cycle[i-1], nodes_in_cycle[i])))
				#queue nodes
				for i in range(1,len(nodes_in_cycle)):
					if self.verbose: print("Appending node "+str(nodes_in_cycle[i])+" to exploration queue")
					self.explore_queue.append(nodes_in_cycle[i])
					self.directions_for_queue.append(cycle_directions[i-1])
					self.coordinates_for_queue.append(cycle_coordinates[i-1])
				#update explored directions in the cycle
				for i in range(len(nodes_in_cycle)):
					if self.verbose: print("Direction "+cycle_directions[i]+" explored for node "+str(nodes_in_cycle[i]))
					self.explored_directions[nodes_in_cycle[i]].append(cycle_directions[i])
				#remove explored direction for this node from <next_directions>
				if cycle_directions[0] in next_directions:
					next_directions.remove(cycle_directions[0])
		if self.verbose: print("Out of cycles")
		#after the cycles have been processed, other edges can be expanded
		sorted_edge_list = list()
		#links to carbon atoms should be explored first, followed by nitrogen, oxygen, fluorine, and hydrogen
		atom_species_ordering = ["C", "N", "O", "F", "H"]
		for a_s in atom_species_ordering:
			for e in edge_list:
				if G.nodes[e[1]]["info"] == a_s:
					sorted_edge_list.append(e)
		#explore edges
		for e in sorted_edge_list:
			if self.verbose: print("Exploring edge "+str(e)+" towards node "+str(e[1]))
			#check if the edge has been closed
			if MoleculeDrawer.edgeInvariant(e) in self.closed_edges:
				if self.verbose: print("Edge already explored")
				continue
			#otherwise explore the edge
			self.closed_edges.append(MoleculeDrawer.edgeInvariant(e))
			#calculate direction
			next_dir = None
			#try fetching a direction from the queue
			if next_directions:
				next_dir = next_directions.pop(0)
			#if the queue is empty (valence error), a random not yet explored direction is taken
			else:
				for nndd in ["up", "top-left", "left", "bottom_left", "down", "bottom-right", "right", "top-right"]:
					if nndd not in self.explored_directions[node]:
						next_dir = nndd
			if self.verbose: print("Edge direction is "+next_dir)
			self.explored_directions[node].append(next_dir)
			#retrieve atom types
			atom_type_source = G.nodes[node]["info"]
			atom_type_destination = G.nodes[e[1]]["info"]
			#calculate new coordinates
			new_next_coords = MoleculeDrawer.stride(next_coords, next_dir, atom_type_source, atom_type_destination)
			#append destination node to exploration queue
			if self.verbose: print("Appending node "+str(e[1])+" to exploration queue")
			self.explore_queue.append(e[1])
			self.directions_for_queue.append(next_dir)
			self.coordinates_for_queue.append(new_next_coords)
		#terminate node exploration
		if self.verbose: print("Node "+str(node)+" fully explored, closing")
		return
		
	#recursive method to check for cycles
	@staticmethod
	def cycleSeeker(G, node, nodes_visited, edges_visited, verbose):
		if verbose: print("CS: looking for cycles from node "+str(node))
		cycles_detected = list()
		#check if a circular path starts and ends in this node, and, in case, return it as a one element list
		if nodes_visited:
			if node == nodes_visited[0]:
				if verbose: print("CS: node "+str(node)+" completes a cycle")
				cycles_detected.append(nodes_visited)
				return cycles_detected
		#check if a circular path ends in this node, but starts elsewhere
		if node in nodes_visited:
			if verbose: print("CS: cycle detected further away, returning")
			return cycles_detected
		#otherwise add the node to the visited nodes
		nodes_visited.append(node)
		#if more than six nodes were visited without detecting a cycle, return False
		if len(nodes_visited)>6:
			if verbose: print("CS: this cycle seeker instance visited more than six nodes without finding cycles, closing")
			return cycles_detected
		#check all the outcoming links that were not visited
		outcoming_edges = G.edges(node)
		if verbose: print("CS: outcoming edges are "+str(outcoming_edges))
		for oe in outcoming_edges:
			if MoleculeDrawer.edgeInvariant(oe) not in edges_visited:
				#append both edge directions to the list of visited edges
				edges_visited.append(MoleculeDrawer.edgeInvariant(oe))
				next_node = oe[1]
				if verbose: print("CS: processing edge "+str(oe)+" towards node "+str(next_node))
				#launch cycleSeeker copy on next node
				res = MoleculeDrawer.cycleSeeker(G, next_node, nodes_visited.copy(), edges_visited.copy(), verbose)
				if res:
					cycles_detected = cycles_detected + res
		return cycles_detected

	#method that returns directions and coordinates for a three-atom cycle
	@staticmethod
	def layoutTriangle(G, current_direction, coords_zero):
		directions = list()
		coordinates = list()
		if current_direction == "none":
			directions = ["top-left", "right", "bottom-left"]
			coordinates.append([coords_zero[0]-0.6, coords_zero[1]+0.8])
			coordinates.append([coords_zero[0]+0.6, coords_zero[1]+0.8])
		elif current_direction == "up":
			directions = ["top-left", "right", "bottom-left"]
			coordinates.append([coords_zero[0]-0.6, coords_zero[1]+0.8])
			coordinates.append([coords_zero[0]+0.6, coords_zero[1]+0.8])
		elif current_direction == "top-left":
			directions = ["left", "top-right", "down"]
			coordinates.append([coords_zero[0]-1, coords_zero[1]])
			coordinates.append([coords_zero[0], coords_zero[1]+1])
		elif current_direction == "left":
			directions = ["bottom-left", "up", "bottom-right"]
			coordinates.append([coords_zero[0]-0.8, coords_zero[1]-0.6])
			coordinates.append([coords_zero[0]-0.8, coords_zero[1]+0.6])
		elif current_direction == "bottom-left":
			directions = ["down", "top-left", "right"]
			coordinates.append([coords_zero[0], coords_zero[1]-1])
			coordinates.append([coords_zero[0]-1, coords_zero[1]])
		elif current_direction == "down":
			directions = ["bottom-right", "left", "top-right"]
			coordinates.append([coords_zero[0]+0.6, coords_zero[1]-0.8])
			coordinates.append([coords_zero[0]-0.6, coords_zero[1]-0.8])
		elif current_direction == "bottom-right":
			directions = ["right", "bottom-left", "up"]
			coordinates.append([coords_zero[0]+1, coords_zero[1]])
			coordinates.append([coords_zero[0], coords_zero[1]-1])
		elif current_direction == "right":
			directions = ["top-right", "down", "top-left"]
			coordinates.append([coords_zero[0]+0.8, coords_zero[1]+0.6])
			coordinates.append([coords_zero[0]+0.8, coords_zero[1]-0.6])
		elif current_direction == "top-right":
			directions = ["up", "bottom-right", "left"]
			coordinates.append([coords_zero[0], coords_zero[1]+1])
			coordinates.append([coords_zero[0]+1, coords_zero[1]])
		return coordinates, directions

	#method that returns directions and coordinates for a four-atom cycle
	@staticmethod
	def layoutSquare(G, current_direction, coords_zero):
		directions = list()
		coordinates = list()
		if current_direction == "none":
			directions = ["left", "up", "right", "down"]
			coordinates.append([coords_zero[0]-1, coords_zero[1]])
			coordinates.append([coords_zero[0]-1, coords_zero[1]+1])
			coordinates.append([coords_zero[0], coords_zero[1]+1])
		elif current_direction == "up":
			directions = ["top-left", "top-right", "bottom-right", "bottom-left"]
			coordinates.append([coords_zero[0]-0.7, coords_zero[1]+0.7])
			coordinates.append([coords_zero[0], coords_zero[1]+1.4])
			coordinates.append([coords_zero[0]+0.7, coords_zero[1]+0.7])
		elif current_direction == "top-left":
			directions = ["left", "up", "right", "down"]
			coordinates.append([coords_zero[0]-1, coords_zero[1]])
			coordinates.append([coords_zero[0]-1, coords_zero[1]+1])
			coordinates.append([coords_zero[0], coords_zero[1]+1])
		elif current_direction == "left":
			directions = ["bottom-left", "top-left", "top-right", "bottom-right"]
			coordinates.append([coords_zero[0]-0.7, coords_zero[1]-0.7])
			coordinates.append([coords_zero[0]-1.4, coords_zero[1]])
			coordinates.append([coords_zero[0]-0.7, coords_zero[1]+0.7])
		elif current_direction == "bottom-left":
			directions = ["down", "left", "up", "right"]
			coordinates.append([coords_zero[0], coords_zero[1]-1])
			coordinates.append([coords_zero[0]-1, coords_zero[1]-1])
			coordinates.append([coords_zero[0]-1, coords_zero[1]])
		elif current_direction == "down":
			directions = ["bottom-right", "bottom-left", "top-left", "top-right"]
			coordinates.append([coords_zero[0]+0.7, coords_zero[1]-0.7])
			coordinates.append([coords_zero[0], coords_zero[1]-1.4])
			coordinates.append([coords_zero[0]-0.7, coords_zero[1]-0.7])
		elif current_direction == "bottom-right":
			directions = ["right", "down", "left", "up"]
			coordinates.append([coords_zero[0]+1, coords_zero[1]])
			coordinates.append([coords_zero[0]+1, coords_zero[1]-1])
			coordinates.append([coords_zero[0], coords_zero[1]-1])
		elif current_direction == "right":
			directions = ["top-right", "bottom-right", "bottom-left", "top-left"]
			coordinates.append([coords_zero[0]+0.7, coords_zero[1]+0.7])
			coordinates.append([coords_zero[0]+1.4, coords_zero[1]])
			coordinates.append([coords_zero[0]+0.7, coords_zero[1]-0.7])
		elif current_direction == "top-right":
			directions = ["up", "right", "down", "left"]
			coordinates.append([coords_zero[0], coords_zero[1]+1])
			coordinates.append([coords_zero[0]+1, coords_zero[1]+1])
			coordinates.append([coords_zero[0]+1, coords_zero[1]])
		return coordinates, directions

	#method that returns directions and coordinates for a five-atom cycle
	@staticmethod
	def layoutPentagon(G, current_direction, coords_zero):
		directions = list()
		coordinates = list()
		if current_direction == "none":
			directions = ["top-left", "up", "right", "down", "bottom-left"]
			coordinates.append([coords_zero[0]-0.81, coords_zero[1]+0.59])
			coordinates.append([coords_zero[0]-0.5, coords_zero[1]+1.54])
			coordinates.append([coords_zero[0]+0.5, coords_zero[1]+1.54])
			coordinates.append([coords_zero[0]+0.81, coords_zero[1]+0.59])
		elif current_direction == "up":
			directions = ["top-left", "up", "right", "down", "bottom-left"]
			coordinates.append([coords_zero[0]-0.81, coords_zero[1]+0.59])
			coordinates.append([coords_zero[0]-0.5, coords_zero[1]+1.54])
			coordinates.append([coords_zero[0]+0.5, coords_zero[1]+1.54])
			coordinates.append([coords_zero[0]+0.81, coords_zero[1]+0.59])			
		elif current_direction == "top-left":
			directions = ["left", "top-left", "top-right", "bottom-right", "down"]
			coordinates.append([coords_zero[0]-0.99, coords_zero[1]-0.16])
			coordinates.append([coords_zero[0]-1.44, coords_zero[1]+0.73])
			coordinates.append([coords_zero[0]-0.73, coords_zero[1]+1.44])
			coordinates.append([coords_zero[0]+0.16, coords_zero[1]+0.99])			
		elif current_direction == "left":
			directions = ["bottom-left", "left", "up", "right", "bottom-right"]
			coordinates.append([coords_zero[0]-0.59, coords_zero[1]-0.81])
			coordinates.append([coords_zero[0]-1.54, coords_zero[1]-0.5])
			coordinates.append([coords_zero[0]-1.54, coords_zero[1]+0.5])
			coordinates.append([coords_zero[0]-0.59, coords_zero[1]+0.81])			
		elif current_direction == "bottom-left":
			directions = ["down", "bottom-left", "top-left", "top-right", "right"]
			coordinates.append([coords_zero[0]+0.16, coords_zero[1]-0.99])
			coordinates.append([coords_zero[0]-0.73, coords_zero[1]-1.44])
			coordinates.append([coords_zero[0]-1.44, coords_zero[1]-0.73])
			coordinates.append([coords_zero[0]-0.99, coords_zero[1]+0.16])		
		elif current_direction == "down":
			directions = ["bottom-right", "down", "left", "up", "top-right"]
			coordinates.append([coords_zero[0]+0.81, coords_zero[1]-0.59])
			coordinates.append([coords_zero[0]+0.5, coords_zero[1]-1.54])
			coordinates.append([coords_zero[0]-0.5, coords_zero[1]-1.54])
			coordinates.append([coords_zero[0]-0.81, coords_zero[1]-0.59])		
		elif current_direction == "bottom-right":
			directions = ["right", "bottom-right", "bottom-left", "top-left", "up"]
			coordinates.append([coords_zero[0]+0.99, coords_zero[1]+0.16])
			coordinates.append([coords_zero[0]+1.44, coords_zero[1]-0.73])
			coordinates.append([coords_zero[0]+0.73, coords_zero[1]-1.44])
			coordinates.append([coords_zero[0]-0.16, coords_zero[1]-0.99])				
		elif current_direction == "right":
			directions = ["top-right", "right", "down", "left", "top-left"]
			coordinates.append([coords_zero[0]+0.59, coords_zero[1]+0.81])
			coordinates.append([coords_zero[0]+1.54, coords_zero[1]+0.5])
			coordinates.append([coords_zero[0]+1.54, coords_zero[1]-0.5])
			coordinates.append([coords_zero[0]+0.59, coords_zero[1]-0.81])					
		elif current_direction == "top-right":
			directions = ["up", "top-right", "bottom-right", "bottom-left", "left"]
			coordinates.append([coords_zero[0]-0.16, coords_zero[1]+0.99])
			coordinates.append([coords_zero[0]+0.73, coords_zero[1]+1.44])
			coordinates.append([coords_zero[0]+1.44, coords_zero[1]+0.73])
			coordinates.append([coords_zero[0]+0.99, coords_zero[1]-0.16])			
		return coordinates, directions
	
	#method that returns directions and coordinates for a six-atom cycle
	@staticmethod
	def layoutHexagon(G, current_direction, coords_zero):
		directions = list()
		coordinates = list()
		if current_direction == "none":
			directions = ["top-left", "up", "top-right", "bottom-right", "down", "bottom-left"]
			coordinates.append([coords_zero[0]-0.87, coords_zero[1]+0.5])
			coordinates.append([coords_zero[0]-0.87, coords_zero[1]+1.5])
			coordinates.append([coords_zero[0], coords_zero[1]+2.0])
			coordinates.append([coords_zero[0]+0.87, coords_zero[1]+1.5])
			coordinates.append([coords_zero[0]+0.87, coords_zero[1]+0.5])
		elif current_direction == "up":
			directions = ["top-left", "up", "top-right", "bottom-right", "down", "bottom-left"]
			coordinates.append([coords_zero[0]-0.87, coords_zero[1]+0.5])
			coordinates.append([coords_zero[0]-0.87, coords_zero[1]+1.5])
			coordinates.append([coords_zero[0], coords_zero[1]+2.0])
			coordinates.append([coords_zero[0]+0.87, coords_zero[1]+1.5])
			coordinates.append([coords_zero[0]+0.87, coords_zero[1]+0.5])	
		elif current_direction == "top-left":
			directions = ["left", "top-left", "up", "right", "bottom-right", "down"]
			coordinates.append([coords_zero[0]-0.97, coords_zero[1]-0.26])
			coordinates.append([coords_zero[0]-1.67, coords_zero[1]+0.45])
			coordinates.append([coords_zero[0]-1.41, coords_zero[1]+1.41])
			coordinates.append([coords_zero[0]-0.45, coords_zero[1]+1.67])
			coordinates.append([coords_zero[0]+0.26, coords_zero[1]+0.97])	
		elif current_direction == "left":
			directions = ["bottom-left", "left", "top-left", "top-right", "right", "bottom-right"]
			coordinates.append([coords_zero[0]-0.5, coords_zero[1]-0.87])
			coordinates.append([coords_zero[0]-1.5, coords_zero[1]-0.87])
			coordinates.append([coords_zero[0]-2.0, coords_zero[1]])
			coordinates.append([coords_zero[0]-1.5, coords_zero[1]+0.87])
			coordinates.append([coords_zero[0]-0.5, coords_zero[1]+0.87])
		elif current_direction == "bottom-left":
			directions = ["down", "bottom-left", "left", "up", "top-right", "right"]
			coordinates.append([coords_zero[0]+0.26, coords_zero[1]-0.97])			
			coordinates.append([coords_zero[0]-0.45, coords_zero[1]-1.67])
			coordinates.append([coords_zero[0]-1.41, coords_zero[1]-1.41])
			coordinates.append([coords_zero[0]-1.67, coords_zero[1]-0.45])
			coordinates.append([coords_zero[0]-0.97, coords_zero[1]+0.26])
		elif current_direction == "down":
			directions = ["bottom-right", "down", "bottom-left", "top-left", "up", "top-right"]
			coordinates.append([coords_zero[0]+0.87, coords_zero[1]-0.5])
			coordinates.append([coords_zero[0]+0.87, coords_zero[1]-1.5])
			coordinates.append([coords_zero[0], coords_zero[1]-2.0])
			coordinates.append([coords_zero[0]-0.87, coords_zero[1]-1.5])
			coordinates.append([coords_zero[0]-0.87, coords_zero[1]-0.5])
		elif current_direction == "bottom-right":
			directions = ["right", "bottom-right", "down", "left", "top-left", "up"]
			coordinates.append([coords_zero[0]+0.97, coords_zero[1]+0.26])
			coordinates.append([coords_zero[0]+1.67, coords_zero[1]-0.45])
			coordinates.append([coords_zero[0]+1.41, coords_zero[1]-1.41])
			coordinates.append([coords_zero[0]+0.45, coords_zero[1]-1.67])
			coordinates.append([coords_zero[0]-0.26, coords_zero[1]-0.97])
		elif current_direction == "right":
			directions = ["top-right", "right", "bottom-right", "bottom-left", "left", "top-left"]
			coordinates.append([coords_zero[0]+0.5, coords_zero[1]+0.87])
			coordinates.append([coords_zero[0]+1.5, coords_zero[1]+0.87])
			coordinates.append([coords_zero[0]+2.0, coords_zero[1]])
			coordinates.append([coords_zero[0]+1.5, coords_zero[1]-0.87])
			coordinates.append([coords_zero[0]+0.5, coords_zero[1]-0.87])		
		elif current_direction == "top-right":
			directions = ["up", "top-right", "right", "down", "bottom-left", "left"]
			coordinates.append([coords_zero[0]-0.26, coords_zero[1]+0.97])			
			coordinates.append([coords_zero[0]+0.45, coords_zero[1]+1.67])
			coordinates.append([coords_zero[0]+1.41, coords_zero[1]+1.41])
			coordinates.append([coords_zero[0]+1.67, coords_zero[1]+0.45])
			coordinates.append([coords_zero[0]+0.97, coords_zero[1]-0.26])
		return coordinates, directions

