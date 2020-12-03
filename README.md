# MG2N2
This repository contains the code for training and using the Molecule Generative Graph Neural Network, a neural network model for the generation of small molecular graphs.
The model is composed of three Graph Neural Network modules and generates the molecular graphs with a sequential algorithm.
A First-In-First-Out node expansion queue keeps track of the nodes to be expanded. 
The node generator module decides if to generate a new neighbor for the node currently under focus and, in case, its type.
The edge classifier decides the type of edge connecting the focused node to its new neighbor.
The linker module decides if any other edge should be added between the new node and the rest of the graph.

# cite
If you make use of this code for your publication, please cite:
[PAPER REFERENCE GOES HERE]

# requirements
To run the code, you will need a python>2.6 distribution with the following python packages:
Tensorflow
Numpy
Scipy
Matplotlib
Networkx

# installation
No installation procedure is required, this repository is ready-to-use.

# data setup
QM9 data is provided in a "raw" format, to save space and downloading time. The script "translate_dataset.py" will process the dataset into the right format.

# utility scripts
The "graph_decomposition.py" and "molecule_drawer.py" python files contain utilities for the other scripts.

# network training
The script "train_generator.py" trains the generator module.
The script "train_bond_classifier.py" trains the edge classifier module.
The script "train_linker.py" trains the extra edge generator module.
The hyparparameters of each module are declared in the first part of the corresponding script. The trained modules are saved into a "Temp/Modules" folder.

# graph generation
The script "generate_graphs.py" generates a batch of graphs, exploiting the three modules which had been previously saved in the "Temp/Modules" folder.
The hyperparameters of the modules are declared at the beginning of the script.


