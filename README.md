Growing hierarchical self-organizing map
========================================

Growing hierarchical self-organizing map implementation for unsupervised learning.

GHSOM creates data driven hierarchical mapping of the input data which preserves topological properties of the input data. The basic self-organizing map (som)  implementation is based on python minisom package with slight modifications to speed and nan-handling. 

In practice som is a one-layer neural network algorithm that utilizes competitive learning. This means in practice that the map is initialized as AxB matrix where all nodes are 'compete' of data points during the training process. Original som has two major disadvantages: how to choose map size (A and B) and inability to model hierarchical structures often present in the data. 

GHSOM tries to overcomes these challenges. The map size problem is addressed with the letter G (=Growing), which basically means that instead of running the training algorithm for n number of iterations, the training algorithm is run lamda iterations at a time and after each lamda iterations, stopping criterion (global stopping criterion) is checked. If stopping criteria is not satisfied, new row or column of map nodes is added between the node with highest error and its most dissimilar node. This process is continued until stopping criterion is  met or maximum iterations is reached. Letter H (=Hierarchical) addresses the  hierarchy issue by adding second stopping criterion (local stopping criterion) to be checked after lamda iterations. If local stopping criterion is satisfied, new child map is created for nodes not satisfying global stopping criterion. Thus by altering stopping local stopping criterion the user can create desired map hierarchy strucuture, either with deep hierarchy of small maps or shallow  hierarchy of large maps, or something in between. 


USAGE
=====

Clone the repository and run `pip install .` on the folder where `setup.py`is  located. Then you can import the module normally `import ghsom.ghsom as ghsom`.

## Running GHSOM

GHSOM object needs following attributes set:
* input_len: `int`: Number of input features

Optionnally, following attributes can be set
* learning_rate: `float`: Learning rate parameter which controlls the convergence of the training algorithm
* sigma: `float`: Controls the size of the neighborhood in the weight update during training process
* decay_function: `function`: Decay function decreases learnning rate and sigma during training process

The training algorithm needs following parameters specified:
* data: `numpy array`: Data matrix, features on columns, 
* lamda: `int`: How many iterations before decision made of growing the map / adding new layer of maps. Good idea is to have the lamda such that each vector in the input data is processed at least once. 
* num_iterations: `int`: How many iterations in total for each map. Good idea is to set this is lamda multiplied with some value
* tau_0: `float`: How much of the total variation in the data needs to be explained by each node -> determines the total size of maps and nodes. Default value is 0.01
* tau_u: `float`: How much of the variation in the upper node needs to explained by each map on average -> controls the deepness of the hierarchy. Default value is 0.2
* map_min_n: `int`: How many data points need to be at least in order to create a child map
* verbose: `boolean`: Print the progress of training process to stdout
* labels: `list`: List of lables to store to ghsom object. Labels should be in same order as data array
* data_has_nan: `boolean`: does data contain null values. If no null values are present, faster distance function can be used during training process.


After the GHSOM has been trained, it's tree hierarchy can be plotted with method `plot_tree()` while individual maps can be plotted with `plot_map()` and individual nodes with `plot_node()`. Method `create_hierarchy_table` can be used to create a RELEX-style csv for each label and it's clusters for each hierarchy level. This can be read in to RELEX via messages tab. Method `predict()` can be used to predict the hierarchy table for any new data. 

Trained GHSOM object can be saved with function `save_model`and saved model can be loaded with function `load_model` from ghsom_utils module.

## Running GSOM

Module ghsom.gsom contains gsom, which can be used to cluster input data without hierarchical structure. Same attributes can be set to GSOM object than GHSOM object.

The training algorithm for GSOM needs following parameters specified:
* data: `numpy array`: Data matrix, features on columns, 
* lamda: `int`: How many iterations before decision made of growing the map / adding new layer of maps. Good idea is to have the lamda such that each vector in the input data is processed at least once. 
* num_iterations: `int`: How many iterations in total for each map. Good idea is to set this is lamda multiplied with some value
* tau_0: `float`: How much of the total variation in the data needs to be explained by each node -> determines the total size of maps and nodes. Default value is 0.01
* tau_u: `float`: How much of the variation in the upper node needs to explained by each map on average -> controls the deepness of the hierarchy. Default value is 0.2
* verbose: `boolean`: Print the progress of training process to stdout
* data_has_nan: `boolean`: does data contain null values. If no null values are present, faster distance function can be used during training process.
