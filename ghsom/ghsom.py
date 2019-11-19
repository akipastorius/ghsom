""" Growing hierarchical Self-Organizing Map implementation. The basic SOM
implementation is based on minisom package implementation with slight
modifications for speed."""
from .ghsom_utils import asymptotic_decay, exponential_decay
from .gsom import gsom
from numpy import (array, reshape, nan, nanmean, vstack, isnan)
import pandas as pd
from anytree import Node, RenderTree

class ghsom(object):
    """ Initializes Growing Hierarchical Self-Organizing Map.
    Parameters
    ----------
    input_len : int
        number of features in the data

    learning rate: float
        Initial learning rate, decreased during the
        training with decay_function. Controlls the convergence of the
        training process

    sigma: float
        Initial sigma, decreased during the training with decay_function.
        Controls the size of neighborhood in weight update

    decay_function : function
        Function that reduces learning_rate and sigma at each iteration
        the default function is exponential decay with distribution
        parameter 0.05

        For growing som, the max_iterations is actually lamda parameter,
        after lamda iterations, learning rate and sigma parameters are
        resetted.


    random_seed : int, optional (default=None)
        Random seed to use.

    """

    def __init__(self,
                 input_len: int,
                 learning_rate: float=1.0,
                 sigma: float=1.0,
                 decay_function=exponential_decay,
                 neighborhood_function: str='gaussian',
                 random_seed: int=None):

        self._input_len = input_len
        self._decay_function = decay_function
        self.neighborhood_function = neighborhood_function
        self._learning_rate = learning_rate
        self._sigma = sigma
        root = gsom(x=1,
                    y=1,
                    input_len=input_len,
                    learning_rate = self._learning_rate
                    )
        root.name = 'root'
        self.tree = []
        self.tree.append(Node('root', parent = None,
                                      **{'som': root,
                                         'parent_node': None}))


    def _check_nan_values(self, data):
        if isnan(data).any():
            raise ValueError(
                    'nan in the data array and data_has_nan is False')


    def train(self,
              data: array,
              num_iteration: int=1,
              lamda: int=1,
              tau_0: float=0.01,
              tau_u: float=0.1,
              map_min_n: int=1,
              verbose: bool=False,
              labels: list=None,
              data_has_nan: bool=False,
              train_type: str='sequence'):

        """Trains using all the vectors in data sequentially.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_iterations : int
            Maximum number of iterations (one iteration per sample).

        lamda : int
            During training, after lamda iterations, map is grown by one unit

        tau_0 : double
            Parameter for global stopping criteria

        tau_u : double
            Parameter for local stopping criteria

        map_size_min : int
            Minimum number of datapoints that are required in order to create
            child map

        verbose : bool (default=False)
            If True the status of the training
            will be printed at each iteration.

        train_type: str
            train data either randomly or sequentially

        """

        """Set root map weights to mean of the data. Calculate qe_0 and create
        first child map to the tree"""
        if data_has_nan:
            pass
        else:
            self._check_nan_values(data)

        if labels==None:
            labels = [nan] * data.shape[0]

        self.tree[0].som._weights = reshape(
                nanmean(data, axis = 0),
                newshape = self.tree[0].som._weights.shape
                )

        qe_0 = list(self.tree[0].som.get_qe_map(data).values())[0]

        self.tree[0].labels = labels
        labmap = self.tree[0].som.labels_map(data, labels)
        self.add_map(parent=self.tree[0],
                     parent_node=(0,0),
                     data=data,
                     qe_u=qe_0,
                     labels = labmap[(0,0)])

        node = self.get_next_untrained_node()
        """Loop the tree until all leaf maps are trained"""
        while node != []:
            node.som.train(data=node.data,
                           num_iteration=num_iteration,
                           lamda=lamda,
                           qe_0=qe_0,
                           qe_u=node.qe_u,
                           tau_0=tau_0,
                           tau_u=tau_u,
                           data_has_nan = data_has_nan,
                           verbose=verbose,
                           train_type = train_type)

            if node.som.status == 'add_level':
                """Add new child map for all nodes in the current map for which
                the quantization error is greater than qe_0 * tau_0, and
                which satisfy the map_min_n criterion. Global orientation
                of child maps is missing entirely from this implementation,
                but in our case I think its not necessary.
                """
                qe_map = node.som.get_qe_map(node.data)
                map_nodes = {qm: qm for qm in qe_map if
                             qe_map[qm] > qe_0 * tau_0}.keys()

                winmap = node.som.win_map(node.data)
                labmap = node.som.labels_map(node.data, node.labels)
                for mn in map_nodes:
                    new_data = vstack(winmap[mn])
                    par_map = [i for i in self.tree if
                               i.som.name == node.som.name][0]
                    if new_data.shape[0] <= map_min_n:
                        continue

                    self.add_map(parent=par_map, parent_node=mn,
                                      data=new_data,
                                      qe_u=qe_map[mn],
                                      labels=labmap[mn])

            node = self.get_next_untrained_node()

    def add_map(self, parent, parent_node, data, qe_u, labels=None):
        """Initializes a new gsom object and adds it along with necessary data
        (data array, qe_u parameter, labels) to tree as child node to given
        parent node"""
        som = gsom(x=2,
                   y=2,
                   input_len=self._input_len,
                   learning_rate=self._learning_rate,
                   sigma = self._sigma,
                   decay_function=self._decay_function
                   )
        som.random_weights_init(data)
        som.name = '{0}'.format(len(self.tree))
        self.tree.append(Node(som.name,
                              parent = parent,
                              **{'som': som,
                                 'parent_node': parent_node,
                                 'data': data,
                                 'qe_u': qe_u,
                                 'labels': labels}))

    def get_next_untrained_node(self):
        """Returns next untrained map in the tree
        """
        node = [i for i in self.tree if i.som.status == 'init'
           and i.name != 'root']
        if node == []:
            return node
        else:
            return node[0]

    def plot_tree(self):
        """Plots tree structure
        """
        for pre, fill, node in RenderTree(self.tree[0]):
            print("%s%s" % (pre, node.name))

    def plot_map(self,
                 data: array,
                 output_path: str=None):
        """Plots heatmaps for all maps in the tree
        """
        for row in RenderTree(self.tree[0]):
            if row.node.name == 'root':
                continue

            row.node.som.plot_map(
                    row.node.data,
                    output_path=output_path,
                    parent_map=row.node.parent.name,
                    parent_node=row.node.parent_node)

    def plot_node(self,
                  data: array,
                  output_path: str=None):
        """Plots time-series figures for all nodes for all maps in the tree
        """
        for row in RenderTree(self.tree[0]):
            if row.node.name == 'root':
                continue
            row.node.som.plot_node(
                    row.node.data,
                    output_path=output_path,
                    parent_map=row.node.parent.name,
                    parent_node=row.node.parent_node)

    def create_hierarchy_table(self,
                               output_path: str=None):
        """Creates hierarchy table in DataFrame / csv format from data used in
        the training process by looping each tree layer, maps on each layer and
        nodes on each map"""
        tree_depth = self.tree[0].height
        list1 = ['cluster_level_'] * tree_depth
        list2 = list(range(tree_depth))
        columns = [m+str(n) for m,n in zip(list1,list2)]
        labels = self.tree[0].labels
        df = pd.DataFrame(index = labels, columns = columns)

        for i in range(tree_depth):
            nodes = [m for m in self.tree if m.depth == i+1]
            for j, n in zip(range(len(nodes)),nodes):
                som_i = n.som
                data_i = n.data
                labels_i = n.labels
                lab_map = som_i.labels_map(data_i, labels_i)
                for lm in lab_map:
                    code = '{0}_{1}_{2}_{3}'.format(i, som_i.name, lm[0], lm[1])
                    keys = lab_map[lm]
                    df.loc[keys,df.columns[i]] = code

            if i > 0:
                idx = df.loc[:,columns[i]].isnull()
                df.loc[idx,columns[i]] = df.loc[idx,columns[i-1]]

        if output_path != None:
            df.to_csv(path_or_buf=output_path + 'results.csv',
                      sep=';',
                      index=True,
                      index_label='label')
            return
        return df

    def predict(self, data: array, labels: list, output_path: str=None):
        """Predicts hierarchy table for any data by finding the winner map and
        node for each data point on each tree layer"""
        tree_depth = self.tree[0].height
        list1 = ['cluster_level_'] * tree_depth
        list2 = list(range(tree_depth))
        columns = [m+str(n) for m,n in zip(list1,list2)]
        df = pd.DataFrame(index = labels, columns = columns)

        for i, x in zip(labels,data):
            parent_node = (0,0)
            parent_name = 'root'
            for row in RenderTree(self.tree[1]):
                if ((row.node.parent_node == parent_node) and
                (row.node.parent.name == parent_name)):
                    wn = row.node.som.winner(x)
                    depth = row.node.depth-1
                    name = row.node.som.name
                    code = '{0}_{1}_{2}_{3}'.format(depth, name, wn[0], wn[1])
                    df.loc[i,columns[depth]] = code
                    parent_node = wn
                    parent_name = name

        for i in range(tree_depth):
            if i > 0:
                idx = df.loc[:,columns[i]].isnull()
                df.loc[idx,columns[i]] = df.loc[idx,columns[i-1]]

        if output_path != None:
                df.to_csv(path_or_buf=output_path + 'results.csv',
                          sep=';',
                          index=True,
                          index_label='label')
                return

        return df
