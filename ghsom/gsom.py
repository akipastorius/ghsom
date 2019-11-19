""" Growing Self-Organizing Map implementation. The basic SOM
implementation is based on minisom package implementation with slight
modifications for speed."""
from .ghsom_utils import (exponential_decay, asymptotic_decay, qe_sum_1d, qe_sum_1d_nan, qe_sum_2d,
                   _incremental_index_verbose)
from numpy import (array, unravel_index, nditer, linalg, random,
                   power, exp, pi, zeros, arange, outer, meshgrid,
                   logical_and, mean, cov, argsort, linspace, transpose,
                   einsum, vstack, nanmin, nanmax, isnan, nan_to_num)
from collections import defaultdict
from warnings import warn
import matplotlib.pyplot as plt

class gsom(object):
    def __init__(self,
                 input_len: int,
                 x: int=2,
                 y: int=2,
                 learning_rate: float=1.0,
                 sigma: float=1.0,
                 neighborhood_function: str='gaussian',
                 decay_function=exponential_decay,
                 random_seed: int=None):
        """Initializes a Self Organizing Maps.

        Parameters
        ----------
        x : int
            x dimension of the SOM.

        y : int
            y dimension of the SOM.

        input_len : int
            Number of the elements of the vectors in input.

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

            A custom decay function will need to to take in input
            three parameters in the following order:

            1. learning rate
            2. current iteration
            3. maximum number of iterations allowed


        neighborhood_function : function, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map
            possible values: 'gaussian', 'mexican_hat', 'bubble'

        random_seed : int, optional (default=None)
            Random seed to use.
        """
        self._distance_function=qe_sum_1d
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        self._x = x
        self._y = y
        self.status = 'init'
        self.name = None
        self._weights = random.rand(x, y, input_len)*2-1
        for i in range(x):
            for j in range(y):
                # normalization
                norm = qe_sum_1d(self._weights[i, j])
                self._weights[i, j] = self._weights[i, j] / norm

        self._activation_map = zeros((x, y))
        self._neigx = arange(x) # used to evaluate the neighborhood function
        self._neigy = arange(y)  # used to evaluate the neighborhood function
        self._decay_function = decay_function
        self._lamda_count = 1
        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in \
        ['triangle', 'bubble'] and divmod(self._sigma, 1)[1] != 0:
            warn('sigma should be an integer when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]

    def get_weights(self):
        """Returns the weights of the neural network."""
        return self._weights

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x.

        """
        d = nan_to_num(x - self._weights)
        self._activation_map = linalg.norm(d, axis = 2)

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self._activation_map

    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*pi*sigma*sigma
        ax = exp(-power(self._neigx-c[0], 2)/d)
        ay = exp(-power(self._neigy-c[1], 2)/d)
        return outer(ax, ay)  # the external product gives a matrix

    def _mexican_hat(self, c, sigma):
        """Mexican hat centered in c."""
        xx, yy = meshgrid(self._neigx, self._neigy)
        p = power(xx-c[0], 2) + power(yy-c[1], 2)
        d = 2*pi*sigma*sigma
        return exp(-p/d)*(1-2/d*p)

    def _bubble(self, c, sigma):
        """Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        ax = logical_and(self._neigx > c[0]-sigma,
                         self._neigx < c[0]+sigma)
        ay = logical_and(self._neigy > c[1]-sigma,
                         self._neigy < c[1]+sigma)
        return outer(ax, ay)*1.

    def _triangle(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        triangle_x = (-abs(c[0] - self._neigx)) + sigma
        triangle_y = (-abs(c[1] - self._neigy)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return outer(triangle_x, triangle_y)

    def _check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError('num_iteration must be > 1')

    def _check_nan_values(self, data):
        if isnan(data).any():
            raise ValueError(
                    'nans in the data array and data_has_nan is False')

    def _check_input_len(self, data):
        """Checks that the data in input is of the correct shape."""
        data_len = len(data[0])
        if self._input_len != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self._input_len)
            raise ValueError(msg)

    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)

    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.

        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig)*eta
        # w_new = eta * neighborhood_function * (x-w)
        d = nan_to_num(x-self._weights)
        self._weights += einsum('ij, ijk->ijk', g, d)

    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        self._check_input_len(data)
        q = zeros(data.shape)
        for i, x in enumerate(data):
            q[i] = self._weights[self.winner(x)]
        return q

    def random_weights_init(self, data):
        """Initializes the weights of the SOM
        picking random samples from data."""
        self._check_input_len(data)
        it = nditer(self._activation_map, flags=['multi_index'])
        u_i = list(range(len(data)))
        while not it.finished:
            if len(u_i) == 0:
                u_i = list(range(len(data)))
            rand_i = random.choice(u_i)
            u_i.remove(rand_i)
            self._weights[it.multi_index] = data[rand_i]
            it.iternext()


    def pca_weights_init(self, data):
        """Initializes the weights to span the first two principal components.

        This initialization doesn't depend on random processes and
        makes the training process converge faster.

        It is strongly reccomended to normalize the data before initializing
        the weights and use the same normalization for the training data.
        """
        if self._input_len == 1:
            msg = 'The data needs at least 2 features for pca initialization'
            raise ValueError(msg)
        self._check_input_len(data)
        if len(self._neigx) == 1 or len(self._neigy) == 1:
            msg = 'PCA initialization inappropriate:' + \
                  'One of the dimensions of the map is 1.'
            warn(msg)
        pc_length, pc = linalg.eig(cov(transpose(data)))
        pc_order = argsort(-pc_length)
        for i, c1 in enumerate(linspace(-1, 1, len(self._neigx))):
            for j, c2 in enumerate(linspace(-1, 1, len(self._neigy))):
                self._weights[i, j] = c1*pc[pc_order[0]] + c2*pc[pc_order[1]]

    def train(self,
              data: array,
              qe_0: float,
              qe_u: float,
              num_iteration: int=1,
              lamda: int=1,
              tau_0: float=0.01,
              tau_u: float=0.1,
              verbose: bool=False,
              data_has_nan: bool=False,
              train_type: str='sequence'):

        """Trains using all the vectors in data sequentially or randomly.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_iterations : int
            Maximum number of iterations (one iteration per sample).

        lamda : int
            During training, after lamda iterations, map is grown by one unit

        qe0 : double
            global error, used in global stopping criterion

        qeu : double
            error of the upper unit, used in local stopping criterion

        tau1 : double
            parameter for local stopping criteria

        tau2 : double
            parameter for global stopping criteria

        verbose : bool (default=False)
            If True the status of the training
            will be printed at each iteration.

        train_type: str
            train data either randomly or sequentially

        """

        if train_type not in ['random', 'sequence']:
            train_type  = 'random'

        if data_has_nan:
            self._distance_function=qe_sum_1d_nan
        else:
            self._check_nan_values(data)
            self._distance_function=qe_sum_1d

        self._check_iteration_number(num_iteration)
        self._check_input_len(data)


        iterations = range(num_iteration)
        if verbose:
            iterations = _incremental_index_verbose(num_iteration)

        for iteration in iterations:
            if train_type == 'random':
                idx = random.randint(len(data))
            elif train_type == 'sequence':
                idx = iteration % len(data)

            if self._lamda_count == lamda:
                mqe = self.get_mqe(data)
                qe = array(list(self.get_qe_map(data).values()))
                if (qe < qe_0 * tau_0).all():
                    self.status = 'trained'
                    if verbose:
                        print(' - sufficient resolution achieved for {0}'
                          .format(self.name))
                    return

                elif (mqe < qe_u * tau_u):
                    self.status = 'add_level'
                    if verbose:
                        print(' - new child map required for {0}'.format(self.name))
                    return

                elif (self._x * self._y) <= data.shape[0]:
                    if iteration < num_iteration - 1:
                        self.grow_map(data)
                        self._lamda_count = 0
                    else:
                        pass

            self.update(data[idx], self.winner(data[idx]),
                        self._lamda_count, lamda)

            self._lamda_count += 1

        if verbose:
            print(' - max iterations reached for {0}'.format(self.name))

        self.status = 'max_iterations'
        return


    def distance_map(self):
        """Returns the distance map of the weights.
        Each cell is the normalised sum of the distances between
        a neuron and its neighbours."""
        um = zeros((self._weights.shape[0], self._weights.shape[1]))
        it = nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                    if (ii >= 0 and ii < self._weights.shape[0] and
                            jj >= 0 and jj < self._weights.shape[1]):
                        w_1 = self._weights[ii, jj, :]
                        w_2 = self._weights[it.multi_index]
                        um[it.multi_index] += self._distance_function(w_1-w_2)
            it.iternext()
        um = um/um.max()
        return um

    def activation_response(self, data):
        """Returns a matrix where the element i,j is the number of times
        that the neuron i,j have been winner."""
        self._check_input_len(data)
        a = zeros((self._weights.shape[0], self._weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        self._check_input_len(data)
        error = 0
        for x in data:
            error += self._distance_function(x-self._weights[self.winner(x)])
        return error/len(data)

    def win_map(self, data):
        """Returns a dictionary wm where wm[(i,j)] is a list
        with all the patterns that have been mapped in the position i,j."""
        self._check_input_len(data)
        winmap = defaultdict(list)
        for x in data:
            winmap[self.winner(x)].append(x)
        return winmap

    def labels_map(self, data, labels):
        """Returns a dictionary wm where wm[(i,j)] is a dictionary
        that contains the number of samples from a given label
        that have been mapped in position i,j.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        label : np.array or list
            Labels for each sample in data.
        """
        self._check_input_len(data)
        if not len(data) == len(labels):
            raise ValueError('data and labels must have the same length.')
        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[self.winner(x)].append(l)
        return winmap

    def get_qe_map(self, data):
        """Returns the quantization error for each node in the map
        """
        winmap = self.win_map(data)
        w = self.get_weights()
        qe_map = {wm: qe_sum_2d(nan_to_num(winmap[wm] - w[wm])) for wm in winmap}
        return qe_map

    def get_mqe(self,data):
        """Returns the mean quantization error of the map calculated as
        mean over quantization errors of nodes in the map
        """
        qe_map = self.get_qe_map(data)
        mqe = mean(list(qe_map.values()))
        return mqe

    def get_error_unit(self, data):
        """Returns the node with highest quantization error"""
        qemap = self.get_qe_map(data)
        e = max(qemap.keys(), key=(lambda key: qemap[key]))
        return e

    def get_dissimilar_neighbor(self, node):
        """Returns the most dissimilar neighbor for given node"""
        x = node[0]
        y = node[1]
        w = self.get_weights()
        X = self._x
        Y = self._y
        neighbors = [(x2, y2) for x2 in range(x-1, x+2)
                                for y2 in range(y-1, y+2)
                                if (-1 < x < X and
                                    -1 < y < Y and
                                    (x != x2 or y != y2) and
                                    (0 <= x2 < X) and
                                    (0 <= y2 < Y) and
                                    (x == x2 or y == y2))]
        if len(neighbors) == 0:
            d = (node[0], node[1]+1)
            return d

        if len(neighbors) == 1:
            d = (node[0]+1, node[1])
            return d

        error = 0
        for neighbor in neighbors:
            dist = self._distance_function(w[node] - w[neighbor])
            if dist > error:
                d = neighbor
                error = dist

        return d

    def insert_units(self, e, d):
        """Inserts a row or column between the error unit and its most
        dissimilar neighbor"""
        w = self.get_weights()
        if e[0] == d[0]:
            dim = 'col'
            idx = max(e[1], d[1])
            self._y += 1
            self._neigy = arange(self._y)
        elif e[1] == d[1]:
            dim = 'row'
            idx = max(e[0], d[0])
            self._x += 1
            self._neigx = arange(self._x)

        self._activation_map = zeros((self._x, self._y))
        nw = zeros((self._x, self._y, w.shape[2]))
        if dim == 'row':
            nw[range(0,idx),:,:] = w[range(0,idx),:,:]
            nw[range(idx+1,nw.shape[0]),:,:] = w[range(idx, w.shape[0]),:,:]
            nw[idx,:,:] = (nw[idx-1,:,:] + nw[idx+1,:,:])/2

        if dim == 'col':
            nw[:,range(0,idx),:] = w[:,range(0,idx),:]
            nw[:,range(idx+1,nw.shape[1]),:] = w[:,range(idx, w.shape[1]),:]
            nw[:,idx,:] = (nw[:,idx-1,:] + nw[:,idx+1,:])/2

        self._weights = nw

    def grow_map(self, data):
        """Calculates the error unit e, i.e. map node with highest quantization
        error, and adds row / column of nodes between error unit and its most
        dissimilar neighbor d, and adjustes sigma"""
        e = self.get_error_unit(data)
        d = self.get_dissimilar_neighbor(e)
        self.insert_units(e,d)

    def plot_map(self, data, parent_map='', parent_node='',
                             output_path=None):
        """Plots the heatmap of the som and corresponding datapoints"""
        dmap = self.distance_map()
        plt.figure(figsize=(self._x*4,self._y*4))
        plt.rcParams.update({'font.size': 12})
        plt.title('{0}, parent map {1}, parent node {2}'
                  .format(self.name, parent_map, parent_node))
        plt.pcolor(dmap.T, cmap='Blues', vmin=0, vmax=2)
        wmap = self.win_map(data)
        for wm in wmap:
            d = vstack(wmap[wm])
            t = linspace(wm[0], wm[0]+1,d.shape[1])
            min_d = nanmin(d, axis = 1)[:, None]
            max_d = nanmax(d, axis = 1)[:, None]
            d_s = (d - min_d) / (max_d - min_d) + wm[1]
            plt.plot(t, d_s.transpose(), color = [0.5,0.5,0.5,0.5], lw = 0.5)

        if output_path != None:
            plt.savefig('{0}/heatmap_{1}.png'.format(output_path, self.name))
            plt.close()



    def plot_node(self, data, parent_map='', parent_node='',
                              output_path=None):
        """Plots each node of the map as separate figure"""
        wmap = self.win_map(data)
        for i, wm in zip(range(len(wmap)),wmap):
            d = vstack(wmap[wm])
            plt.figure(figsize=(10,10))
            plt.rcParams.update({'font.size': 12})
            plt.title('{0}, node {1}, n = {2}, parent map {3}, parent node {4}'
                      .format(self.name, wm, len(d), parent_map, parent_node))
            plt.plot(d.transpose(), color = (0.5, 0.5, 0.5), lw = 0.5)
            if output_path != None:
                plt.savefig('{0}/time_series_{1}_node_{2}.png'.format(
                        output_path, self.name, i))
                plt.close()



