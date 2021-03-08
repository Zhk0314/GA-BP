import random
import numpy as np
import pandas as pd
import copy
import time
from sklearn.preprocessing import OneHotEncoder


def compute_cost(A2, Y):
    # 需要重新看一下你的公式哪里有错误，这个计算不出来向量相乘是什么，正交嘛？
    """
    compute cost
    (计算成本函数)
    :param A2: The sigmoid output of the second activation, of shape (1, number of examples) (第2层激活函数sigmoid函数输出向量)
    :param Y: "true" labels vector of shape (1, number of examples) (正确标签向量)
    :return:
    cost: cross-entropy cost
    """
    m = Y.shape[0]  # number of example 这里应该是取y的第一维度的长度
    cost = - np.sum(np.multiply(np.log(A2), Y) + np.multiply(np.log(1. - A2), 1. - Y)) / m
    # cost = np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))/(-m)

    cost = np.squeeze(cost)  # squeeze()函数的功能是：从矩阵shape中，去掉维度为1的。例如一个矩阵是的shape是（5， 1），使用过这个函数后，结果为（5，）。

    assert (isinstance(cost, float))  # 若cost不是float型 则直接报异常

    return np.sum(A2.squeeze() - Y.squeeze()) / m


class Network(object):

    def __init__(self, sizes):

        '''The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.'''

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # helper variables
        self.bias_nitem = sum(sizes[1:])
        self.weight_nitem = sum([self.weights[i].size for i in range(self.num_layers - 2)])
        self.cost_ = []

    def feedforward(self, a):
        '''Return the output of the network if ``a`` is input.'''
        for index, _ in enumerate(self.biases):
            b, w = self.biases[index], self.weights[index]
            if index == 0:
                a = np.tanh(np.dot(w, a) + b)
            else:
                a = np.dot(w, a) + b
        a = self.sigmoid(a)
        return a

    def sigmoid(self, z):
        '''The sigmoid function.'''
        return 1.0 / (1.0 + np.exp(-z))

    def score(self, X, y):

        '''
        @X = data to test
        @y = data-label to test
        @returns = score of network prediction (less is better)
        @ref: https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
        '''

        total_score = 0
        for i in range(X.shape[0]):
            predicted = self.feedforward(X[i].reshape(-1, 1))
            actual = y[i].reshape(-1, 1)
            total_score += np.sum(np.power(predicted - actual, 2) / 2)  # mean-squared error
        return total_score

    def accuracy(self, X, y):

        '''
        @X = data to test
        @y = data-label to test
        @returns = accuracy (%) (more is better)
        '''
        accuracy = 0
        for i in range(X.shape[0]):
            output = self.feedforward(X[i].reshape(-1, 1))
            accuracy += int(np.argmax(output) == np.argmax(y[i]))
        return accuracy / X.shape[0] * 100

    def __str__(self):
        s = "\nBias:\n\n" + str(self.biases)
        s += "\nWeights:\n\n" + str(self.weights)
        s += "\n\n"
        return s

    def return_cost(self, X, y):
        """
        :param X: dev data
        :param y:
        :return: cost score
        """
        out_put_list = []
        for i in range(X.shape[0]):
            output = self.feedforward(X[i].reshape(-1, 1))
            out_put_list.append([np.argmax(output)])
        return compute_cost(np.array(out_put_list), np.array(y))

    def nn_model(self, X, y, learning_rate, dev_x, dev_y, iter=150000):
        """
        :param X: the train data features
        :param y: the train data labels
        :param learning_rate:
        :param dev_x: dev data x
        :param dev_y: dev data y
        :param iter: numbers of iteration
        :return:None
        """
        for i in range(iter):
            index = np.random.randint(X.shape[0] - 1)
            a = [X[index]]
            for index1, _ in enumerate(self.biases):
                b, w = self.biases[index1], self.weights[index1]
                if index1 == 0:
                    a.append(np.tanh(np.dot(w, a[index1])))
                else:
                    a.append(np.dot(w, a[index1]))
            if i % 1000 == 0:
                self.cost_.append(self.return_cost(dev_x, dev_y))  #
                # print(i)
            error = np.sum([r1 - r2 for r1, r2 in zip(y[index], self.sigmoid(a[-1]))])
            deltas = [[1 / 2 * error * rq for rq in self.sigmoid(a[-1])]]
            # 开始反向计算误差，更新权重
            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer
                dW2 = (1 / 2) * np.dot(self.weights[l].T, deltas[-1]) * a[l]
                deltas.append(dW2)
            deltas.reverse()
            for i1 in range(len(self.weights)):
                layer = np.atleast_2d(a[i1])
                delta = np.atleast_2d(deltas[i1])
                self.weights[i1] += learning_rate * delta.T.dot(layer)
            for i1 in range(len(self.biases)):
                rq = learning_rate * a[i1 + 1]
                for r1 in range(len(self.biases[i1])):
                    self.biases[i1][r1] += rq[r1]


class NNGeneticAlgo:

    def __init__(self, n_pops, net_size, mutation_rate, crossover_rate, retain_rate, X, y):

        '''
        n_pops   = How much population do our GA need to create
        net_size = Size of neural network for population members
        mutation_rate = probability of mutating all bias & weight inside our network
        crossover_rate = probability of cross-overing all bias & weight inside out network
        retain_rate = How many to retain our population for the best ones
        X = our data to test accuracy
        y = our data-label to test accuracy
        '''

        self.n_pops = n_pops
        self.net_size = net_size
        self.nets = [Network(self.net_size) for i in range(self.n_pops)]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.retain_rate = retain_rate
        self.X = X[:]
        self.y = y[:]

    def get_random_point(self, type):

        '''
        @type = either 'weight' or 'bias'
        @returns tuple (layer_index, point_index)
            note: if type is set to 'weight', point_index will return (row_index, col_index)
        '''
        nn = self.nets[0]
        layer_index, point_index = random.randint(0, nn.num_layers - 2), 0
        if type == 'weight':
            row = random.randint(0, nn.weights[layer_index].shape[0] - 1)
            col = random.randint(0, nn.weights[layer_index].shape[1] - 1)
            point_index = (row, col)
        elif type == 'bias':
            point_index = random.randint(0, nn.biases[layer_index].size - 1)
        return (layer_index, point_index)

    def get_all_scores(self):
        return [net.score(self.X, self.y) for net in self.nets]

    def get_all_accuracy(self):
        return [net.accuracy(self.X, self.y) for net in self.nets]

    def crossover(self, father, mother):
        '''
        @father = neural-net object representing father
        @mother = neural-net object representing mother
        @returns = new child based on father/mother genetic information
        '''

        # make a copy of father 'genetic' weights & biases information
        nn = copy.deepcopy(father)
        # cross-over bias
        for _ in range(self.nets[0].bias_nitem):
            # get some random points
            layer, point = self.get_random_point('bias')
            # replace genetic (bias) with mother's value
            if random.uniform(0, 1) < self.crossover_rate:
                nn.biases[layer][point] = mother.biases[layer][point]
        # cross-over weight
        for _ in range(self.nets[0].weight_nitem):
            # get some random points
            layer, point = self.get_random_point('weight')
            # replace genetic (weight) with mother's value
            if random.uniform(0, 1) < self.crossover_rate:
                nn.weights[layer][point] = mother.weights[layer][point]

        return nn

    def mutation(self, child):

        '''
        @child_index = neural-net object to mutate its internal weights & biases value
        @returns = new mutated neural-net
        '''

        nn = copy.deepcopy(child)

        # mutate bias
        for _ in range(self.nets[0].bias_nitem):
            # get some random points
            layer, point = self.get_random_point('bias')
            # add some random value between -0.5 and 0.5
            if random.uniform(0, 1) < self.mutation_rate:
                nn.biases[layer][point] += random.uniform(-0.5, 0.5)

        # mutate weight
        for _ in range(self.nets[0].weight_nitem):
            # get some random points
            layer, point = self.get_random_point('weight')
            # add some random value between -0.5 and 0.5
            if random.uniform(0, 1) < self.mutation_rate:
                nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5)

        return nn

    def evolve(self):

        # calculate score for each population of neural-net
        score_list = list(zip(self.nets, self.get_all_scores()))

        # sort the network using its score
        score_list.sort(key=lambda x: x[1])

        # exclude score as it is not needed anymore
        score_list = [obj[0] for obj in score_list]

        # keep only the best one
        retain_num = int(self.n_pops * self.retain_rate)
        score_list_top = score_list[:retain_num]

        # return some non-best ones
        retain_non_best = int((self.n_pops - retain_num) * self.retain_rate)
        for _ in range(random.randint(0, retain_non_best)):
            score_list_top.append(random.choice(score_list[retain_num:]))

        # breed new childs if current population number less than what we want
        while len(score_list_top) < self.n_pops:

            # 随机遍历抽样法、
            father = random.choice(score_list_top)
            mother = random.choice(score_list_top)

            if father != mother:
                new_child = self.crossover(father, mother)
                new_child = self.mutation(new_child)
                score_list_top.append(new_child)

        # copy our new population to current object
        self.nets = score_list_top


def main():
    # load data from iris.csv into X and y
    df = pd.read_csv("wang1.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # convert y into one-hot encoded format
    y = y.reshape(-1, 1)
    y_ori = y
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()

    # parameters
    N_POPS = 30
    NET_SIZE = [12, 10, 3]
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.4
    RETAIN_RATE = 0.4
    # start our neural-net & optimize it using genetic algorithm
    nnga = NNGeneticAlgo(N_POPS, NET_SIZE, MUTATION_RATE, CROSSOVER_RATE, RETAIN_RATE, X, y)
    start_time = time.time()

    for i in range(0):  # 这里设置为0可以之间进入BP的模型训练，学习率调高，迭代次数同时多调一点
        if i % 10 == 0:
            print("Current iteration : {}".format(i + 1))
            print("Time taken by far : %.1f seconds" % (time.time() - start_time))
            print("Current top member's network accuracy: %.2f%%\n" % nnga.get_all_accuracy()[0])
        nnga.evolve()
    nnga.nets[0].nn_model(X, y, 0.001, X, y_ori)
    print('after back progation :', nnga.nets[0].accuracy(X, y))
    import matplotlib.pyplot as plt
    plt.plot(nnga.nets[0].cost_)
    plt.show()


if __name__ == "__main__":
    main()
