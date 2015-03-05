import itertools

import numpy as np
from pylab import imread, figure, savefig, imshow, close, gray


class Node(object):
    """
    Base-class for Nodes in a factor graph. Only instantiate sub-classes of Node.
    """

    def __init__(self, name):
        # A name for this Node, for printing purposes
        self.name = name

        # Neighbours in the graph, identified with their index in this list.
        # i.e. self.neighbours contains neighbour 0 through len(self.neighbours) - 1.
        self.neighbours = []

        # Reset the node-state (not the graph topology)
        self.reset()

    def reset(self):
        # Incoming messages; a dictionary mapping neighbours to messages.
        # That is, it maps  Node -> np.ndarray.
        self.in_msgs = {}

        # A set of neighbours for which this node has pending messages.
        # We use a python set object so we don't have to worry about duplicates.
        self.pending = set([])

    def add_neighbour(self, nb):
        self.neighbours.append(nb)

    def send_sp_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_sp_msg not implemented in base-class Node')

    def send_ms_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_ms_msg not implemented in base-class Node')

    def receive_msg(self, other, msg):
        # Store the incoming message, replacing previous messages from the same node
        # print "\t %s received message from %s: %s" % (self, other, msg)
        self.in_msgs[other] = msg

        for neighbour in set(self.neighbours):

            # heb ik van mijn ander andere neighbours alle messages binnen
            if all((other_neighbour in self.in_msgs) for other_neighbour in (set(self.neighbours) - {neighbour})):
                self.pending.add(neighbour)
                # print "\t %s now has a pending message for %s" % (self, ', '.join([str(p) for p in self.pending]))

    def __str__(self):
        # This is printed when using 'print node_instance'
        return self.name


class Variable(Node):
    def __init__(self, name, num_states):
        """
        Variable node constructor.
        Args:
            name: a name string for this node. Used for printing. 
            num_states: the number of states this variable can take.
            Allowable states run from 0 through (num_states - 1).
            For example, for a binary variable num_states=2,
            and the allowable states are 0, 1.
        """
        self.num_states = num_states

        # Call the base-class constructor
        super(Variable, self).__init__(name)

    def set_observed(self, observed_state):
        """
        Set this variable to an observed state.
        Args:
            observed_state: an integer value in [0, self.num_states - 1].
        """
        # Observed state is represented as a 1-of-N variable
        # Could be 0.0 for sum-product, but log(0.0) = -inf so a tiny value is preferable for max-sum
        self.observed_state[:] = 0.0000000000001
        self.observed_state[observed_state] = 1.0

    def set_latent(self):
        """
        Erase an observed state for this variable and consider it latent again.
        """
        # No state is preferred, so set all entries of observed_state to 1.0
        # Using this representation we need not differentiate between observed and latent
        # variables when sending messages.
        self.observed_state[:] = 1.0

    def reset(self):
        super(Variable, self).reset()
        self.observed_state = np.ones(self.num_states)

    def marginal(self, Z=None):
        """
        Compute the marginal distribution of this Variable.
        It is assumed that message passing has completed when this function is called.
        Args:
            Z: an optional normalization constant can be passed in. If None is passed, Z is computed.
        Returns: marginal, Z. The first is a numpy array containing the normalized marginal distribution.
         Z is either equal to the input Z, or computed in this function (if Z=None was passed).
        """
        marginal = np.ones(self.num_states)
        for neighbour in self.neighbours:
            marginal *= self.in_msgs[neighbour]
        marginal *= self.observed_state
        if Z is None:
            Z = np.sum(marginal)

        marginal /= Z
        return marginal, Z

    def map_state(self):
        """
        Compute the map_state  of this Variable.
        """
        summed = np.sum(np.array(self.in_msgs.values()), axis=0)
        summed += np.log(self.observed_state)
        return np.argmax(summed)

    def send_sp_msg(self, other):
        """
        Variable -> Factor message for sum-product
        :param other:
        :return:
        """
        if len(self.neighbours) == 1:
            msg = np.ones(self.num_states)
        else:
            # if True:
            neighbours = set(self.neighbours)
            receiving_neighbours = neighbours - {other}

            for receiv_node in receiving_neighbours:
                if receiv_node not in self.in_msgs:
                    raise Exception('did not receive message for node %s' % str(receiv_node))

            received_msgs = [self.in_msgs.get(n) for n in receiving_neighbours]

            if len(received_msgs) > 1:
                msg = np.ones(self.num_states)
                for received_msg in received_msgs:
                    msg *= received_msg
            else:
                msg = received_msgs[0]
        msg *= self.observed_state
        other.receive_msg(self, msg)
        self.pending.remove(other)

    def send_ms_msg(self, other):
        # TODO: implement Variable -> Factor message for max-sum
        neighbours = set(self.neighbours)
        receiving_neighbours = neighbours - {other}
        # receiving_i = [self.neighbour  s.index(n) for n in receiving_neighbours]

        for receiv_node in receiving_neighbours:
            if receiv_node not in self.in_msgs:
                raise Exception('did not receive message for node %s' % str(receiv_node))

        received_msgs = [self.in_msgs.get(n) for n in receiving_neighbours]

        if not receiving_neighbours:
            msg = np.zeros(self.num_states)
        else:
            summed_msgs = np.add.reduce(received_msgs)
            msg = summed_msgs

        msg += np.log(self.observed_state)
        if self.name == "Bronchitis":
            print msg
        # print "\t Send msg from Var [%s] to Fac [%s] (%s)" % (self.name, other.name, str(msg))

        other.receive_msg(self, msg)


class Factor(Node):
    def __init__(self, name, f, neighbours):
        """
        Factor node constructor.
        Args:
            name: a name string for this node. Used for printing
            f: a numpy.ndarray with N axes, where N is the number of neighbours.
               That is, the axes of f correspond to variables, and the index along that axes corresponds to a value
               of that variable.
               Each axis of the array should have as many entries as the corresponding neighbour variable has states.
            neighbours: a list of neighbouring Variables. Bi-directional connections are created.
        """
        # Call the base-class constructor
        super(Factor, self).__init__(name)

        assert len(neighbours) == f.ndim, 'Factor function f should accept as many arguments as this Factor node has ' \
                                          'neighbours'

        for nb_ind in range(len(neighbours)):
            nb = neighbours[nb_ind]
            assert f.shape[nb_ind] == nb.num_states, 'The range of the factor function f is invalid for input %i %s' % (
                nb_ind, nb.name)
            self.add_neighbour(nb)
            nb.add_neighbour(self)

        self.f = f

    def send_sp_msg(self, other):
        """
        Factor -> Variable message for sum-product
        :param other:
        :return:
        """
        neighbours = set(self.neighbours)
        receiving_neighbours = list(neighbours - {other})
        mes_i = [receiving_neighbours.index(n) for n in receiving_neighbours]
        fac_i = [self.neighbours.index(n) for n in receiving_neighbours]

        for receiv_node in receiving_neighbours:
            if receiv_node not in self.in_msgs:
                raise Exception('did not receive message for node %s' % str(receiv_node))

        received_msgs = [self.in_msgs.get(n) for n in receiving_neighbours]

        a = np.array(np.multiply.reduce(np.ix_(*received_msgs)))
        msg = np.tensordot(a, self.f, axes=(mes_i, fac_i))
        other.receive_msg(self, msg)
        self.pending.remove(other)

    def send_ms_msg(self, other):
        # TODO: implement Factor -> Variable message for max-sum
        neighbours = set(self.neighbours)
        receiving_neighbours = neighbours - {other}
        receiving_i = [self.neighbours.index(n) for n in receiving_neighbours]

        for receiv_node in receiving_neighbours:
            if receiv_node not in self.in_msgs:
                raise Exception('did not receive message for node %s' % str(receiv_node))

        received_msgs = [self.in_msgs.get(n) for n in receiving_neighbours]

        if not receiving_neighbours:
            msg = np.log(self.f)
        else:
            summed_msgs = np.add.reduce(np.ix_(*received_msgs))
            summed_msgs_f = np.log(self.f) + summed_msgs

            msg = np.amax(summed_msgs_f, axis=tuple(receiving_i))
        # print "\t Send msg from Fac [%s] to Var [%s] (%s)" % (self.name, other.name, str(msg))
        other.receive_msg(self, msg)


def send_pending(node):
    if node.pending:
        for pending_neigh in list(node.pending):
            node.send_sp_msg(pending_neigh)


def send_pending_ms(node):
    if node.pending:
        for pending_neigh in list(node.pending):
            node.send_ms_msg(pending_neigh)


def sum_product(node_list):
    # Begin to end
    for node in node_list:
        send_pending(node)

    # End to begin
    for node in node_list[::-1]:
        send_pending(node)


def ms_product(node_list):
    for node in node_list:
        send_pending_ms(node)

    # End to begin
    for node in node_list[::-1]:
        send_pending_ms(node)


def print_marginal(node):
    marginal, z = node.marginal()
    print "Marginal for %s is %s with z %s" % (node.name, marginal, z)


def print_map_state(node):
    map_state = node.map_state()
    print "Map state for %s is %s" % (node.name, map_state)


def init_network():
    Influenza = Variable("Influenza", 2)
    Smokes = Variable("Smokes", 2)
    SoreThroat = Variable("SoreThroat", 2)
    Fever = Variable("Fever", 2)
    Bronchitis = Variable("Bronchitis", 2)
    Coughing = Variable("Coughing", 2)
    Wheezing = Variable("Wheezing", 2)
    f_3dim = np.zeros((2, 2, 2))
    f_3dim[0, 0, 0] = 0.9999
    f_3dim[0, 0, 1] = 0.0001
    f_3dim[0, 1, 0] = 0.3
    f_3dim[0, 1, 1] = 0.7
    f_3dim[1, 0, 0] = 0.1
    f_3dim[1, 0, 1] = 0.9
    f_3dim[1, 1, 0] = 0.01
    f_3dim[1, 1, 1] = 0.99
    f_0 = Factor("f_0", np.array([[0.999, 0.001], [0.7, 0.3]]), [Influenza, SoreThroat])
    f_1 = Factor("f_1", np.array([[0.95, 0.05], [0.1, 0.9]]), [Influenza, Fever])
    f_2 = Factor("f_2", f_3dim, [Influenza, Smokes, Bronchitis])
    f_3 = Factor("f_3", np.array([[0.93, 0.07], [0.2, 0.8]]), [Bronchitis, Coughing])
    f_4 = Factor("f_4", np.array([[0.999, 0.001], [0.4, 0.6]]), [Bronchitis, Wheezing])
    f_5 = Factor("f_5", np.array([0.95, 0.05]), [Influenza])
    f_6 = Factor("f_6", np.array([0.8, 0.2]), [Smokes])
    return Bronchitis, Coughing, Fever, Influenza, Smokes, SoreThroat, Wheezing, f_0, f_1, f_2, f_3, f_4, f_5, f_6


def test_sum_product():
    Bronchitis, Coughing, Fever, Influenza, Smokes, SoreThroat, Wheezing, f_0, f_1, f_2, f_3, f_4, f_5, \
    f_6 = init_network()

    # prior factors
    node_list = [f_5, f_6, Smokes, SoreThroat, Fever, f_0, f_1, Influenza, f_2, Coughing, f_3, Wheezing, f_4,
                 Bronchitis]
    f_5.pending.add(Influenza)
    f_6.pending.add(Smokes)
    SoreThroat.pending.add(f_0)
    Fever.pending.add(f_1)
    Coughing.pending.add(f_3)
    Wheezing.pending.add(f_4)

    Bronchitis.set_observed(1)
    Influenza.set_observed(0)

    sum_product(node_list)

    print_marginal(Influenza)
    print_marginal(Bronchitis)
    print_marginal(Coughing)
    print_marginal(Wheezing)
    print_marginal(Fever)
    print_marginal(SoreThroat)
    print_marginal(Smokes)


def test_ms_product():
    Bronchitis, Coughing, Fever, Influenza, Smokes, SoreThroat, Wheezing, f_0, f_1, f_2, f_3, f_4, f_5, \
    f_6 = init_network()
    node_list = [f_5, f_6, Smokes, SoreThroat, Fever, f_0, f_1, Influenza, f_2, Coughing, f_3, Wheezing, f_4,
                 Bronchitis]

    f_5.pending.add(Influenza)
    f_6.pending.add(Smokes)
    SoreThroat.pending.add(f_0)
    Fever.pending.add(f_1)
    Coughing.pending.add(f_3)
    Wheezing.pending.add(f_4)

    # Coughing.set_observed(1)
    # Wheezing.set_observed(1)
    # Fever.set_observed(0)
    # SoreThroat.set_observed(1)
    # Influenza.set_observed(1)
    Smokes.set_observed(1)
    ms_product(node_list)

    print_map_state(Influenza)
    print_map_state(Bronchitis)
    print_map_state(Coughing)
    print_map_state(Wheezing)
    print_map_state(Fever)
    print_map_state(SoreThroat)
    print_map_state(Smokes)


def load_images():
    # Load the image and binarize
    im = np.mean(imread('dalmatian1.png'), axis=2) < 0.5
    im = im[0:40, 0:40]
    # imshow(im)
    # gray()

    # Add some noise
    noise = np.random.rand(*im.shape) > 0.9
    noise_im = np.logical_xor(noise, im)

    test_im = np.ones((10, 10))
    # test_im[5:8, 3:8] = 1.0
    # test_im[5,5] = 1.0
    # figure()
    # imshow(test_im)

    # Add some noise
    noise = np.random.rand(*test_im.shape) > 0.9
    noise_test_im = np.logical_xor(noise, test_im)
    # figure()
    # imshow(noise_test_im)
    return im, noise_im, test_im, noise_test_im


def init_image_graph(im):
    xs = []
    ys = []
    graph = []
    xy_factors = []
    xx_factors = []
    height, width = im.shape
    for row in range(height):
        for col in range(width):
            y = Variable("y[%d,%d]" % (row, col), 2)
            x = Variable("x[%d,%d]" % (row, col), 2)

            # f = np.array([[0.95, 0.05], [0.05, 0.95]])  # TODO: wat moet dit worden?
            f = np.array([[0.9999, 0.0001], [0.0001, 0.9999]])  # TODO: wat moet dit worden?

            factor = Factor("f " + str(row) + "," + str(col), f, [y, x])

            y.set_observed(im[row, col])
            y.pending.add(factor)

            x.set_latent()

            ys.append(y)
            graph.append(y)
            graph.append(factor)
            xs.append(x)
            xy_factors.append(factor)

    for i in range(0, len(xs)):
        graph.append(xs[i])
        f1 = np.array([[0.60, 0.40], [0.40, 0.60]])  # TODO: wat moet dit worden
        if (i + 1) % width != 0:
            right_neighbour_factor = Factor("f(%s->%s)" % (xs[i].name, xs[i + 1].name), f1, [xs[i], xs[i + 1]])
            graph.append(right_neighbour_factor)
            xx_factors.append(right_neighbour_factor)
            # Initialize a message to the left:
            xs[i].in_msgs[right_neighbour_factor] = np.array([np.log(.5), np.log(.5)])
            xs[i].pending.add(right_neighbour_factor)

        if i < (height - 1) * width:
            down_neighbour_factor = Factor("f(%s->%s)" % (xs[i].name, xs[i + width].name), f1, [xs[i], xs[i + width]])
            graph.append(down_neighbour_factor)
            xx_factors.append(down_neighbour_factor)
            # Initialize a message upwards:
            xs[i].in_msgs[down_neighbour_factor] = np.array([np.log(.5), np.log(.5)])
            xs[i].pending.add(down_neighbour_factor)

    # xs[0].pending.add(xx_factors[0])
    # xs[0].pending.add(xx_factors[1])
    # xs[0].in_msgs[xx_factors[0]] = np.array([np.log(1), np.log(1)])
    # xs[0].in_msgs[xx_factors[1]] = np.array([np.log(1), np.log(1)])
    return graph, xs, ys


def create_map_im(shape, xs):
    new_img = np.zeros(shape)
    for i, p in enumerate(itertools.product(range(shape[0]), range(shape[1]))):
        row, col = p
        new_img[row][col] = xs[i].map_state()

    return new_img


def save_image(im, name):
    figure()
    gray()
    # im[0][0] = 0.
    imshow(im)
    savefig("%s.png" % (name))
    close()


def run_denoising(noise_im, im):
    graph, xs, ys = init_image_graph(noise_im)
    print "starting algorithm..."
    for i in range(20):
        for node in graph:
            # print "- " + node.name
            send_pending_ms(node)
        print "----- NEXT RUN (%d) ------" % (i + 1)
        new_img = create_map_im(im.shape, xs)
        save_image(new_img, "iters/im_%d" % i)
    print new_img
    print im
    print noise_im
    # new_img = create_map_im(im.shape, xs)
    save_image(im, "im")
    save_image(noise_im, "noise_im")
    save_image(new_img, "new_img")


def test_image_denoising():
    im, noise_im, test_im, noise_test_im = load_images()
    # run_denoising(noise_test_im, test_im)
    run_denoising(noise_im, im)


if __name__ == '__main__':
    # try:
    test_image_denoising()
    # except:
    # print "doei"