import numpy as np


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
            msgs = []
            summed_msgs = np.add.reduce(received_msgs)
            print receiving_i
            for i in receiving_i:
                msgs.append(np.log(self.f[i]) + summed_msgs[i])
            msg = max(msgs)
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

    f_5.send_ms_msg(Influenza)


if __name__ == '__main__':
    # try:
    test_ms_product()
    # except:
    # print "doei"