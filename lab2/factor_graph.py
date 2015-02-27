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
        # Store the incomming message, replacing previous messages from the same node
        self.in_msgs[other] = msg

        for neighbour in (set(self.neighbours) - {other}):
            # heb ik van mijn ander andere neighbours alle messages binnen
            if all(self.in_msgs[other_neighbour] for other_neighbour in (set(self.neighbours) - {other, neighbour})):
                self.pending.add(neighbour)
    
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
        self.observed_state[:] = 0.000001
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
            # TODO:
            marginal = np.dot(marginal, self.in_msgs[neighbour])
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
        neighbours = set(self.neighbours)
        receiving_neighbours = neighbours - {other}

        for receiv_node in receiving_neighbours:
            if receiv_node not in self.in_msgs:
                raise Exception('did not receive message for node %s' % str(receiv_node))

        received_msgs = [self.in_msgs.get(n) for n in receiving_neighbours]

        # TODO: check deze implementatie als received_msgs klaar is
        msg = np.multiply.reduce(np.ix_(*received_msgs))
        other.receive_msg(self, msg)

    def send_ms_msg(self, other):
        # TODO: implement Variable -> Factor message for max-sum
        pass

class Factor(Node):
    def __init__(self, name, f, neighbours):
        """
        Factor node constructor.
        Args:
            name: a name string for this node. Used for printing
            f: a numpy.ndarray with N axes, where N is the number of neighbours.
               That is, the axes of f correspond to variables, and the index along that axes corresponds to a value of that variable.
               Each axis of the array should have as many entries as the corresponding neighbour variable has states.
            neighbours: a list of neighbouring Variables. Bi-directional connections are created.
        """
        # Call the base-class constructor
        super(Factor, self).__init__(name)

        assert len(neighbours) == f.ndim, 'Factor function f should accept as many arguments as this Factor node has neighbours'
        
        for nb_ind in range(len(neighbours)):
            nb = neighbours[nb_ind]
            assert f.shape[nb_ind] == nb.num_states, 'The range of the factor function f is invalid for input %i %s' % (nb_ind, nb.name)
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
        receiving_neighbours = neighbours - {other}
        receiving_i = [self.neighbours.index(n) for n in receiving_neighbours]

        for receiv_node in receiving_neighbours:
            if receiv_node not in self.in_msgs:
                raise Exception('did not receive message for node %s' % str(receiv_node))
        
        received_msgs = [self.in_msgs.get(n) for n in receiving_neighbours]

        # TODO: check deze implementatie als received_msgs klaar is
        a = np.multiply.reduce(np.ix_(*received_msgs))
        msg = np.tensordot(a, self.f, axes=(receiving_i, receiving_i))
        other.receive_msg(self, msg)

    def send_ms_msg(self, other):
        # TODO: implement Factor -> Variable message for max-sum
        pass


def send_pending(node):
    if node.pending:
        for pending_neigh in node.pending:
            node.send_sp_msg(pending_neigh)


def sum_product(node_list):
    # Begin to end
    for node in node_list:
        send_pending(node)

    # End to begin
    for node in node_list[::-1]:
        send_pending(node)

def test_sum_product():
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

