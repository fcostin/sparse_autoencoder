import numpy
import scipy.optimize

def sup_norm(x):
    return numpy.max(numpy.abs(x))

def sigmoid(z):
    """
    nonlinear activation function

    n.b. gradient of sigmoid(z) is (sigmoid(z)) * (1 - sigmoid(z)) -- nice!
    """
    return 1.0 / (1.0 + numpy.exp(-z))

def psi(x, w):
    return numpy.dot(w[:, :-1], x) + w[:, -1]

class Net(object):
    def __init__(self, layer_shapes, lmbda, examples = None):
        self.n_layers = len(layer_shapes)
        self.layer_shapes = layer_shapes
        self.layer_sizes = map(numpy.product, layer_shapes)
        self.n_weights = sum(self.layer_sizes)
        self.lmbda = lmbda
        if examples is None:
            examples = []
        self.examples = examples
        super(Net, self).__init__()

    def flatten_weights(self, weights):
        flat_weights = numpy.zeros((self.n_weights, ), dtype = numpy.float)
        i = 0
        for (w, size) in zip(weights, self.layer_sizes):
            flat_weights[i:i + size] = numpy.ravel(w)
            i += size
        return flat_weights

    def unflatten_weights(self, flat_weights):
        weights = []
        i = 0
        for size, shape in zip(self.layer_sizes, self.layer_shapes):
            weights.append(
                numpy.reshape(flat_weights[i:i + size], shape)
            )
            i += size
        return weights

    def backprop(self, weights):
        grad_w_j = [0.0] * self.n_layers
        grad_b_j = [0.0] * self.n_layers
        for (x, y) in self.examples:
            # 1. feed forward
            activation = [x]
            for w in weights:
                z = psi(activation[-1], w)
                activation.append(sigmoid(z))
            # 2. output layer
            delta = [
                -(y - activation[-1]) * activation[-1] * (1.0 - activation[-1])
            ]
            # 3. hidden layers
            for w, a in reversed(zip(weights, activation)):
                delta.append(
                    numpy.dot(w[:, :-1].T, delta[-1]) * a * (1.0 - a)
                )
            delta = list(reversed(delta))
            # 4. compute partial derivatives
            for i in xrange(self.n_layers):
                d = delta[i + 1]
                a = activation[i]
                grad_w_j[i] += numpy.multiply.outer(d, a)
                grad_b_j[i] += d
        return (grad_w_j, grad_b_j)
    
    def evaluate_objective(self, weights):
        n_examples = len(self.examples)
        r = numpy.zeros((n_examples, self.layer_shapes[-1][0]), dtype = numpy.float)
        for i, (x, y) in enumerate(self.examples):
            activation = [x]
            for w in weights:
                z = psi(activation[-1], w)
                activation.append(sigmoid(z))
            r[i, :] = (activation[-1] - y) ** 2
        error_term = 0.5 * numpy.mean(r ** 2)
        # n.b. weights for bias nodes are excempt from regularisation
        penalty_term = 0.5 * numpy.sum(numpy.sum(w[:, :-1] ** 2) for w in weights)
        obj = error_term + self.lmbda * penalty_term
        print ' -- obj : %e' % obj
        return obj
    
    def evaluate_gradient(self, weights):
        n_examples = len(self.examples)
        if n_examples < 1:
            raise ValueError('need at least 1 example')
        grad_w_j, grad_b_j = self.backprop(weights)
        grad_objective = []
        for i in xrange(self.n_layers):
            # n.b. weights for bias nodes are excempt from regularisation
            grad_objective.append(
                numpy.hstack((
                    (grad_w_j[i] / float(n_examples)) + self.lmbda * weights[i][:, :-1],
                    (grad_b_j[i] / float(n_examples))[:, numpy.newaxis],
                ))
            )
        return grad_objective

def test_gradient(f, grad_f, x_0, h, tol):
    n = len(x_0)
    g = grad_f(x_0)
    approx_g = numpy.zeros(g.shape, dtype = g.dtype)
    for i in xrange(n):
        print 'checking %d of %d' % (i, n)
        e_i = numpy.zeros(x_0.shape, dtype = x_0.dtype)
        e_i[i] = 1.0
        approx_g[i] = (f(x_0 + (h * e_i)) - f(x_0 - (h * e_i))) / (2.0 * h)
    error = sup_norm(g - approx_g)
    print error
    if error >= tol:
        print 'g:'
        print g
        print 'approx_g:'
        print approx_g
        print 'residual:'
        print g - approx_g
        1/0

def test_flatten_unflatten(net):
    noise = lambda shape : numpy.random.normal(0.0, 1.0, shape)
    weights = [noise(s) for s in net.layer_shapes]
    weights_tilde = net.unflatten_weights(net.flatten_weights(weights))
    assert all(numpy.all(x == y) for (x, y) in zip(weights, weights_tilde))

def main():
    m = 1 # n input (& output) nodes
    n = 1 # n hidden nodes

    x = numpy.random.uniform(-1.0, 1.0, (m, ))
    y = numpy.random.uniform(-1.0, 1.0, (m, ))

    weights = [
        numpy.random.normal(0.0, 0.1, (n, m + 1)),
    #    numpy.random.normal(0.0, 0.01, (n, n + 1)),
        numpy.random.normal(0.0, 0.1, (m, n + 1)),
    ]

    weights = [
        numpy.random.normal(0.0, 0.1, (m, m + 1)),
    ]
    
    examples = [(x, y)] * 1

    net = Net(map(lambda x : x.shape, weights), lmbda = 0.1, examples = examples)
    print 'evaluate objective'
    obj = net.evaluate_objective(weights)
    print '\tobj %s' % str(obj)

    f = lambda w : net.evaluate_objective(net.unflatten_weights(w))
    grad_f = lambda w : net.flatten_weights(net.evaluate_gradient(net.unflatten_weights(w)))

    test_flatten_unflatten(net)

    test_gradient(f, grad_f, net.flatten_weights(weights), h = 1.0e-4, tol = 1.0e-5)
    
    print 'minimise via cg'
    result = scipy.optimize.fmin_cg(
        f = f,
        x0 = net.flatten_weights(weights),
        fprime = grad_f,
    )
    w_opt = result[0]
    obj_opt = result[1]
    print 'obj_opt : %s' % str(obj_opt)


def profile(func):
    import cProfile, pstats
    p = cProfile.Profile()
    p.runcall(func)
    s = pstats.Stats(p)
    s.sort_stats('cum').print_stats(25)

if __name__ == '__main__':
    profile(main)

