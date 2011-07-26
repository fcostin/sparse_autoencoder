import numpy
import scipy.optimize

def memoise(cache = None):
    if cache is None:
        cache = {}
    def memoiser(f):
        def memoised_f(*args):
            if args not in cache:
                cache[args] = f(*args)
            return cache[args]
        return memoised_f
    return memoiser

def sup_norm(x):
    return numpy.max(numpy.abs(x))

def sigmoid(z):
    """
    nonlinear activation function

    n.b. gradient of sigmoid(z) is (sigmoid(z)) * (1 - sigmoid(z)) -- nice!
    """
    return 1.0 / (1.0 + numpy.exp(-z))

def kl_div(p, q):
    """
    KL divergence
    """
    return p * numpy.log(p / q) + (1.0 - p) * numpy.log((1.0 - p)/(1.0 - q))

def kl_div_prime(p, q):
    """
    derivative of KL divergence wrt second argument q
    """
    return -p/q + (1.0 - p)/(1.0 - q)

class Net(object):
    def __init__(self, layer_shapes, lmbda = 0.0, beta = 0.0, examples = None):
        self.n_layers = len(layer_shapes)
        self.layer_shapes = layer_shapes
        self.layer_sizes = map(numpy.product, layer_shapes)
        self.n_weights = sum(self.layer_sizes)
        self.lmbda = lmbda
        self.beta = beta
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

    def make_alpha(self, weights, x):
        # seed inputs with x then compute forward pass
        alpha = {-1 : x}
        for i in xrange(self.n_layers):
            z = numpy.hstack((alpha[i - 1], 1.0))
            alpha[i] = sigmoid(numpy.dot(weights[i], z))
        return alpha
    
    def make_alpha_prime(self, alpha):
        # compute alpha_prime for back pass
        alpha_prime = {}
        for i in xrange(self.n_layers - 1, -1, -1):
            alpha_prime[i] = alpha[i] * (1.0 - alpha[i])
        return alpha_prime

    def make_delta(self, weights, y, alpha, alpha_prime, delta_bias = None):
        # compute delta for back pass
        delta = {}
        for i in xrange(self.n_layers - 1, -1, -1):
            if i == self.n_layers - 1:
                delta[i] = alpha_prime[i] * (alpha[i] - y)
            else:
                z = numpy.dot(weights[i + 1][:, :-1].T, delta[i + 1])
                if delta_bias and i in delta_bias:
                    z += delta_bias[i]
                delta[i] = z * alpha_prime[i]
        return delta
    
    def make_grad_w(self, alpha, delta):
        # compute grad_w for back pass
        grad_w = {}
        for i in xrange(self.n_layers - 1, -1, -1):
            z = numpy.hstack((alpha[i - 1], 1.0))
            grad_w[i - 1] = numpy.outer(delta[i], z)
        return grad_w

    def propagate(self, weights, prop_back, delta_bias = None):
        """
        arguments:
            weights : list of 2d weight arrays (matrices)
            prop_back : bool, should we backprop after forwardprop?
            delta_bias : (optional) mapping of int -> bias vector, where
                the integer k satisifies 0 <= k < n_layers
                and, if present, the shape of the bias vector for a layer
                k matches the shape of the delta vector for that k
        returns:
            list of derivatives of net output with respect to layer weights,
            where the i-th item is a 2d array of the same shape as the
            i-th weight array, giving the corresponding partial derivatives.
        """

        net_grad_w = [0.0] * self.n_layers
        net_square_error = 0.0
        for (x, y) in self.examples:
            alpha = self.make_alpha(weights, x)
            alpha_prime = self.make_alpha_prime(alpha)
            delta = self.make_delta(weights, y, alpha, alpha_prime)
            grad_w = self.make_grad_w(alpha, delta)
            # accumulate into net weight derivatives
            for i in xrange(self.n_layers - 1, -1, -1):
                net_grad_w[i] += grad_w[i - 1]
            # update objective
            y_pred = alpha[self.n_layers - 1]
            net_square_error += numpy.sum((y_pred - y) ** 2)
        if prop_back:
            return net_square_error, net_grad_w
        else:
            return net_square_error
    
    def obj_from_net_square_error(self, weights, net_square_error):
        n_examples = len(self.examples)
        # compute objective function from net square error
        error_term = 0.5 * net_square_error / float(n_examples)
        # n.b. weights for bias nodes are excempt from regularisation
        penalty_term = 0.5 * numpy.sum(numpy.sum(w[:, :-1] ** 2) for w in weights)
        objective = error_term + self.lmbda * penalty_term
        return objective

    def grad_obj_from_grad_w(self, weights, grad_w):
        n_examples = len(self.examples)
        # compute grad objective in terms of gradient of net square error
        grad_objective = []
        for i in xrange(self.n_layers):
            w_i = grad_w[i][:, :-1]
            bias_i = grad_w[i][:, -1]
            # n.b. weights for bias nodes are excempt from regularisation
            grad_objective.append(
                numpy.hstack((
                    (w_i / float(n_examples)) + self.lmbda * weights[i][:, :-1],
                    (bias_i / float(n_examples))[:, numpy.newaxis],
                ))
            )
        return grad_objective

    def evaluate_objective_and_gradient(self, weights):
        net_square_error, grad_w = self.propagate(weights, prop_back = True)
        obj = self.obj_from_net_square_error(weights, net_square_error)
        grad_obj = self.grad_obj_from_grad_w(weights, grad_w)
        return obj, grad_obj

def make_gradient_approx(f, h):
    def approx_grad_f(x_0):
        n = len(x_0)
        approx_g = numpy.zeros(x_0.shape, dtype = x_0.dtype)
        for i in xrange(n):
            e_i = numpy.zeros(x_0.shape, dtype = x_0.dtype)
            e_i[i] = 1.0
            approx_g[i] = (f(x_0 + (h * e_i)) - f(x_0 - (h * e_i))) / (2.0 * h)
        return approx_g
    return approx_grad_f
     
def assert_gradient_works(f, grad_f, x_0, h, tol):
    n = len(x_0)
    g = grad_f(x_0)
    approx_grad_f = make_gradient_approx(f, h)
    approx_g = approx_grad_f(x_0)
    error = sup_norm(g - approx_g) / sup_norm(approx_g)
    if error >= tol:
        print 'g:'
        print g
        print 'g_numeric:'
        print approx_g
        print 'relative residual:'
        print (g - approx_g) / numpy.abs(approx_g)
        raise ValueError('gradient is bad')

def test_flatten_unflatten(net):
    noise = lambda shape : numpy.random.normal(0.0, 1.0, shape)
    weights = [noise(s) for s in net.layer_shapes]
    weights_tilde = net.unflatten_weights(net.flatten_weights(weights))
    assert all(numpy.all(x == y) for (x, y) in zip(weights, weights_tilde))

def lbfgs(f_and_grad_f, x_0):
    w_opt, obj_opt, info = scipy.optimize.fmin_l_bfgs_b(
        func = f_and_grad_f,
        x0 = x_0,
    )
    if info['warnflag'] != 0:
        raise RuntimeError('cvgc failure, warnflag is %d' % info['warnflag'])
    return (w_opt, obj_opt)

def main():
    m = 7 # n input nodes
    n = 2 # n hidden nodes
    o = 11 # n output nodes

    x = numpy.random.uniform(-1.0, 1.0, (m, ))
    y = numpy.random.uniform(-1.0, 1.0, (o, ))

    weights = [
        numpy.random.normal(0.0, 0.1, (n, m + 1)),
        numpy.random.normal(0.0, 0.1, (o, n + 1)),
    ]
    
    examples = [(x, y)] * 17

    net = Net(map(lambda x : x.shape, weights), lmbda = 0.1, examples = examples)

    # enable to sanity-check consistency of objective and gradient
    if True:
        def test_f(flat_w):
            w = net.unflatten_weights(flat_w)
            f, _ = net.evaluate_objective_and_gradient(w)
            return f

        def test_grad_f(flat_w):
            w = net.unflatten_weights(flat_w)
            _, grad_f = net.evaluate_objective_and_gradient(w)
            return net.flatten_weights(grad_f)

        assert_gradient_works(
            test_f,
            test_grad_f,
            net.flatten_weights(weights),
            h = 1.0e-4,
            tol = 1.0e-5,
        )

    def f_and_grad_f(flat_w):
        w = net.unflatten_weights(flat_w)
        f, grad_f = net.evaluate_objective_and_gradient(w)
        print '-- obj : %e' % f
        return (f, net.flatten_weights(grad_f))

    if False:
        test_flatten_unflatten(net)
        assert_gradient_works(
            f,
            grad_f,
            net.flatten_weights(weights),
            h = 1.0e-4,
            tol = 1.0e-5
        )

    print 'minimising objective with l-bfgs'
    w_opt, obj_opt = lbfgs(f_and_grad_f, net.flatten_weights(weights))
    print 'obj_opt : %s' % str(obj_opt)

def profile(func):
    import cProfile, pstats
    p = cProfile.Profile()
    p.runcall(func)
    s = pstats.Stats(p)
    s.sort_stats('cum').print_stats(25)

if __name__ == '__main__':
    profile(main)

