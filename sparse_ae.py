import numpy
import scipy.optimize
import itertools

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
    def __init__(self, layer_shapes, lmbda = 0.0, beta = 0.0, rho = 0.05, examples = None, example_weights = None, verbose = False):
        self.n_layers = len(layer_shapes)
        self.layer_shapes = layer_shapes
        self.layer_sizes = map(numpy.product, layer_shapes)
        self.n_weights = sum(self.layer_sizes)
        self.lmbda = lmbda
        self.beta = beta
        self.rho = rho
        if examples is None:
            examples = []
        self.examples = examples
        if example_weights is None:
            example_weights = numpy.ones((len(self.examples), ), dtype = numpy.int)
        self.example_weights = example_weights
        self.net_example_weight = float(numpy.sum(self.example_weights))
        self.verbose = verbose
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

    def make_rho_hat(self, weights):
        # compute mean activity over all training inputs for each hidden node
        rho_hat = {}
        for (x, y), example_weight in itertools.izip(self.examples, self.example_weights):
            alpha = self.make_alpha(weights, x)
            # only apply this penalty to hidden layers:
            # ignore first (i == -1) and last (i == self.n_layers - 1)
            for i in xrange(self.n_layers - 1):
                rho_hat[i] = rho_hat.get(i, 0.0) + example_weight * alpha[i]
        for i in rho_hat:
            rho_hat[i] = rho_hat[i] / self.net_example_weight
        return rho_hat

    def propagate(self, weights):
        """
        arguments:
            weights : list of 2d weight arrays (matrices)
        returns:
            (net_square_error, net_grad_w, sparsity_penalty_term)
            list of derivatives of net output with respect to layer weights,
            where the i-th item is a 2d array of the same shape as the
            i-th weight array, giving the corresponding partial derivatives.
        """

        delta_bias = {}
        sparsity_penalty_term = 0.0
        if self.beta > 0.0:
            rho_hat = self.make_rho_hat(weights)
            # compute derivative of sparsity penalty
            for i in rho_hat:
                delta_bias[i] = self.beta * kl_div_prime(
                    self.rho,
                    rho_hat[i],
                )
            for i in rho_hat:
                sparsity_penalty_term += numpy.sum(kl_div(self.rho, rho_hat[i]))
                sparsity_penalty_term *= self.beta
        elif self.beta < 0.0:
            raise ValueError('beta must be non-negative')

        net_grad_w = [0.0] * self.n_layers
        net_square_error = 0.0
        for (x, y), example_weight in itertools.izip(self.examples, self.example_weights):
            alpha = self.make_alpha(weights, x)
            alpha_prime = self.make_alpha_prime(alpha)
            delta = self.make_delta(weights, y, alpha, alpha_prime, delta_bias)
            grad_w = self.make_grad_w(alpha, delta)
            # accumulate into net weight derivatives
            for i in xrange(self.n_layers - 1, -1, -1):
                net_grad_w[i] += example_weight * grad_w[i - 1]
            # update objective
            y_pred = alpha[self.n_layers - 1]
            net_square_error += example_weight * numpy.sum((y_pred - y) ** 2)

        return net_square_error, net_grad_w, sparsity_penalty_term
    
    def obj_from_net_square_error(self, weights, net_square_error, sparsity_penalty_term):
        # compute objective function from net square error
        error_term = 0.5 * net_square_error / self.net_example_weight
        # n.b. weights for bias nodes are excempt from regularisation
        penalty_term = 0.5 * numpy.sum(numpy.sum(w[:, :-1] ** 2) for w in weights)
        objective = error_term + self.lmbda * penalty_term + sparsity_penalty_term
        if self.verbose:
            print 'J\tNET\t%.8e\tmse\t%.2e\tcoef\t%.2e\tsparsity\t%.2e' % (
                objective,
                error_term,
                self.lmbda * penalty_term,
                sparsity_penalty_term,
            )
        return objective

    def grad_obj_from_grad_w(self, weights, grad_w):
        # compute grad objective in terms of gradient of net square error
        grad_objective = []
        for i in xrange(self.n_layers):
            w_i = grad_w[i][:, :-1]
            bias_i = grad_w[i][:, -1]
            # n.b. weights for bias nodes are excempt from regularisation
            grad_objective.append(
                numpy.hstack((
                    (w_i / self.net_example_weight) + self.lmbda * weights[i][:, :-1],
                    (bias_i / self.net_example_weight)[:, numpy.newaxis],
                ))
            )
        return grad_objective

    def evaluate_objective_and_gradient(self, weights):
        net_square_error, grad_w, sparsity_penalty_term = self.propagate(weights)
        obj = self.obj_from_net_square_error(weights, net_square_error, sparsity_penalty_term)
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
        factr = 1e12, # factor for tolerance between successive func values, 1e7 is moderate accuracy
    )
    if info['warnflag'] != 0:
        raise RuntimeError('cvgc failure, warnflag is %d' % info['warnflag'])
    return (w_opt, obj_opt)

def main():
    m = 9 # n input nodes
    n = 5 # n hidden nodes
    o = 17 # n output nodes

    weights = [
        numpy.random.normal(0.0, 0.01, (n, m + 1)),
        numpy.random.normal(0.0, 0.01, (o, n + 1)),
    ]
    
    n_examples = 20
    examples = []
    for i in xrange(n_examples):
        x = numpy.random.uniform(-1.0, 1.0, (m, ))
        y = numpy.random.uniform(-1.0, 1.0, (o, ))
        examples.append((x, y))

    net = Net(
        map(lambda x : x.shape, weights),
        lmbda = 1.0,
        beta = 1.0,
        examples = examples,
        verbose = True,
    )

    # enable to sanity-check consistency of objective and gradient
    if True:
        print 'checking consistency of objective and gradient'
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
        print '\tobjective & gradient are approximately consistent'

    def f_and_grad_f(flat_w):
        w = net.unflatten_weights(flat_w)
        f, grad_f = net.evaluate_objective_and_gradient(w)
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

