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
            activation = {0 : x}
            for i in xrange(self.n_layers):
                w = weights[i]
                z = psi(activation[i], w)
                activation[i + 1] = sigmoid(z)
            # 2. output layer
            delta = {
                self.n_layers : -(y - activation[self.n_layers]) * activation[self.n_layers] * (1.0 - activation[self.n_layers])
            }
            # 3. hidden layers
            for i in xrange(self.n_layers - 1, 0, -1):
                w = weights[i - 1]
                a = activation[i]
                delta[i] = numpy.dot(w[:, :-1], delta[i + 1]) * a * (1.0 - a)
            # 4. compute partial derivatives
            for i in xrange(self.n_layers):
                d = delta[i + 1]
                a = activation[i]
                grad_w_j[i] += numpy.multiply.outer(d, a)
                grad_b_j[i] += d
        return (grad_w_j, grad_b_j)

    def backprop_v2(self, weights):

        alpha_cache = {}
        alpha_prime_cache = {}
        delta_cache = {}
        grad_w_cache = {}

        cachery = (alpha_cache, alpha_prime_cache, delta_cache, grad_w_cache)

        @memoise(alpha_cache)
        def eval_alpha(i):
            assert i > -1 #rely on manual caching of i = -1 case
            # tack on bias input
            x = numpy.hstack((eval_alpha(i - 1), 1.0))
            return sigmoid(psi(eval_alpha(i-1), weights[i]))
        
        @memoise(alpha_prime_cache)
        def eval_alpha_prime(i):
            a = eval_alpha(i)
            return a * (1.0 - a)
        
        @memoise(delta_cache)
        def eval_delta(i):
            assert 0 <= i < self.n_layers
            if i == self.n_layers - 1:
                return eval_alpha_prime(i) * (eval_alpha(i) - y)
            else:
                delta = numpy.hstack((eval_delta(i + 1), 1.0))
                return numpy.dot(weights[i], delta) * eval_alpha_prime(i)

        @memoise(grad_w_cache)
        def eval_grad_w(i):
            assert -1 <= i < self.n_layers -1
            x = numpy.hstack((eval_alpha(i), 1.0))
            return numpy.outer(eval_delta(i + 1), x)
        
        net_grad_w = [0.0] * self.n_layers
        for (x, y) in self.examples:
            alpha_cache[(-1, )] = x
            for i in xrange(self.n_layers):
                net_grad_w[i] += eval_grad_w(i - 1)
            for cache in cachery:
                cache.clear()
        return net_grad_w

    def evaluate_objective(self, weights):
        n_examples = len(self.examples)
        r = 0.0
        for i, (x, y) in enumerate(self.examples):
            activation = [x]
            for w in weights:
                z = psi(activation[-1], w)
                activation.append(sigmoid(z))
            r += numpy.sum((activation[-1] - y) ** 2)
        error_term = 0.5 * r / float(n_examples)
        # n.b. weights for bias nodes are excempt from regularisation
        penalty_term = 0.5 * numpy.sum(numpy.sum(w[:, :-1] ** 2) for w in weights)
        obj = error_term + self.lmbda * penalty_term
        return obj
    
    def evaluate_gradient(self, weights):
        n_examples = len(self.examples)
        if n_examples < 1:
            raise ValueError('need at least 1 example')
        grad_w = self.backprop_v2(weights)
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
     

def test_gradient(f, grad_f, x_0, h, tol):
    n = len(x_0)
    g = grad_f(x_0)
    approx_grad_f = make_gradient_approx(f, h)
    approx_g = approx_grad_f(x_0)
    error = sup_norm(g - approx_g)
    print error
    if error >= tol:
        print 'g:'
        print g
        print 'g_numeric:'
        print approx_g
        print 'relative residual:'
        print (g - approx_g) / numpy.abs(approx_g)

def test_flatten_unflatten(net):
    noise = lambda shape : numpy.random.normal(0.0, 1.0, shape)
    weights = [noise(s) for s in net.layer_shapes]
    weights_tilde = net.unflatten_weights(net.flatten_weights(weights))
    assert all(numpy.all(x == y) for (x, y) in zip(weights, weights_tilde))

def main():
    m = 5 # n input (& output) nodes
    n = 2 # n hidden nodes

    x = numpy.random.uniform(-1.0, 1.0, (m, ))
    y = numpy.random.uniform(-1.0, 1.0, (m, ))

    weights = [
        numpy.random.normal(0.0, 0.1, (n, m + 1)),
        #     numpy.random.normal(0.0, 0.1, (n, n + 1)),
        numpy.random.normal(0.0, 0.1, (m, n + 1)),
    ]
    
    examples = [(x, y)] * 17

    net = Net(map(lambda x : x.shape, weights), lmbda = 0.0, examples = examples)
    print 'evaluate objective'
    obj = net.evaluate_objective(weights)
    print '\tobj %s' % str(obj)

    def f(w):
        obj = net.evaluate_objective(net.unflatten_weights(w))
        print '-- obj : %e' % obj
        return obj

    grad_f = lambda w : net.flatten_weights(net.evaluate_gradient(net.unflatten_weights(w)))
    # grad_f = make_gradient_approx(f, h = 1e-5)

    test_flatten_unflatten(net)

    test_gradient(f, grad_f, net.flatten_weights(weights), h = 1.0e-4, tol = 1.0e-5)
    
    print 'minimise via cg'
    w_opt = scipy.optimize.fmin_cg(
        f = f,
        x0 = net.flatten_weights(weights),
        fprime = grad_f,
    )
    obj_opt = f(w_opt)
    print 'obj_opt : %s' % str(obj_opt)


def profile(func):
    import cProfile, pstats
    p = cProfile.Profile()
    p.runcall(func)
    s = pstats.Stats(p)
    s.sort_stats('cum').print_stats(25)

if __name__ == '__main__':
    profile(main)

