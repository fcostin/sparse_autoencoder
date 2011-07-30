"""
semi-finished trial bash of sparse autoencoder on weekly edit count data
"""

import sparse_ae
import numpy
import hashlib

def get_data(n):
    assert n > 0
    x = numpy.load('usr_edits_per_week.npy')
    return x[:, -n:]

def reduce_data(data):
    """
    arguments:
        data : shape (n_usrs, n_features) array of features
    returns:
        (reduced_data, weights)
    where
        reduced_data : shape (n_unique, n_features) array of unique features
        weights : shape (n_unique, ) array of multiplicy of unique features in data
    """
    count = {}
    unique = {}
    hash_features = lambda x : hashlib.md5(x).digest()
    for i in xrange(data.shape[0]):
        x = data[i, :]
        key = hash_features(x)
        if key not in count:
            unique[key] = x
            count[key] = 1
        else:
            count[key] += 1

    n_unique = len(unique)
    reduced_data = numpy.zeros((n_unique, data.shape[1]), dtype = data.dtype)
    weights = numpy.zeros((n_unique, ), dtype = numpy.int)
    for i, k in enumerate(unique):
        reduced_data[i, :] = unique[k]
        weights[i] = count[k]
    return reduced_data, weights

def phi(z):
    return numpy.log(1.0 + z)

def phi_inv(z):
    return numpy.exp(z) - 1.0

class Preprocessor(object):
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.lo = None
        self.hi = None
        super(Preprocessor, self).__init__()

    def fit(self, training_data):
        training_data = numpy.asarray(training_data)
        self.mu = numpy.mean(training_data, axis = 0)
        self.sigma = numpy.std(training_data, axis = 0)
        self.lo = numpy.min(training_data, axis = 0)
        self.hi = numpy.max(training_data, axis = 0)
        return self

    def forward_transform(self, data):
        data = numpy.asarray(data)
        # d = (data - self.mu[numpy.newaxis, :]) / (self.sigma[numpy.newaxis, :])
        return (data - self.lo[numpy.newaxis, :]) / (self.hi - self.lo)[numpy.newaxis, :]

    def reverse_transform(self, data):
        d = numpy.asarray(data)
        return (data * (self.hi - self.lo)[numpy.newaxis, :]) + self.lo[numpy.newaxis, :]

def main():

    n_input = 100 # number of input nodes
    n_hidden = 50 # number of hidden nodes
    n_output = 100

    usr_data = get_data(n_input)
    # reduced_data, example_weights = reduce_data(usr_data)
    reduced_data = phi(usr_data)
    example_weights = None
    print 'reduced feature data set has shape %s' % str(reduced_data.shape)

    # thin dataset to speed up code testing
    # XXX this should be weighted by example weight for reduced data
    training_mask = numpy.random.uniform(0.0, 1.0, reduced_data.shape[0]) < 0.15
    training_data = reduced_data[training_mask, :]
    test_data = reduced_data[numpy.logical_not(training_mask), :]

    zeta = Preprocessor().fit(training_data)
    zeta_training_data = zeta.forward_transform(training_data)
    zeta_test_data = zeta.forward_transform(test_data)

    # autoencode with neural net (y = x)
    examples = [(x, x) for x in zeta_training_data]

    weights = [
        numpy.random.normal(0.0, 0.01, (n_hidden, n_input + 1)),
        numpy.random.normal(0.0, 0.01, (n_hidden, n_hidden + 1)),
        numpy.random.normal(0.0, 0.01, (n_hidden, n_hidden + 1)),
        numpy.random.normal(0.0, 0.01, (n_hidden, n_hidden + 1)),
        numpy.random.normal(0.0, 0.01, (n_hidden, n_hidden + 1)),
        numpy.random.normal(0.0, 0.01, (n_output, n_hidden + 1)),
    ]
    # zero the initial bias weights
    for i in xrange(len(weights)):
        weights[i][:, -1] = 0.0
    
    net = sparse_ae.Net(
        map(lambda x : x.shape, weights),
        lmbda = 1.0e-6,
        beta = 1.0e-3,
        rho = 0.05,
        examples = examples,
        example_weights = example_weights,
        verbose = True,
    )

    def f_and_grad_f(flat_w):
        w = net.unflatten_weights(flat_w)
        f, grad_f = net.evaluate_objective_and_gradient(w)
        return (f, net.flatten_weights(grad_f))

    print 'minimising objective with l-bfgs'
    w_opt, obj_opt = sparse_ae.lbfgs(
        f_and_grad_f,
        net.flatten_weights(weights),
        max_iters = 100,
    )

    w_opt = net.unflatten_weights(w_opt)
    print 'obj_opt : %s' % str(obj_opt)
    obj_opt_check, _ = net.evaluate_objective_and_gradient(w_opt)
    print 'obj_opt check #1 : %s' % str(obj_opt_check)
    check_predict = net.predict(w_opt, zeta_training_data)
    check_net_square_error = 0.0
    for i in xrange(len(check_predict)):
        check_net_square_error += numpy.sum((check_predict[i] - zeta_training_data[i]) ** 2)
    obj_check = net.obj_from_net_square_error(w_opt, check_net_square_error, sparsity_penalty_term = 0.0)
    print 'obj_opt check #2 : %s' % str(obj_check)

    def compute_rmsle(x, y):
        assert len(x) == len(y)
        r = 0.0
        for i in xrange(len(x)):
            r += numpy.sum((x[i] - y[i]) ** 2)
        return (r / float(len(x))) ** 0.5


    print 'evaulating on training data'
    predicted_training_data = net.predict(w_opt, zeta_training_data)
    training_err = compute_rmsle(predicted_training_data, zeta_training_data)
    print '\ttraining RMSLE %e' % training_err
    print 'evaluating on test data'
    predicted_test_data = net.predict(w_opt, zeta_test_data)
    test_err = compute_rmsle(predicted_test_data, zeta_test_data)
    print '\ttest RMSLE %e' % test_err

    print 'evaulating on training data (UN-PREPROCESSED)'
    predicted_training_data = zeta.reverse_transform(net.predict(w_opt, zeta_training_data))
    training_err = compute_rmsle(predicted_training_data, training_data)
    print '\ttraining RMSLE %e' % training_err
    print 'evaluating on test data (UN-PREPROCESSED)'
    predicted_test_data = zeta.reverse_transform(net.predict(w_opt, zeta_test_data))
    test_err = compute_rmsle(predicted_test_data, test_data)
    print '\ttest RMSLE %e' % test_err

if __name__ == '__main__':
    main()
