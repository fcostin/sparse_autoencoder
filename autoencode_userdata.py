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

def fit_preprocessing_transform(training_data):
    mu = numpy.mean(training_data, axis = 0)
    sigma = numpy.std(training_data, axis = 0)
    def preprocessing_transform(d):
        d = phi(training_data)
        d = (d - mu[numpy.newaxis, :]) / (sigma[numpy.newaxis, :])
        return d
    return preprocessing_transform

def main():

    n_input = 100 # number of input nodes
    n_hidden = 25 # number of hidden nodes
    n_output = 100

    usr_data = get_data(n_input)
    # reduced_data, example_weights = reduce_data(usr_data)
    reduced_data = usr_data
    example_weights = None
    print 'reduced feature data set has shape %s' % str(reduced_data.shape)

    # thin dataset to speed up code testing
    # XXX this should be weighted by example weight for reduced data
    training_mask = numpy.random.uniform(0.0, 1.0, reduced_data.shape[0]) < 0.2
    training_data = reduced_data[training_mask, :]
    testing_data = reduced_data[numpy.logical_not(training_mask), :]

    zeta = fit_preprocessing_transform(training_data)
    training_data = zeta(training_data)
    testing_data = zeta(testing_data)

    # autoencode with neural net (y = x)
    examples = [(x, x) for x in training_data]

    weights = [
        numpy.random.normal(0.0, 0.01, (n_hidden, n_input + 1)),
        numpy.random.normal(0.0, 0.01, (n_output, n_hidden + 1)),
    ]
    
    net = sparse_ae.Net(
        map(lambda x : x.shape, weights),
        lmbda = 1.0,
        beta = 1.0,
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
    w_opt, obj_opt = sparse_ae.lbfgs(f_and_grad_f, net.flatten_weights(weights))
    print 'obj_opt : %s' % str(obj_opt)

if __name__ == '__main__':
    main()
