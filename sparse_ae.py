import numpy

def sigmoid(z):
    """
    n.b. gradient of sigmoid(z) is (sigmoid(z)) * (1 - sigmoid(z)) -- nice!
    """
    return 1.0 / (1.0 + numpy.exp(-z))

def test_sigmoid():
    # sanity check identity for sigmoid derivative
    f = sigmoid
    z = numpy.linspace(-1, 1, 101)
    h = 1.0e-4
    gf = (f(z + h) - f(z - h)) / (2.0 * h)
    gf2 = f(z) * (1.0 - f(z))
    r = numpy.linalg.norm(gf - gf2, ord = numpy.inf)
    assert r < 1.0e-8

def psi(x, w):
    """
    psi makes linear combinations of x using weights w (with bias)
    inputs:
        x : shape (m, ) input vector
        w : shape (n, m + 1) weight matrix (m+1th weight is for implicit bias node)
    output:
        z : shape (n, ) vector
    """
    b = w[:, -1]
    y = numpy.dot(w[:, :-1], x)
    return b + y

def phi(x, w):
    """
    phi evaluates a layer of neurons using inputs x and weights w
    """
    return sigmoid(psi(x, w))

def grad_phi(x, w):
    """
    evaluates grad_w phi(x; w)

    inputs:
        x : shape (m, ) input vector
        w : shape (n, m + 1) weight matrix
    output:
        g : shape (n, n, m + 1) matrix of partial derivatives
    """
    # compute sigmoid' psi using identity sigmoid' = sigmoid*(1-sigmoid)
    y = phi(x, w)
    grad_sigmoid_psi = y * (1.0 - y)
    n, m_plus_one = w.shape
    g = numpy.zeros((n, n, m_plus_one))
    for i in xrange(n):
        g[i, i, :] = grad_sigmoid_psi[i] * numpy.hstack((x, 1))
    return g

def obj_j(x, y, w):
    return 0.5 * ((phi(x, w) - y) ** 2)

def grad_obj_j(x, y, w):
    return (phi(x, w) - y)[:, numpy.newaxis, numpy.newaxis] * grad_phi(x, w)

def test_gradient_code():
    test_sigmoid()
    
    m = 13
    n = 7

    x = numpy.random.uniform(-1.0, 1.0, (m, ))
    y = numpy.random.uniform(-1.0, 1.0, (n, ))
    w = numpy.random.normal(0.0, 1.0, (n, m + 1))

    f = obj_j(x, y, w)
    g = grad_obj_j(x, y, w)

    eps = 1.0e-6
    
    approx_g = numpy.zeros((n, n, m + 1))
    for i in xrange(n):
        for j in xrange(m + 1):
            h = numpy.zeros((n, m + 1))
            h[i, j] = 1.0
            approx_g[:, i, j] = (obj_j(x, y, w + (eps * h)) - obj_j(x, y, w - (eps * h))) / (2.0 * eps)

    print 'g versus g~'
    print '||g~|| = %.3f' % numpy.linalg.norm(approx_g.ravel(), ord = numpy.inf)
    print '||g|| = %.3f' % numpy.linalg.norm(g.ravel(), ord = numpy.inf)
    print '||g~ - g|| = %e' % numpy.linalg.norm((approx_g - g).ravel(), ord = numpy.inf)

    for i in xrange(n):
        print (g / approx_g)[i, i, 0]


    gp = grad_phi(x, w)
    approx_gp = numpy.zeros((n, n, m + 1))
    for i in xrange(n):
        for j in xrange(m + 1):
            h = numpy.zeros((n, m + 1))
            h[i, j] = 1.0
            approx_gp[:, i, j] = (phi(x, w + (eps * h)) - phi(x, w - (eps * h))) / (2.0 * eps)
    print 'gp versus gp~'
    print '||gp~|| = %.3f' % numpy.linalg.norm(approx_gp.ravel(), ord = numpy.inf)
    print '||gp|| = %.3f' % numpy.linalg.norm(gp.ravel(), ord = numpy.inf)
    print '||gp~ - gp|| = %e' % numpy.linalg.norm((approx_gp - gp).ravel(), ord = numpy.inf)
    
def main():
    test_gradient_code()

if __name__ == '__main__':
    main()
