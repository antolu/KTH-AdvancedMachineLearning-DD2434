import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
import scipy
import random

sigma = 1.0
tau = 0.5
l_list = list((0.1, 0.5, 1, 5))
l = 0.5
beta = 3.5

def computeGaussian(X, mu, sigma) :
    return np.exp(-0.5 * np.transpose(X - mu) @ np.linalg.inv(sigma) @ (X - mu)) / ((2 * pi) * np.linalg.det(sigma)**(0.5))

# Compute kernel of xi, xj
def kernel(xi, xj, sigma, l) :
    return sigma**2 * np.exp(-np.dot(xi - xj, xi - xj) / l**2)

# Compute gram matrix
def compute_gram_matrix(x, sigma, l) : 
    N = x.shape[0]
    gram = np.zeros((N, N))

    i = 0
    j = 0

    while i < N :
        while j < N :
            val = kernel(x[i], x[j], sigma, l)
            gram[i, j] = val
            gram[j, i] = val

            j += 1
        i += 1
        j = i

    return gram

def compute_posterior(K, var) :
    N = K.shape[0]
    noise = var * np.identity(N)

    C = K + noise
    return C

# Returns mean and covariance of predictive posterior
# for vector x with unknown targets
def get_posterior_params(K, X, x, f) :
    N = K.shape[0]
    M = x.shape[0]
    KxX = np.zeros((M, N))
    
    i = 0
    j = 0

    while i < M :
        while j < N :
            val = kernel(X[j], x[i], sigma, l)
            KxX[i, j] = val
            j += 1

        i += 1
        j = 0
    
    c = compute_gram_matrix(x, sigma, l)

    mean = KxX @ np.linalg.inv(K) @ f
    variance = c - KxX @ np.linalg.inv(K) @ np.transpose(KxX)

    return mean, variance

# Linear regressor
def f(w, x) :
    return w[0] * x + w[1]

def g(x) :
    N = x.shape[0]
    return h(x) + np.random.multivariate_normal(np.zeros(N), beta * np.identity(N))

# function without noise
def h(x) :
    return (2 + (0.5 * x - 1)**2) * np.sin(3 * x)

# First generate some data
Ntargets = 201
a = np.array([1.5, 1])
xData = np.linspace(-2, 2, Ntargets)
targets = f(a, xData)
targetsWithNoise = targets + np.random.multivariate_normal(np.zeros(Ntargets), sigma * np.identity(Ntargets))

ones = np.ones(Ntargets)

xVec = np.column_stack((xData, ones))

muPrior = np.zeros(Ntargets)

## Plot samples of prior with different length scales

for L in l_list :
    covPrior = compute_gram_matrix(xData, sigma, L)
    for i in range(0, 10) :
        randoms = np.random.multivariate_normal(muPrior, covPrior)
        
    #     pb.plot(xData, randoms)
    #     pb.title("l = " + str(l))

    # pb.show()
    # pb.savefig("l=" + str(l) + ".eps")

#############################################
# Generate some more non-linear data
#############################################

X = np.linspace(-4, 6, 9)
nonlinear_targets = g(X)

###########################
# Sample from posterior
###########################

x_test = np.linspace(-10, 10, 100)

gram = compute_gram_matrix(X, sigma, l)

mean, variance = get_posterior_params(gram, X, x_test, nonlinear_targets.reshape(-1, 1))
f = np.random.multivariate_normal(np.ravel(mean), variance)

pb.plot(x_test, f, label="Test data")

# Plot training points
pb.plot(X, nonlinear_targets, "s", label="Training data")

# Plot means
pb.plot()

# Plot variance
pb.gca().fill_between(x_test.flat, np.ravel(mean) - 2 * np.diag(variance), np.ravel(mean) + 2 * np.diag(variance), color="#dddddd")

## Plot line of g(x) for reference 
# x = np.linspace(-10, 10, 1000)
# y = h(x)
# pb.plot(x, y)

pb.legend()
pb.show()

########################
# Plot posterior
########################

for i in range(0, 5) :
    randoms = np.random.multivariate_normal(np.zeros((9)), gram)
    pb.plot(X, randoms)

pb.plot(X, nonlinear_targets, "s", label="Training data")
pb.title("Priors")
pb.legend()
pb.show()