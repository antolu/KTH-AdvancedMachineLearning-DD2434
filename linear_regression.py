import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
import scipy
import random

sigma = 0.3
tau = 0.5

def computeGaussian(X, mu, sigma) :
    return np.exp(-0.5 * np.transpose(X - mu) @ np.linalg.inv(sigma) @ (X - mu)) / ((2 * pi) * np.linalg.det(sigma)**(0.5))

def f(w, x) :
    return w[0] * x + w[1]

# First generate some data
Ntargets = 201
a = np.array([1.5, 1])
xData = np.linspace(-2, 2, Ntargets)
targets = f(a, xData)
targetsWithNoise = targets + np.random.multivariate_normal(np.zeros(Ntargets), sigma * np.identity(Ntargets))

#################
# Prior
#################

Ngrid = 100
muPrior = np.array([[0], [0]])
covPrior = tau * np.identity(2)

w0 = w1 = np.linspace(-2, 2, num=Ngrid)
W0, W1 = np.meshgrid(w0, w1)
zs = np.array([computeGaussian(np.array([x, y]), np.ravel(muPrior), covPrior) for x,y in zip(np.ravel(W0), np.ravel(W1))])
prior = zs.reshape(W0.shape)


pb.pcolormesh(W0, W1, prior)
ax1 = pb.axes()
ax1.set_aspect('equal', 'datalim')
ax1.margins(0.1)
pb.show()

##################
# Posterior
##################

for i in range(0, 6) :
    ## Sample point from generated data
    randIdx = random.randrange(0, Ntargets)
    randx = xData[randIdx]
    target = targetsWithNoise[randIdx]

    xPosterior = np.array([[randx], [1]])
    xDot = xPosterior @ np.transpose(xPosterior)
    # print("random point")
    # print(xPosterior)
    # print(xDot)

    covPosterior = np.linalg.inv(np.linalg.inv(covPrior) + (1/sigma) * (xDot))
    muPosterior = covPosterior @ (np.linalg.inv(covPrior) @ muPrior + (1/sigma) * xPosterior * target) 
    
    # + (1/sigma) * target * np.linalg.inv( + (1/sigma) * (xDot)) @ xPosterior

    x = y = np.linspace(-3, 3, num=Ngrid)
    X, Y = np.meshgrid(x, y)

    posterior = np.array([computeGaussian(np.array([x, y]), np.ravel(muPosterior), covPosterior) for x,y in zip(np.ravel(X), np.ravel(Y))])
    posterior = posterior.reshape(X.shape)

    pb.pcolormesh(X, Y, posterior)
    ax2 = pb.axes()
    ax2.set_aspect('equal', 'datalim')
    ax2.margins(0.1)
    pb.show()

    ## Draw samples from the posterior

    for i in range(0, 6) :
        w = np.random.multivariate_normal(np.ravel(muPosterior), covPosterior)
        y = f(w, xData)
        pb.plot(xData, y)

    pb.plot(xData, targetsWithNoise, "+")
    pb.show()

    covPrior = covPosterior
    muPrior = muPosterior

    print("mu")
    print(muPrior)
    print("covariance")
    print(covPrior)


# # To sample from a multivariable Gaussian
# f = np.random.multivariate_normal(mu, K)

# # To compute a distance matrix between two sets of vectors
# D = cdist(x1, x2)

# # To compute the exponential of all elements in a matrix
# E = np.exp(D)