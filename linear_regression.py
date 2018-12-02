import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
import scipy
import random

sigma = 1.0
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
# pb.show()
pb.savefig("prior.eps")
pb.clf()

##################
# Posterior
##################

## Sample random points
N = 6
xPoints = np.zeros((N, 1))
yPoints = np.zeros((N, 1))

randIdx = (-105, -30, 170, 50, 80, 10)
i = 0

for idx in randIdx :
    randx = xData[idx]
    target = targetsWithNoise[idx]

    xPoints[i, 0] = randx
    yPoints[i, 0] = target

    i += 1

# Compute posterior
for i in range(0, N) :
    xPosterior = np.array([[xPoints[i, 0]], [1]])
    xDot = xPosterior @ np.transpose(xPosterior)

    covPosterior = np.linalg.inv(np.linalg.inv(covPrior) + (1/sigma) * (xDot))
    muPosterior = covPosterior @ (np.linalg.inv(covPrior) @ muPrior + (1/sigma) * xPosterior * yPoints[i, 0]) 

    x = y = np.linspace(-3, 3, num=Ngrid)
    X, Y = np.meshgrid(x, y)

    posterior = np.array([computeGaussian(np.array([x, y]), np.ravel(muPosterior), covPosterior) for x,y in zip(np.ravel(X), np.ravel(Y))])
    posterior = posterior.reshape(X.shape)

    pb.pcolormesh(X, Y, posterior)
    ax2 = pb.axes()
    ax2.set_aspect('equal', 'datalim')
    ax2.margins(0.1)
    pb.savefig("posterior_iteration_" + str(i) + ".eps")
    # pb.show()
    pb.clf()

    ## Draw samples from the posterior

    for j in range(0, 6) :
        w = np.random.multivariate_normal(np.ravel(muPosterior), covPosterior)
        y = f(w, xData)
        pb.plot(xData, y)

    pb.plot(xData, targetsWithNoise, "+", label="dataset")
    pb.plot(xPoints[0:i+1, 0], yPoints[0:i+1, 0], "bs", label="training data")
    pb.legend()
    pb.savefig("samples_iteration_" + str(i) + ".eps")
    # pb.show()
    pb.clf()

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