import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
import scipy
import random

sigma = 0.3
tau = 0.5
l_list = list((0.1, 0.5, 1, 5))

def computeGaussian(X, mu, sigma) :
    return np.exp(-0.5 * np.transpose(X - mu) @ np.linalg.inv(sigma) @ (X - mu)) / ((2 * pi) * np.linalg.det(sigma)**(0.5))

def kernel(xi, xj, sigma, l) :
    return sigma**2 * np.exp(-np.dot(xi - xj, xi - xj) / l**2)

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

def f(w, x) :
    return w[0] * x + w[1]

# First generate some data
Ntargets = 201
a = np.array([1.5, 1])
xData = np.linspace(-2, 2, Ntargets)
targets = f(a, xData)
targetsWithNoise = targets + np.random.multivariate_normal(np.zeros(Ntargets), sigma * np.identity(Ntargets))

ones = np.ones(Ntargets)

xVec = np.column_stack((xData, ones))

gram = compute_gram_matrix(xData, sigma, l)

muPrior = np.zeros(Ntargets)
covPrior = gram

for l in l_list :
    for i in range(0, 10) :
        covPrior = compute_gram_matrix(xData, sigma, l)
        randoms = np.random.multivariate_normal(muPrior, covPrior)
        
        pb.plot(xData, randoms)
        pb.title("l = " + str(l))

    pb.show()
    pb.savefig("l=" + str(l) + ".eps")
