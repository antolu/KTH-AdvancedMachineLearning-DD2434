import pylab as pb
import numpy as np
import scipy as sp
import scipy.optimize as opt
import math as m

sigma = 0.0001

def J(N, M) :
    ret = np.zeros((N, M))
    
    for i in range(0, N) :
        for j in range(0, M) :
            if i == j :
                ret[i, j] = 1
    
    return ret

def f(W, *args) :

    y, sigma, Jji, Jij = args

    W.reshape((10, 2))

    C = W @ W.T + sigma * np.identity(D)
    Cinv = np.linalg.inv(C)
    # return the value of the objective at x
    fw = N * D / 2 * m.log(2 * m.pi) + N / 2 * m.log(np.linalg.det(C)) + 1/2 * np.trace(Cinv @ y @ y.T)

    print(fw)
    return np.array([fw])

def dfx(W, *args) :
    y, sigma, Jji, Jij = args
    
    N = 200
    D = 10

    W = W.reshape((10, 2))

    C = W @ W.T + sigma * np.identity(D)
    Cinv = np.linalg.inv(C)
    dW = W @ Jji + Jij @ W.T

    d = N/2 * np.trace(Cinv @ dW) - 1/2 * np.trace(Cinv @ y @ y.T @ Cinv @ dW)
    arr = np.array([d])
    print(d)
    return arr
    # return the gradient of the objective at x

def f_nonlin(x) :
    vcol= np.sin(x) + x * np.cos(x)
    hcol = np.cos(x) + x * np.sin(x)
    return np.hstack((vcol, hcol))

def f_lin(x, A) :
    return x @ np.transpose(A)

######################
## Generate data
######################
N = 200
D = 10

A = np.random.normal(0, 1, size=20).reshape((10, 2))

x = np.linspace(0, 4*m.pi, N).reshape(-1, 1)
fnonlin = f_nonlin(x)
flin = f_lin(fnonlin, A)

Y = flin.T

# Plot original curve
# pb.plot(fnonlin[1, :], fnonlin[0, :])
# pb.show()
# exec(open("representation_learning.py").read())

######################
## Optimize W
######################

Jji = J(2, 10)
Jij = J(10, 2)

x0 = np.random.normal(0, 1, size=20)
# x0 = np.zeros((20))

W_star = opt.fmin_cg(f, x0, fprime=dfx, args=(Y, sigma, Jji, Jij))

W = W_star.reshape((10, 2))

x1 = np.zeros((N))
x2 = np.zeros((N))

for i in range(0, N) :
    y = Y[:, i].reshape(-1, 1)
    mu = np.zeros((10, 1))

    muPosterior = np.linalg.inv(W.T @ W + sigma * np.identity(2)) @ W.T @ (y - mu)
    covPosterior = sigma * np.linalg.inv(W.T @ W + sigma * np.identity(2))

    x = np.random.multivariate_normal(muPosterior.ravel(), covPosterior)

    x1[i] = x[0]
    x2[i] = x[1]

xx = Y.T @ W @ np.linalg.inv(W.T @ W)

pb.plot(fnonlin[:, 0], fnonlin[:, 1], label="Original curve")
pb.plot(x1, x2, label="Estimated curve")
# pb.plot(xx[:, 0], xx[:, 1], label="Estimated curve")
pb.legend()
pb.show()

# for i in range(0, N) :
