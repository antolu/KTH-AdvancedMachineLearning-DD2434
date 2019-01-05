import numpy as np
import math as m
import scipy.stats

def normal_distr(mu, sigma, x) :
    return 1/(m.sqrt(2*m.pi*sigma)) * m.exp(((x - mu)**2)/(2*sigma))

Atr = np.array([[1/4, 3/4], [3/4, 1/4]])

def get_transition(state1, m1, state2):
    # calculate A((r1, m1), (r2, m1+1)) (for test purpose we set below)
    return Atr[state1][state2]


def get_emission(state, m, emission, mus, beta, players):
    # calculate O(m, o) (for test purpose we set below)
    O = normal_distr(mus[players[0]-1][(state,m)] + mus[players[1]-1][(state,m)], beta, emission)
    return O


def get_init():
    # provide an array containing the initial state probability having size R (for test purpose we set below)
    pi = np.array([0.2, 0.8])
    # number of rows
    R = pi.shape[0]
    return pi, R


def forward(observations, mus, beta, players):
    pi, R = get_init()
    M = len(observations)
    alpha = np.zeros((M, R))

    # base case
    O = []
    for r in range(R):
        O.append(get_emission(r, 0, observations[0], mus, beta, players))
    alpha[0, :] = pi * O[:]

    # recursive case
    for m in range(1, M):
        for r2 in range(R):
            for r1 in range(R):
                transition = get_transition(r1, m, r2)
                emission = get_emission(r2, m, observations[m], mus, beta, players)
                alpha[m, r2] += alpha[m - 1, r1] * transition * emission


    return (alpha, np.sum(alpha[M - 1, :]))


def backward(observations, mus, bet, players):
    pi, R = get_init()
    M = len(observations)
    beta = np.zeros((M, R))

    # base case
    beta[M - 1, :] = 1

    # recursive case
    for m in range(M - 2, -1, -1):
        for r1 in range(R):
            for r2 in range(R):
                transition = get_transition(r1, m, r2)
                emission = get_emission(r2, m, observations[m + 1], mus, bet, players)
                beta[m, r1] += beta[m + 1, r2] * transition * emission

    O = []
    for r in range(R):
        O.append(get_emission(r, m, observations[0], mus, bet, players))

    return beta, np.sum(pi * O[:] * beta[0, :])



# test examples
# print(forward(get_init, get_transition, get_emission, [0, 0, 1, 1, 1, 1]))
# print(backward(get_init, get_transition, get_emission, [0, 0, 1, 1, 1, 1]))




