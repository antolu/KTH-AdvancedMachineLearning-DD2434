import pickle
import numpy as np
from forward_backward import *

def forwardbackward(observations, mus, beta) :
    players = list(observations.keys())

    gammas = {}

    for player in players : 
        sequences = observations[player]
        
        rounds = list(sequences.keys())
        gamma_per_round = {}

        for round_ in rounds :
            alpha, norm = forward(sequences[round_], mus, beta, player)
            bet, temp = backward(sequences[round_], mus, beta, player)
            gamma_per_round[round_-1] = alpha * bet
        
        gammas[player] = gamma_per_round
    
    return gammas

def get_gamma_sum(gammas, i, m, n, np) :
    gamma_sum = 0

    for r in range(R) :
        gammas[(n, np)]
        gamma_sum += gammas[(n, np)][r][m, i]

    return gamma_sum

def load_obj(name ):
    with open('./' + name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')
sequence_outputs = load_obj("sequence_output")

R = 10
N = 20
M = 30

# Initialize parameters
mus = {}
beta = 20
gammas = {}

for n in range(N) :
    mus[n] = {}
    for i in range(2) :
        for m in range(M) :
            mus[n][(i,m)] = 15+i*5

# Sort and count players
players = list(sequence_outputs.keys())

player_list = {}
player_list_inv = {}
for i in range(N) :
    player_list[i] = list()
    player_list_inv[i] = list()

for player in players : 
    player_list[player[0] - 1].append(player[1])
for player in players : 
    player_list_inv[player[1] - 1].append(player[0])

############################################################
# Iterate EM
############################################################

for j in range(100) :
    old_mus = mus
    oldbeta = beta
    print(j)
    gammas = forwardbackward(sequence_outputs, mus, beta)

    nom = 0
    denom = 0

    for i in range(2) :
        for m in range(M) :
            A = np.zeros((N, N))
            b = np.zeros(N)
            for n in range(N) : 
                gamma_n_np = np.zeros((N))

                for npr in player_list[n] :
                    gamma_sum = get_gamma_sum(gammas, i, m, n+1, npr)
                    gamma_n_np[npr-1] = gamma_sum
                    
                    A[n, npr-1] = gamma_sum * mus[npr-1][(i,m)]

                    # Calculate b for n(n, npr)
                    for r in range(R) :
                        b[n] += gammas[(n+1, npr)][r][m, i] * sequence_outputs[(n+1, npr)][r+1][m]
                    
                for npr in player_list_inv[n] :
                    gamma_sum = get_gamma_sum(gammas, i, m, npr, n+1)
                    gamma_n_np[npr-1] = gamma_sum
                    
                    A[n, npr-1] = gamma_sum * mus[npr-1][(i,m)]

                    # Calculate b for n(n, npr)
                    for r in range(R) :
                        b[n] += gammas[(npr, n+1)][r][m, i] * sequence_outputs[(npr, n+1)][r+1][m]
                
                A[n, n] = np.sum(gamma_n_np) * mus[n][(i,m)]

            solved_mus = np.linalg.solve(A, b)

            # Assign mus
            for n in range(N) :
                mus[n][(i,m)] = solved_mus[n]

        # Do some convergence checking
        for m in range(M) :
            for n in range(N) :
                for npr in range(N) :
                    for r in range(R) :
                        if (n, npr) in gammas :
                            nom += gammas[(n+1, npr+1)][r][m, i] * (sequence_outputs[(n+1, npr+1)][r+1][m] - mus[n][(i,m)] - mus[npr][(i,m)])**2 
                            denom += gammas[(n+1, npr+1)][r][m, i]
    
    beta = nom/(2*denom)

print(mus)