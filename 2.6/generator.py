import pickle
import random
import itertools
import numpy as np
from collections import defaultdict


#Variables you need
M = 30
N = 20
R = 10
beta = np.pi
sigma = 5
# mean for each subfield with respect to the player
mu = np.array([np.random.normal(10, sigma, (M,N)), 
               np.random.normal(23, sigma, (M,N))])
print ("Mean matrix shape: ",mu.shape)


#Function to save dictionary in a pickle file
def save_obj(obj, name ):
    with open('./'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def generate_observations(): 
    '''
    Function to generate observations at each subfield with respect to each player
    '''
    observations = np.zeros((2,M,N))
    for i in range(2):
        for j in range(M):
            for k in range(N):
                observations[i,j,k] = np.random.normal(mu[i,j,k], beta, 1)
    return observations

def generate_data(observations):            
    '''
    Function to generate tournament data
    '''
    # Generate Players List and list of pairs
    players_list = [i+1 for i in range (N)]
    players_combinations = list(itertools.combinations(players_list, 2))
    
    output_sequences = defaultdict(dict)
    for combination in players_combinations:
        output_sequences[combination][i+1] = []
        for i in range(R):
            sequence = []
            table_idx = 0     #If we are on left side of subfield or on right side
            for j in range(M):
                choice = random.choice([1/float(4),3/float(4)])
                if choice == 3/float(4):
                    table_idx = 1 - table_idx
                sequence.append(observations[table_idx][j][combination[0]-1]+observations[table_idx][j][combination[1]-1])

            output_sequences[combination][i+1] = sequence
    return output_sequences

if __name__ == '__main__':
    
    observations = generate_observations()
    output_sequences = generate_data(observations)
    #Save the dictionary
    #save_obj(output_sequences,"sequence_output")