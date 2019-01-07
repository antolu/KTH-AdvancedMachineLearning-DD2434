import pickle
from ex_2_3 import Node, load_params, load_sample, print_tree
from helper import Tree
import numpy as np

"""
    The data is stored in Newick form:
        [A,B,[C,D]E]F;
        
           ________A 
          |
        F |________B
          |          ________C
          |         |
          |________E
                    |________D
                    
    But we are working in python. So we will work with lists that look
    the following way:
        
    Tree  = ['F', pF, ['A', pA, [], 'B', pB, [], 'E', pE, ['C', pC, [], 'D', pD, []]]]
        
    Each variable has a name (in string format), a list of categorical parameters,
    e.g. pF = [0.3, 0.2, 0.5], and a list of child nodes.
"""






"""
  Load the parameters of each tree and three samples from it.
  The data is stored in a dictionary. 
  
  The key indicates
      k := the number of categorical variables
      md:= maximum depth of the generated tree
      mb:= maximum branching factor of each node
      alpha:= the alpha values used to draw the parameters from a Dirichlet
  
  None of these numbers is important for your implementation but cab be 
  used for interpretation.
"""


my_data_path = ''

with open(my_data_path + 'tree_with_CPD.pkl', 'rb') as handle:
    params = pickle.load(handle, encoding='latin1')

with open(my_data_path + 'tree_with_leaf_samples.pkl', 'rb') as handle:
    samples = pickle.load(handle, encoding='latin1')



"""
    Construct a tree with parameters from the loaded parameter dict.
"""
params_list = params
root = load_params(params)



"""
    Load a matching sample into the tree.
"""
load_sample(root, samples)



"""
Print the tree (not very sophisticated). Structure: nodename_parentname
"""
# print_tree(root)

"""
Print the tree with sample (not very sophisticated). Structure: nodename_parentname:sample
"""
print_tree(root, print_sample = True)

print()

####################################################################################################
#
# Own code 
#
####################################################################################################

def rec(node, i, s_dict) :
    decendants = node.descendants
    distr = node.cat
    # print("node sample: " + str(node.sample))

    # Check if leaf
    if decendants == [] :
        if node.sample == i :
            return 1
        else :
            return 0

    s = 1
    if node in s_dict[i] :
        s = s_dict[i][node]
    else :
        for dec in decendants :

            ssum = 0

            for j in range(0, len(dec.cat[i])) :
                ssum += dec.cat[i][j] * rec(dec, j, s_dict)
            
            s *= ssum

        s_dict[i][node] = s

    return s

def compute_posterior(node, s_dict, p_dict) :
    decendants = node.descendants
    distr = node.cat

    # Check if leaf
    if decendants == [] :
        return

    for j in range(0, len(distr)) :
        if node in p_dict[j] :
            continue
        else : # compute posterior
            p = 0
            s = s_dict[j][node]
            for i in range(0, len(distr[j])) :
                cat = distr[j][i] 
                p += s * cat
            p_dict[j][node] = p

        for dec in decendants :
            compute_posterior(dec, s_dict, p_dict)
    return

def normalize_posterior(node, p_dict) :
    decendants = node.descendants
    distr = node.cat

    # Check if leaf
    if decendants == [] :
        return

    # determine normalizer
    denom = 0
    for i in range(0, len(distr)) :
        denom += p_dict[i][node]

    # normalize
    for i in range(0, len(distr)) :
        p_dict[i][node] = p_dict[i][node] / denom

    # normalize children
    for dec in decendants :
        normalize_posterior(dec, p_dict)

def sample(node, p_dict, samples) :
    descendants = node.descendants
    distr = node.cat

    if descendants == [] :
        return

    p_cat = list()
    for i in range(0,len(distr[0])) : 
        p_cat.append(p_dict[i][node])

    node.sample = np.random.multinomial(1, p_cat)[0]

    for dec in descendants : 
        sample(dec, p_dict, samples)

# only one tree provided
# root, params, sample already loaded

s_dict = {0:{}, 1:{}}
p_dict = {0:{}, 1:{}}

root_decendants = root.descendants
root_distr = root.cat

# Calculate p(beta|X)
ssum = 0.0
for i in range(0, len(root_distr[0])) :
    s = root_distr[0][i] * rec(root, i, s_dict)
    s_dict[i][root] = s
    ssum += s

# Print it
val = ssum
print("Value" + str(val))

####################################################################################################
#
# Posterior
#
####################################################################################################

# root
denom = 0
for k in range(0, 2) :
    s = s_dict[k][root]
    posterior = s * root_distr[0][k]
    denom += posterior
    p_dict[k][root] = posterior

# normalize root posterior 
for j in range(0, 2) :
    p_dict[j][root] = p_dict[j][root] / denom


for dec in root.descendants :
    # Calculate values for the rest of the nodes
    compute_posterior(dec, s_dict, p_dict)

for dec in root.descendants :
    # Normalize values for the rest of the nodes
    normalize_posterior(dec, p_dict)

for dec in root.descendants :
    for j in range(0, 2) :
        print(p_dict[j][dec])

samples = {}

sample(root, p_dict, samples)

print_tree(root, print_sample = True)