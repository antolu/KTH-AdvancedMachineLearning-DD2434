import pickle
from ex_2_3 import Node, load_params, load_sample, print_tree
from ex_2_3_tree_helper import Tree
import sys

sys.setrecursionlimit(1500)

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

with open(my_data_path + 'tree_params.pickle', 'rb') as handle:
    params = pickle.load(handle, encoding='latin1')

with open(my_data_path + 'tree_samples.pickle', 'rb') as handle:
    samples = pickle.load(handle, encoding='latin1')



"""
    Construct a tree with parameters from the loaded parameter dict.
"""
params_list = list(params.keys())
params_name = params_list[0]     
params = params[params_name]
root = load_params(params)



"""
    Load a matching sample into the tree.
"""
samples_name = params_name + '_sample_1'
sample = samples[samples_name]
load_sample(root, sample)



"""
Print the tree (not very sophisticated). Structure: nodename_parentname
"""
# print_tree(root)

"""
Print the tree with sample (not very sophisticated). Structure: nodename_parentname:sample
"""
# print_tree(root, print_sample = True)

"""
Use tree object:
"""

print("\nTree")

t = Tree()    
    

my_data_path = ''

with open(my_data_path + 'tree_params.pickle', 'rb') as handle:
    params = pickle.load(handle, encoding='latin1')

key = list(params.keys())[0]    
    
"""
Load params into tree
"""
t.load_params(params[key])
# t.print_tree()        

"""
Generate a random tree
"""
t.create_random_tree(3)
# t.print_tree() 

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
    # print(str(node.name) + " " + str(i) + " " + str(s))
    return s

for k in range(0, len(params_list)) : 

    # Build tree
    params_name = params_list[k]     
    parameters = params[params_name]
    root = load_params(parameters)

    # Load samples onto tree
    samples_name = params_name + '_sample_1'
    sample = samples[samples_name]
    load_sample(root, sample)

    s_dict = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}}

    root_decendants = root.descendants
    root_distr = root.cat

    # Calculate p(beta|X)
    ssum = 0.0
    for i in range(0, len(root_distr[0])) :
        s = root_distr[0][i] * rec(root, i, s_dict)
        print(s)
        ssum += s

    # Print it
    val = ssum
    print("p(beta|X) iter " + str(k + 1) + ": " + str(val))