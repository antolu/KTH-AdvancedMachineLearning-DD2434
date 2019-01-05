import pickle

def load_obj(name ):
    with open('./' + name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')
sequence_outputs = load_obj("sequence_output")

R = 10
N = 20
M = 30

players = list(sequence_outputs.keys())

# print(players)
# 190 players

sequence = sequence_outputs[players[0]]

# sequence : dict
# key: r
# M long sequence for each r

print(len(sequence[1]))