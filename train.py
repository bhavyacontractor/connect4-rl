from helper import *

b = board()
g = game(b)
# states, actions, value = g.get_episode()
g.train(20000, 0.999)