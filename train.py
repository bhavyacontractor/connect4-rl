from helper import *

b = board()
g = game(b)
# states, actions, value = g.get_episode()
g.train(10000, 0.9)