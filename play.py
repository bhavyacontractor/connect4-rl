from helper import *

b = board()
model = network_linear(b.rows, b.cols)
model.load_state_dict(torch.load("./model.pt"))

p = play(model)
p.start()