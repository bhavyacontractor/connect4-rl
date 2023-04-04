from helper import *

b = board()
model = network_linear(b.rows, b.cols)
model.load_state_dict(torch.load("./model_10000.pt", map_location=torch.device("cpu")))

p = play(model)
p.start()