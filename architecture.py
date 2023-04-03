from board import *

class network_linear(nn.Module):
    def __init__(self, r, c):
        super().__init__()
        # input wil be the 6 x 7 board
        # output will be a vector of size 7 giving q-values of for each action
        self.rows = r
        self.cols = c
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        self.flatten = nn.Flatten(start_dim=0, end_dim=2)

        linear_input_size = self.rows * self.cols * 32
        self.linear1 = nn.Linear(linear_input_size, 50)
        self.linear2 = nn.Linear(50, 50)
        self.linear3 = nn.Linear(50, 50)
        self.linear4 = nn.Linear(50, 7)

        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out))
        out = F.leaky_relu(self.conv5(out))
        out = F.leaky_relu(self.conv6(out))
        out = F.leaky_relu(self.conv7(out))

        out = self.flatten(out)
        out = F.leaky_relu(self.linear1(out))
        out = F.leaky_relu(self.linear2(out))
        out = F.leaky_relu(self.linear3(out))
        out = self.linear4(out)
        return out
    
    def training_step(self, state, value):
        out = self(state)                  
        loss = F.mse_loss(out, value) 
        return loss
    
    def fit(self, epochs, lr, states, actions, values_train, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(self.parameters(), lr)
        for epoch in range(epochs):
            # self.train()
            train_losses = []
            for state, action, value in zip(states, actions, values_train):
                # flat_state = [item for sublist in state for item in sublist]
                temp = []
                for i in range(self.cols):
                    temp.append(0)
                temp[action] = value
                # print(len(flat_state), len(flat_state) - self.cols + actions[action])
                # print(action, len(actions))
                # flat_state[len(flat_state) - self.cols + action] = 1
                flat_state = torch.Tensor(state)
                flat_state = flat_state[None, :]
                # print("Flat_state shape", flat_state.shape)
                # value = torch.Tensor([value])
                # print(value)
                target = torch.Tensor(temp)
                loss = self.training_step(flat_state, target)
                # print(loss)
                train_losses.append(loss)
                loss.backward()
                nn.utils.clip_grad_value_(self.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()
            
            # result = self.evaluate(val_loader)
            # result['train_loss'] = torch.stack(train_losses).mean().item()
            # model.epoch_end(epoch, result)
            # history.append(result)
            # print("Epoch: ", epoch, "Loss: ", torch.stack(train_losses).mean().item())
        # print()
        return history