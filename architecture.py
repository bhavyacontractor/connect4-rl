from board import *

class network_linear(nn.Module):
    def __init__(self, r, c):
        super().__init__()
        self.rows = r
        self.cols = c
        self.linear1 = nn.Linear((r+1)*c, 24)
        self.linear2 = nn.Linear(24, 12)
        self.linear3 = nn.Linear(12, 6)
        self.linear4 = nn.Linear(6, 1)
        
    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out
    
    def training_step(self, state, value):
        out = self(state)                  
        loss = F.mse_loss(out, value) 
        return loss
    
    def fit(self, epochs, lr, states, actions, values, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(self.parameters(), lr)
        for epoch in range(epochs):
            # self.train()
            train_losses = []
            for state, action in zip(states, actions):
                flat_state = [item for sublist in state for item in sublist]
                for i in range(self.cols):
                    flat_state.append(0)
                # print(len(flat_state), len(flat_state) - self.cols + actions[action])
                # print(action, len(actions))
                flat_state[len(flat_state) - self.cols + action] = 1
                flat_state = torch.Tensor(flat_state)
                values = torch.Tensor([values])
                # print(values)
                loss = self.training_step(flat_state, values)
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
    
class network_linear(nn.Module):
    def __init__(self, r, c):
        super().__init__()
        self.rows = r
        self.cols = c
        self.linear1 = nn.Linear((r+1)*c, 24)
        self.linear2 = nn.Linear(24, 12)
        self.linear3 = nn.Linear(12, 6)
        self.linear4 = nn.Linear(6, 1)
        
    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out
    
    def training_step(self, state, value):
        out = self(state)                  
        loss = F.mse_loss(out, value) 
        return loss
    
    def fit(self, epochs, lr, states, actions, values, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(self.parameters(), lr)
        for epoch in range(epochs):
            # self.train()
            train_losses = []
            for state, action in zip(states, actions):
                flat_state = [item for sublist in state for item in sublist]
                for i in range(self.cols):
                    flat_state.append(0)
                # print(len(flat_state), len(flat_state) - self.cols + actions[action])
                # print(action, len(actions))
                flat_state[len(flat_state) - self.cols + action] = 1
                flat_state = torch.Tensor(flat_state)
                values = torch.Tensor([values])
                # print(values)
                loss = self.training_step(flat_state, values)
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