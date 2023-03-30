import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class board:
    def __init__(self, r=6, c=7):
        self.rows = r
        self.cols = c
        self.state = []
        self.game_end = False
        for i in range(r):
            temp = []
            for j in range(c):
                temp.append(0)
            self.state.append(temp)

    def reset(self):
        self.state = []
        self.game_end = False
        for i in range(self.rows):
            temp = []
            for j in range(self.cols):
                temp.append(0)
            self.state.append(temp)

    def check_valid(self, action):
        if self.state[self.rows - 1][action] == 0:
            return True
        else:
            return False

    def move(self, player, action):
        temp = 0
        while self.state[temp][action] != 0:
            temp += 1
        self.state[temp][action] = player
        self.check_end()

    def check_end(self):
        for i in range(self.rows):
            for j in range(self.cols - 3):
                if((self.state[i][j] == self.state[i][j+1]) and (self.state[i][j] == self.state[i][j+2]) and (self.state[i][j] == self.state[i][j+3]) and ((self.state[i][j] == 1) or (self.state[i][j] == 2))):
                    self.game_end = True
                    return
                
        for i in range(self.rows - 3):
            for j in range(self.cols):
                if((self.state[i][j] == self.state[i+1][j]) and (self.state[i][j] == self.state[i+2][j]) and (self.state[i][j] == self.state[i+3][j]) and ((self.state[i][j] == 1) or (self.state[i][j] == 2))):
                    self.game_end = True
                    return
                
        for i in range(self.rows - 3):
            for j in range(self.cols - 3):
                if((self.state[i][j] == self.state[i+1][j+1]) and (self.state[i][j] == self.state[i+2][j+2]) and (self.state[i][j] == self.state[i+3][j+3]) and ((self.state[i][j] == 1) or (self.state[i][j] == 2))):
                    self.game_end = True
                    return

        for i in range(self.rows - 1, 2, -1):
            for j in range(self.cols - 3):
                if((self.state[i][j] == self.state[i-1][j+1]) and (self.state[i][j] == self.state[i-2][j+2]) and (self.state[i][j] == self.state[i-3][j+3]) and ((self.state[i][j] == 1) or (self.state[i][j] == 2))):
                    self.game_end = True
                    return
                        
        self.game_end = False
    
    def get_action_and_move(self, player): 
        temp = []
        for i in range(self.cols):
            temp.append(i)
        action = random.choice(temp)

        while not self.check_valid(action):
            action = random.choice(temp)
        
        self.move(player, action)

        return action
    
class network(nn.Module):
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
    
    # def validation_step(self, batch):
    #     images, labels = batch 
    #     out = self(images)                    # Generate predictions
    #     loss = F.cross_entropy(out, labels)   # Calculate loss
    #     acc = accuracy(out, labels)           # Calculate accuracy
    #     return {'val_loss': loss, 'val_acc': acc}
        
    # def validation_epoch_end(self, outputs):
    #     batch_losses = [x['val_loss'] for x in outputs]
    #     epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    #     batch_accs = [x['val_acc'] for x in outputs]
    #     epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    #     return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    # def epoch_end(self, epoch, result):
    #     print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

    # @torch.no_grad()
    # def evaluate(self, val_loader):
    #     self.eval()
    #     outputs = [model.validation_step(batch) for batch in val_loader]
    #     return model.validation_epoch_end(outputs)

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
                optimizer.step()
                optimizer.zero_grad()
            
            # result = self.evaluate(val_loader)
            # result['train_loss'] = torch.stack(train_losses).mean().item()
            # model.epoch_end(epoch, result)
            # history.append(result)
            # print("Epoch: ", epoch, "Loss: ", train_losses)
        # print()
        return history

class game:
    def __init__(self, b):
        self.model = network(b.rows, b.cols)
        self.board = b

    def get_episode(self):
        self.board.reset()
        states = []
        actions = []
        i = 0
        while not self.board.game_end:
            temp = []
            for sublist in self.board.state:
                ls = []
                for item in sublist:
                    ls.append(item)
                temp.append(ls)
            states.append(temp)
            actions.append(self.board.get_action_and_move((i%2) + 1))
            flag = 0
            for j in range(self.board.cols):
                if(self.board.state[self.board.rows - 1][j] == 0):
                    flag = 1
                    break
            if(flag == 0):
                return states, actions, 0

            i += 1
        states.append(self.board.state)
        # for state in states:
        #     print(state)

        if i%2 == 0:
            return states, actions, 2
        else:
            return states, actions, 1

    def train(self, episodes):
        for i in range(episodes):
            states, actions, player = self.get_episode()

            if player == 1:
                value1 = 100
                value2 = -100
            elif player == 2:
                value2 = 100
                value1 = -100
            elif player == 0:
                value1 = 0
                value2 = 0

            states_train1 = []
            actions_train1 = []
            states_train2 = []
            actions_train2 = []
            for j in range(1, len(actions) + 1):
                if(j%2 == 0):
                    states_train2.append(states[j-1])
                    actions_train2.append(actions[j-1])
                else:
                    states_train1.append(states[j-1])
                    actions_train1.append(actions[j-1])
            
            self.model.fit(10, 0.5, states_train1, actions_train1, value1)
            self.model.fit(10, 0.5, states_train2, actions_train2, value2)
            print(f"{i}th Episode Trained")
        torch.save(self.model.state_dict(), "./model.pt")

class play:
    def __init__(self, m):
        self.model = m
        self.board = board()
    
    def start(self):
        i = 1
        while not self.board.game_end:
            self.display()

            if(i%2 == 1):
                print("Your move: ", end="")
                action = int(input())
                print()
                self.board.move(1, action)
            else:
                values = []
                for j in range(self.board.cols):
                    flat_state = [item for sublist in self.board.state for item in sublist]
                    for k in range(self.board.cols):
                        flat_state.append(0)
                    flat_state[len(flat_state) - self.board.cols + j] = 1
                    flat_state = torch.Tensor(flat_state)
                    values.append(self.model(flat_state))
                    print(values)

                print(values)
                
                print("Model move: ", action)
                self.board.move(2, action)
            
            i += 1
        self.display()
        print("Game Ended")

    def display(self):
        for i in range(self.board.rows-1, -1, -1):
            for j in range(self.board.cols):
                print(self.board.state[i][j], end=" ")
            print()
        print()