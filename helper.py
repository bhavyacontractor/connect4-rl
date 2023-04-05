from architecture import *

class game:
    def __init__(self, b):
        self.teacher = network_linear()
        self.student = network_linear()
        self.teacher.load_state_dict(self.student.state_dict())
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

    def train(self, episodes, gamma):
        for i in tqdm(range(episodes), desc="Games played"):
            states, actions, player = self.get_episode()

            if player == 1:
                value1 = 1
                value2 = -1
            elif player == 2:
                value2 = 1
                value1 = -1
            elif player == 0:
                value1 = 0
                value2 = 0

            states_train1 = []
            actions_train1 = []
            values_train1 = []
            values_train2 = []
            states_train2 = []
            actions_train2 = []
            for j in range(1, len(actions) + 1):
                if(j%2 == 0):
                    states_train2.append(states[j-1])
                    actions_train2.append(actions[j-1])
                    if(j < len(actions)):
                        value = torch.Tensor(states[j])[None, :]
                        value = self.teacher(value).detach().numpy()
                        value = max(value)
                        values_train2.append(gamma*value)
                    else:
                        values_train2.append(value2)

                else:
                    states_train1.append(states[j-1])
                    actions_train1.append(actions[j-1])
                    values_train1.append(value1)
                    if(j < len(actions)):
                        value = torch.Tensor(states[j])[None, :]
                        value = self.teacher(value).detach().numpy()
                        value = max(value)
                        values_train1.append(gamma*value)
                    else:
                        values_train1.append(value1)
                
            self.student.fit(2, 0.001, states_train1, actions_train1, values_train1)
            # print([params.data for params in self.model.parameters()])
            self.student.fit(2, 0.001, states_train2, actions_train2, values_train2)
            # print(f"{i}th Episode Trained", end="\r")

            if(i%100 == 0):
                self.teacher.load_state_dict(self.student.state_dict())
        
        self.teacher.load_state_dict(self.student.state_dict())
        torch.save(self.teacher.state_dict(), "./model_10000.pt")

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
                state = torch.Tensor(self.board.state)[None, :]
                values = self.model(state).detach().numpy()
                # for j in range(self.board.cols):
                #     # flat_state = [item for sublist in self.board.state for item in sublist]
                #     # for k in range(self.board.cols):
                #     #     flat_state.append(0)
                #     # flat_state[len(flat_state) - self.board.cols + j] = 1
                #     flat_state = torch.Tensor(flat_state)
                #     values.append(self.model(flat_state).detach().numpy())
                #     # print(values)
                
                action = np.argmax(np.array(values))
                while self.board.state[self.board.rows - 1][action] != 0:
                    action = np.argmax(np.array(values))
                    values[action] = -1e9
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