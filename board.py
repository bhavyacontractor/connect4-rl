from tqdm import tqdm
import numpy as np
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
    
    def get_action_and_move(self, player, last_move): 
        p = random.uniform(0, 1)

        if(p < 0.5):
            return (last_move+1)%(self.cols)
        
        temp = []
        for i in range(self.cols):
            temp.append(i)
        action = random.choice(temp)

        while not self.check_valid(action):
            action = random.choice(temp)
        
        self.move(player, action)

        return action