from GTDQN import PokerStateConverter
from Poker_Env import PokerEnv
from typing import Tuple, Dict
import random 
from collections import deque 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import copy 
import numpy as np 

class DQNAgent:
    """Traditional DQN agent adapted for poker"""
    
    def __init__(self, grid_size: Tuple[int, int], action_size: int, hidden_size: int = 256):
        self.grid_size = grid_size
        self.action_size = action_size
        self.state_converter = PokerStateConverter(grid_size)
        
        # Networks
        self.policy_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size[0] * grid_size[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self.target_net = copy.deepcopy(self.policy_net)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def act(self, state: Dict, env: PokerEnv) -> str:
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(env.get_valid_actions(state))
            
        state_grid = self.state_converter.state_to_grid(state, env)
        state_tensor = torch.FloatTensor(state_grid).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            valid_actions = env.get_valid_actions(state)
            valid_q = q_values[0][valid_actions]
            return valid_actions[valid_q.argmax().item()]
    
    def train(self, batch):
        """Train on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        state_batch = torch.FloatTensor(np.array(states))
        action_batch = torch.LongTensor(actions)
        reward_batch = torch.FloatTensor(rewards)
        next_state_batch = torch.FloatTensor(np.array(next_states))
        done_batch = torch.FloatTensor(dones)
        
        current_q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q = self.target_net(next_state_batch).max(1)[0].detach()
        target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
        
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
