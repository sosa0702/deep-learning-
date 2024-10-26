import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple
from Poker_Env import PokerEnv
import copy 

class PokerStateConverter:
    """Converts poker states into grid representations for CNN processing"""
    
    def __init__(self, grid_size=(7, 13)):
        self.grid_size = grid_size  # 7 rows (suits + extra), 13 columns (ranks)
        self.rank_map = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, 
                        '9':7, '10':8, 'J':9, 'Q':10, 'K':11, 'A':12}
        self.suit_map = {'♥':0, '♦':1, '♣':2, '♠':3}

    def state_to_grid(self, player_state: Dict, env: PokerEnv) -> np.ndarray:
        """Convert poker state to 3D grid (channels, height, width)"""
        # Initialize 3 channels: player cards, community cards, betting info
        grid = np.zeros((3, *self.grid_size))
        
        # Channel 0: Player's hand
        for card in player_state['hand']:
            rank, suit = card
            grid[0, self.suit_map[suit], self.rank_map[rank]] = 1
        
        # Channel 1: Community cards
        for card in env.community_cards:
            rank, suit = card
            grid[1, self.suit_map[suit], self.rank_map[rank]] = 1
        
        # Channel 2: Betting information
        # Normalize all values between 0 and 1
        max_chips = env.starting_chips * env.num_players
        grid[2, 4, :] = player_state['chips'] / max_chips  # Player chips
        grid[2, 5, :] = env.pot / max_chips  # Pot size
        grid[2, 6, :] = player_state['bet'] / max_chips  # Current bet
        
        return grid

class PokerGTDQN(nn.Module):
    """Enhanced GT-DQN with CNN+LSTM for poker state processing"""
    
    def __init__(self, grid_size: Tuple[int, int], action_size: int, hidden_size: int = 256):
        super(PokerGTDQN, self).__init__()
        
        self.grid_size = grid_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # CNN for spatial feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        conv_out_size = self._get_conv_output_size()
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=conv_out_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Regret tracking
        self.register_buffer('cumulative_regret', torch.zeros(action_size))
        self.register_buffer('average_strategy', torch.ones(action_size) / action_size)
        
    def _get_conv_output_size(self):
        """Calculate CNN output size"""
        x = torch.zeros(1, 3, *self.grid_size)
        return int(np.prod(self.conv_layers(x).shape))
    
    def forward(self, state, hidden=None):
        batch_size = state.size(0)
        
        # CNN feature extraction
        x = self.conv_layers(state)
        
        # LSTM processing
        x = x.unsqueeze(1)  # Add sequence dimension
        lstm_out, hidden = self.lstm(x, hidden)
        x = lstm_out[:, -1]  # Take last sequence output
        
        # Dueling architecture
        value = self.value(x)
        advantage = self.advantage(x)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values, hidden
        
    def update_regret(self, action, new_regret):
        """Update regret values and ensure they are bounded."""
        if action not in self.regret_values:
            self.regret_values[action] = 0
        
        # Update regret with new value
        self.regret_values[action] += new_regret
        
        # Apply bounding to regret values
        self.regret_values[action] = np.clip(self.regret_values[action], -self.regret_bound, self.regret_bound)

        # Optional: Return the updated regret for logging or further analysis
        return self.regret_values[action]
        
    def get_mixed_strategy(self):
        """Calculate mixed strategy probabilities from regret"""
        positive_regret = F.relu(self.cumulative_regret)
        sum_regret = positive_regret.sum()
        if sum_regret > 0:
            strategy = positive_regret / sum_regret
        else:
            strategy = torch.ones_like(positive_regret) / len(positive_regret)
        return strategy

class GTDQNAgent:
    def __init__(self, grid_size, action_size, hidden_size=256, memory_size=10000):
        # Network and state processing parameters
        self.grid_size = grid_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Initialize state converter
        self.state_converter = PokerStateConverter(grid_size)  # Instantiate here
        
        # Initialize neural networks
        self.policy_net = PokerGTDQN(grid_size, action_size, hidden_size)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=memory_size)
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # LSTM state handling
        self.hidden_state = None
        
        # Strategy tracking
        self.strategy_profile = {}  # Information set -> strategy mapping
        self.minimum_regret = 1e-6  
        
    def reset_episode(self):
        """Reset LSTM hidden state for new episode"""
        self.hidden_state = None
        
    def get_state_key(self, state, env):
        """Create a unique key for the current game state"""
        # Include player's hand, community cards, and betting information
        hand = ','.join(f"{r}{s}" for r, s in state['hand'])
        community = ','.join(f"{r}{s}" for r, s in env.community_cards)
        return f"{hand}|{community}|{state['chips']}|{state['bet']}|{env.pot}"
    
    def act(self, state, env, training=True):
        """Select action using epsilon-greedy policy with GT considerations"""
        # Epsilon-greedy exploration during training
        if training and random.random() < self.epsilon:
            return random.choice(env.get_valid_actions(state))
        
        # Convert state to tensor format
        state_grid = self.state_converter.state_to_grid(state, env)  # Use self.state_converter
        state_tensor = torch.FloatTensor(state_grid).unsqueeze(0)
        
        # Get Q-values and update hidden state
        with torch.no_grad():
            q_values, self.hidden_state = self.policy_net(state_tensor, self.hidden_state)
            
        # Get valid actions and their Q-values
        valid_actions = env.get_valid_actions(state)
        valid_q = q_values[0][valid_actions]
        
        # Calculate action probabilities using softmax
        action_probs = F.softmax(valid_q / 0.1, dim=0)  # Temperature parameter of 0.1
        
        # Update strategy profile
        state_key = self.get_state_key(state, env)
        self.strategy_profile[state_key] = action_probs.numpy()
        
        # Select action with highest Q-value
        return valid_actions[valid_q.argmax().item()]

    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def train(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(states))
        action_batch = torch.LongTensor(actions)
        reward_batch = torch.FloatTensor(rewards)
        next_state_batch = torch.FloatTensor(np.array(next_states))
        done_batch = torch.FloatTensor(dones)
        
        # Get current Q values
        current_q_values, _ = self.policy_net(state_batch)
        current_q = current_q_values.gather(1, action_batch.unsqueeze(1))
        
        # Get next Q values from target net
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_state_batch)
            next_q = next_q_values.max(1)[0]
            
        # Calculate target Q values
        target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
        
        # Calculate loss and update policy network
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update regret tracking in policy network
        self.policy_net.update_regret(current_q.mean(), current_q_values)
        
    def update_target_network(self):
        """Update target network with policy network's weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def get_mixed_strategy(self, state_key):
        """Get mixed strategy for a given state"""
        if state_key in self.strategy_profile:
            return self.strategy_profile[state_key]
        return np.ones(self.action_size) / self.action_size  # Uniform strategy if state not seen
        
    def save(self, path):
        """Save model weights and strategy profile"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'strategy_profile': self.strategy_profile
        }, path)
        
    def load(self, path):
        """Load model weights and strategy profile"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.strategy_profile = checkpoint['strategy_profile']
