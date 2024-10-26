from Poker_Env import PokerEnv
import math 
from typing import Dict
from collections import defaultdict
import random
import numpy as np
import copy 


class MCTSNode:
    """Monte Carlo Tree Search node for CFR agent"""
    
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_actions = None
        self.ucb_constant = 2.0
        
    def ucb_score(self, parent_visits):
        """Calculate UCB score for node selection"""
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits + 
                self.ucb_constant * math.sqrt(math.log(parent_visits) / self.visits))
        
    def expand(self, action, next_state):
        """Add a child node"""
        child = MCTSNode(next_state, parent=self)
        self.children[action] = child
        return child

class CFRAgent:
    """Advanced CFR agent with MCTS integration"""
    
    def __init__(self, env: PokerEnv):
        self.env = env
        self.regret_sum = defaultdict(lambda: np.zeros(len(env.get_valid_actions({}))))
        self.strategy_sum = defaultdict(lambda: np.zeros(len(env.get_valid_actions({}))))
        self.mcts_iterations = 100
        
    def get_info_set(self, state: Dict) -> str:
        """Convert state to information set string"""
        hand = ','.join(f"{r}{s}" for r, s in state['hand'])
        community = ','.join(f"{r}{s}" for r, s in self.env.community_cards)
        return f"{hand}|{community}|{state['chips']}|{state['bet']}|{self.env.pot}"
    
    def mcts_search(self, state: Dict) -> int:
        """Perform MCTS to find best action"""
        root = MCTSNode(state)
        
        for _ in range(self.mcts_iterations):
            node = root
            sim_env = copy.deepcopy(self.env)
            
            # Selection
            while node.untried_actions is None or len(node.untried_actions) == 0:
                if not node.children:
                    break
                node = max(node.children.values(), 
                          key=lambda n: n.ucb_score(node.visits))
            
            # Expansion
            if node.untried_actions is None:
                node.untried_actions = sim_env.get_valid_actions(node.state)
            if node.untried_actions:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                next_state = self.simulate_action(sim_env, node.state, action)
                node = node.expand(action, next_state)
            
            # Simulation
            reward = self.rollout(sim_env, node.state)
            
            # Backpropagation
            while node:
                node.visits += 1
                node.value += reward
                node = node.parent
        
        # Return best action
        return max(root.children.items(), 
                  key=lambda x: x[1].visits)[0]
    
    def get_strategy(self, info_set: str) -> np.ndarray:
        """Get current strategy for an information set"""
        regrets = self.regret_sum[info_set]
        positive_regrets = np.maximum(regrets, 0)
        sum_positive = np.sum(positive_regrets)
        
        if sum_positive > 0:
            strategy = positive_regrets / sum_positive
        else:
            strategy = np.ones_like(regrets) / len(regrets)
            
        return strategy
    
    def act(self, state: Dict) -> str:
        """Choose action using CFR+MCTS"""
        info_set = self.get_info_set(state)
        strategy = self.get_strategy(info_set)
        
        if random.random() < 0.1:  # Exploration
            return self.mcts_search(state)
        else:
            return np.random.choice(self.env.get_valid_actions(state), p=strategy)
