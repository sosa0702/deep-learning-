import numpy as np
from Poker_Env import PokerEnv
from GTDQN import GTDQNAgent

class PokerTrainingModule:
    def __init__(self, env: PokerEnv, agent, num_episodes: int = 10000, batch_size: int = 32):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.memory = []  # For experience replay
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration factor
        self.epsilon_decay = 0.995  # Decay for epsilon
        self.epsilon_min = 0.1  # Minimum epsilon value
        self.reward_list = []  # To track rewards over episodes

    def train(self):
        for episode in range(self.num_episodes):
            self.env.reset_round()
            total_reward = 0
            done = False
            
            while not done:
                current_player = self.env.current_player
                state = self.get_state()  # Implement a method to get the current state
                
                # Agent selects action based on the state
                action = self.agent.act(state, self.env)  # Pass env as argument
                
                # Apply the selected action
                valid_action = self.env.apply_action(current_player, action)
                
                if valid_action:
                    # Reward calculation
                    reward = self.calculate_reward()
                    total_reward += reward
                    
                    # Update regret based on the reward and action taken
                    self.agent.update_regret(action, reward)  # Ensure you have an update_regret method in GTDQNAgent
                    
                    # Store the experience
                    next_state = self.get_state()  # Get the new state after action
                    self.memory.append((state, action, reward, next_state, done))
                    
                    # Learn from the experience if enough memory
                    if len(self.memory) > self.batch_size:
                        self.replay()
                
                # Check if the game is over
                done = not self.env.game_active
            
            # Decay epsilon after each episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.reward_list.append(total_reward)
            print(f"Episode {episode + 1}/{self.num_episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")

    def calculate_reward(self) -> float:
        """Calculate the reward based on the outcome of the game."""
        if self.env.players[self.env.current_player]['status'] == 'folded':
            return -1  # Negative reward for folding
        # Add other reward conditions as necessary
        return 0

    def replay(self):
        """Sample a batch of experiences and train the agent."""
        batch = np.random.choice(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.agent.predict(next_state))  # Predict future rewards
            
            # Update the agent's model
            target_f = self.agent.predict(state)
            target_f[0][action] = target  # Update the target for the action taken
            self.agent.fit(state, target_f)  # Train the model

    def get_state(self) -> np.ndarray:
        """Transform the game state into a format suitable for the agent."""
        # Implement this method based on the state representation used by your agent
        return np.array([])  # Replace with actual state representation

if __name__ == "__main__":
    grid_size = (5, 5) 
    action_size = 4 
    env = PokerEnv(num_players=4, starting_chips=999, small_blind=10)
    agent = GTDQNAgent(grid_size=grid_size, action_size=action_size)  
    trainer = PokerTrainingModule(env, agent)
    trainer.train()
