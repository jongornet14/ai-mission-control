import requests
import numpy as np

# Simple Q-learning agent
class SimpleAgent:
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.epsilon = 0.9
        
    def get_state(self, obs):
        # Discretize CartPole observation to simple state
        cart_pos, cart_vel, pole_angle, pole_vel = obs
        state = (
            int(cart_pos * 10) // 2,  # Bin cart position
            int(pole_angle * 100) // 5  # Bin pole angle
        )
        return state
    
    def act(self, obs):
        state = self.get_state(obs)
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.choice([0, 1])  # Explore
        
        # Get Q-values for this state
        q_values = self.q_table.get(state, [0, 0])
        return np.argmax(q_values)  # Exploit
    
    def learn(self, obs, action, reward, next_obs, done):
        state = self.get_state(obs)
        next_state = self.get_state(next_obs)
        
        # Initialize Q-values if new state
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0]
        
        # Q-learning update
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state]) if not done else 0
        new_q = current_q + self.learning_rate * (reward + 0.95 * next_max_q - current_q)
        self.q_table[state][action] = new_q

# Train the agent
agent = SimpleAgent()

# Create environment
response = requests.post("http://localhost:50053/create/CartPole-v1")
session_id = response.json()["session_id"]

print("Training agent...")
for episode in range(100):
    # Reset
    response = requests.post(f"http://localhost:50053/reset/{session_id}")
    obs = response.json()["observation"]
    
    total_reward = 0
    for step in range(200):
        # Agent acts
        action = agent.act(obs)
        
        # Step environment
        response = requests.post(
            f"http://localhost:50053/step/{session_id}",
            json={"action": int(action)}  # Convert to regular Python int
        )
        data = response.json()
        
        next_obs = data["observation"]
        reward = data["reward"]
        done = data["done"]
        
        # Agent learns
        agent.learn(obs, action, reward, next_obs, done)
        
        obs = next_obs
        total_reward += reward
        
        if done:
            break
    
    # Decay exploration
    agent.epsilon *= 0.995
    
    if episode % 20 == 0:
        print(f"Episode {episode}: {total_reward} reward, ε={agent.epsilon:.3f}")

print("Training done!")