import numpy as np
from rl_agent import QLearningAgent

print("--- Running Isolated Brain Test ---")

# 1. Initialize the agent with dummy semantic actions
actions = ["forward", "turn_left", "attack"]
agent = QLearningAgent(actions=actions, alpha=0.1, gamma=0.9, epsilon=0.0) # Set epsilon=0 to test exploitation

# 2. Test Q-table initialization
initial_state = "front:air_wood:0"
print(f"Initial Q-values for unseen state: {agent.get_q_values(initial_state)}")

# 3. Simulate a Bellman update (Let's pretend moving 'forward' found 'wood' and gave a +50 reward)
next_state = "front:log_wood:1"
agent.update(state=initial_state, action="forward", reward=50.0, next_state=next_state)

print("\n--- After Update ---")
print(f"Updated Q-values for {initial_state}: {agent.get_q_values(initial_state)}")
print(f"Q-Table Contents: {agent.q_table}")

# 4. Ensure it exploits the best known action
chosen = agent.choose_action(initial_state)
print(f"\nChosen action (Should be 'forward'): {chosen}")
assert chosen == "forward", "Brain math error!"
print("\n🟢 Brain logic works perfectly!")