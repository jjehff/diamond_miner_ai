import numpy as np
import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.3):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state):
        q_values = self.get_q_values(state)
        
        # pick random action
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # pick  action with the highest Q-value
        # if tie, np.argmax picks first one
        best_action_index = np.argmax(q_values)
        return self.actions[best_action_index]

    def update(self, state, action, reward, next_state):
        current_q_values = self.get_q_values(state)
        action_idx = self.actions.index(action)
        old_q = current_q_values[action_idx]
        
        next_q_values = self.get_q_values(next_state)
        max_next_q = np.max(next_q_values)

        new_q = old_q + self.alpha * (reward + (self.gamma * max_next_q) - old_q)
        self.q_table[state][action_idx] = new_q