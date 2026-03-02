import random
import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists('plots'):
    os.makedirs('plots')

class RiverSwim:
    LEFT = 0
    RIGHT = 1

    def __init__(self, n_states=6, gamma=0.99, m=5):
        self.n_states = n_states
        self.current_state = 0
        self.m = m
        self.R_max = 1.0  # R-MAX Optimisme [cite: 243]

        self.states = list(range(n_states))
        self.actions = [self.LEFT, self.RIGHT]

        # R-MAX Tabeller
        self.N = np.zeros((self.n_states, len(self.actions), self.n_states))
        self.N_sa = np.zeros((self.n_states, len(self.actions)))
        self.T_hat = np.zeros((self.n_states, len(self.actions), self.n_states))
        
        # Initialiser model med optimisme (selv-loops) indtil m besøg [cite: 242]
        for s in range(self.n_states):
            for a in range(len(self.actions)):
                self.T_hat[s, a, s] = 1.0

        self.R_sum = np.zeros((self.n_states, len(self.actions)))
        # R-MAX: Alle ukendte handlinger antages at give max belønning [cite: 243]
        self.R_hat = np.full((self.n_states, len(self.actions)), self.R_max)

        self.V = np.zeros(self.n_states)
        self.pi = np.zeros(self.n_states, dtype=int)
        self.gamma = gamma

        self.small_reward = 5 / 1000
        self.large_reward = 1.0
        self.T = self._build_transitions()

    def _build_transitions(self):
        n = self.n_states
        T = {}
        for s in range(n):
            T[s] = {}
            if s == 0:
                T[s][self.LEFT] = [(1.0, s)]
            else:
                T[s][self.LEFT] = [(1.0, s - 1)]

            if s == 0:
                T[s][self.RIGHT] = [(0.6, s + 1), (0.4, s)]
            elif s == n - 1:
                T[s][self.RIGHT] = [(0.6, s), (0.4, s - 1)]
            else:
                T[s][self.RIGHT] = [(0.35, s + 1), (0.60, s), (0.05, s - 1)]
        return T

    def get_reward(self, state):
        if state == 0: return self.small_reward
        elif state == self.n_states - 1: return self.large_reward
        return 0.0

    def reset(self):
        self.current_state = 0
        return self.current_state

    def step(self, action):
        transitions = self.T[self.current_state][action]
        rand = random.random()
        cumulative = 0.0
        next_state = self.current_state
        for prob, s_next in transitions:
            cumulative += prob
            if rand <= cumulative:
                next_state = s_next
                break

        # Opdater R-MAX model hvis tærskel m ikke er nået [cite: 242]
        if self.N_sa[self.current_state][action] < self.m:
            self.N[self.current_state][action][next_state] += 1
            self.N_sa[self.current_state][action] += 1
            self.R_sum[self.current_state][action] += self.get_reward(next_state)

            if self.N_sa[self.current_state][action] == self.m:
                self.R_hat[self.current_state][action] = self.R_sum[self.current_state][action] / self.m
                self.T_hat[self.current_state, action, :] = self.N[self.current_state, action, :] / self.m
        
        reward = self.get_reward(next_state)
        self.current_state = next_state
        return next_state, reward, False

    def policy_iteration(self, theta=1e-6, max_iter=100):
        for _ in range(max_iter):
            self._policy_evaluation(theta)
            if self._policy_improvement(): break

    def _policy_evaluation(self, theta=1e-6):
        while True:
            delta = 0
            for s in self.states:
                v_old = self.V[s]
                a = self.pi[s]
                self.V[s] = self.R_hat[s, a] + self.gamma * np.sum(self.T_hat[s, a, :] * self.V)
                delta = max(delta, abs(v_old - self.V[s]))
            if delta < theta: break

    def _policy_improvement(self):
        policy_stable = True
        for s in self.states:
            old_action = self.pi[s]
            q_values = [self.R_hat[s, a] + self.gamma * np.sum(self.T_hat[s, a, :] * self.V) for a in self.actions]
            self.pi[s] = np.argmax(q_values)
            if old_action != self.pi[s]: policy_stable = False
        return policy_stable

    def choose_action(self):
        # R-MAX: Handl altid grådigt i forhold til den optimistiske model [cite: 302, 311]
        return self.pi[self.current_state]

    def train(self, n_steps=500000, update_interval=1000, early_stop_window=20000, early_stop_threshold=1e-6, verbose=True):
        self.reset()
        total_reward = 0.0
        reward_history = [] 

        for t in range(1, n_steps + 1):
            action = self.choose_action()
            _, reward, _ = self.step(action)
            total_reward += reward
            reward_history.append(total_reward)

            if t % update_interval == 0:
                self.policy_iteration()

            if verbose and t % 50000 == 0:
                print(f"  Step {t:6d} | Reward Rate: {(total_reward/t):.4f}")

            # Early stopping baseret på reward rate stabilitet [cite: 35]
            if t >= 2 * early_stop_window:
                recent_rate = (reward_history[-1] - reward_history[-early_stop_window]) / early_stop_window
                prev_rate = (reward_history[-early_stop_window] - reward_history[-2 * early_stop_window]) / early_stop_window
                if abs(recent_rate - prev_rate) < early_stop_threshold and recent_rate > 0.01:
                    if verbose: print(f"  Konvergeret ved trin {t}")
                    break
        return reward_history

if __name__ == "__main__":
    river_lengths = [5, 10, 15, 20]
    n_repetitions = 5
    max_steps = 500000
    all_results = {}

    for n_states in river_lengths:
        print(f"\nTester flodlængde: {n_states}")
        histories = [RiverSwim(n_states=n_states).train(n_steps=max_steps) for _ in range(n_repetitions)]
        all_results[n_states] = histories

    # PLOTTING
    plt.figure(figsize=(10, 6))
    for n_states in river_lengths:
        histories = all_results[n_states]
        max_len = max(len(h) for h in histories)
        padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in histories])
        
        avg_cumulative = np.mean(padded, axis=0)
        # Beregn Reward Rate (belønning pr. skridt) [cite: 37]
        reward_rate = np.diff(avg_cumulative)
        
        # Udjævning (Moving Average) for pænere grafer
        window = 5000
        smoothed = np.convolve(reward_rate, np.ones(window)/window, mode='valid')
        
        plt.plot(range(len(smoothed)), smoothed, label=f'T = {n_states}')

    # Tilføj teoretisk max (0.6 reward pr. skridt i s_last) som reference [cite: 17, 21]
    plt.axhline(y=0.6, color='r', linestyle='--', alpha=0.3, label='Teoretisk Max (s_last)')
    plt.xlabel('Trin')
    plt.ylabel('Gennemsnitlig Belønning pr. Trin (Glidende gennemsnit)')
    plt.title('R-MAX Læringskurve: RiverSwim')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig('plots/learning-curve.png') # Gem med det krævede navn 
    plt.show()