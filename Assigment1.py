import random
import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists('plots'):
    os.makedirs('plots')

class RiverSwim:
    LEFT = 0
    RIGHT = 1

    def __init__(self, n_states=6, gamma=0.999, m=50):
        self.n_states = n_states
        self.current_state = 0
        self.m = m
        self.R_max = 1.0

        self.states = list(range(n_states))
        self.actions = [self.LEFT, self.RIGHT]

        # R-MAX Tables
        self.N_sa = np.zeros((self.n_states, len(self.actions)))
        self.N_sas = np.zeros((self.n_states, len(self.actions), self.n_states))
        self.T_hat = np.zeros((self.n_states, len(self.actions), self.n_states))
        
        for s in range(self.n_states):
            for a in range(len(self.actions)):
                self.T_hat[s, a, s] = 1.0

        self.R_sum = np.zeros((self.n_states, len(self.actions)))
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
            T[s][self.LEFT] = [(1.0, max(0, s - 1))]
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

        if self.N_sa[self.current_state][action] < self.m:
            self.N_sas[self.current_state][action][next_state] += 1
            self.N_sa[self.current_state][action] += 1
            self.R_sum[self.current_state][action] += self.get_reward(next_state)

            if self.N_sa[self.current_state][action] == self.m:
                self.R_hat[self.current_state][action] = self.R_sum[self.current_state][action] / self.m
                self.T_hat[self.current_state, action, :] = self.N_sas[self.current_state, action, :] / self.m
        
        reward = self.get_reward(next_state)
        self.current_state = next_state
        return next_state, reward

    def policy_iteration(self, theta=1e-6):
        while True:
            while True:
                delta = 0
                for s in self.states:
                    v = self.V[s]
                    a = self.pi[s]
                    self.V[s] = self.R_hat[s, a] + self.gamma * np.sum(self.T_hat[s, a, :] * self.V)
                    delta = max(delta, abs(v - self.V[s]))
                if delta < theta: break
            stable = True
            for s in self.states:
                old_a = self.pi[s]
                self.pi[s] = np.argmax([self.R_hat[s, a] + self.gamma * np.sum(self.T_hat[s, a, :] * self.V) for a in self.actions])
                if old_a != self.pi[s]: stable = False
            if stable: break

    def train(self, n_steps=10000, early_stop_target =  1e-4, window_size = 1000):
        self.reset()
        reward_history = []
        visit_counts = np.zeros(self.n_states)
        total_reward = 0
        

        for t in range(1, n_steps + 1):
            action = self.pi[self.current_state]
            visit_counts[self.current_state] += 1
            old_policy = self.pi.copy()
            old_known = np.sum(self.N_sa >= self.m)
            _, r = self.step(action)
            new_known = np.sum(self.N_sa >= self.m)
            
            total_reward += r
            reward_history.append(r) # Immediate reward for rolling average

            if new_known > old_known:
                self.policy_iteration()

            all_known = np.all(self.N_sa >= self.m)

            if all_known and self.current_state == self.n_states - 1:
                print(f"All states are known and we are in the last state at step {t}")
                break


        remaining_steps = n_steps - len(reward_history)
        if remaining_steps > 0:
            last_reward = reward_history[-1] if reward_history else 0
            reward_history.extend([last_reward] * remaining_steps)   
                
        return reward_history, visit_counts / n_steps

if __name__ == "__main__":
    lengths = [5, 10, 15, 20]
    results = {}

    for T in lengths:
        print(f"Træner T={T}...")
        reps = [RiverSwim(n_states=T).train() for _ in range(5)]
        results[T] = reps

    # --- PLOT 1: Læringskurve (Reward Rate over tid) ---
    plt.figure(figsize=(10, 5))
    for T in lengths:
        # Gennemsnit over 5 repetitioner
        rates = [rep[0] for rep in results[T]]
        avg_rate = np.mean(rates, axis=0)
        
        # Udjævning for læsbarhed
        window = 2000
        smoothed = np.convolve(avg_rate, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=f'T={T}')

    plt.axhline(y=0.005, color='gray', linestyle='--', alpha=0.5, label='Start Reward (s1)')
    plt.title('Gennemsnitlig belønning pr. trin (Læringskurve)')
    plt.xlabel('Trin')
    plt.ylabel('Reward pr. trin')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('learning-curve.png')
    plt.show()

    # --- PLOT 2: Hvor ender agenten? (State Visitation Distribution) ---
    plt.figure(figsize=(10, 5))
    for T in lengths:
        # Gennemsnitlige besøg over 5 repetitioner
        visitations = [rep[1] for rep in results[T]]
        # For at plotte dem sammen, bruger vi normaliserede x-akser (0 til 1) eller bare index
        avg_vis = np.mean(visitations, axis=0)
        
        # Vi viser kun s1 og s_last for at se om den finder målet
        plt.bar(np.arange(len(avg_vis)) + (T*0.02), avg_vis, alpha=0.6, label=f'T={T}', width=0.4)

    plt.title('Hvor opholder agenten sig? (State Distribution)')
    plt.xlabel('Tilstand (0 = s1, højeste index = s_last)')
    plt.ylabel('% af tiden brugt i tilstanden')
    plt.legend()
    plt.savefig('state-visitation.png')
    plt.show()