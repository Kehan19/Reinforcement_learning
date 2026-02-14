import random
import numpy as np
import matplotlib.pyplot as plt
import os
if not os.path.exists('plots'):
    os.makedirs('plots')


# =====================================================================
# RiverSwim MDP — Assignment 1, Exercise 1
# =====================================================================
# States:       s_1 ... s_n  (dynamic, default n=6)
# Actions:      0 = LEFT, 1 = RIGHT
# Rewards:      s_1 → 5/1000,  s_last → 1
#
# Transitions (RIGHT action):
#   s_1:          0.6 → s_2,   0.4 → stay
#   s_2 to s_5:   0.35 → right, 0.6 → stay, 0.05 → left
#   s_6:          0.6 → stay,  0.4 → left
#
# Transitions (LEFT action):
#   Always deterministic: move one step left (stay at s_1)
# =====================================================================


class RiverSwim:
    LEFT = 0
    RIGHT = 1

    def __init__(self, n_states=6, gamma=0.9):
        self.n_states = n_states
        self.current_state = 0  # start at s_1

        # --- States ---
        self.states = list(range(n_states))

        # --- Actions ---
        self.actions = [self.LEFT, self.RIGHT]

        # --- Exercise 2: Estimated MDP tables ---
        self.N = np.zeros((self.n_states, len(self.actions), self.n_states))

        self.N_sa = np.zeros((self.n_states, len(self.actions)))

        self.T_hat = np.zeros((self.n_states, len(self.actions), self.n_states))

        self.R_sum = np.zeros((self.n_states, len(self.actions)))

        self.R_hat = np.ones((self.n_states, len(self.actions)))

        self.V = np.zeros(self.n_states)

        self.pi = np.zeros(self.n_states, dtype=int)

        # --- Discount factor ---
        self.gamma = gamma

        # --- Rewards ---
        # Only non-zero at s_1 and s_last
        self.small_reward = 5 / 1000   # r at s_1
        self.large_reward = 1.0        # r at s_last

        # --- Transition table ---
        # T[state][action] = list of (probability, next_state)
        self.T = self._build_transitions()

    def _build_transitions(self):

        n = self.n_states
        T = {}

        for s in range(n):
            T[s] = {}

            # --- LEFT action: deterministic, move left ---
            if s == 0:
                T[s][self.LEFT] = [(1.0, s)]          # stay at s_1
            else:
                T[s][self.LEFT] = [(1.0, s - 1)]      # move left

            # --- RIGHT action: stochastic ---
            if s == 0:
                # Leftmost state (s_1)
                T[s][self.RIGHT] = [
                    (0.6, s + 1),   # move right to s_2
                    (0.4, s),       # stay at s_1
                ]
            elif s == n - 1:
                # Rightmost state (s_last)
                T[s][self.RIGHT] = [
                    (0.6, s),       # stay at s_last
                    (0.4, s - 1),   # drift left
                ]
            else:
                # Intermediate states (s_2 ... s_{n-1})
                T[s][self.RIGHT] = [
                    (0.35, s + 1),  # move right
                    (0.60, s),      # stay (pushed back by current)
                    (0.05, s - 1),  # drift left
                ]

        return T

    def get_reward(self, state):
        """
        Return reward for arriving at a state.
        Non-zero only at s_1 and s_last.
        """
        if state == 0:
            return self.small_reward       # 5/1000
        elif state == self.n_states - 1:
            return self.large_reward       # 1.0
        else:
            return 0.0

    def reset(self):
        """Reset to starting state s_1."""
        self.current_state = 0
        return self.current_state

    def step(self, action):
        """
        Take an action. Returns (next_state, reward, done).
        """
        transitions = self.T[self.current_state][action]

        # Sample next state from transition probabilities
        rand = random.random()
        cumulative = 0.0
        next_state = self.current_state
        for prob, s_next in transitions:
            cumulative += prob
            if rand <= cumulative:
                next_state = s_next
                break

        
        self.N[self.current_state][action][next_state] += 1
        self.N_sa[self.current_state][action] += 1
        # Recompute ALL transition probs for this (s, a) so they sum to 1.0
        self.T_hat[self.current_state, action, :] = self.N[self.current_state, action, :] / self.N_sa[self.current_state, action]
        self.R_sum[self.current_state][action] += self.get_reward(next_state)
        #self.R_hat[self.current_state][action] = self.R_sum[self.current_state][action] / self.N_sa[self.current_state][action]

        
        # Reward depends on which state we land in
        reward = self.get_reward(next_state)

        # Only update the reward estimate if we've actually seen a reward > 0
        # This keeps the 'hope' (the 1.0) alive for states we haven't reached yet.
        if reward > 0:
            self.R_hat[self.current_state][action] = self.R_sum[self.current_state][action] / self.N_sa[self.current_state][action]

        self.current_state = next_state
        done = False  # non-episodic (continuing task)
        return next_state, reward, done

    # =================================================================
    # Exercise 2: Policy Iteration on the ESTIMATED MDP (T_hat, R_hat)
    # =================================================================

    def policy_iteration(self, theta=1e-6, max_iter=100):
        """
        Run policy iteration on the estimated MDP (T_hat, R_hat).
        Updates self.V and self.pi.

        theta:    convergence threshold for policy evaluation
        max_iter: max number of policy eval/improve cycles
        """
        for i in range(max_iter):
            # --- Step 1: Policy Evaluation ---
            # Compute V(s) for the current policy self.pi
            self._policy_evaluation(theta)

            # --- Step 2: Policy Improvement ---
            # Update self.pi greedily based on V
            policy_stable = self._policy_improvement()

            # If the policy didn't change, we've converged
            if policy_stable:
                break

    def _policy_evaluation(self, theta=1e-6):
        """
        Step 1: Evaluate the current policy.
        Compute V(s) = R_hat[s, pi(s)] + gamma * sum_s' T_hat[s, pi(s), s'] * V(s')
        Iterate until V converges (max change < theta).
        """
        while True:
            delta = 0
            for s in self.states:
                v_old = self.V[s]
                a = self.pi[s]  # action chosen by current policy

                # Bellman equation for policy pi
                # V(s) = R(s,a) + gamma * sum over s' of T(s,a,s') * V(s')
                self.V[s] = self.R_hat[s, a] + self.gamma * np.sum(
                    self.T_hat[s, a, :] * self.V
                )

                delta = max(delta, abs(v_old - self.V[s]))

            # Converged when values barely change
            if delta < theta:
                break

    def _policy_improvement(self):
        """
        Step 2: Improve the policy.
        For each state, pick the action that maximizes:
            Q(s, a) = R_hat[s, a] + gamma * sum_s' T_hat[s, a, s'] * V(s')

        Returns True if the policy is stable (no changes).
        """
        policy_stable = True

        for s in self.states:
            old_action = self.pi[s]

            # Compute Q-value for each action
            q_values = np.zeros(len(self.actions))
            for a in self.actions:
                q_values[a] = self.R_hat[s, a] + self.gamma * np.sum(
                    self.T_hat[s, a, :] * self.V
                )

            # Pick the best action
            self.pi[s] = np.argmax(q_values)

            # Check if the policy changed
            if old_action != self.pi[s]:
                policy_stable = False

        return policy_stable

    # =================================================================
    # Behavior Policy: epsilon-greedy
    # =================================================================

    def choose_action(self, epsilon=0.1):
        """
        Epsilon-greedy behavior policy.

        With probability (1 - epsilon): follow the learned policy pi(s)  (exploit)
        With probability epsilon:       pick a random action              (explore)
        """
        if random.random() < epsilon:
            # Explore: random action
            return random.choice(self.actions)
        else:
            # Exploit: follow the current best policy
            return self.pi[self.current_state]

    # =================================================================
    # Training Loop: ties everything together
    # =================================================================

    def train(self, n_steps=500000, epsilon_start=1.0, epsilon_end=0.01,
              update_interval=1000, early_stop_window=50000, early_stop_threshold=1e-7,
              verbose=True):
        """
        Online model-based RL training loop.

        Early stopping: compares average reward/step over two consecutive
        windows. If the improvement is below 'early_stop_threshold', training stops.
        """
        self.reset()
        total_reward = 0.0
        reward_history = []  # track total reward at each step

        for t in range(1, n_steps + 1):
            # Decay epsilon linearly
            # 1. Set the static exploration period
            static_period = 10000

            if t <= static_period:
                epsilon = 0.5
            else:
                # 2. Calculate how far we are into the decay period
                decay_steps = n_steps - static_period
                current_decay_step = t - static_period
                
                # 3. Cosine Decay Formula
                # This decays from 0.5 down to epsilon_end
                cosine_out = 0.5 * (1 + np.cos(np.pi * current_decay_step / decay_steps))
                epsilon = epsilon_end + (0.5 - epsilon_end) * cosine_out
            

            # 1. Choose action (epsilon-greedy)
            action = self.choose_action(epsilon)

            # 2. Take a step (updates N, T_hat, R_hat internally)
            next_state, reward, done = self.step(action)
            total_reward += reward

            # 3. Periodically run policy iteration
            if t % update_interval == 0:
                self.policy_iteration()

            # Track progress
            reward_history.append(total_reward)

            # Print progress
            if verbose and t % 1000 == 0:
                policy_str = "".join(
                    ["L" if self.pi[s] == 0 else "R" for s in self.states]
                )
                print(
                    f"  Step {t:6d} | "
                    f"Total reward: {total_reward:8.2f} | "
                    f"\u03b5={epsilon:.3f} | "
                    f"Policy: [{policy_str}]"
                )

            # --- Early stopping ---
            # Compare avg reward rate in two consecutive windows
            if t >= 2 * early_stop_window:
                # Reward earned in the previous window vs the one before it
                recent = reward_history[-1] - reward_history[-early_stop_window]
                previous = reward_history[-early_stop_window] - reward_history[-2 * early_stop_window]

                # Average reward per step in each window
                rate_recent = recent / early_stop_window
                rate_previous = previous / early_stop_window

                # If the rate hasn't improved significantly, stop
                if abs(rate_recent - rate_previous) < early_stop_threshold:
                    if verbose:
                        print(f"  Early stop at step {t} | "
                              f"reward rate flattened ({rate_previous:.4f} → {rate_recent:.4f})")
                    break

        return reward_history


# =====================================================================
# Exercise 3: Run experiments with varying river lengths
# =====================================================================
if __name__ == "__main__":
    river_lengths = [5, 10, 15, 20]   # s_1 to s_T
    n_repetitions = 5
    max_steps = 1000000
    gamma = 0.99

    print("=" * 60)
    print("  RiverSwim — Exercise 3: Varying River Lengths")
    print("=" * 60)
    print(f"  Lengths: {river_lengths}")
    print(f"  Repetitions: {n_repetitions}")
    print(f"  Max steps: {max_steps}")
    print(f"  Gamma: {gamma}")

    # Store results for plotting
    # Key = n_states, Value = list of reward histories (one per rep)
    all_results = {}

    for n_states in river_lengths:
        print(f"\n{'=' * 60}")
        print(f"  River length: {n_states} states")
        print("=" * 60)

        histories = []
        for rep in range(1, n_repetitions + 1):
            print(f"\n  --- Repetition {rep}/{n_repetitions} ---")
            env = RiverSwim(n_states=n_states, gamma=gamma)
            history = env.train(
                n_steps=max_steps,
                epsilon_start=1.0,
                epsilon_end=0.1,
                update_interval=1000,
                early_stop_window=10000,
                early_stop_threshold=1e-7,
                verbose=True,
            )
            histories.append(history)

            
            # Print final policy for this rep
            policy_str = "".join(
                ["L" if env.pi[s] == 0 else "R" for s in env.states]
            )
            print(f"  Final policy: [{policy_str}]  "
                  f"Steps: {len(history)}  "
                  f"Total reward: {history[-1]:.2f}")
            

            # After training is done, run one last iteration with NO randomness
            env.policy_iteration()
            final_policy = "".join(["L" if env.pi[s] == 0 else "R" for s in env.states])
            print(f"The True Calculated Policy is: [{final_policy}]")


        all_results[n_states] = histories

    # =====================================================================
    # Plot: Learning curves averaged across repetitions
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("  Generating plot...")
    print("=" * 60)

    plt.figure(figsize=(12, 7))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    for idx, n_states in enumerate(river_lengths):
        histories = all_results[n_states]

        # Pad shorter histories to max length (early-stopped runs)
        max_len = max(len(h) for h in histories)
        padded = []
        for h in histories:
            if len(h) < max_len:
                # Extend with the final value (reward stays flat after stopping)
                h_padded = h + [h[-1]] * (max_len - len(h))
            else:
                h_padded = h
            padded.append(h_padded)

        # Average across repetitions
        avg_reward = np.mean(padded, axis=0)
        std_reward = np.std(padded, axis=0)

        steps = np.arange(1, max_len + 1)

        # Plot mean with shaded std region
        plt.plot(steps, avg_reward, label=f'T = {n_states} states',
                 color=colors[idx], linewidth=2)
        plt.fill_between(steps,
                         avg_reward - std_reward,
                         avg_reward + std_reward,
                         alpha=0.2, color=colors[idx])

    plt.xlabel('Time Step', fontsize=13)
    plt.ylabel('Total Reward (averaged over 5 runs)', fontsize=13)
    plt.title('RiverSwim Learning Curves — Varying River Lengths', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plot_path = 'plots/learning_curves.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\n  Plot saved to: {plot_path}")
    plt.show()
