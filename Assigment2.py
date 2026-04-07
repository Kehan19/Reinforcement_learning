import numpy as np
import matplotlib.pyplot as plt

class OverestimationGridWorld:
    def __init__(self, state_size=10, env_type="standard"):
        """
        state_size: Tunable parameter for the state space size.
        env_type: 'standard', 'high_variance', or 'long_safe_path' [1, 2].
        """
        # For the 'long_safe_path' environment, we artificially extend the state size
        self.state_size = state_size * 2 if env_type == "long_safe_path" else state_size
        self.env_type = env_type
        
        # State 0 is start, State 1 is the Lure State, States 2 to state_size are the Safe Path
        self.current_state = 0
        self.n_actions = 10 
        
    def reset(self):
        self.current_state = 0
        return self.current_state
        
    def step(self, action):
        done = False
        reward = 0.0
        
        if self.current_state == 0:
            if action == 0: # Move to safe path
                self.current_state = 2
            else: # Move to Lure state
                self.current_state = 1
                
        elif self.current_state == 1: # The Lure State
            # Expected reward is 0, but with high variance noise [2]
            std_dev = 4.0 if self.env_type == "high_variance" else 2.0
            reward = np.random.normal(loc=0.0, scale=std_dev)
            done = True 
            
        elif self.current_state >= 2: # The Safe Path
            if self.current_state == self.state_size - 1:
                reward = 0.1 # Small guaranteed positive reward at the end [2]
                done = True
            else:
                self.current_state += 1
                
        return self.current_state, reward, done


class EnsembleQLearningAgents:
    def __init__(self, n_states, n_actions, algorithm="DQN", ensemble_size=2, alpha=0.1, gamma=0.99):
        self.algorithm = algorithm
        self.n_actions = n_actions
        self.ensemble_size = ensemble_size if algorithm in ["Q-ensemble-min", "REDQ"] else 2
        self.alpha = alpha
        self.gamma = gamma
        
        # Initialize an ensemble of Q-tables
        self.Q_tables = [np.zeros((n_states, n_actions)) for _ in range(self.ensemble_size)]

    def select_action(self, state, epsilon=0.1):
        # Average across all Q-tables in the ensemble for action selection
        avg_Q = np.mean(self.Q_tables, axis=0)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(avg_Q[state])

    def update(self, state, action, reward, next_state, done):
        # Determine the action that the agent *would* take in the next state
        avg_Q_next = np.mean(self.Q_tables, axis=0)
        best_next_action = np.argmax(avg_Q_next[next_state])

        # 1. DQN: Double Q-learning (Decoupled selection and evaluation)
        if self.algorithm == "DQN":
            if np.random.rand() < 0.5:
                # Update Q0, evaluate with Q1
                best_action_q0 = np.argmax(self.Q_tables[0][next_state])
                target_Q = self.Q_tables[1][next_state, best_action_q0]
                target = reward if done else reward + self.gamma * target_Q
                self.Q_tables[0][state, action] += self.alpha * (target - self.Q_tables[0][state, action])
            else:
                # Update Q1, evaluate with Q0
                best_action_q1 = np.argmax(self.Q_tables[1][next_state])
                target_Q = self.Q_tables[0][next_state, best_action_q1]
                target = reward if done else reward + self.gamma * target_Q
                self.Q_tables[1][state, action] += self.alpha * (target - self.Q_tables[1][state, action])

        # 2. DQN-min: TD3 style min-clipping on two tables
        elif self.algorithm == "DQN-min":
            # Fixed to correctly access table 0 and table 1
            target_Q = min(self.Q_tables[0][next_state, best_next_action], 
                           self.Q_tables[1][next_state, best_next_action])
            target = reward if done else reward + self.gamma * target_Q
            
            # Use 50% probability mask to update each table to ensure they decorrelate
            for i in range(2):
                if np.random.rand() < 0.5:
                    self.Q_tables[i][state, action] += self.alpha * (target - self.Q_tables[i][state, action])

        # 3. Q-ensemble-min: Min-clipping across ALL ensemble members
        elif self.algorithm == "Q-ensemble-min":
            all_next_Qs = [q[next_state, best_next_action] for q in self.Q_tables]
            target_Q = np.min(all_next_Qs)
            target = reward if done else reward + self.gamma * target_Q
            
            # Use 50% probability mask to update each table to ensure they decorrelate
            for i in range(self.ensemble_size):
                if np.random.rand() < 0.5:
                    self.Q_tables[i][state, action] += self.alpha * (target - self.Q_tables[i][state, action])

        # 4. REDQ: Min-clipping on a randomly chosen pair from the ensemble
        elif self.algorithm == "REDQ":
            idx1, idx2 = np.random.choice(self.ensemble_size, size=2, replace=False)
            target_Q = min(self.Q_tables[idx1][next_state, best_next_action], 
                           self.Q_tables[idx2][next_state, best_next_action])
            target = reward if done else reward + self.gamma * target_Q
            
            # Use 50% probability mask to update each table to ensure they decorrelate
            for i in range(self.ensemble_size):
                if np.random.rand() < 0.5:
                    self.Q_tables[i][state, action] += self.alpha * (target - self.Q_tables[i][state, action])


def run_experiment(env_type="standard", state_size=10, ensemble_size=5, episodes=2000):
    algorithms = ["DQN", "DQN-min", "Q-ensemble-min", "REDQ"]
    results = {}
    gamma = 0.99
    
    # Calculate True optimal value baseline for the plots [6]
    actual_state_size = state_size * 2 if env_type == "long_safe_path" else state_size
    true_optimal_value = 0.1 * (gamma ** (actual_state_size - 2)) 

    for algo in algorithms:
        env = OverestimationGridWorld(state_size=state_size, env_type=env_type)
        # Ensure we pass the correct scaled state size to the agent
        agent = EnsembleQLearningAgents(n_states=env.state_size, 
                                        n_actions=env.n_actions, 
                                        algorithm=algo, 
                                        ensemble_size=ensemble_size,
                                        gamma=gamma)
        
        estimated_start_values = []
        
        for ep in range(episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(state, epsilon=0.1)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
            
            # Track perceived value of the start state (s_0) [6]
            avg_Q_start = np.mean(agent.Q_tables, axis=0) 
            perceived_value = np.max(avg_Q_start[0])
            estimated_start_values.append(perceived_value)
            
        results[algo] = estimated_start_values
        
    return results, true_optimal_value


def plot_overestimation_profiles(results, true_optimal_value, title="Overestimation Bias Profiles"):
    plt.figure(figsize=(10, 6))
    
    for algo, values in results.items():
        plt.plot(values, label=f"{algo} Predicted Return", alpha=0.8)
        
    plt.axhline(y=true_optimal_value, color='r', linestyle='--', label="True Optimal Return V*(s0)")
    
    plt.title(title)
    plt.xlabel("Training Episodes")
    plt.ylabel("Estimated Value of Start State max_a Q(s0, a)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()


if __name__ == "__main__":
    # --- 1. Test across all 3 environment configurations [1] ---
    environments = ["standard", "high_variance", "long_safe_path"]
    for env in environments:
        print(f"Running {env} Environment...")
        results, true_val = run_experiment(env_type=env, state_size=10, ensemble_size=5)
        plot_overestimation_profiles(results, true_val, f"{env}_Env_Bias_Profile")

    # --- 2. Evaluate the effect of state size (increments of 5) [3] ---
    state_sizes = [5, 10, 15, 20]
    for s_size in state_sizes:
        print(f"Running standard env with state size {s_size}...")
        results, true_val = run_experiment(env_type="standard", state_size=s_size, ensemble_size=5)
        plot_overestimation_profiles(results, true_val, f"State_Size_{s_size}_Profile")

    # --- 3. Evaluate the effect of ensemble size (increments of 1) [3] ---
    ensemble_sizes = [2, 3, 4, 5]
    for e_size in ensemble_sizes:
        print(f"Running standard env with ensemble size {e_size}...")
        results, true_val = run_experiment(env_type="standard", state_size=10, ensemble_size=e_size)
        plot_overestimation_profiles(results, true_val, f"Ensemble_Size_{e_size}_Profile")
        
    print("Experiments complete. Plots saved to the current directory.")
