
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import gym
from gym import spaces
import copy

from trading_env import TradingEnv, Agent
from state import shared_state
import threading

class Actor(nn.Module):
    def __init__(self, shared_encoder, n_assets, hidden_dim=64):
        super(Actor, self).__init__()
        self.shared_encoder = shared_encoder  # Use the shared encoder instance
        
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.out = nn.Linear(128, 2 * n_assets)  # Output size is 2 * n_assets (price_pct and qty_frac for each asset)

    def forward(self, x):
        x = self.shared_encoder(x)  # Get the shared encoded representation
        x = F.relu(self.fc1(x))
        return self.out(x)  # Output shape: (2 * n_assets,)


class Critic(nn.Module):
    def __init__(self, shared_encoder, obs_dim, n_assets, hidden_dim=64):
        super(Critic, self).__init__()
        self.shared_encoder = shared_encoder  # Use the shared encoder instance
        
        # Define the first layer that accepts both state (obs) and action (a)
        self.fcs = nn.Linear(hidden_dim + (2 * n_assets), 128)  # State + action space size
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)  # Output is a single value: the Q-value

    def forward(self, s, a):
        x = self.shared_encoder(s)
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fcs(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1) 

class SharedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_lstm=True):
        super(SharedEncoder, self).__init__()
        self.use_lstm = use_lstm
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # LSTM or GRU selection based on the use_lstm flag
        if self.use_lstm:
            # Set input_size to 605 (features per observation), batch_first is true
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Fully connected layer after RNN to generate the final state encoding
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # Ensure x has the shape (batch_size, seq_len, input_dim)
        if len(x.shape) == 2:  # If the input is of shape (batch_size, input_dim), we add a seq_len dimension
            x = x.unsqueeze(0)  # Adding seq_len dimension: (batch_size, 1, input_dim)

        self.rnn.flatten_parameters()

        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass through the RNN (LSTM or GRU)
        if self.use_lstm:
            c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
            out, _ = self.rnn(x, (h0, c0))  # LSTM returns the output and the hidden state
        else:
            out, _ = self.rnn(x, h0)  # GRU returns the output and the hidden state
        
        # Use the last time step's output for the next layers (or you can pool across time steps)
        out = out[:, -1, :]  # Select the last time step output
        out = F.relu(self.fc(out))  # Apply fully connected layer
        return out

# ===== ReplayBuffer =====
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        S, A, R, S2, D = map(np.array, zip(*batch))
        return S, A, R, S2, D

    def __len__(self):
        return len(self.buf)


def train_rl(algorithm="TD3"):
    """
    Train the agent with the selected reinforcement learning algorithm (DDPG, TD3, SAC).
    
    Parameters:
    - algorithm (str): The RL algorithm to use ("DDPG", "TD3", or "SAC").
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets = ["AAPL", "GOOG", "TSLA", "MSFT"]
    env = TradingEnv(assets, fee_rate=-0.001)
    env.reset()

    n_agents = len(env.agents)
    n_assets = len(assets)

    shared_encoder = SharedEncoder(env.observation_space.shape[0], hidden_dim=64, use_lstm=True).to(device)
    actor_list = [Actor(shared_encoder, n_assets).to(device) for _ in range(n_agents)]
    critic = Critic(shared_encoder, env.observation_space.shape[0], n_assets).to(device)
    critic_tgt = copy.deepcopy(critic)

    buf = ReplayBuffer()
    a_opt = [torch.optim.AdamW(actor.parameters(), lr=1e-4) for actor in actor_list]
    c_opt = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    gamma, tau = 0.99, 0.005

    if algorithm != "DDPG":
        critic_2 = Critic(shared_encoder, env.observation_space.shape[0], n_assets).to(device)
        critic_2_tgt = copy.deepcopy(critic_2)
        c_opt_2 = torch.optim.AdamW(critic_2.parameters(), lr=1e-3)

    num_eps = 200
    step_counter = 0 

    for ep in range(num_eps):
        obs_n = env.reset()
        ep_rewards = [0] * n_agents

        while True:
            actions = []
            batch_obs = torch.tensor(obs_n, dtype=torch.float32, device=device)
            batch_actions = actor_list[0](batch_obs).detach().cpu().numpy()

            # Exploration for DDPG/TD3/SAC
            actions = batch_actions + np.array([env.agents[i].noise() for i in range(n_agents)])

            next_obs_n, rewards_n, done, _ = env.step(actions)

            with shared_state.lock:

                shared_state.market = env.market
                shared_state.agents = env.agents

                total_val = sum(agent.value() for agent in shared_state.agents)

                # Track wealth share for each agent separately at each step
                for agent in shared_state.agents:
                    shared_state.step_wealth_history.append((
                        ep,
                        step_counter,  # Continuous step count
                        agent.name,  # Agent name
                        agent.value() / total_val  # Wealth share for the agent
                    ))

                step_counter += 1

            if any(done):
                print(f"Episode {ep} finished due to no trades for 20 consecutive periods.")
                break  # Exit the loop if any agent is done (i.e., episode is over)

            for i, (ob, ac, rw, nob) in enumerate(zip(obs_n, actions, rewards_n, next_obs_n)):
                buf.push(ob, ac, rw, nob, done[i])

            obs_n = next_obs_n

            if len(buf) >= 256:
                S, A, R, S2, D = buf.sample(256)
                S_t = torch.tensor(S, dtype=torch.float32, device=device)
                A_t = torch.tensor(A, dtype=torch.float32, device=device)
                R_t = torch.tensor(R, dtype=torch.float32, device=device)
                S2_t = torch.tensor(S2, dtype=torch.float32, device=device)

                with torch.no_grad():
                    current_agent_idx = ep % len(actor_list)
                    A2 = actor_list[current_agent_idx](S2_t)

                    Q2_1 = critic_tgt(S2_t, A2)

                    if algorithm != "DDPG":
                        Q2_2 = critic_2_tgt(S2_t, A2)
                        mult = torch.min(Q2_1, Q2_2) if algorithm == "TD3" else Q2_1 - Q2_2
                        target_q = R_t + gamma * mult
                    else:
                        target_q = R_t + gamma * Q2_1

                # Critic update for all algorithms (DDPG, TD3, SAC)
                c_opt.zero_grad()
                loss_c = F.mse_loss(critic(S_t, A_t), target_q)
                loss_c.backward()
                c_opt.step()

                # Actor update for all algorithms
                for i, actor in enumerate(actor_list):
                    a_opt[i].zero_grad()
                    loss_a = -critic(S_t, actor(S_t)).mean()
                    if algorithm == "SAC":
                        # SAC uses an entropy regularization term
                        loss_a -= 0.1 * critic_2(S_t, actor(S_t)).mean()
                    loss_a.backward()
                    a_opt[i].step()

                for p, pt in zip(critic.parameters(), critic_tgt.parameters()):
                    pt.data.mul_(1 - tau).add_(tau * p.data)

                if algorithm != "DDPG":
                    c_opt_2.zero_grad()
                    c2 = critic_2(S2_t, A2)
                    loss_c2 = F.mse_loss(c2, target_q.detach())
                    loss_c2.backward()
                    c_opt_2.step()

                    for p, pt in zip(critic_2.parameters(), critic_2_tgt.parameters()):
                        pt.data.mul_(1 - tau).add_(tau * p.data)

            total_value = sum(ag.value() for ag in env.agents)
            if any(ag.value() / total_value < 0.05 for ag in env.agents):
                break

            finals = [ag.value() for ag in env.agents]

            winner_idx = np.argmax(finals)
            loser_idx = np.argmin(finals)

            if loser_idx != winner_idx:
                actor_list[loser_idx].load_state_dict(actor_list[winner_idx].state_dict())

            tot0 = sum(env.init_vals)
            tot1 = sum(finals)
            deltas = [float((fv / tot1) - (iv / tot0)) for iv, fv in zip(env.init_vals, finals)]
            shares = [float(fv / tot1) for fv in finals]
            print(f"Ep {ep} | Δ-Shares: {[round(d, 4) for d in deltas]} | Shares: {[round(s, 4) for s in shares]} | Total Wealth: {sum(finals)}")
