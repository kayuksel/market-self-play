import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from collections import deque
import gym
from gym import spaces
import copy

from trading_env import TradingEnv, Agent
from state import shared_state
import threading

class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.25):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.mish(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        return F.mish(x + residual)


class Actor(nn.Module):
    def __init__(self, shared_encoder, n_assets, in_dim=128, hidden_dim=128, depth=2, dropout=0.25):
        super(Actor, self).__init__()
        self.shared_encoder = shared_encoder
        self.res_blocks = nn.Sequential(*[
            ResidualMLPBlock(in_dim, hidden_dim, dropout) for _ in range(depth)
        ])
        self.price_head = nn.Linear(in_dim, n_assets)
        self.qty_head = nn.Linear(in_dim, n_assets)

    def forward(self, x):
        x = self.shared_encoder(x)  # (batch, in_dim)
        x = self.res_blocks(x)
        price_pct = torch.tanh(self.price_head(x))   # ∈ [-1, 1]
        qty_frac = torch.sigmoid(self.qty_head(x))   # ∈ [0, 1]
        return torch.cat([price_pct, qty_frac], dim=1)  # (batch, 2*n_assets)

class Critic(nn.Module):
    def __init__(self, shared_encoder, obs_dim, n_assets, in_dim=128, hidden_dim=128, depth=2, dropout=0.25):
        super(Critic, self).__init__()
        self.shared_encoder = shared_encoder
        self.res_blocks = nn.Sequential(*[
            ResidualMLPBlock(in_dim + 2 * n_assets, hidden_dim, dropout) for _ in range(depth)
        ])
        self.head = nn.Linear(in_dim + 2 * n_assets, 1)

    def forward(self, s, a):
        s_encoded = self.shared_encoder(s)  # (batch, in_dim)
        x = torch.cat([s_encoded, a], dim=1)
        x = self.res_blocks(x)
        return self.head(x).squeeze(-1)
 

class SharedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.25, use_lstm=True):
        super(SharedEncoder, self).__init__()
        self.use_lstm = use_lstm
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Apply dropout only if num_layers > 1
        rnn_dropout = dropout if num_layers > 1 else 0.0

        if self.use_lstm:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout
            )
        
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        self.rnn.flatten_parameters()

        # Initialize hidden state with shape (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)

        if self.use_lstm:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        out = out[:, -1, :]  # Last timestep
        return F.relu(self.fc(out))


# ===== ReplayBuffer =====
def default_to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(device).type(dtype)
    else:
        return torch.tensor(x, dtype=dtype, device=device)

class ReplayBuffer:
    """
    A simple uniform (non-prioritized) replay buffer for experience replay.
    """
    def __init__(self, capacity=20000, device="cuda"):
        self.capacity = capacity
        self.device = device

        # storage pointers
        self.ptr = 0
        self.size = 0

        # buffers (lazy init)
        self.s_buf = None
        self.a_buf = None
        self.r_buf = None
        self.s2_buf = None
        self.d_buf = None

    def push(self, s, a, r, s2, d):
        # to tensor
        s = default_to_tensor(s, self.device)
        a = default_to_tensor(a, self.device)
        r = default_to_tensor(r, self.device).unsqueeze(0)
        s2 = default_to_tensor(s2, self.device)
        d = default_to_tensor(d, self.device).unsqueeze(0)

        # lazy initialization of buffers
        if self.s_buf is None:
            obs_shape = s.shape
            act_shape = a.shape
            self.s_buf  = torch.zeros((self.capacity, *obs_shape),  device=self.device, dtype=s.dtype)
            self.a_buf  = torch.zeros((self.capacity, *act_shape), device=self.device, dtype=a.dtype)
            self.r_buf  = torch.zeros((self.capacity, 1),           device=self.device, dtype=r.dtype)
            self.s2_buf = torch.zeros((self.capacity, *obs_shape),  device=self.device, dtype=s2.dtype)
            self.d_buf  = torch.zeros((self.capacity, 1),           device=self.device, dtype=d.dtype)

        # store transition
        self.s_buf[self.ptr]  = s
        self.a_buf[self.ptr]  = a
        self.r_buf[self.ptr]  = r
        self.s2_buf[self.ptr] = s2
        self.d_buf[self.ptr]  = d

        # advance pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # uniformly sample indices
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)

        # fetch batches
        S  = self.s_buf[idx]
        A  = self.a_buf[idx]
        R  = self.r_buf[idx].squeeze(1)
        S2 = self.s2_buf[idx]
        D  = self.d_buf[idx].squeeze(1)

        return S, A, R, S2, D, idx

    def __len__(self):
        return self.size

def train_rl(algorithm: str = "TD3", num_eps: int = 200):
    """
    Train the agent with the selected RL algorithm using AMP (autocast & GradScaler).
    Supports DDPG, TD3, SAC with Prioritized Experience Replay and shared encoder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Environment ---
    assets = ["AAPL", "GOOG", "TSLA", "MSFT"]
    env = TradingEnv(assets, fee_rate=-0.001)
    obs_n = env.reset()

    n_agents = len(obs_n)
    n_assets = len(assets)
    gamma, tau = 0.99, 0.005

    # --- Models ---
    shared_encoder = SharedEncoder(
        input_dim=env.observation_space.shape[0],
        hidden_dim=128,
        use_lstm=False
    ).to(device)

    actor_list = [Actor(shared_encoder, n_assets).to(device) for _ in range(n_agents)]
    critic = Critic(
        shared_encoder,
        obs_dim=env.observation_space.shape[0],
        n_assets=n_assets
    ).to(device)
    critic_tgt = copy.deepcopy(critic)

    if algorithm != "DDPG":
        critic_2 = Critic(
            shared_encoder,
            obs_dim=env.observation_space.shape[0],
            n_assets=n_assets
        ).to(device)
        critic_2_tgt = copy.deepcopy(critic_2)

    # --- Optimizers & AMP scaler ---
    a_opt = [torch.optim.AdamW(actor.parameters(), lr=1e-4) for actor in actor_list]
    c_opt = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    if algorithm != "DDPG":
        c_opt_2 = torch.optim.AdamW(critic_2.parameters(), lr=1e-3)
    scaler = GradScaler()

    # --- Replay Buffer ---
    buf = ReplayBuffer(capacity=20000, device=device)

    step_counter = 0

    for ep in range(num_eps):
        if ep > 0:
            obs_n = env.reset()

        while True:
            # --- Batched shared-encoder pass (inference) ---
            obs_array = np.stack(obs_n).astype(np.float32)
            obs_tensor = torch.from_numpy(obs_array).to(device)
            with torch.no_grad(), autocast('cuda'):
                x_enc = shared_encoder(obs_tensor)
                actions_list = []
                for i, actor in enumerate(actor_list):
                    x_i = x_enc[i:i+1]
                    x = actor.res_blocks(x_i)
                    price_pct = torch.tanh(actor.price_head(x))
                    qty_frac = torch.sigmoid(actor.qty_head(x))
                    actions_list.append(torch.cat([price_pct, qty_frac], dim=1).squeeze(0))
                batch_actions = torch.stack(actions_list, dim=0).cpu().numpy()

            # Exploration noise
            actions = batch_actions + np.stack([agent.noise() for agent in env.agents])
            next_obs_n, rewards_n, done, _ = env.step(actions)

            # Logging
            with shared_state.lock:
                shared_state.market = env.market
                shared_state.agents = env.agents
                total_val = sum(a.value() for a in env.agents)
                for agent in env.agents:
                    shared_state.step_wealth_history.append((ep, step_counter, agent.name, agent.value() / total_val))
                step_counter += 1

            # Store transition
            for ob, ac, rw, nob, d in zip(obs_n, actions, rewards_n, next_obs_n, done):
                buf.push(ob, ac, rw, nob, d)
            obs_n = next_obs_n

            # --- Learning step with AMP ---
            if len(buf) >= 256:
                S, A, R, S2, D, weights = buf.sample(256)

                # Compute target Q-values
                with torch.no_grad(), autocast('cuda'):
                    agent_idx = ep % n_agents
                    A2 = actor_list[agent_idx](S2)
                    Q2_1 = critic_tgt(S2, A2)
                    if algorithm != "DDPG":
                        Q2_2 = critic_2_tgt(S2, A2)
                        if algorithm == "TD3":
                            target_q = R + gamma * torch.min(Q2_1, Q2_2)
                        else:  # SAC
                            target_q = R + gamma * (Q2_1 - Q2_2)
                    else:
                        target_q = R + gamma * Q2_1

                # Critic 1 update
                c_opt.zero_grad()
                with autocast('cuda'):
                    td_errors = critic(S, A) - target_q
                    loss_c1 = (weights * td_errors.pow(2)).mean()
                scaler.scale(loss_c1).backward()
                scaler.unscale_(c_opt)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                scaler.step(c_opt)

                # Critic 2 update
                if algorithm != "DDPG":
                    c_opt_2.zero_grad()
                    with autocast('cuda'):
                        loss_c2 = F.mse_loss(critic_2(S, A), target_q)
                    scaler.scale(loss_c2).backward()
                    scaler.unscale_(c_opt_2)
                    torch.nn.utils.clip_grad_norm_(critic_2.parameters(), 1.0)
                    scaler.step(c_opt_2)

                # Delayed policy update
                if step_counter % 2 == 0:
                    for i, actor in enumerate(actor_list):
                        a_opt[i].zero_grad()
                        with autocast('cuda'):
                            q_val = critic(S, actor(S))
                            loss_a = -q_val.mean()
                            if algorithm == "SAC":
                                q_val2 = critic_2(S, actor(S))
                                loss_a -= 0.1 * q_val2.mean()
                        scaler.scale(loss_a).backward()
                        scaler.unscale_(a_opt[i])
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                        scaler.step(a_opt[i])

                    # Soft target updates
                    for p, pt in zip(critic.parameters(), critic_tgt.parameters()):
                        pt.data.mul_(1 - tau)
                        pt.data.add_(tau * p.data)
                    if algorithm != "DDPG":
                        for p, pt in zip(critic_2.parameters(), critic_2_tgt.parameters()):
                            pt.data.mul_(1 - tau)
                            pt.data.add_(tau * p.data)

                scaler.update()

            # Evolutionary weight copying on mixed outcomes
            finals = [ag.value() for ag in env.agents]
            n_rep = max(1, int(n_agents * 0.2))
            losers = np.argsort(finals)[:n_rep]
            winners = np.argsort(finals)[-n_rep:]

            for loser, winner in zip(losers, winners[::-1]):
                for p_l, p_w in zip(actor_list[loser].parameters(), actor_list[winner].parameters()):
                    p_l.data.copy_(p_w.data)

            if any(done):
                print(f"Episode {ep} ended due to insufficient trades.")
                break

        # End of episode logging
        tot0, tot1 = sum(env.init_vals), sum([ag.value() for ag in env.agents])
        deltas = [(fv/tot1) - (iv/tot0) for iv, fv in zip(env.init_vals, finals)]
        shares = [fv/tot1 for fv in finals]
        print(
            f"Ep {ep} | Δ-Shares: {[round(d,4) for d in deltas]} | "
            f"Shares: {[round(s,4) for s in shares]} | Total Wealth: {round(tot1,2)}"
        )

    print("Training complete.")
