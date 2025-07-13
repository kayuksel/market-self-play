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

    shared_encoder = SharedEncoder(env.observation_space.shape[0], hidden_dim=128, use_lstm=False).to(device)
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

                # Critic update
                c_opt.zero_grad()
                loss_c = F.mse_loss(critic(S_t, A_t), target_q)
                loss_c.backward()
                c_opt.step()

                # Actor update
                for i, actor in enumerate(actor_list):
                    a_opt[i].zero_grad()
                    loss_a = -critic(S_t, actor(S_t)).mean()
                    if algorithm == "SAC":
                        # SAC uses an entropy regularization term
                        loss_a -= 0.1 * critic_2(S_t, actor(S_t)).mean()
                    loss_a.backward()
                    a_opt[i].step()

                # Soft update targets
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

            # Collect final wealths
            finals = [ag.value() for ag in env.agents]

            # ===== Replace bottom 20% with top 20% =====
            frac = 0.2
            n_rep = max(1, int(n_agents * frac))
            sorted_idx = np.argsort(finals)
            losers = sorted_idx[:n_rep]
            winners = sorted_idx[-n_rep:]
            for loser_idx, winner_idx in zip(losers, winners[::-1]):
                actor_list[loser_idx].load_state_dict(
                    actor_list[winner_idx].state_dict()
                )
            # ============================================

            tot0 = sum(env.init_vals)
            tot1 = sum(finals)
            deltas = [float((fv / tot1) - (iv / tot0)) for iv, fv in zip(env.init_vals, finals)]
            shares = [float(fv / tot1) for fv in finals]
            print(f"Ep {ep} | Δ-Shares: {[round(d, 4) for d in deltas]} | Shares: {[round(s, 4) for s in shares]} | Total Wealth: {sum(finals)}")
