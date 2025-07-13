import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import gym
from gym import spaces
import copy

# ===== Market =====
class Market:
    def __init__(self, assets, fee_rate=0.0):
        self.assets = assets
        self.fee_rate = fee_rate
        self.reset()

    def reset(self):
        self.prices = {a: 1.0 for a in self.assets}
        self.executed_volume = {a: 0 for a in self.assets}  # Initialize total volume for VWAP
        self.executed_value = {a: 0 for a in self.assets}  # Initialize total value for VWAP
        self.buy_orders = []
        self.sell_orders = []

    def add_agents(self, agents):
        self.agents = agents

    def get_prices(self):
        return np.array([self.prices[a] for a in self.assets], dtype=np.float32)

    def place_order(self, agent, side, asset_idx, price, qty):
        (self.buy_orders if side=='buy' else self.sell_orders).append({
            'agent': agent, 'asset': asset_idx, 'price': price, 'qty': qty
        })

    def get_order_book_entries(self, asset_idx):
        """
        Returns a list of individual orders for a specific asset, with fields:
        - price
        - qty
        - agent (name)
        - side: 'buy' or 'sell'
        """
        orders = []
        for order in self.buy_orders:
            if order['asset'] == asset_idx:
                orders.append({
                    'price': order['price'],
                    'qty': order['qty'],
                    'agent': order['agent'].name,
                    'side': 'buy'
                })
        for order in self.sell_orders:
            if order['asset'] == asset_idx:
                orders.append({
                    'price': order['price'],
                    'qty': order['qty'],
                    'agent': order['agent'].name,
                    'side': 'sell'
                })
        return orders


    def match_orders(self):
        trade_occurred = False
        buyers = {asset: set() for asset in self.assets}  # Track unique buyers for each asset
        sellers = {asset: set() for asset in self.assets}  # Track unique sellers for each asset
        executed_volumes = {asset: 0 for asset in self.assets}
        executed_values = {asset: 0 for asset in self.assets}

        # Tracking VWAP for the current batch of executed orders
        current_executed_value = {a: 0 for a in self.assets}
        current_executed_volume = {a: 0 for a in self.assets}

        for idx, asset in enumerate(self.assets):
            while True:
                # Get buy orders (bids) and sell orders (asks) for this asset
                bids = [o for o in self.buy_orders if o['asset'] == idx]
                asks = [o for o in self.sell_orders if o['asset'] == idx]

                if not bids or not asks:
                    break

                # Sort the orders
                best_bid = max(bids, key=lambda o: o['price'])
                best_ask = min(asks, key=lambda o: o['price'])

                if best_bid['price'] < best_ask['price']:
                    break

                # Execute the trade at the ask price
                trade_price = best_ask['price']
                trade_qty = min(best_bid['qty'], best_ask['qty'])

                max_afford = best_bid['agent'].money / (trade_price * (1 + self.fee_rate))
                trade_qty = min(trade_qty, max_afford)

                avail = best_ask['agent'].get_asset_quantity(asset)
                trade_qty = min(trade_qty, avail)

                if trade_qty <= 0:
                    break

                cost = trade_price * trade_qty * (1 + self.fee_rate)
                proceeds = trade_price * trade_qty

                best_bid['agent'].money -= cost
                best_bid['agent'].portfolio[asset] = best_bid['agent'].portfolio.get(asset, 0) + trade_qty

                best_ask['agent'].money += proceeds
                best_ask['agent'].portfolio[asset] = best_ask['agent'].portfolio.get(asset, 0) - trade_qty

                best_bid['qty'] -= trade_qty
                best_ask['qty'] -= trade_qty

                if best_bid['qty'] <= 0:
                    self.buy_orders.remove(best_bid)
                if best_ask['qty'] <= 0:
                    self.sell_orders.remove(best_ask)

                # Update executed trade totals for both overall and current VWAP calculation
                self.executed_value[asset] += trade_price * trade_qty
                self.executed_volume[asset] += trade_qty
                current_executed_value[asset] += trade_price * trade_qty
                current_executed_volume[asset] += trade_qty

                trade_occurred = True
                buyers[asset].add(best_bid['agent'])
                sellers[asset].add(best_ask['agent'])
                executed_volumes[asset] += trade_qty
                executed_values[asset] += trade_price * trade_qty

        # After matching all orders, update the VWAP for each asset based on executed trades
        for asset in self.assets:
            if self.executed_volume[asset] > 0:
                # Calculate overall VWAP from all executed trades
                overall_vwap = self.executed_value[asset] / self.executed_volume[asset]
                
                # Calculate current executed VWAP from the current batch of executed trades
                current_executed_vwap = current_executed_value[asset] / current_executed_volume[asset] if current_executed_volume[asset] > 0 else 0

                self.prices[asset] = overall_vwap  # Update last trade price with overall VWAP

                # Print both overall VWAP and current executed VWAP details
                if executed_volumes[asset] > 0:
                    buyers_list = ', '.join([buyer.name for buyer in buyers[asset]])
                    sellers_list = ', '.join([seller.name for seller in sellers[asset]])
                    print(f"Asset: {asset} | VWAP: {overall_vwap} | Executed VWAP: {current_executed_vwap} | Executed Volume: {executed_volumes[asset]} | Buyers: {buyers_list} | Sellers: {sellers_list}")

        return trade_occurred

    def get_order_book_histogram(self, asset_idx, nbins=100, price_window_pct=0.1, exclude_agent=None):
        """
        Returns a single histogram with:
        - Positive volumes for buy orders.
        - Negative volumes for sell orders.
        """
        asset = self.assets[asset_idx]
        anchor = self.prices[asset]
        lo, hi = anchor * (1 - price_window_pct), anchor * (1 + price_window_pct)
        bins = np.linspace(lo, hi, nbins + 1)

        # Ensure bins are in increasing order (in case of any reverse price_window_pct)
        bins = np.sort(bins)

        # Get buy and sell orders
        bids = [o for o in self.buy_orders if o['asset'] == asset_idx]
        asks = [o for o in self.sell_orders if o['asset'] == asset_idx]

        # If excluding an agent's own orders, filter them out
        if exclude_agent:
            bids = [o for o in bids if o['agent'] != exclude_agent]
            asks = [o for o in asks if o['agent'] != exclude_agent]

        # Calculate buy and sell volumes
        buy_vols = np.histogram([o['price'] for o in bids], bins=bins, weights=[o['qty'] for o in bids])[0] if bids else np.zeros(nbins, dtype=np.float32)
        sell_vols = np.histogram([o['price'] for o in asks], bins=bins, weights=[o['qty'] for o in asks])[0] if asks else np.zeros(nbins, dtype=np.float32)

        # Combine buy and sell volumes into a single set of volumes:
        # Buy orders will be positive, and sell orders will be negative
        combined_vols = buy_vols - sell_vols  # Subtract sell_vols to get a negative volume for sells

        # Normalize by total market volume (combining buy and sell volumes)
        total_buy_vol = sum(o['qty'] for o in bids)
        total_sell_vol = sum(o['qty'] for o in asks)
        total_market_vol = total_buy_vol + total_sell_vol

        if total_market_vol == 0:  # Avoid division by zero
            total_market_vol = 1  # To prevent division by zero

        combined_vols /= total_market_vol  # Normalize

        return combined_vols.astype(np.float32)


class TradingEnv(gym.Env):
    def __init__(self, assets, fee_rate=0.0, nbins=100, price_window_pct=0.1):
        super().__init__()
        self.assets = assets
        self.market = Market(assets, fee_rate)
        self.nbins = nbins
        self.pct = price_window_pct
        self.no_trade_counter = 0  # Track periods without executed trades
        self.history_len = 5  # Sequence length of 5

        # Observation: [order-book histograms] for each asset
        #             + [normalized portfolio allocations (assets + cash)] for each agent
        obs_dim = (
            len(assets) * (2 * nbins)  # buy + sell volumes per bin for each asset
            + len(assets) + 2           # normalized holdings + cash for each agent
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )

        # Action: [price_pct, qty_frac] for each asset
        self.action_space = spaces.Box(
            low=np.array([-1.] * len(assets) + [-1.] * len(assets), dtype=np.float32),  # price_pct and qty_frac for each asset
            high=np.array([1.] * len(assets) + [1.] * len(assets), dtype=np.float32),
            dtype=np.float32
        )

        self.obs_history = []  # To store history of observations for each agent

    def reset(self):
        self.market.reset()
        self.agents = []
        self.obs_history = []
        
        for i in range(5):
            init_port = {a: 20 for a in self.assets}
            ag = Agent(f"agent{i}", 20.0, init_port, self.market)
            ag.reset()
            ag.action_low, ag.action_high = self.action_space.low, self.action_space.high
            self.agents.append(ag)
        
        self.market.add_agents(self.agents)
        self.init_vals = [ag.value() for ag in self.agents]
        self.no_trade_counter = 0  # Reset counter at the start of a new episode
        
        # Initialize observation history for each agent as an empty list of sequences
        self.obs_history = [[self._obs(ag)] for ag in self.agents]  # Start with the first observation
        
        return [self._get_sequence_obs(i) for i in range(len(self.agents))]

    def step(self, actions):
        # 1) Agents place orders for all assets (price_pct and qty_frac for each asset)
        for ag, act in zip(self.agents, actions):
            ag.act(act)  # The act method will apply actions for all assets simultaneously
        
        # 2) Match all orders and check if any trade occurred
        trade_occurred = self.market.match_orders()

        # Check if any trades were executed
        if trade_occurred:
            self.no_trade_counter = 0  # Reset counter if trades occurred
        else:
            self.no_trade_counter += 1  # Increment counter if no trades

        # 3) Compute observations and rewards
        obs, rewards = [], []
        for ag in self.agents:
            new_val = ag.value()
            reward = new_val - self.init_vals[self.agents.index(ag)]
            obs.append(self._obs(ag))  # No need to pass asset_idx, as it's based on all assets
            rewards.append(reward)

        # Update the observation history
        for i, ag in enumerate(self.agents):
            if len(self.obs_history[i]) >= self.history_len:
                self.obs_history[i].pop(0)  # Keep only the last 5 observations
            self.obs_history[i].append(self._obs(ag))

            # If the history is smaller than 5, repeat the most recent observation to fill the history
            while len(self.obs_history[i]) < self.history_len:
                self.obs_history[i].insert(0, self._obs(ag))  # Insert the latest observation at the beginning

        # 4) Reshape observations into (batch_size, seq_len, input_size)
        sequence_obs = [self._get_sequence_obs(i) for i in range(len(self.agents))]

        # Reset the episode if no trades happened for 15 consecutive periods
        done = [False] * len(self.agents)
        if self.no_trade_counter >= 20:
            done = [True] * len(self.agents)  # Indicate the episode is done
            self.reset()  # Reset the environment if no trades for 15 periods

        return sequence_obs, rewards, done, {}

    def _get_sequence_obs(self, agent_idx):
        # Ensure the history has exactly 5 observations
        if len(self.obs_history[agent_idx]) < self.history_len:
            while len(self.obs_history[agent_idx]) < self.history_len:
                self.obs_history[agent_idx].insert(0, self.obs_history[agent_idx][-1])  # Repeat the last observation
        
        # Concatenate the last 5 observations for the agent (Shape: (5, 605))
        seq_obs = np.array(self.obs_history[agent_idx])  # Shape should be (5, 605)

        # Reshape to (1, seq_len, input_size) where seq_len = 5 and input_size = 605
        seq_obs = seq_obs.reshape(self.history_len, -1)  # Reshaped to (5, 605)
        
        return seq_obs

    def _obs(self, ag):
        wealth = ag.value() + 1e-6  # Adding a small value to avoid division by zero

        all_hists = []
        for idx in range(len(self.assets)):
            # Get combined histogram (buy - sell volumes) for all agents
            combined_vols_all = self.market.get_order_book_histogram(
                idx, nbins=self.nbins, price_window_pct=self.pct, exclude_agent=None
            )

            # Get combined histogram (buy - sell volumes) excluding the current agent's orders
            combined_vols_other = self.market.get_order_book_histogram(
                idx, nbins=self.nbins, price_window_pct=self.pct, exclude_agent=ag
            )

            # Concatenate both histograms (all agents' perspective + other agents' perspective)
            all_hists.append(np.concatenate([combined_vols_all, combined_vols_other], axis=0))

        hist_obs = np.concatenate(all_hists, dtype=np.float32)

        # 2) Normalized portfolio allocations [assets..., cash]
        holdings_vals = [ag.get_asset_quantity(a) * self.market.prices[a] for a in self.assets]
        cash_val = ag.money
        alloc = np.array(holdings_vals + [cash_val], dtype=np.float32) / wealth

        # 3) Wealth share
        total_wealth = sum(ag.value() for ag in self.agents)
        wealth_share = ag.value() / total_wealth

        # Return a 1D observation of shape (605,)
        return np.concatenate([hist_obs, alloc, np.array([wealth_share], dtype=np.float32)], axis=0)

# ===== OUNoise =====
class OUNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.3):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(size) * mu  # Noise size should match the action space size

    def reset(self):
        self.state = np.ones_like(self.state) * self.mu

    def __call__(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state  # Return noise

# ===== Agent =====
class Agent:
    def __init__(self, name, money, portfolio, market):
        self.name = name
        self.money = money
        self.portfolio = dict(portfolio)
        self.market = market
        self.noise = OUNoise(2 * len(self.market.assets))  # Noise for all assets' actions

    def reset(self):
        self.noise.reset()

    def value(self):
        """Compute the total value of the agent (money + portfolio value)"""
        return self.money + sum(self.market.prices[a] * q for a, q in self.portfolio.items())

    def get_asset_quantity(self, a):
        """Get the quantity of a specific asset in the portfolio"""
        return self.portfolio.get(a, 0)

    def cancel_opposite_orders(self, side):
        """Cancel all orders of the opposite type (buy/sell)"""
        opposite_side = 'sell' if side == 'buy' else 'buy'
        if opposite_side == 'buy':
            self.market.buy_orders = [o for o in self.market.buy_orders if o['agent'] != self]
        else:
            self.market.sell_orders = [o for o in self.market.sell_orders if o['agent'] != self]

    def act(self, cont_act):
        """Agent takes actions for all assets simultaneously"""
        cont_act = np.nan_to_num(cont_act)  # Clean any NaNs in actions
        cont_act = np.clip(cont_act, self.action_low, self.action_high)  # Clip actions within valid range

        # Loop over each asset and execute actions for each
        for i in range(len(self.market.assets)):
            price_pct = cont_act[i]  # price_pct for the i-th asset
            qty_frac = cont_act[len(self.market.assets) + i]  # qty_frac for the i-th asset
            
            # Get the asset index and apply the action
            idx = i
            asset = self.market.assets[idx]
            mid_price = self.market.prices[asset]
            price = mid_price * (1 + np.tanh(price_pct))  # Price adjustment using price_pct

            # Determine whether it's a buy or sell order
            if qty_frac >= 0:  # Buy order
                side = 'buy'
                max_q = self.money / price if price > 0 else 0  # Max quantity that can be bought
                qty = float(qty_frac) * max_q  # Adjusted quantity to buy
            else:  # Sell order
                side = 'sell'
                available = self.get_asset_quantity(asset)  # Available quantity to sell
                qty = float(-qty_frac) * available  # Adjusted quantity to sell

            # Cancel opposite orders (if any)
            self.cancel_opposite_orders(side)

            # Place the order for this asset
            self.place_order(idx, price, qty, side)

    def place_order(self, asset_idx, price, qty, side):
        """Place an order in the market (buy or sell)"""
        self.market.place_order(self, side, asset_idx, price, qty)