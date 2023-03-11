import gym, random, pdb
import numpy as np

class Agent:
    def __init__(self, name, money, assets={}):
        self.name = name
        self.money = money
        self.assets = assets

    def place_buy_order(self, asset, price, quantity):
        return True if self.money >= price * quantity else False

    def place_sell_order(self, asset, price, quantity):
        return True if asset in self.assets and self.assets[asset] >= quantity else False

    def get_asset_quantity(self, asset):
        return self.assets[asset] if asset in self.assets else 0

class Market:
    def __init__(self, assets):
        self.assets = assets
        self.buy_orders = {asset: [] for asset in assets}
        self.sell_orders = {asset: [] for asset in assets}
        self.prices = {asset: 0 for asset in assets}

    def add_buy_order(self, agent, asset, price, quantity):
        self.buy_orders[asset].append((price, agent, quantity))
        self.buy_orders[asset] = sorted(self.buy_orders[asset], key=lambda x: x[0], reverse=True)

    def add_sell_order(self, agent, asset, price, quantity):
        self.sell_orders[asset].append((price, agent, quantity))
        self.sell_orders[asset] = sorted(self.sell_orders[asset], key=lambda x: x[0])

    def match_orders(self):
        for asset in self.assets:
            self.buy_orders[asset].sort(key=lambda x: -x[0])
            self.sell_orders[asset].sort(key=lambda x: x[0])
            i = 0
            j = 0
            while i < len(self.buy_orders[asset]) and j < len(self.sell_orders[asset]):
                buy_order = self.buy_orders[asset][i]
                sell_order = self.sell_orders[asset][j]

                if buy_order[0] > sell_order[0]:
                    quantity = min(buy_order[2], sell_order[2])
                    buy_agent = buy_order[1]
                    sell_agent = sell_order[1]
                    price = (sell_order[0] + buy_order[0]) / 2.0

                    if buy_agent.place_buy_order(asset, price, quantity) and sell_agent.place_sell_order(asset, price, quantity):
                        print(f"{buy_agent.name} and {sell_agent.name} exchanged {quantity} units of {asset} at {price}")

                        if asset in buy_agent.assets:
                            buy_agent.assets[asset] += quantity
                        else:
                            buy_agent.assets[asset] = quantity

                        sell_agent.assets[asset] -= quantity
                        buy_agent.money -= quantity * price
                        sell_agent.money += quantity * price

                        self.prices[asset] = price
                        if buy_order[2] > quantity:
                            self.add_buy_order(buy_agent, asset, buy_order[0], buy_order[2] - quantity)
                        else:
                            self.buy_orders[asset].pop(i)
                        if sell_order[2] > quantity:
                            self.add_sell_order(sell_agent, asset, sell_order[0], sell_order[2] - quantity)
                        else:
                            self.sell_orders[asset].pop(j)
                    else:
                        break
                i+=1
                j+=1

    def get_total_asset_value(self):
        total_value = 0
        for agent in agents:
            for asset in self.prices:
                total_value += agent.get_asset_quantity(asset) * self.prices[asset]
        return total_value


    def get_total_market_value(self):
        return sum([agent.money for agent in agents]) + self.get_total_asset_value()

    def recalibrate_market(self):
        fix = 1000.0 / self.get_total_market_value()
        for asset in self.assets: self.prices[asset] = self.prices[asset] * fix
        for agent in agents: agent.money = agent.money * fix


    def run_iteration(self):
        for agent in agents:
            buy_or_sell = random.choice(["buy", "sell"])
            if buy_or_sell == "buy":
                asset = random.choice(self.assets)
                price = random.uniform(self.prices[asset]*0.8, self.prices[asset]*1.2)
                quantity = random.uniform(0, agent.money/price)
                if agent.place_buy_order(asset, price, quantity):
                    self.add_buy_order(agent, asset, price, quantity)
            else:
                if agent.assets:
                    asset = random.choice(list(agent.assets.keys()))
                    price = random.uniform(self.prices[asset]*0.8, self.prices[asset]*1.2)
                    quantity = random.uniform(0, agent.assets[asset])
                    if agent.place_sell_order(asset, price, quantity):
                        self.add_sell_order(agent, asset, price, quantity)
        self.match_orders()
        print(self.get_total_market_value())

    def create_random_assets(self):
        """helper function that creates a random dictionary of assets and quantities"""
        random_assets = {}
        for asset in self.assets:
            random_assets[asset] = random.randint(0, 100)
        return random_assets

class TradingEnv(gym.Env):
    def __init__(self, assets):
        self.assets = assets
        self.market = Market(assets)
        self.agents = [Agent("agent" + str(i), 50, self.market.create_random_assets()) for i in range(5)]
        self.observation_space = gym.spaces.Box(low=0, high=float('inf'), shape=(len(assets) * 2 + 1,))
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(2),gym.spaces.Discrete(len(assets)),gym.spaces.Discrete(100), gym.spaces.Discrete(100)))

    def step(self, actions):
        self.market.match_orders()
        total_market_value = self.market.get_total_market_value()
        for i, action in enumerate(actions):
            agent = self.agents[i]
            if action[0] == 0:
                if agent.place_buy_order(self.assets[action[1]], action[2], action[3]):
                    self.market.add_buy_order(agent, self.assets[action[1]], action[2], action[3])
            elif action[0] == 1:
                if agent.place_sell_order(self.assets[action[1]], action[2], action[3]):
                    self.market.add_sell_order(agent, self.assets[action[1]], action[2], action[3])

        self.market.match_orders()
        highest_bid = [max([order[0] for order in self.market.buy_orders[asset]]) if len(self.market.buy_orders[asset])>0 else 0 for asset in self.assets]
        lowest_ask = [min([order[0] for order in self.market.sell_orders[asset]]) if len(self.market.sell_orders[asset])>0 else 0 for asset in self.assets]
        observation = highest_bid + lowest_ask + [total_market_value]
        agent_values = [agent.money + sum([agent.get_asset_quantity(asset) * self.market.prices[asset] for asset in self.assets]) for agent in self.agents]
        rewards = [agent_value / total_market_value for agent_value in agent_values]
        done = False
        info = {}
        return observation, rewards, done, info

    def reset(self):
        self.market = Market(self.assets)
        self.agents = [Agent("agent" + str(i), 50, self.market.create_random_assets()) for i in range(5)]

        highest_bid = [max([order[0] for order in self.market.buy_orders[asset]]) if len(self.market.buy_orders[asset])>0 else 0 for asset in self.assets]
        lowest_ask = [min([order[0] for order in self.market.sell_orders[asset]]) if len(self.market.sell_orders[asset])>0 else 0 for asset in self.assets]
        total_market_value = self.market.get_total_market_value()
        observation = highest_bid + lowest_ask + [total_market_value]
        return observation

assets = ["asset1", "asset2", "asset3"]

# Create environment
env = TradingEnv(assets)

# Reset the environment
observation = env.reset()
done = False

for i in range(1000):
    # Generate random actions
    actions = [np.random.randint(0, 2, 4) for _ in range(5)]
    # Take a step in the environment
    observation, rewards, done, info = env.step(actions)
    print("Step: ", i)
    print("Rewards: ", rewards)
    print("Observation: ", observation)

'''
agents = [Agent("A", 250), Agent("B", 250), Agent("C", 250), Agent("D", 250)]
market = Market(["gold", "silver", "bronze"])

for agent in agents: agent.assets = market.create_random_assets()

for asset in market.assets:
    market.prices[asset] = random.randint(10, 20)

for i in range(10000):
    market.run_iteration()
    print("Iteration ", i+1)
    for agent in agents:
        print(agent.name, "has money:", agent.money, "and assets:", agent.assets)
        print("Market prices:", market.prices)
'''