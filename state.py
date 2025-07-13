import threading

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.market = None
        self.agents = None
        self.episode = 0
        self.step_wealth_history = []  # Stores wealth share at each step (episode, step, agent_wealth_share)

shared_state = SharedState()
