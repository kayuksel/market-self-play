
# Market Self-Play

This repository provides a framework for simulating market environments where multiple agents engage in self-play trading using reinforcement learning (RL). The environment is based on a continuous double auction (CDA) model, allowing agents to place buy and sell orders, interact with a limit order book (LOB), and learn optimal trading strategies through self-play.

## Features

- **Continuous Double Auction (CDA) Environment**: Simulates a market where agents can place limit orders and interact with the LOB.
- **Multi-Agent Reinforcement Learning (MARL)**: Supports training of multiple agents using RL algorithms.
- **Zero-Sum Game Setup**: Agents compete in a zero-sum environment, where the gain of one agent is the loss of another.
- **Modular Design**: Easily extendable components for agents, environment, and training loops.

## File Structure

```plaintext
market-self-play/
│
├── trading_env.py      # Defines the trading environment
├── rl_training.py      # Contains the RL agent and training loop
├── state.py            # Manages the state representation
├── main.py             # Entry point for running the simulation
└── README.md           # Project documentation
```

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.
