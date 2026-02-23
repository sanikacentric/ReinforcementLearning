# Reinforcement Learning - From Scratch in Python

A hands-on Jupyter Notebook implementation of **Reinforcement Learning (RL)** fundamentals, built from scratch using Python. This project walks through the core anatomy of an RL system — Agent, Environment, Policy, and Rewards — with working code examples.

## What is Reinforcement Learning?

Reinforcement Learning is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment** to maximize cumulative **rewards**. Unlike supervised learning, the agent is not told what to do — it discovers the best strategy through trial and error.

## Core Concepts Covered

### The RL Anatomy
- **Agent** — A piece of code that implements a policy and takes actions
- **Environment** — A model of the external world that provides observations and rewards
- **State** — The current situation observed by the agent
- **Action** — What the agent does in each state
- **Reward** — Feedback signal that guides the agent toward the goal
- **Policy** — The agent's strategy mapping states to actions

## What's in the Notebook

```
Tutorial_2_Simple_Reinforcement_Learning_Code.ipynb
```

1. **Simple RL Implementation** — A dummy environment that gives random rewards regardless of action (teaching the basic loop)
2. **Agent Class** — Implements a basic policy with action selection
3. **Environment Class** — Handles action inputs, returns observations and rewards
4. **Training Loop** — Iterative agent-environment interaction cycle
5. **Policy Improvement** — How the agent updates its strategy based on received rewards

## Code Structure

```python
class Environment:
    def step(self, action):
        # Returns (observation, reward, done)
        reward = random.random()
        return observation, reward, False

class Agent:
    def select_action(self, observation):
        # Returns action based on current policy
        return random.choice(self.actions)

# Training loop
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs)
        obs, reward, done = env.step(action)
        agent.update_policy(reward)
```

## Key Learning Outcomes

- Understand the agent-environment feedback loop
- Implement a basic RL agent from scratch without any ML frameworks
- Grasp state, action, reward, and policy concepts
- Build intuition for more advanced RL algorithms (Q-Learning, Deep RL)

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Jupyter Notebook | Interactive experimentation |
| NumPy | Numerical computations |

## Getting Started

```bash
git clone https://github.com/sanikacentric/ReinforcementLearning.git
cd ReinforcementLearning
jupyter notebook
```
Open `Tutorial_2_Simple_Reinforcement_Learning_Code.ipynb` and run all cells.

## Prerequisites

```bash
pip install jupyter numpy
```

## Next Steps After This Repo

- **Q-Learning** — Tabular method for discrete state/action spaces
- **Deep Q-Network (DQN)** — Neural network as the value function approximator
- **Policy Gradient Methods** — Directly optimize the policy
- **Actor-Critic** — Combine value function and policy gradient approaches

## License

Open source — for educational purposes.
