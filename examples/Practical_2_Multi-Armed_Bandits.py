
"""
Practical 2: Multi-Armed Bandits
================================
"""

# # Practical 2: Multi-Armed Bandits

# %%


import numpy as np


# ## Basic Environment

# %%


class Environment():
    def __init__(self):
        self.state = 0

    def step(self, a):
        self.state += 1
        reward = -1
        next_observation = 10 * self.state
        return next_observation , reward
        
    def reset(self):
        self.state = 0
        return self.state
    
    def render(self):
        print(self.state)


# %%


env = Environment()
obs = env.reset()
for _ in range(10):
    obs, rew = env.step(0)
    env.render()


# ## Multi-Armed Bandits

# %%


class BanditEnv():
    def __init__(self):
        self.state = 0
        self.n_actions = 6
        self.n_states = 0
        self.probs = np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.7])
        self.rewards = np.array([1.0, 0.1, 2.0, 0.5, 6.0, 70.])
    
    def step(self, a):
        reward = -25
        if np.random.uniform() < self.probs[a]:
            reward += self.rewards[a]
        return self.state, reward
    
    def reset(self):
        self.state = 0
        return self.state


# %%


class Agent():
    def __init__(self, env):
        self.env = env
        
    def sample_action(self, observation=0):
        return np.random.randint(self.env.n_actions)


# %%


env = BanditEnv()
agent = Agent(env)
o = env.reset()
money = 0
money_per_machine = np.zeros(env.n_actions)
usage_per_machine = np.zeros(env.n_actions)
for episode in range(1000):
    a = agent.sample_action(o)
    o, r = env.step(a)
    money += r
    money_per_machine[a] += r
    usage_per_machine[a] += 1
print("about " + str(money))
print("about " + str(money_per_machine/usage_per_machine))


# ###  Solving Multi-Armed Bandits

# %%


class Agent():

    def __init__(self, env, alpha=None, epsilon=0):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.k = np.zeros(self.env.n_actions)
        self.q = np.zeros(self.env.n_actions)

    @property
    def q_corrected(self):
        if self.alpha is None:
            return self.q
        else:
            return self.q / (1 - (1 - self.alpha)**self.k + 1e-8)

    def put_data(self, action, reward):
        self.k[action] += 1
        if self.alpha is None:
            # exact average
            if self.k[action] == 1:
                self.q[action] = r
            else:
                self.q[action] += (r - self.q[action]) / self.k[action]
        else:
            # smoothing average
            self.q[action] += self.alpha * (r - self.q[action])

            self.q[action] = (1 - self.alpha) * self.q[action] + self.alpha * r
        
    def sample_action(self, state=0, epsilon=0.4):
        if np.random.rand() < self.epsilon:
            return np.random.randint(env.n_actions)
        else:
            return np.argmax(self.q_corrected)

env = BanditEnv()
agent = Agent(env=env, alpha=0.1, epsilon=0.1)
s = env.reset()
for episode in range(1000):
    a = agent.sample_action(s)
    s, r = env.step(a)
    # learn to estimate the value of each action
    agent.put_data(a, r)

print(agent.k)
print(agent.q)
print(agent.q_corrected)


# ## Multi-Armed Multi-Room Bandits

# %%


class BanditEnv():
    def __init__(self):
        self.n_actions = 6+1
        self.n_states  = 3
        self.probs = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.4, 0.2, 0.1, 0.1, 0.1, 0.7],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        self.rewards = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
            [0.0, 1.0, 0.1, 2.0, 0.5, 6.0, 70.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        ])

    def step(self, s, a):
        r = -25

        # if first action taken, move to next state with no reward
        if a == 0:
            return (s+1)%3, 0
        # else pull the slot machine and get the reward, stay in current room
        if np.random.uniform() < self.probs[s, a]:
            r += self.rewards[s,a]

        return s, r
    
    def reset(self):
        return 0
        
class Agent():
    def __init__(self):
        self.q = np.zeros([env.n_states, env.n_actions])

    def sample_action(self, state=0, epsilon=0.4):
        if np.random.rand() > epsilon:
            return np.argmax(self.q[state,:])
        else:
            return np.random.randint(env.n_actions)

env = BanditEnv()
s = env.reset()
agent = Agent()
for episode in range(1000):
    a = agent.sample_action(s)
    next_s,r = env.step(s,a)
    
    agent.q[s,a] += 0.1 * (r - agent.q[s,a])
    s = next_s
    
print(agent.q)

