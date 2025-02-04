
"""
Practical 4: DynamicProgramming
===============================
"""

# # Practical 4: DynamicProgramming

# %%


import numpy as np
import rldurham as rld


# ## Frozen Lake Environment

# %%


env = rld.make(
    'FrozenLake-v1',         # small version
    # 'FrozenLake8x8-v1',    # larger version
    # desc=["GFFS", "FHFH", "FFFH", "HFFG"],  # custom map
    render_mode="rgb_array", # for rendering as image/video
    is_slippery=False,       # warning: slippery=True results in complex dynamics
)
rld.env_info(env, print_out=True)
rld.seed_everything(42, env)
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


# %%


# render the environment (requires render_mode="rgb_array")
rld.render(env)


# %%


# helper function that can also plot policies and value functions
rld.plot_frozenlake(env=env,
                    v=np.random.uniform(0, 1, 16),
                    policy=np.random.uniform(0, 1, (16, 4)), 
                    draw_vals=True)


# %%


def uniform_policy(env):
    return np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
rld.plot_frozenlake(env=env, policy=uniform_policy(env))


# ## Policy Evaluation

# %%


def policy_eval_step(env, policy, gamma, v_init=None):
    if v_init is None:
        v_init = np.zeros(env.observation_space.n)
    v = np.zeros(env.observation_space.n)
    for s_from in range(env.observation_space.n):
        for a in range(env.action_space.n):
            pi = policy[s_from, a]
            for p, s_to, r, done in env.P[s_from][a]:
                v[s_from] += pi * p * (r + gamma * v_init[s_to])
    return v


# %%


v = np.zeros(env.observation_space.n)


# %%


v = policy_eval_step(env, uniform_policy(env), 1, v)
rld.plot_frozenlake(env, v, uniform_policy(env), draw_vals=True)


# %%


def policy_eval_step_inplace(env, policy, gamma, v_init=None):
    if v_init is None:
        v_init = np.zeros(env.observation_space.n)
    v = v_init.copy() # opearate on copy in-place
    for s_from in reversed(range(env.observation_space.n)):  # reverse order of states
        v_s_from = 0  # compute value for this state
        for a in range(env.action_space.n):
            pi = policy[s_from, a]
            for p, s_to, r, done in env.P[s_from][a]:
                v_s_from += pi * p * (r + gamma * v[s_to])  # use the values we also update
        v[s_from] = v_s_from  # update
    return v


# %%


v = np.zeros(env.observation_space.n)


# %%


v = policy_eval_step_inplace(env, uniform_policy(env), 1, v)
rld.plot_frozenlake(env, v, uniform_policy(env), draw_vals=True)


# %%


def policy_evaluation(env, policy, gamma, v_init=None, print_iter=False, atol=1e-8, max_iter=10**10):
    if v_init is None:
        v_init = np.zeros(env.observation_space.n)
    v = v_init
    for i in range(1, max_iter + 1):
        new_v = policy_eval_step(env, policy, gamma, v)
        # new_v = policy_eval_step_inplace(env, policy, gamma, v)
        if np.allclose(v, new_v, atol=atol):
            break
        v = new_v
    if print_iter:
        print(f"{i} iterations")
    return v


# %%


v = policy_evaluation(env, uniform_policy(env), 1, print_iter=True)
rld.plot_frozenlake(env, v, uniform_policy(env), draw_vals=True)


# ## Policy Improvement

# %%


def q_from_v(env, v, s, gamma):
    q = np.zeros(env.action_space.n)
    for a in range(env.action_space.n):
        for p, s_to, r, done in env.P[s][a]:
            q[a] += p * (r + gamma * v[s_to])
    return q


# %%


def policy_improvement(env, v, gamma, deterministic=False):
    policy = np.zeros([env.observation_space.n, env.action_space.n]) / env.action_space.n
    for s in range(env.observation_space.n):
        q = q_from_v(env, v, s, gamma)
        if deterministic:
            # deterministic policy
            policy[s][np.argmax(q)] = 1
        else:
            # stochastic policy with equal probability on maximizing actions
            best_a = np.argwhere(q==np.max(q)).flatten()
            policy[s, best_a] = 1 / len(best_a)
    return policy


# %%


env = rld.make('FrozenLake8x8-v1', is_slippery=False)
rld.seed_everything(42, env)
gamma = 1
policy = uniform_policy(env)


# %%


v = policy_evaluation(env, policy, gamma=gamma)
rld.plot_frozenlake(env, v=v, policy=policy, draw_vals=True)


# %%


policy = policy_improvement(env, v, gamma=gamma)
rld.plot_frozenlake(env, v=v, policy=policy, draw_vals=True)


# ## Policy Iteration

# %%


env = rld.make('FrozenLake8x8-v1', is_slippery=False)
rld.seed_everything(42, env)
policy = uniform_policy(env)
gamma = 1


# %%


v = policy_evaluation(env, policy, gamma=gamma)
rld.plot_frozenlake(env, v=v, policy=policy, draw_vals=True)
print(v)
policy = policy_improvement(env, v, gamma=gamma)
rld.plot_frozenlake(env, v=v, policy=policy, draw_vals=True)

