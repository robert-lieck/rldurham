from typing import Optional, Iterable, Union
from collections import deque
from math import sqrt

import numpy as np
import lightning as lt
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython import display as disp


def seed_everything(seed: Optional[int] = None,
                    env: Optional[gym.Env] = None,
                    seed_env: bool = True,
                    seed_actions: bool = True,
                    other: Iterable = (),
                    workers: bool = False) -> Union[int, tuple]:
    """
    This uses the `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_ ``seed_everything`` method to seed
    pseudo-random number generators in pytorch, numpy, and python.random. If ``env`` is provided, it optionally
    resets the environment and seeds its action space with the same seed.

    :param seed: integer value, see ``seed_everything`` `documentation <https://pytorch-lightning.readthedocs.io/en/1.7.7/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything>`_ for details
    :param env: ``gymnasium`` environment
    :param seed_env: whether to reset/seed ``env`` (this is done by calling ``reset(seed=seed)``)
    :param seed_actions: whether to seed the action space of ``env``
    :param other: other objects ``x`` to call ``x.seed(seed)`` on
    :param workers: passed on to ``seed_everything``
    :return: ``(seed, observation, info)`` if ``env`` was reset, otherwise, just ``seed``
    """
    seed = lt.seed_everything(seed=seed, workers=workers)
    obs_inf = ()
    if env is not None:
        if seed_env:
            obs_inf = env.reset(seed=seed)
        if seed_actions:
            env.action_space.seed(seed=seed)
    for x in other:
        x.seed(seed)
    if obs_inf:
        return (seed,) + obs_inf
    else:
        return seed


class Env(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._rldurham_obs = None
        self._rldurham_reward = None
        self._rldurham_terminated = None
        self._rldurham_truncated = None
        self._rldurham_info = None

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        (self._rldurham_obs,
         self._rldurham_reward,
         self._rldurham_terminated,
         self._rldurham_truncated,
         self._rldurham_info) = (obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info


def make(*args, **kwargs):
    return Env(gym.make(*args, **kwargs))


class Recorder(gym.Wrapper, gym.utils.RecordConstructorArgs):
    # see RecordEpisodeStatistics for inspiration

    def __init__(self, env, key="recorder", full_stats=False, smoothing=None):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        self._key = key
        self._full_stats = full_stats
        self._smoothing = smoothing

        self._episode_count = 0
        self._episode_started = False

        self._episode_reward_sum = 0
        self._episode_squared_reward_sum = 0
        self._episode_length = 0

        if self._smoothing:
            self._episode_reward_sum_queue = deque(maxlen=smoothing)
            self._episode_length_queue = deque(maxlen=smoothing)

        self._episode_full_stats = []

    def step(self, action):
        self._episode_started = True
        obs, reward, terminated, truncated, info = super().step(action)

        self._episode_reward_sum += reward
        self._episode_squared_reward_sum += reward ** 2
        self._episode_length += 1
        if self._full_stats:
            self._episode_full_stats.append((obs, reward, terminated, truncated, info))

        if terminated or truncated:
            assert self._key not in info
            i = {
                "idx": self._episode_count,
                "length": self._episode_length,
                "r_sum": self._episode_reward_sum,
                "r_mean": self._episode_reward_sum / self._episode_length,
                "r_std": sqrt(self._episode_squared_reward_sum / self._episode_length - (self._episode_reward_sum / self._episode_length) ** 2),
            }
            if self._smoothing is not None:
                self._episode_reward_sum_queue.append(self._episode_reward_sum)
                self._episode_length_queue.append(self._episode_length)
                assert len(self._episode_reward_sum_queue) == len(self._episode_length_queue)
                queue_length = len(self._episode_reward_sum_queue)
                length_ = sum(self._episode_length_queue) / queue_length
                r_sum_ = sum(self._episode_reward_sum_queue)
                r_mean_ = r_sum_ / queue_length
                r_std_ = sqrt(sum([r ** 2 for r in self._episode_reward_sum_queue]) / queue_length - r_mean_ ** 2)
                i["length_"] = length_
                i["r_sum_"] = r_sum_
                i["r_mean_"] = r_mean_
                i["r_std_"] = r_std_

            if self._full_stats:
                i['full'] = self._episode_full_stats
            info[self._key] = i

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self._episode_reward_sum = 0
        self._episode_squared_reward_sum = 0
        self._episode_length = 0
        self._episode_full_stats = []
        if self._episode_started:
            self._episode_count += 1
            self._episode_started = False
        return super().reset(seed=seed, options=options)


class InfoTracker:
    """
    Track info by recursively descending into dict and creating lists of info values.
    """

    def __init__(self):
        self.info = {}

    def track(self, info):
        if self.info:
            self._update(new_info=info, tracked_info=self.info)
        else:
            self._create(new_info=info, tracked_info=self.info)

    def _check(self, new_info, tracked_info):
        if set(new_info.keys()) != set(tracked_info.keys()):
            raise ValueError(f"Keys in tracked new info and tracked info are not the same "
                             f"(new: {list(new_info.keys())}, tracked: {list(tracked_info.keys())}")

    def _create(self, new_info, tracked_info):
        for k, v in new_info.items():
            if isinstance(v, dict):
                tracked_info[k] = {}
                self._create(new_info=v, tracked_info=tracked_info[k])
            else:
                tracked_info[k] = [v]

    def _update(self, new_info, tracked_info):
        self._check(new_info=new_info, tracked_info=tracked_info)
        for k in new_info:
            if isinstance(tracked_info[k], dict):
                self._update(new_info=new_info[k], tracked_info=tracked_info[k])
            else:
                tracked_info[k].append(new_info[k])


def plot_tracked_info(tracked_info, show=True, ax=None, key='recorder',
                      length=False, r_sum=False, r_mean=False, r_std=False,
                      length_=False, r_sum_=False, r_mean_=False, r_std_=False,
                      ):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    def get_kwargs(flag, **kwargs):
        if isinstance(flag, dict):
            return {**kwargs, **flag}
        else:
            return kwargs
    idx = tracked_info[key]['idx']

    _length = tracked_info[key]['length']
    _r_sum = np.array(tracked_info[key]['r_sum'])
    _r_mean = np.array(tracked_info[key]['r_mean'])
    _r_std = np.array(tracked_info[key]['r_std'])
    if length:
        ax.plot(idx, _length, **get_kwargs(length, label='length'))
    if r_sum:
        ax.plot(idx, _r_sum, **get_kwargs(r_sum, label='r_sum'))
    if r_mean:
        ax.plot(idx, _r_mean, **get_kwargs(r_mean, label='r_mean'))
    if r_std:
        ax.fill_between(idx, _r_mean - _r_std, _r_mean + _r_std, **get_kwargs(r_std, label='r_std', alpha=0.2, color='tab:grey'))

    _length_ = tracked_info[key]['length_']
    _r_sum_ = np.array(tracked_info[key]['r_sum_'])
    _r_mean_ = np.array(tracked_info[key]['r_mean_'])
    _r_std_ = np.array(tracked_info[key]['r_std_'])
    if length_:
        ax.plot(idx, _length_, **get_kwargs(length_, label='length_'))
    if r_sum_:
        ax.plot(idx, _r_sum_, **get_kwargs(r_sum_, label='r_sum_'))
    if r_mean_:
        ax.plot(idx, _r_mean_, **get_kwargs(r_mean_, label='r_mean_'))
    if r_std_:
        ax.fill_between(idx, _r_mean_ - _r_std_, _r_mean_ + _r_std_, **get_kwargs(r_std_, label='r_std_', alpha=0.2, color='tab:grey'))

    ax.set_xlabel('episode index')
    ax.legend()
    if show:
        plt.show()
        disp.clear_output(wait=True)
    if fig is not None:
        return fig, ax
