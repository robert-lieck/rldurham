from typing import Optional, Iterable, Union
from collections import deque
from math import sqrt
import os
import time

import numpy as np
import pandas as pd
import lightning as lt
import gymnasium as gym
from gymnasium import error, logger
import matplotlib.pyplot as plt
from IPython import display as disp

import ale_py  # required for atari games in "ALE" namespace to work


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


def render(env, sleep=0):
    disp.clear_output(wait=True)
    plt.imshow(env.render())
    plt.show()
    if sleep:
        time.sleep(sleep)


def env_info(env):
    discrete_act = hasattr(env.action_space, 'n')
    discrete_obs = hasattr(env.observation_space, 'n')
    act_dim = env.action_space.n if discrete_act else env.action_space.shape[0]
    obs_dim = env.observation_space.n if discrete_obs else env.observation_space.shape[0]
    return discrete_act, discrete_obs, act_dim, obs_dim


class RLDurhamEnv(gym.Wrapper):

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
    return RLDurhamEnv(gym.make(*args, **kwargs))


class VideoRecorder(gym.wrappers.RecordVideo):

    def __init__(self, *args, name_func, **kwargs):
        super().__init__(*args, **kwargs)
        self.name_func = name_func

    def stop_recording(self):
        # MODIFIED from gym.wrappers.RecordVideo
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "gymnasium[other]"`'
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{self.name_prefix}{self.name_func()}.mp4")  # MODIFIED HERE
            clip.write_videofile(path, logger=moviepy_logger)

        self.recorded_frames = []
        self.recording = False
        self._video_name = None


class Recorder(gym.Wrapper, gym.utils.RecordConstructorArgs):
    # see RecordEpisodeStatistics for inspiration

    def __init__(self, env, info=True, video=False, logs=False, key="recorder",
                 video_folder="videos", video_prefix="xxxx00-agent-video",
                 full_stats=False, smoothing=None):
        gym.utils.RecordConstructorArgs.__init__(self)
        if video:
            env = VideoRecorder(env,
                                video_folder=video_folder,
                                name_prefix=video_prefix,
                                episode_trigger=self._video_episode_trigger,
                                name_func=self._video_name_func)
        gym.Wrapper.__init__(self, env)

        # flags to turn functionality on/off
        self.info = info
        self.video = video
        self.logs = logs

        # other settings
        self._key = key
        self._full_stats = full_stats
        self._smoothing = smoothing

        # flag to ignore episodes without any steps taken
        self._episode_started = False

        # episode stats
        self._episode_count = 0
        self._episode_reward_sum = 0
        self._episode_squared_reward_sum = 0
        self._episode_length = 0
        # logging statistics
        self._episode_count_log = []
        self._episode_reward_sum_log = []
        self._episode_squared_reward_sum_log = []
        self._episode_length_log = []
        # windowed stats for smoothing
        if self._smoothing:
            self._episode_reward_sum_queue = deque(maxlen=smoothing)
            self._episode_length_queue = deque(maxlen=smoothing)
        # full stats
        self._episode_full_stats = []

    def _video_episode_trigger(self, episode_id):
        return self.video

    def _video_name_func(self):
        return f",episode={self._episode_count},score={self._episode_reward_sum}"

    def step(self, action):
        self._episode_started = True
        obs, reward, terminated, truncated, info = super().step(action)

        self._episode_reward_sum += reward
        self._episode_squared_reward_sum += reward ** 2
        self._episode_length += 1
        if self._full_stats:
            self._episode_full_stats.append((obs, reward, terminated, truncated, info))

        if self.info and (terminated or truncated):
            assert self._key not in info
            # add episode stats
            i = {
                "idx": self._episode_count,
                "length": self._episode_length,
                "r_sum": self._episode_reward_sum,
                "r_mean": self._episode_reward_sum / self._episode_length,
                "r_std": sqrt(self._episode_squared_reward_sum / self._episode_length - (self._episode_reward_sum / self._episode_length) ** 2),
            }
            # add smoothing stats
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

    def _finish_episode(self):
        # ignore episodes without any steps taken
        if self._episode_started:
            self._episode_started = False
            # record logging stats
            if self.logs:
                self._episode_count_log.append(self._episode_count)
                self._episode_reward_sum_log.append(self._episode_reward_sum)
                self._episode_squared_reward_sum_log.append(self._episode_squared_reward_sum)
                self._episode_length_log.append(self._episode_length)
            # reset episode stats
            self._episode_count += 1
            self._episode_reward_sum = 0
            self._episode_squared_reward_sum = 0
            self._episode_length = 0
            self._episode_full_stats = []

    def reset(self, *, seed=None, options=None):
        ret = super().reset(seed=seed, options=options)  # first reset super (e.g. to save video)
        self._finish_episode()
        return ret

    def close(self):
        super().close()
        self._finish_episode()

    def write_log(self, folder="logs", file="xxxx00-agent-log.txt"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, file)
        df = pd.DataFrame({
            "count": self._episode_count_log,
            "reward_sum": self._episode_reward_sum_log,
            "squared_reward_sum": self._episode_squared_reward_sum_log,
            "length": self._episode_length_log,
        })
        df.to_csv(path, index=False, sep='\t')


class InfoTracker:
    """
    Track info by recursively descending into dict and creating lists of info values.
    """

    def __init__(self):
        self.info = {}

    def track(self, info, ignore_empty=True):
        if not info and ignore_empty:
            return
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

    def plot(
            self, show=True, ax=None, key='recorder', ignore_empty=True,
            length=False, r_sum=False, r_mean=False, r_std=False,
            length_=False, r_sum_=False, r_mean_=False, r_std_=False
    ):
        if not self.info and ignore_empty:
            return
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        def get_kwargs(flag, **kwargs):
            if isinstance(flag, dict):
                return {**kwargs, **flag}
            else:
                return kwargs
        idx = self.info[key]['idx']

        if length:
            _length = self.info[key]['length']
            ax.plot(idx, _length, **get_kwargs(length, label='length'))
        if r_sum:
            _r_sum = np.array(self.info[key]['r_sum'])
            ax.plot(idx, _r_sum, **get_kwargs(r_sum, label='r_sum'))
        _r_mean = None
        if r_mean:
            _r_mean = np.array(self.info[key]['r_mean'])
            ax.plot(idx, _r_mean, **get_kwargs(r_mean, label='r_mean'))
        if r_std:
            if _r_mean is None:
                _r_mean = np.array(self.info[key]['r_mean'])
            _r_std = np.array(self.info[key]['r_std'])
            ax.fill_between(idx, _r_mean - _r_std, _r_mean + _r_std, **get_kwargs(r_std, label='r_std', alpha=0.2, color='tab:grey'))

        if length_:
            _length_ = self.info[key]['length_']
            ax.plot(idx, _length_, **get_kwargs(length_, label='length_'))
        if r_sum_:
            _r_sum_ = np.array(self.info[key]['r_sum_'])
            ax.plot(idx, _r_sum_, **get_kwargs(r_sum_, label='r_sum_'))
        _r_mean_ = None
        if r_mean_:
            _r_mean_ = np.array(self.info[key]['r_mean_'])
            ax.plot(idx, _r_mean_, **get_kwargs(r_mean_, label='r_mean_'))
        if r_std_:
            if _r_mean_ is None:
                _r_mean_ = np.array(self.info[key]['r_mean_'])
            _r_std_ = np.array(self.info[key]['r_std_'])
            ax.fill_between(idx, _r_mean_ - _r_std_, _r_mean_ + _r_std_, **get_kwargs(r_std_, label='r_std_', alpha=0.2, color='tab:grey'))

        ax.set_xlabel('episode index')
        ax.legend()
        if show:
            disp.clear_output(wait=True)
            plt.show()
        if fig is not None:
            return fig, ax
