from typing import Optional, Iterable, Union
from collections import deque
from math import sqrt
import os
import time

import torch
import numpy as np
import pandas as pd
import lightning as lt
import gymnasium as gym
from gymnasium import error, logger
import matplotlib.pyplot as plt
from IPython import display as disp

# required for atari games in "ALE" namespace to work
import ale_py

# check for most up-to-date version
import rldurham.version_check

# register the custom BipedalWalker environment for coursework
gym.register(
    id="rldurham/Walker",
    entry_point="rldurham.bipedal_walker:BipedalWalker",
)

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


def render(env, clear=False, axis_off=True, show=True, sleep=0):
    if clear:
        disp.clear_output(wait=True)
    plt.imshow(env.render())
    if axis_off:
        plt.axis('off')
    if show:
        plt.show()
    if sleep:
        time.sleep(sleep)


# helper function to draw the frozen lake
def plot_frozenlake(env, v=None, policy=None, col_ramp=1, dpi=175, draw_vals=False, mark_ice=True):
    # set up plot
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams.update({'axes.edgecolor': (0.32, 0.36, 0.38)})
    plt.rcParams.update({'font.size': 4 if env.nrow == 8 else 7})
    gray = np.array((0.32, 0.36, 0.38))
    plt.figure(figsize=(3, 3))
    ax = plt.gca()
    ax.set_xticks(np.arange(env.ncol) - .5)
    ax.set_yticks(np.arange(env.nrow) - .5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(color=(0.42, 0.46, 0.48), linestyle=':')
    ax.set_axisbelow(True)
    ax.tick_params(color=(0.42, 0.46, 0.48), which='both', top='off', left='off', right='off', bottom='off')
    # use zero value as dummy if not provided
    if v is None:
        v = np.zeros(env.observation_space.n)
    # plot values
    plt.imshow(1 - v.reshape(env.nrow, env.ncol) ** col_ramp, cmap='gray', interpolation='none', clim=(0, 1), zorder=-1)
    # function for plotting policy
    def plot_arrow(x, y, dx, dy, v, scale=0.4):
        plt.arrow(x, y, scale * float(dx), scale * float(dy), color=gray + 0.2 * (1 - v), head_width=0.1, head_length=0.1, zorder=1)
    # go through states
    for s in range(env.observation_space.n):
        x, y = s % env.nrow, s // env.ncol
        # print numeric values
        if draw_vals and v[s] > 0:
            vstr = '{0:.1e}'.format(v[s]) if env.nrow == 8 else '{0:.6f}'.format(v[s])
            plt.text(x - 0.45, y + 0.45, vstr, color=(0.2, 0.8, 0.2), fontname='Sans')
        # mark ice, start, goal
        if env.desc.tolist()[y][x] == b'F':
            plt.text(x-0.45,y-0.3, 'ice', color=(0.5, 0.6, 1), fontname='Sans')
            if mark_ice:
                ax.add_patch(plt.Circle((x, y), 0.2, color=(0.7, 0.8, 1), zorder=0))
        elif env.desc.tolist()[y][x] == b'S':
            plt.text(x-0.45,y-0.3, 'start',color=(0.2,0.5,0.5), fontname='Sans',
                     weight='bold')
        elif env.desc.tolist()[y][x] == b'G':
            plt.text(x-0.45,y-0.3, 'goal', color=(0.7,0.2,0.2), fontname='Sans',
                     weight='bold')
            continue # don't plot policy for goal state
        else:
            continue # don't plot policy for holes
        if policy is not None:
            a = policy[s]
            if a[0] > 0.0: plot_arrow(x, y, -a[0],    0., v[s])  # left
            if a[1] > 0.0: plot_arrow(x, y,    0.,  a[1], v[s])  # down
            if a[2] > 0.0: plot_arrow(x, y,  a[2],    0., v[s])  # right
            if a[3] > 0.0: plot_arrow(x, y,    0., -a[3], v[s])  # up
    plt.show()


def env_info(env, print_out=False):
    discrete_act = hasattr(env.action_space, 'n')
    discrete_obs = hasattr(env.observation_space, 'n')
    act_dim = env.action_space.n if discrete_act else env.action_space.shape[0]
    obs_dim = env.observation_space.n if discrete_obs else env.observation_space.shape[0]
    if print_out:
        print(f"actions are {'discrete' if discrete_act else 'continuous'} with {act_dim} dimensions/#actions")
        print(f"observations are {'discrete' if discrete_obs else 'continuous'} with {obs_dim} dimensions/#observations")
        print(f"maximum timesteps is: {env.spec.max_episode_steps}")
    return discrete_act, discrete_obs, act_dim, obs_dim


def check_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'The device is: {device} ', end='')
    if device.type == 'cpu':
        print("(as recommended)")
    else:
        print("(train on the cpu is recommended instead)")
    return device


class RLDurhamEnv(gym.Wrapper):
    """
    Light-weight environment wrapper to enable logging.
    """

    def __init__(self, env, unwrap=True):
        super().__init__(env)
        self.unwrap = unwrap
        self._unscaled_reward = None

    def __getattr__(self, item):
        if self.unwrap:
            return getwrappedattr(self.env, item)
        else:
            raise AttributeError(f"Has no attribute '{item}' "
                                 f"(set unwrap=True to attempt unwrapping the enclosed environments)")

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._unscaled_reward = reward
        return obs, reward, terminated, truncated, info


def getwrappedattr(env, attr, depth=None):
    # try to get the attribute
    try:
        return getattr(env, attr)
    except AttributeError:
        # stop if max depth reached, otherwise, count down depth
        if depth is not None:
            if depth <= 0:
                raise AttributeError("Maximum unwrapping depth reached")
            else:
                depth = depth - 1
    # try to get the wrapped environment
    try:
        env = env.env
    except AttributeError:
        raise AttributeError(f"Attribute '{attr}' not found and cannot further unwrap "
                             f"(no 'env' attribute found, probably not a wrapper)")
    # recursively unwrap
    return getwrappedattr(env, attr, depth)


def make(*args, **kwargs):
    """
    Drop-in replacement for ``gym.make`` to enable logging.
    """
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
        self._episode_reward_sum_unscaled = 0
        self._episode_squared_reward_sum = 0
        self._episode_squared_reward_sum_unscaled = 0
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
        return f",episode={self._episode_count},score={self._episode_reward_sum_unscaled}"

    def step(self, action):
        self._episode_started = True
        obs, reward, terminated, truncated, info = super().step(action)

        self._episode_reward_sum += reward
        self._episode_reward_sum_unscaled += getwrappedattr(self, "_unscaled_reward")
        self._episode_squared_reward_sum += reward ** 2
        self._episode_squared_reward_sum_unscaled += getwrappedattr(self, "_unscaled_reward") ** 2
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
                self._episode_reward_sum_log.append(self._episode_reward_sum_unscaled)
                self._episode_squared_reward_sum_log.append(self._episode_squared_reward_sum_unscaled)
                self._episode_length_log.append(self._episode_length)
            # reset episode stats
            self._episode_count += 1
            self._episode_reward_sum = 0
            self._episode_reward_sum_unscaled = 0
            self._episode_squared_reward_sum = 0
            self._episode_squared_reward_sum_unscaled = 0
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
