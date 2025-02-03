from unittest import TestCase
from itertools import count

import gymnasium as gym
from gymnasium import spaces

import rldurham as rld


class SimpleTestingEnv(gym.Env):
    """
    A simple testing environment with episodes of fixed length and reward equal to the action.
    """

    def __init__(self, episode_length=10):
        super().__init__()
        # Define the action space (integers)
        self.action_space = spaces.Discrete(100)  # Adjust the upper limit as needed
        # Define the observation space (always None, so we use a placeholder)
        self.observation_space = spaces.Box(low=0, high=0, shape=(0,), dtype=float)
        # Episode length
        self.episode_length = episode_length
        self.current_step = 0

    def reset(self, seed=None, options=None):
        # Reset the environment
        self.current_step = 0
        # Return the initial observation (always None) and additional info
        return None, {}

    def step(self, action):
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Calculate reward (equal to the action)
        reward = action
        # Observations are always None
        observation = None
        # Increment the step counter
        self.current_step += 1
        # Check if the episode is done
        terminated = self.current_step >= self.episode_length
        # Check if the episode is truncated (never happens)
        truncated = False
        # Additional info (empty for this environment)
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        # Simple rendering for debugging (optional)
        print(f"Step: {self.current_step}")

    def close(self):
        # Close resources if any (not required here)
        pass


class TestTemplate(TestCase):

    def test_make_recorder_tracker(self):
        episode_length = 10
        env = rld.RLDurhamEnv(SimpleTestingEnv(episode_length=episode_length))
        env = rld.Recorder(env, smoothing=10, logs=True)
        rld.seed_everything(42, env)
        self.assertEqual(env._episode_count, 0)
        env.reset()
        self.assertEqual(env._episode_count, 0)

        info_tracker = rld.InfoTracker()

        for episode_num in range(15):
            obs, info = env.reset()
            for step_count in count():
                action = episode_num * (step_count % 2)  # alternating action 0 and episode_num
                obs, reward, terminated, truncated, info = env.step(action)
                self.assertEqual(env._episode_count, episode_num)
                if terminated or truncated:
                    info_tracker.track(info)
                    break

            # episode-specific info

            self.assertEqual(info['recorder']['idx'], episode_num)
            self.assertEqual(info['recorder']['length'], episode_length)
            # half of the rewards are non-zero
            self.assertEqual(info['recorder']['r_sum'], episode_length * episode_num / 2)
            self.assertEqual(info['recorder']['r_mean'], episode_num / 2)
            # rewards are zero or episode_num (half/half), so always episode_num/2 from mean (also episode_num/2) away
            self.assertEqual(info['recorder']['r_std'], episode_num / 2)

            # smoothed info

            self.assertEqual(info['recorder']['length_'], episode_length)  # constant
            # buffer runs over at some point
            if episode_num <= env._smoothing: # first reward sum is zero, so first rolling over does not change reward sum
                self.assertEqual(info['recorder']['r_sum_'], episode_length * (episode_num * (episode_num + 1)) / 4)
                self.assertEqual(info['recorder']['r_mean_'], episode_length * (episode_num * (episode_num + 1)) / 4 / min(episode_num + 1, env._smoothing))
                # upper bound for std (couldn't do the math in my head...)
                self.assertLessEqual(info['recorder']['r_std_'], episode_length * episode_num / 4)
                r_std_ = info['recorder']['r_std_']  # but stays the same after buffer runs over
            else:
                self.assertNotEqual(info['recorder']['r_sum_'], episode_length * (episode_num * (episode_num + 1)) / 4)
                self.assertNotEqual(info['recorder']['r_mean_'], episode_length * (episode_num * (episode_num + 1)) / 4 / min(episode_num + 1, env._smoothing))
                self.assertEqual(info['recorder']['r_std_'], r_std_)
        env.close()

    def test_logging(self):
        for transform in [False, True]:
            # create environment
            env = rld.RLDurhamEnv(SimpleTestingEnv(episode_length=10))

            # transform reward (before wrapping in recorder)
            if transform:
                reward_scale = 2
                env = gym.wrappers.TransformReward(env, lambda reward: reward_scale * reward)
            else:
                reward_scale = 1

            #  add wrapper to record stats (after rescaling rewards)
            env = rld.Recorder(env, logs=True)

            rld.seed_everything(42, env)
            for episode_num in range(15):
                rsum = 0
                while True:
                    obs, rew, terminated, truncated, _ = env.step(action=episode_num + 1)
                    rsum += rew
                    if terminated or truncated:
                        # get current episode statistics
                        episode_count = rld.getwrappedattr(env, "_episode_count")
                        episode_reward_sum = rld.getwrappedattr(env, "_episode_reward_sum")
                        episode_squared_reward_sum = rld.getwrappedattr(env, "_episode_squared_reward_sum")
                        episode_length = rld.getwrappedattr(env, "_episode_length")
                        # upon reset the log statistics are updated
                        env.reset()
                        # episode count and length should always be equal
                        self.assertEqual(episode_count, rld.getwrappedattr(env, "_episode_count_log")[-1])
                        self.assertEqual(episode_length, rld.getwrappedattr(env, "_episode_length_log")[-1])
                        # reward statistics should be scaled by reward_scale
                        self.assertEqual(rsum, episode_reward_sum)
                        self.assertEqual(episode_reward_sum / reward_scale, rld.getwrappedattr(env, "_episode_reward_sum_log")[-1])
                        self.assertEqual(episode_squared_reward_sum / reward_scale ** 2, rld.getwrappedattr(env, "_episode_squared_reward_sum_log")[-1])

                        break

            env.close()

    def test_cartpole(self):
        env = rld.make("CartPole-v1")
        env = rld.Recorder(env)
        rld.seed_everything(42, env)
        self.assertEqual(env._episode_count, 0)
        env.reset()
        self.assertEqual(env._episode_count, 0)

        for episode_num in range(10):
            obs, info = env.reset()
            episode_over = False
            while not episode_over:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_over = terminated or truncated
                self.assertEqual(env._episode_count, episode_num)
            self.assertEqual(info['recorder']['idx'], episode_num)
        env.close()
