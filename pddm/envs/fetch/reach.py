import os
from gym import utils
# from gym.envs.robotics import fetch_env1
from pddm.envs import fetch_env
import numpy as np

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def do_reset(self, initial_state):   #used for using true dynamic in pddm

        self.sim.set_state(initial_state[0])
        self.goal = initial_state[1]
        self.sim.forward()
        return self._get_obs()

    # *********** Added by Ray get_reward() ***************
    # ----------------------------
    def get_reward(self, observations, goal, actions):

        # print('Test: this is the ReachEnv reward func')

        if np.ndim(observations) == 2:  # for the planner to select actions
            n, m = observations.shape
            assert m == 10
            reward = np.zeros(n)
            dones = np.zeros(n)

            grip_pos = observations[:, 0:3]

            # from ipdb import set_trace;
            # set_trace()

            d_pos = np.linalg.norm(grip_pos - goal, axis=-1)

            index = np.array([i for i in range(n)])
            Idx = index[(d_pos <= self.distance_threshold)]
            reward[Idx] += 100
            dones[Idx] = True

            Idx = index[dones==0]
            reward[Idx] += - d_pos[Idx]

            # Idx = index[diff_angle <= 100]  # if d_pos<=thre_pos and d_vel<=thre_vel:
            # reward[Idx] += 1
            # Idx = index[diff_angle <= 50]  # if d_pos<=thre_pos and d_vel<=thre_vel:
            # reward[Idx] += 2
            # Idx = index[diff_angle <= 10]  # if d_pos<=thre_pos and d_vel<=thre_vel:
            # reward[Idx] += 5
            # Idx = index[diff_angle <= 5]  # if d_pos<=thre_pos and d_vel<=thre_vel:
            # reward[Idx] += 10

            return reward, dones

        else:  # for the real reward when interacting with the environment.
            m = len(observations)
            assert m == 10
            reward = 0

            grip_pos = observations[0:3]
            d_pos = np.linalg.norm(grip_pos - goal, axis=-1)

            # from ipdb import set_trace;
            # set_trace()

            done = False
            if d_pos <= self.distance_threshold:
                reward += 100
                done = True
            else:
                reward += - d_pos

            #     if diff_angle <= 100:
            #         reward += 1
            #     if diff_angle <= 50:
            #         reward += 2
            #     if diff_angle <= 10:
            #         reward += 5
            #     if diff_angle <= 5:
            #         reward += 10
            #
            #     done = False

            return reward, done