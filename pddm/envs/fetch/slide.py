import os
import numpy as np

from gym import utils
# from gym.envs.robotics import fetch_env
from pddm.envs import fetch_env

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'slide.xml')


class FetchSlideEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)



    # *********** Added by Ray ***************
    def do_reset(self, initial_state):   #used for using true dynamic in pddm

        self.sim.set_state(initial_state[0])
        self.goal = initial_state[1]
        self.sim.forward()
        return self._get_obs()

    # *********** Added by Ray ***************
    def do_reset(self, initial_state):   #used for using true dynamic in pddm

        self.sim.set_state(initial_state[0])
        self.goal = initial_state[1]
        self.sim.forward()
        return self._get_obs()

    def get_reward(self, observations, goal, actions):    ######

        # print('Test: this is the PushEnv reward func')

        # from ipdb import set_trace;
        # set_trace()

        if np.ndim(observations)==2:       # for the planner to select actions
            n,m = observations.shape
            assert m == 25
            reward = np.zeros(n)
            dones = np.zeros(n)

            grip_pos = observations[:,:3]
            ag = observations[:,3:6]

            d_grip2ag = np.linalg.norm(grip_pos - ag, axis=-1)
            d_ag2g = np.linalg.norm(ag - goal, axis=-1)

            index = np.array([i for i in range(n)])
            Idx = index[(d_ag2g <= self.distance_threshold)]
            reward[Idx] += 100
            dones[Idx] = True

            Idx = index[(d_ag2g > self.distance_threshold)]
            reward[Idx] += - (d_grip2ag[Idx] + 10*d_ag2g[Idx])

            return reward, dones

        else:      # for the real reward when interacting with the environment.
            m = len(observations)
            assert m == 25
            reward = 0
            done = False

            grip_pos = observations[:3]
            ag = observations[3:6]

            d_grip2ag = np.linalg.norm(grip_pos - ag, axis=-1)
            d_ag2g = np.linalg.norm(ag - goal, axis=-1)

            if d_ag2g <= self.distance_threshold:
                reward += 100
                done = True
            else:
                reward += - (d_grip2ag + 10*d_ag2g)

            return reward, done