import os
from gym import utils
# from gym.envs.robotics import fetch_env
from pddm.envs import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


    #     self.init_qpos = self.data.qpos.ravel().copy()
    #     self.init_qvel = self.data.qvel.ravel().copy()
    #
    # def reset(self, initial_state=None):  ###### rewrote by ray
    #
    #     if initial_state:
    #         return self.do_reset(initial_state)
    #     else:
    #         did_reset_sim = False
    #         while not did_reset_sim:
    #             did_reset_sim = self._reset_sim()
    #         self.goal = self._sample_goal().copy()
    #         obs = self._get_obs()
    #
    #         return obs
    #
    # def reset_model(self):
    #     self.reset_pose = self.init_qpos.copy()
    #     self.reset_vel = self.init_qvel.copy()
    #     self.reset_goal = self._sample_goal().copy()
    #     return self.do_reset(self.reset_pose, self.reset_vel, self.reset_goal)
    #
    def do_reset(self, initial_state):   #used for using true dynamic in pddm

        self.sim.set_state(initial_state[0])
        self.goal = initial_state[1]
        self.sim.forward()
        return self._get_obs()