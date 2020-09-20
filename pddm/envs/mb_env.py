# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class MBEnvWrapper:
    """
    Wrapper for gym environments.
    To be used with this model-based RL codebase (PDDM).
    """

    def __init__(self, env):

        self.env = env.env
        self.unwrapped_env = env.env.env
        self.action_dim = self.unwrapped_env.action_space.shape[0]

        try:
            obs = env.reset()
            self.observation_dim = obs['observation'].shape[0]+obs['desired_goal'].shape[0]
            self.o_dim = obs['observation'].shape[0]
            self.g_dim = obs['desired_goal'].shape[0]
            self.ag_dim = obs['achieved_goal'].shape[0]
            self.info_dim = 1

        except :
            self.observation_dim = self.unwrapped_env.observation_space.shape[0]


    def reset(self, reset_state=None, return_start_state=False):

        if reset_state:
            # reset to specified state
            obs = self.unwrapped_env.do_reset(reset_state)
            # obs = self.env.reset(reset_state)

        else:
            obs = self.env.reset()

            # standard reset call
            ###################################### ## this one can reset timestep limit
            # if hasattr(self.env, 'reset_model'):
            #     obs = self.env.reset_model()
            # else:
            #     obs = self.env.reset()
            ######################################
            # ###################################### ## this one can not reset timestep limit
            # if hasattr(self.unwrapped_env, 'reset_model'):
            #     obs = self.unwrapped_env.reset_model()
            # else:
            #     obs = self.unwrapped_env.reset()
            # ######################################

        #return
        if return_start_state:

            # from ipdb import set_trace;
            # set_trace()

            if hasattr(self.unwrapped_env, 'sim'):
                # reset_state = self.unwrapped_env.initial_state
                    reset_state = self.unwrapped_env.sim.get_state()
            else:
                reset_state = None
            #goal
            if hasattr(self.unwrapped_env, 'goal'):
                reset_goal = self.unwrapped_env.goal
            else:
                reset_goal = None
            return obs, [reset_state, reset_goal]
        else:
            return obs

    def step(self, action):
        # return self.unwrapped_env.step(action)     ## this step method will have no done=true
        return self.env.step(action)  # this one has timestep limit