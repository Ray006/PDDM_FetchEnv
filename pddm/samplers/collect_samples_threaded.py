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

import numpy as np
import multiprocessing

#my imports
from pddm.utils.data_structures import *

class CollectSamples(object):

    def __init__(self, env, policy, visualize_rollouts, dt_from_xml, is_random, random_sampling_params, self_model):
        self.env = env
        self.policy = policy
        self.rollouts = []

        # self.stateDim = self.env.env.observation_space.shape[0]
        # self.actionDim = self.env.env.action_space.shape[0]
        self.stateDim = self.env.observation_dim
        self.actionDim = self.env.action_dim

        self.dt_from_xml = dt_from_xml
        self.is_random = is_random
        self.random_sampling_params = random_sampling_params

        self.self_model = self_model


    def collect_samples(self, num_rollouts, steps_per_rollout):

        #vars
        all_processes = []
        visualization_frequency = num_rollouts / 10
        num_workers = multiprocessing.cpu_count()  #detect number of cores
        pool = multiprocessing.Pool(8)

        #multiprocessing for running rollouts (utilize multiple cores)
        for rollout_number in range(num_rollouts):
            result = pool.apply_async(
                self.do_rollout,
                args=(steps_per_rollout, rollout_number, visualization_frequency),
                callback=self.mycallback)

        pool.close()  #not going to add anything else to the pool
        pool.join()  #wait for the processes to terminate

        return self.rollouts


    def mycallback(self, x):
        self.rollouts.append(x)


    def do_rollout(self, steps_per_rollout, rollout_number, visualization_frequency):

        #init vars
        obs = []
        actions = []
        rewards_per_step = []
        ag, g, info = [], [], []

        #reset env
        observation, starting_state = self.env.reset(return_start_state=True)

        if isinstance(observation, dict):       # if the observation is a dict

            state_dict = observation
            if self.self_model:
                mixed_obs = state_dict['observation']
                obs.append(np.concatenate((mixed_obs[:3],mixed_obs[9:11],mixed_obs[-5:])))
            else:
                obs.append(state_dict['observation'])
            ag.append(state_dict['achieved_goal'])
            g.append(state_dict['desired_goal'])
            o = obs[-1]
        else:
            o = observation
            obs.append(o)

        # from ipdb import set_trace;
        # set_trace()

        prev_action = None
        for step_num in range(steps_per_rollout):

            #decide what action to take
            if self.is_random:
                action, _ = self.policy.get_action(o, prev_action, self.random_sampling_params)
            else:
                action, _ = self.policy.get_action(o)

            actions.append(action)
            prev_action = action.copy()

            #perform the action
            next_observation, reward, done, _info = self.env.step(action)

            if isinstance(next_observation, dict):
                state_dict = next_observation
                if self.self_model:
                    mixed_obs = state_dict['observation']
                    obs.append(np.concatenate((mixed_obs[:3], mixed_obs[9:11], mixed_obs[-5:])))
                else:
                    obs.append(state_dict['observation'])
                ag.append(state_dict['achieved_goal'])
                g.append(state_dict['desired_goal'])
                o = obs[-1]
            else:
                o = next_observation
                obs.append(o)

            rewards_per_step.append(reward)
            info.append(_info)


        if (rollout_number%visualization_frequency)==0:
            print("Completed rollout # ", rollout_number)


        return Rollout(
            np.array(obs), np.array(actions), np.array(rewards_per_step), starting_state,
            np.array(ag), np.array(g), np.array(info))

