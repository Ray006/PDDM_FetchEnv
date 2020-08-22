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
from pddm.data_augmentation.ker_learning_method import ker_learning

class CollectSamples(object):

    def __init__(self, env, policy, visualize_rollouts, dt_from_xml, is_random, random_sampling_params):
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

        self.n_KER =3
        self.ker = ker_learning(self.env.unwrapped_env.spec.id, self.n_KER)

    def collect_samples(self, num_rollouts, steps_per_rollout):

        #vars
        all_processes = []
        visualization_frequency = num_rollouts / 10
        num_workers = multiprocessing.cpu_count()  #detect number of cores
        pool = multiprocessing.Pool(8)

        # x=self.generate_rollouts(steps_per_rollout)

        #multiprocessing for running rollouts (utilize multiple cores)
        # It is important that args=(xx,) ends with ',' or it won't work !
        for rollout_number in range(num_rollouts):
            result = pool.apply_async(
                self.generate_rollouts,
                args=(steps_per_rollout,),
                callback=self.mycallback)

        # #multiprocessing for running rollouts (utilize multiple cores)
        # for rollout_number in range(num_rollouts):
        #     result = pool.apply_async(
        #         self.do_rollout,
        #         args=(steps_per_rollout, rollout_number, visualization_frequency),
        #         callback=self.mycallback)

        pool.close()  #not going to add anything else to the pool
        pool.join()  #wait for the processes to terminate

        # from ipdb import set_trace;
        # set_trace()

        rollout_class_data = self.convert_2_rollouts(self.rollouts)

        return rollout_class_data


    def mycallback(self, x):
        self.rollouts.append(x)

######################### ker #################################
    def convert_2_rollouts(self, Episodes):
        result =[]
        for episodes in Episodes:
            for episode in episodes:
                o, g, a = episode['o'], episode['g'], episode['u']
                # o,g = self._preprocess_og(episode['o'],episode['ag'],episode['g'])
                o = np.concatenate((o[:-1,:],g),axis=1) # this step is important
                result.append(Rollout(observations=np.array(o), actions=np.array(a)))
        return result




    def reset_all_rollouts(self):
        self.obs_dict = self.env.reset()
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.g = self.obs_dict['desired_goal']

    def generate_rollouts(self, T):
        # if self.n_ker and terminate_ker==False:

        # from ipdb import set_trace;
        # set_trace()

        if self.n_KER:
            return self.generate_rollouts_ker(T)
        # else:
        #     return self.generate_rollouts_vanilla()

    # def generate_rollouts_vanilla(self):
    #     """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
    #     policy acting on it accordingly.
    #     """
    #     self.reset_all_rollouts()
    #     # compute observations
    #     o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
    #     ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
    #     o[:] = self.initial_o
    #     ag[:] = self.initial_ag
    #
    #     # generate episodes
    #     obs, achieved_goals, acts, goals, successes = [], [], [], [], []
    #     dones = []
    #     info_values = [np.empty((self.T - 1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in
    #                    self.info_keys]
    #     Qs = []
    #     for t in range(self.T):
    #         policy_output = self.policy.get_actions(
    #             o, ag, self.g,
    #             compute_Q=self.compute_Q,
    #             noise_eps=self.noise_eps if not self.exploit else 0.,
    #             random_eps=self.random_eps if not self.exploit else 0.,
    #             use_target_net=self.use_target_net)
    #
    #         if self.compute_Q:
    #             u, Q = policy_output
    #             Qs.append(Q)
    #         else:
    #             u = policy_output
    #
    #         if u.ndim == 1:
    #             # The non-batched case should still have a reasonable shape.
    #             u = u.reshape(1, -1)
    #
    #         o_new = np.empty((self.rollout_batch_size, self.dims['o']))
    #         ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
    #         success = np.zeros(self.rollout_batch_size)
    #         # compute new states and observations
    #         obs_dict_new, _, done, info = self.venv.step(u)
    #         o_new = obs_dict_new['observation']
    #         ag_new = obs_dict_new['achieved_goal']
    #         success = np.array([i.get('is_success', 0.0) for i in info])
    #
    #         if any(done):
    #             # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever any of the envs returns done
    #             # trick with using vecenvs is not to add the obs from the environments that are "done", because those are already observations
    #             # after a reset
    #             break
    #
    #         for i, info_dict in enumerate(info):
    #             for idx, key in enumerate(self.info_keys):
    #                 info_values[idx][t, i] = info[i][key]
    #
    #         if np.isnan(o_new).any():
    #             self.logger.warn('NaN caught during rollout generation. Trying again...')
    #             self.reset_all_rollouts()
    #             return self.generate_rollouts()
    #
    #         dones.append(done)
    #         obs.append(o.copy())
    #         achieved_goals.append(ag.copy())
    #         successes.append(success.copy())
    #         acts.append(u.copy())
    #         goals.append(self.g.copy())
    #         o[...] = o_new
    #         ag[...] = ag_new
    #     obs.append(o.copy())
    #     achieved_goals.append(ag.copy())
    #
    #     episode = dict(o=obs,
    #                    u=acts,
    #                    g=goals,
    #                    ag=achieved_goals)
    #     for key, value in zip(self.info_keys, info_values):
    #         episode['info_{}'.format(key)] = value
    #
    #     # stats
    #     successful = np.array(successes)[-1, :]
    #     assert successful.shape == (self.rollout_batch_size,)
    #     success_rate = np.mean(successful)
    #     self.success_history.append(success_rate)
    #     if self.compute_Q:
    #         self.Q_history.append(np.mean(Qs))
    #     self.n_episodes += self.rollout_batch_size
    #
    #     return convert_episode_to_batch_major(episode)

    def generate_rollouts_ker(self, T):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()
        self.rollout_batch_size = 1

        info_keys = ['is_success']

        episodes = []
        episodes_batch = []

        # compute observations
        # o = np.empty((self.rollout_batch_size, self.env.o_dim), np.float32)  # observations
        # ag = np.empty((self.rollout_batch_size, self.env.g_dim), np.float32)  # achieved goals
        # o[:] = self.initial_o
        # ag[:] = self.initial_ag

        o = self.initial_o
        ag = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        dones = []
        info_values = [np.empty((T - 1, self.rollout_batch_size, self.env.info_dim), np.float32) for key in
                       info_keys]

        prev_action = None

        for t in range(T):

            #decide what action to take
            if self.is_random:
                action, _ = self.policy.get_action(o, prev_action, self.random_sampling_params)
            else:
                action, _ = self.policy.get_action(o)

            #     set_trace()

            # o_new = np.empty((self.rollout_batch_size, self.env.o_dim))
            # ag_new = np.empty((self.rollout_batch_size, self.env.g_dim))
            # success = np.zeros(self.rollout_batch_size)

            # compute new states and observations, do not return the reward, and get it from her_sampler.py
            obs_dict_new, _, done, info = self.env.step(action)
            o_new = obs_dict_new['observation']
            ag_new = obs_dict_new['achieved_goal']

            info = [info] # here can be an error with different gym verstions
            success = np.array([i.get('is_success', 0.0) for i in info])


            # no need
            if done:
                break

            # no need
            for i, info_dict in enumerate(info):
                for idx, key in enumerate(info_keys):
                    info_values[idx][t, i] = info[i][key]
            # no need
            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            dones.append(done)
            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(action.copy())
            goals.append(self.g.copy())

            prev_action = action.copy()

            o[...] = o_new
            ag[...] = ag_new

        obs.append(o.copy())
        achieved_goals.append(ag.copy())

        # ----------------Kaleidoscope ER---------------------------
        original_ka_episodes = self.ker.ker_process(obs, acts, goals, achieved_goals)
        # ----------------end---------------------------

        # ----------------pack up as transition---------------------------
        for (obs, acts, goals, achieved_goals) in original_ka_episodes:
            episode = dict(o=obs,
                           u=acts,
                           g=goals,
                           ag=achieved_goals)
            for key, value in zip(info_keys, info_values):
                episode['info_{}'.format(key)] = value
            episodes.append(episode)
        # ----------------end---------------------------

        # from ipdb import set_trace;
        # set_trace()
        # stats
        # successful = np.array(successes)[-1, :]
        # assert successful.shape == (self.rollout_batch_size,)
        # success_rate = np.mean(successful)
        # self.success_history.append(success_rate)
        #
        # mul_factor = 1
        # self.n_episodes += (mul_factor * self.rollout_batch_size)

        # ----------------format processing---------------------------
        # return dict: ['o', 'u', 'g', 'ag', 'info_is_success']
        for episode in episodes:
            episode_batch = self.convert_episode_to_batch_major(episode)
            episodes_batch.append(episode_batch)
        # ----------------end---------------------------

        return episodes_batch

    def convert_episode_to_batch_major(self, episode):
        """Converts an episode to have the batch dimension in the major (first)
        dimension.
        """
        episode_batch = {}
        for key in episode.keys():
            val = np.array(episode[key]).copy()
            # make inputs batch-major instead of time-major
            # episode_batch[key] = val.swapaxes(0, 1)
            episode_batch[key] = val

        return episode_batch











    def do_rollout(self, steps_per_rollout, rollout_number, visualization_frequency):

        #init vars
        observations = []
        actions = []
        rewards_per_step = []

        #reset env
        observation, starting_state = self.env.reset(return_start_state=True)

        from ipdb import set_trace;
        set_trace()

        prev_action = None
        for step_num in range(steps_per_rollout):

            #decide what action to take
            if self.is_random:
                action, _ = self.policy.get_action(observation, prev_action, self.random_sampling_params)
            else:
                action, _ = self.policy.get_action(observation)


            #perform the action
            next_observation, reward, terminal, _ = self.env.step(action)
            rewards_per_step.append(reward)

            o = observation['observation']
            ag = observation['achieved_goal']

            #keep tracks of observations + actions
            observations.append(observation)
            actions.append(action)
            prev_action = action.copy()


            #update the observation
            observation = next_observation

        if (rollout_number%visualization_frequency)==0:
            print("Completed rollout # ", rollout_number)

        return Rollout(
            np.array(observations), np.array(actions),
            np.array(rewards_per_step), starting_state)



    # original
    # def do_rollout(self, steps_per_rollout, rollout_number, visualization_frequency):
    #
    #     #init vars
    #     observations = []
    #     actions = []
    #     rewards_per_step = []
    #
    #     #reset env
    #     observation, starting_state = self.env.reset(return_start_state=True)
    #
    #     from ipdb import set_trace;
    #     set_trace()
    #
    #     prev_action = None
    #     for step_num in range(steps_per_rollout):
    #
    #         #decide what action to take
    #         if self.is_random:
    #             action, _ = self.policy.get_action(observation, prev_action, self.random_sampling_params)
    #         else:
    #             action, _ = self.policy.get_action(observation)
    #
    #         # if self.env.unwrapped_env.spec.id == 'MB_FetchPush-v1':
    #         #     o = observation['observation']
    #         #     ag = observation['achieved_goal']
    #
    #
    #         #keep tracks of observations + actions
    #         observations.append(np.copy(observation))
    #         actions.append(action)
    #         prev_action = action.copy()
    #
    #         #perform the action
    #         next_observation, reward, terminal, _ = self.env.step(action)
    #         rewards_per_step.append(reward)
    #
    #         #update the observation
    #         observation = next_observation
    #
    #     if (rollout_number%visualization_frequency)==0:
    #         print("Completed rollout # ", rollout_number)
    #
    #
    #     # we need list not array
    #     # if self.env ===
    #
    #     return Rollout(
    #         np.array(observations), np.array(actions),
    #         np.array(rewards_per_step), starting_state)
