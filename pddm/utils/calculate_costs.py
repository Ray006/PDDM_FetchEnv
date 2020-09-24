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
from ipdb import set_trace

# def training_reward_func(observations, goal):  ######
#
#     # print('Test: this is the PushEnv reward func')
#     distance_threshold = 0.05
#     count_ray = 0
#
#     # from ipdb import set_trace;
#     # set_trace()
#
#     if np.ndim(observations) == 2:  # for the planner to select actions
#         n, m = observations.shape
#         assert m == 25
#         reward = np.zeros(n)
#         dones = np.zeros(n)
#
#         grip_pos = observations[:, :3]
#         ag = observations[:, 3:6]
#
#         # ****************** meaningful traj?
#         set_trace()
#
#         ag = np.array(ag)
#         delta_movement = np.linalg.norm(ag[1:] - ag[0], axis=2)  # compare with the object starting pos
#         if any(delta_movement > 0.05):  # if the object is moved
#             count_ray += 1
#             # print('move the ag:', count_ray)
#         # ****************** meaningful traj?
#
#
#         d_grip2ag = np.linalg.norm(grip_pos - ag, axis=-1)
#
#         index = np.array([i for i in range(n)])
#
#         # Idx = index[(d_ag2g > distance_threshold)]
#         # reward[Idx] += - d_grip2ag[Idx]
#
#         return reward, dones

def cost_per_step(pt, prev_pt, goal, costs, actions, dones, reward_func):

    # from ipdb import set_trace;
    # set_trace()



    # step_rews, step_dones = training_reward_func(pt, goal)

    step_rews, step_dones = reward_func(pt, goal, actions)

    dones = np.logical_or(dones, step_dones)

    step_rews[dones > 0] = 100 # if done, step_rews=100
    costs -= step_rews


    # costs[dones > 0] += 500
    # costs[dones == 0] -= step_rews[dones == 0]

    return costs, dones

missing_rate = 1.0
def calculate_costs(resulting_states_list, goal, actions, reward_func,
                    evaluating, take_exploratory_actions):
    """Rank various predicted trajectories (by cost)

    Args:
        resulting_states_list :
            predicted trajectories
            [ensemble_size, horizon+1, N, statesize]
        actions :
            the actions that were "executed" in order to achieve the predicted trajectories
            [N, h, acsize]
        reward_func :
            calculates the rewards associated with each state transition in the predicted trajectories
        evaluating :
            determines whether or not to use model-disagreement when selecting which action to execute
            bool
        take_exploratory_actions :
            determines whether or not to use model-disagreement when selecting which action to execute
            bool

    Returns:
        cost_for_ranking : cost associated with each candidate action sequence [N,]
    """

    ensemble_size = len(resulting_states_list)
    tiled_actions = np.tile(actions, (ensemble_size, 1, 1))

    ###########################################################
    ## some reshaping of the predicted trajectories to rate
    ###########################################################

    N = len(resulting_states_list[0][0])

    #resulting_states_list is [ensSize, H+1, N, statesize]
    resulting_states = []
    for timestep in range(len(resulting_states_list[0])): # loops over H+1
        all_per_timestep = []
        for entry in resulting_states_list: # loops over ensSize
            all_per_timestep.append(entry[timestep])
        all_per_timestep = np.concatenate(all_per_timestep)  #[ensSize*N, statesize]
        resulting_states.append(all_per_timestep)
    #resulting_states is now [H+1, ensSize*N, statesize]

    ###########################################################
    ## calculate costs associated with each predicted trajectory
    ######## treat each traj from each ensemble as just separate trajs
    ###########################################################

    #init vars for calculating costs
    costs = np.zeros((N * len(resulting_states_list),))
    prev_pt = resulting_states[0]
    dones = np.zeros((N * len(resulting_states_list),))

    tiled_goal = np.tile(goal, (prev_pt.shape[0],1))

    use_sampled_goal = False
    # use_sampled_goal = True

    if use_sampled_goal:
        global missing_rate
        change = 0.01
        n,m = tiled_goal.shape
        variance = 0.1 * missing_rate  # if missing_rate=100%, means all action sequences are failed, need a larger variance.
        sampled_goal = tiled_goal + np.random.randn(n,m)*variance

    # from ipdb import set_trace;
    # set_trace()

    #accumulate cost over each timestep
    for pt_number in range(len(resulting_states_list[0]) - 1):

        #array of "current datapoint" [(ensemble_size*N) x state]
        pt = resulting_states[pt_number + 1]
        #update cost at the next timestep of the H-step rollout
        actions_per_step = tiled_actions[:, pt_number]

        if use_sampled_goal:            #G2
            pre_ag = prev_pt[:,3:6]
            ag = pt[:,3:6]
            index = (np.linalg.norm((ag - pre_ag),axis=1) > change)
            tiled_goal[index] = sampled_goal[index]

        costs, dones = cost_per_step(pt, prev_pt, tiled_goal, costs, actions_per_step, dones, reward_func)
        #update
        prev_pt = np.copy(pt)

    ###########################################################
    ## assigns costs associated with each predicted trajectory
    ####### need to consider each ensemble separately again
    ####### perform ranking based on either
    #"mean costs" over ensemble predictions (for a given action sequence A)
    # or
    #"model disagreement" over ensemble predictions (for a given action sequence A)
    ###########################################################

    # from ipdb import set_trace;
    # set_trace()

    #consolidate costs (ensemble_size*N) --> (N)
    new_costs = []
    for i in range(N):
        # 1-a0 1-a1 1-a2 ... 2-a0 2-a1 2-a2 ... 3-a0 3-a1 3-a2...
        new_costs.append(costs[i::N])  #start, stop, step

    #mean and std cost (across ensemble) [N,]
    mean_cost = np.mean(new_costs, 1)
    std_cost = np.std(new_costs, 1)

    if use_sampled_goal:
        _, H, _ = actions.shape  # get horizon lenth
        missing_rate = len(mean_cost[mean_cost == H]) / len(mean_cost)
        print('missing_rate:', missing_rate)

    # from ipdb import set_trace;
    # set_trace()

    #rank by rewards
    if evaluating:
        cost_for_ranking = mean_cost
    #sometimes rank by model disagreement, and sometimes rank by rewards
    else:
        if take_exploratory_actions:
            cost_for_ranking = mean_cost - 4 * std_cost
            print("   ****** taking exploratory actions for this rollout")
        else:
            cost_for_ranking = mean_cost

    return cost_for_ranking, mean_cost, std_cost
