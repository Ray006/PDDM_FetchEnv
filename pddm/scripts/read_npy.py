import numpy as np
import os
# from pddm.utils.data_structures import Rollout
from pddm.utils.data_structures import *

# ds = []
# d = np.load('obs.npy', allow_pickle=True)
# ds.append(d)


def get_rollouts(dir):

    rollouts_act = []
    rollouts_obs = []
    rollouts_info = []
    rollouts = []

    files = os.listdir(dir)

    for file in files:
        if file.startswith('acs'):
            file = os.path.join(dir, file)
            rollouts_act = np.load(file, allow_pickle=True)
        if file.startswith('obs'):
            file = os.path.join(dir, file)
            rollouts_obs = np.load(file, allow_pickle=True)
        if file.startswith('info'):
            file = os.path.join(dir, file)
            rollouts_info = np.load(file, allow_pickle=True)


    for i in range(len(rollouts_act)):
        act = rollouts_act[i]
        info = rollouts_info[i]

        obs = []
        ag = []
        g = []
        obs_list = rollouts_obs[i]
        for obs_dict in obs_list:
            obs.append(obs_dict['observation'])
            ag.append(obs_dict['achieved_goal'])
            g.append(obs_dict['desired_goal'])

        # from ipdb import set_trace
        # set_trace()

        rollout = Rollout(observations=np.array(obs), actions=np.array(act), achieved_goals=np.array(ag), desired_goals=np.array(g), info=np.array(info))
        rollouts.append(rollout)

    return rollouts

if __name__ == '__main__':
    dir = 'pddm/pickandplace_demo_data/'
    get_rollouts(dir)
