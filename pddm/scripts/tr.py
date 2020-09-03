
import gym
from ipdb import set_trace;
import numpy as np

def test_env():


    env = gym.make('FetchPush-v1')
    high = env.env.action_space.high
    low = env.env.action_space.low
    obs=[]
    dof = 1
    for i in range(2000):
        env.reset()                 #### reset() in mb_env.py ####
        a = env.env.action_space.sample()
        a = np.zeros_like(a)
        for j in range(10):
            print(a)
            # a = np.ones_like(a)

            # a[0] = 0.1
            for t in range(20):
                o, r, done, env_info = env.step(a)  #### step() in mb_env.py  ####
                # print('o:',t, o['observation'][:3])
                o=o['observation']
                print('o:',t, np.concatenate((o[:3], o[9:11], o[-5:])))
                obs.append(np.concatenate((o[:3], o[9:11], o[-5:])))
                env.env.render()
                # if done:
                #     print('done in i and t:',done,i,t)
                #     break
            if i%2 == 0:
                a[dof] += 0.1
            else:
                a[dof] -= 0.1



        from ipdb import set_trace;
        set_trace()

test_env()