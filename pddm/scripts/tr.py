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

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import numpy.random as npr
import tensorflow as tf
import pickle
import sys
import argparse
import traceback


def test_env_use_true_dynamic():

    import gym

    #env = gym.make('Pusher-v0')
    #env = gym.make('FetchPush-v1')
    env = gym.make('CartPole-v1')


    from ipdb import set_trace;
    set_trace()

    
    for i in range(2000):
        env.reset()
        for t in range(100):
            env.render()
            a = env.env.action_space.sample()
            o, r, done, env_info = env.step(a)  #### step() in mb_env.py  ####
            # if done:
            #     print('done in i and t:', done, i, t)
            #     break

        from ipdb import set_trace;
        set_trace()

if __name__ == '__main__':
    # test_env()
    test_env_use_true_dynamic()


