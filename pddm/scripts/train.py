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

# from ipdb import set_trace;
# set_trace()

#my imports
from pddm.policies.policy_random import Policy_Random
from pddm.utils.helper_funcs import *
from pddm.regressors.dynamics_model import Dyn_Model
from pddm.policies.mpc_rollout import MPCRollout
from pddm.utils.loader import Loader
from pddm.utils.saver import Saver
from pddm.utils.data_processor import DataProcessor
from pddm.utils.data_structures import *
from pddm.utils.convert_to_parser_args import convert_to_parser_args
from pddm.utils import config_reader
from pddm.data_augmentation.ker_learning_method import ker_learning

SCRIPT_DIR = os.path.dirname(__file__)
n_KER = 3

def run_job(args, save_dir=None):

    # Continue training from an existing iteration
    # if args.continue_run>-1:
    #     save_dir = os.path.join(SCRIPT_DIR, args.continue_run_filepath)
    id = args.env_name
    KER = ker_learning(id, n_KER)

    tf.reset_default_graph()
    with tf.Session(config=get_gpu_config(args.use_gpu, args.gpu_frac)) as sess:

        ##############################################
        ### initialize some commonly used parameters (from args)
        ##############################################

        env_name = args.env_name
        continue_run = args.continue_run
        K = args.K
        num_iters = args.num_iters
        num_trajectories_per_iter = args.num_trajectories_per_iter
        horizon = args.horizon

        ### set seeds
        npr.seed(args.seed)
        tf.set_random_seed(args.seed)

        #######################
        ### hardcoded args
        #######################

        ### data types
        args.tf_datatype = tf.float32
        args.np_datatype = np.float32

        ### supervised learning noise, added to the training dataset
        args.noiseToSignal = 0.01

        ### these are for *during* MPC rollouts,
        # they allow you to run the H-step candidate actions on the real dynamics
        # and compare the model's predicted outcomes vs. the true outcomes
        execute_sideRollouts = False
        plot_sideRollouts = True

        ########################################
        ### create loader, env, rand policy
        ########################################
        # from ipdb import set_trace;
        # set_trace()

        loader = Loader(save_dir)
        env, dt_from_xml = create_env(env_name)
        args.dt_from_xml = dt_from_xml
        random_policy = Policy_Random(env.env)

        #doing a render here somehow allows it to not produce a seg fault error later when visualizing
        if args.visualize_MPC_rollout:
            render_env(env)
            render_stop(env)

        #################################################
        ### initialize or load in info
        #################################################

        #check for a variable which indicates that we should duplicate each data point
        #e.g., for baoding, since ballA/B are interchangeable, we store as 2 different points
        if 'duplicateData_switchObjs' in dir(env.unwrapped_env):
            duplicateData_switchObjs = True
            indices_for_switching = [env.unwrapped_env.objInfo_start1, env.unwrapped_env.objInfo_start2,
                                    env.unwrapped_env.targetInfo_start1, env.unwrapped_env.targetInfo_start2]
        else:
            duplicateData_switchObjs = False
            indices_for_switching=[]

        #initialize data processor
        data_processor = DataProcessor(args, duplicateData_switchObjs, indices_for_switching)

        # ######################### PickAndPlace demo #################################
        # demo = False
        # # demo = True
        # if demo:
        #     from pddm.scripts.read_npy import get_rollouts
        #     addr = 'pddm/pickandplace_demo_data/1000/'
        #     rollouts_trainRand = get_rollouts(addr)

        # onPol train/val data
        rollouts_trainOnPol = []
        rollouts_valOnPol = []

        # lists for saving
        trainingLoss_perIter = []
        rew_perIter = []
        scores_perIter = []
        trainingData_perIter = []



        # Get the first training random data

        # training
        rollouts_trainRand = collect_random_rollouts(env, random_policy, args.num_rand_rollouts_train, args.rand_rollout_length, dt_from_xml, args)
        # validation
        rollouts_valRand = collect_random_rollouts(env, random_policy, args.num_rand_rollouts_val, args.rand_rollout_length, dt_from_xml, args)

        # convert (rollouts --> dataset)
        dataset_trainRand = data_processor.convertRolloutsToDatasets(rollouts_trainRand)
        dataset_valRand = data_processor.convertRolloutsToDatasets(rollouts_valRand)

        ### check data dims
        inputSize, outputSize, acSize = check_dims(dataset_trainRand, env)
        ### amount of data
        numData_train_rand = get_num_data(rollouts_trainRand)
        print("    amount of random data: ", numData_train_rand)


        ##############################################
        ### dynamics model + controller
        ##############################################
        dyn_models = Dyn_Model(inputSize, outputSize, acSize, sess, params=args)
        mpc_rollout = MPCRollout(env, dyn_models, random_policy, execute_sideRollouts, plot_sideRollouts, args)

        ### init TF variables
        sess.run(tf.global_variables_initializer())

        ##############################################
        ###  saver
        ##############################################
        saver = Saver(save_dir, sess)
        saver.save_initialData(args, rollouts_trainRand, rollouts_valRand)

        # mean/std of all data
        data_processor.update_stats(dyn_models, dataset_trainRand)

        # preprocess datasets to mean0/std1 + clip actions
        preprocessed_data_trainRand = data_processor.preprocess_data(dataset_trainRand)
        preprocessed_data_valRand = data_processor.preprocess_data(dataset_valRand)
        # convert datasets (x,y,z) --> training sets (inp, outp)
        inputs, outputs = data_processor.xyz_to_inpOutp(preprocessed_data_trainRand)
        inputs_val, outputs_val = data_processor.xyz_to_inpOutp(preprocessed_data_valRand)

        # re-initialize all vars (randomly) if training from scratch
        ##restore model if doing continue_run
        if args.warmstart_training:
            if continue_run > 0:
                restore_path = save_dir + '/models/model_aggIter' + str(continue_run - 1) + '.ckpt'
                saver.tf_saver.restore(sess, restore_path)
                print("\n\nModel restored from ", restore_path, "\n\n")
        else:
            sess.run(tf.global_variables_initializer())

        inputs_onPol = inputs
        outputs_onPol = outputs
        inputs_val_onPol = inputs_val
        outputs_val_onPol = outputs_val

        nEpoch_use = args.nEpoch_init
        ## first time to train the model
        training_loss, training_lists_to_save = dyn_models.train(
            inputs_onPol,
            outputs_onPol,
            nEpoch_use,
            inputs_val_onPol=inputs_val_onPol,
            outputs_val_onPol=outputs_val_onPol, )
        print("   first training loss: ", training_loss)

        ##############################################
        ### THE MAIN LOOP
        ##############################################
        firstTime = True
        counter = 0
        while counter < num_iters:

            #saving rollout info
            list_rewards = []
            list_scores = []
            list_mpes = []
            rollouts_info_train = []
            rollouts_info_test = []

            if not args.print_minimal:
                print("\n#####################################")
                print("performing collecting rollouts... iter ", counter)
                print("#####################################\n")

            # from ipdb import set_trace;
            # set_trace()

            rollouts_train = []
            for rollout_num in range(40):  # for collecting trajs
                print("\n####################### collecting trajs #", rollout_num)
                starting_observation, starting_state = env.reset(return_start_state=True)  #reset env randomly
                rollout_info = mpc_rollout.collecting_trajs(
                    starting_state,
                    starting_observation,
                    controller_type='rand',
                    take_exploratory_actions=False)

                rollouts_info_train.append(rollout_info)
                rollout = Rollout(rollout_info['observations'],
                                  rollout_info['actions'],
                                  None,None)

                rollouts_train.append(rollout)

            # from ipdb import set_trace;
            # set_trace()

            ## here to reflect rollouts_train, but not the rollouts_val
            ######################### ker #################################
            if n_KER:
                Data = rollouts_train
                rollouts_train = []
                for data in Data:
                    original_ka_episodes = KER.ker_process(data.states, data.actions, data.g, data.ag)
                    for episode in original_ka_episodes:
                        rollout = Rollout(observations=np.array(episode[0]), actions=np.array(episode[1]),
                                          desired_goals=np.array(episode[2]), achieved_goals=np.array(episode[3]))
                        rollouts_train.append(rollout)
            ######################### ker #################################



            #aggregate into training data

            rollouts_trainOnPol = rollouts_trainOnPol + rollouts_train
            dataset_trainOnPol = data_processor.convertRolloutsToDatasets(rollouts_trainOnPol)   #convert (rollouts --> dataset)
            # mean/std of all data
            data_processor.update_stats(dyn_models, dataset_trainOnPol)
            preprocessed_data_trainOnPol = data_processor.preprocess_data(dataset_trainOnPol) #preprocess datasets to mean0/std1 + clip actions
            inputs_onPol, outputs_onPol = data_processor.xyz_to_inpOutp(preprocessed_data_trainOnPol)  #convert datasets (x,y,z) --> training sets (inp, outp)

            if (not (args.print_minimal)):
                numData_train_onPol = get_num_data(rollouts_trainOnPol)    # amount of data
                print("\n#####################################")
                print("Training the dynamics model..... iteration ", counter)
                print("#####################################\n")
                print("    amount of onPol data: ", numData_train_onPol)


            #####################################
            ## Training the model
            #####################################

            #re-initialize all vars (randomly) if training from scratch
            ##restore model if doing continue_run
            if args.warmstart_training:
                if firstTime:
                    if continue_run>0:
                        restore_path = save_dir + '/models/model_aggIter' + str(continue_run-1) + '.ckpt'
                        saver.tf_saver.restore(sess, restore_path)
                        print("\n\nModel restored from ", restore_path, "\n\n")
            else:
                sess.run(tf.global_variables_initializer())


            #number of training epochs
            if counter==0: nEpoch_use = args.nEpoch_init
            else: nEpoch_use = args.nEpoch

            if args.always_use_savedModel:
                if continue_run > 0:
                    restore_path = save_dir + '/models/model_aggIter' + str(continue_run - 1) + '.ckpt'
                else:
                    restore_path = save_dir + '/models/finalModel.ckpt'

                saver.tf_saver.restore(sess, restore_path)
                print("\n\nModel restored from ", restore_path, "\n\n")

                # empty vars, for saving
                training_loss = 0
                training_lists_to_save = dict(
                    training_loss_list=0,
                    val_loss_list_rand=0,
                    val_loss_list_onPol=0,
                    val_loss_list_xaxis=0,
                    rand_loss_list=0,
                    onPol_loss_list=0, )
            else:
                ## train model
                training_loss, training_lists_to_save = dyn_models.train(
                    inputs_onPol,
                    outputs_onPol,
                    nEpoch_use,
                    inputs_val_onPol=inputs_val_onPol,
                    outputs_val_onPol=outputs_val_onPol, )

            #####################################
            ## Testing the model and planner
            #####################################
            if not args.print_minimal:
                print("\n#####################################")
                print("performing on-policy MPC rollouts... iter ", counter)
                print("#####################################\n")

            rollouts_val = []
            for rollout_num in range(10):   # for testing, 10 rollouts in total
                print("\n####################### testing trajs #", rollout_num)

                # from ipdb import set_trace;
                # set_trace()

                starting_observation, starting_state = env.reset(return_start_state=True) #reset env randomly
                rollout_info = mpc_rollout.perform_rollout(
                    starting_state,
                    starting_observation,
                    controller_type=args.controller_type,
                    take_exploratory_actions=False)

                if len(rollout_info['observations']) > K:
                    list_rewards.append(rollout_info['rollout_rewardTotal'])
                    list_scores.append(rollout_info['rollout_meanFinalScore'])
                    list_mpes.append(np.mean(rollout_info['mpe_1step']))
                    rollouts_info_test.append(rollout_info)

                rollout = Rollout(rollout_info['observations'],
                                  rollout_info['actions'],
                                  rollout_info['rollout_rewardTotal'],
                                  rollout_info['starting_state'])

                rollouts_val.append(rollout)

            # visualize, if desired
            if args.visualize_MPC_rollout:
                print("\n\nPAUSED FOR VISUALIZATION. Continue when ready to visualize.")
                import IPython
                IPython.embed()
                for vis_index in range(len(rollouts_info_test)):
                    visualize_rendering(rollouts_info_test[vis_index], env, args)


            if counter==0: rollouts_valOnPol = []   # need to clean the random val data
            rollouts_valOnPol = rollouts_valOnPol + rollouts_val
            dataset_valOnPol = data_processor.convertRolloutsToDatasets(rollouts_valOnPol)   #convert (rollouts --> dataset)
            preprocessed_data_valOnPol = data_processor.preprocess_data(dataset_valOnPol)    #preprocess datasets to mean0/std1 + clip actions
            inputs_val_onPol, outputs_val_onPol = data_processor.xyz_to_inpOutp(preprocessed_data_valOnPol)  #convert datasets (x,y,z) --> training sets (inp, outp)



            #########################################################
            ### save everything about this iter of model training
            #########################################################
            trainingData_perIter.append(numData_train_onPol)
            trainingLoss_perIter.append(training_loss)

            #init vars for this iteration
            saver_data = DataPerIter()
            saver.iter_num = counter

            ### stage relevant info for saving
            saver_data.training_numData = trainingData_perIter
            saver_data.training_losses = trainingLoss_perIter
            saver_data.training_lists_to_save = training_lists_to_save
            # Note: the on-policy rollouts include curr iter's rollouts
            # (so next iter can be directly trained on these)
            saver_data.train_rollouts_onPol = rollouts_trainOnPol
            saver_data.val_rollouts_onPol = rollouts_valOnPol
            saver_data.normalization_data = data_processor.get_normalization_data()
            saver_data.counter = counter

            ### save all info from this training iteration
            saver.save_model()
            saver.save_training_info(saver_data)

            #########################################################
            ### save everything about this iter of MPC rollouts
            #########################################################

            # append onto rewards/scores
            rew_perIter.append([np.mean(list_rewards), np.std(list_rewards)])
            scores_perIter.append([np.mean(list_scores), np.std(list_scores)])

            # save
            saver_data.rollouts_rewardsPerIter = rew_perIter
            saver_data.rollouts_scoresPerIter = scores_perIter
            saver_data.rollouts_info = rollouts_info_train
            # saver_data.rollouts_info = rollouts_info_test
            saver.save_rollout_info(saver_data)
            counter = counter + 1

            firstTime = False
        return


def main():

    #####################
    # training args
    #####################

    parser = argparse.ArgumentParser(
        # Show default value in the help doc.
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument(
        '-c',
        '--config',
        nargs='*',
        help=('Path to the job data config file. This is specified relative '
            'to working directory'))

    parser.add_argument(
        '-o',
        '--output_dir',
        default='output',
        help=
        ('Directory to output trained policies, logs, and plots. A subdirectory '
         'is created for each job. This is speficified relative to  '
         'working directory'))

    parser.add_argument('--use_gpu', action="store_true")
    parser.add_argument('-frac', '--gpu_frac', type=float, default=0.7)
    general_args = parser.parse_args()

    #####################
    # job configs
    #####################

    # from ipdb import set_trace;
    # set_trace()

    # Get the job config files
    jobs = config_reader.process_config_files(general_args.config)
    assert jobs, 'No jobs found from config.'

    # Create the output directory if not present.
    output_dir = general_args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.abspath(output_dir)

    # Run separate experiment for each variant in the config
    for index, job in enumerate(jobs):

        #add an index to jobname, if there is more than 1 job
        if len(jobs)>1:
            job['job_name'] = '{}_{}'.format(job['job_name'], index)

        #convert job dictionary to different format
        args_list = config_dict_to_flags(job)
        args = convert_to_parser_args(args_list)

        #copy some general_args into args
        args.use_gpu = general_args.use_gpu
        args.gpu_frac = general_args.gpu_frac

        #directory name for this experiment
        job['output_dir'] = os.path.join(output_dir, job['job_name'])

        ################
        ### run job
        ################

        try:
            run_job(args, job['output_dir'])
        except (KeyboardInterrupt, SystemExit):
            print('Terminating...')
            sys.exit(0)
        except Exception as e:
            print('ERROR: Exception occured while running a job....')
            traceback.print_exc()

if __name__ == '__main__':
    main()
