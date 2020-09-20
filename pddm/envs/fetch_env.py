import numpy as np

from gym.envs.robotics import rotations, robot_env, utils

# from gym.envs.robotics import rotations, utils
# from pddm.envs import robot_env

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
            self, model_path, n_substeps, gripper_extra_height, block_gripper,
            has_object, target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------
    # sparse or no sparse defined by gym
    def compute_reward(self, achieved_goal, goal, info=None):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)

        self.reward_type = 'dense'  ##### test

        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # Added by Ray get_reward() and get_score()
    # ----------------------------
    def get_reward(self, observations, goal, actions):  ###### FetchReach, V9, grip -> to goal or to object, dense

        # if not self.has_object:
        #     ag_index = 0
        # else:
        #     ag_index = 3

        if np.ndim(observations) == 2:  # for the planner to select actions
            n, m = observations.shape
            assert m == 10
            reward = np.zeros(n)
            dones = np.zeros(n)

            grip_pos = observations[:, 0:3]

            # from ipdb import set_trace;
            # set_trace()

            d_pos = np.linalg.norm(grip_pos - goal, axis=-1)

            index = np.array([i for i in range(n)])
            Idx = index[(d_pos <= self.distance_threshold)]
            reward[Idx] += 100
            dones[Idx] = True

            Idx = index[dones==0]
            reward[Idx] += - d_pos[Idx]

            # Idx = index[diff_angle <= 100]  # if d_pos<=thre_pos and d_vel<=thre_vel:
            # reward[Idx] += 1
            # Idx = index[diff_angle <= 50]  # if d_pos<=thre_pos and d_vel<=thre_vel:
            # reward[Idx] += 2
            # Idx = index[diff_angle <= 10]  # if d_pos<=thre_pos and d_vel<=thre_vel:
            # reward[Idx] += 5
            # Idx = index[diff_angle <= 5]  # if d_pos<=thre_pos and d_vel<=thre_vel:
            # reward[Idx] += 10

            return reward, dones

        else:  # for the real reward when interacting with the environment.
            m = len(observations)
            assert m == 10
            reward = 0

            grip_pos = observations[0:3]
            d_pos = np.linalg.norm(grip_pos - goal, axis=-1)
            # from ipdb import set_trace;
            # set_trace()

            done = False
            if d_pos <= self.distance_threshold:
                reward += 100
                done = True
            else:
                reward += - d_pos

            #     if diff_angle <= 100:
            #         reward += 1
            #     if diff_angle <= 50:
            #         reward += 2
            #     if diff_angle <= 10:
            #         reward += 5
            #     if diff_angle <= 5:
            #         reward += 10
            #
            #     done = False

            return reward, done

    # # Added by Ray get_reward() and get_score()
    # # ----------------------------
    # def get_reward(self, observations, starting_state, goal, actions):  ###### FetchReach, V9, pos only
    #
    #     # if not self.has_object:
    #     #     ag_index = 0
    #     # else:
    #     #     ag_index = 3
    #
    #     thre_pos = self.distance_threshold  # 0.05
    #     thre_vel = self.velocity_threshold  # 0.001
    #     thre_angle = self.angle_threshold  # 5.0
    #
    #     if np.ndim(observations) == 2:  # for the planner to select actions
    #         n, m = observations.shape
    #         assert m == 10
    #         reward = np.zeros(n)
    #         dones = np.zeros(n)
    #
    #         grip_pos = observations[:, 0:3]
    #         grip_velp_vec = observations[:, -5:-2]
    #         grip_velp = np.linalg.norm(grip_velp_vec, axis = 1)   ###################
    #
    #         gripper_state = observations[:, 3:5]
    #         gripper_vel = observations[:, -2:]
    #
    #         goal_pos = goal[:, :3]
    #         goal_dir_vec = goal[:, 3:6]
    #         goal_vel = goal[:, -1]
    #
    #         # from ipdb import set_trace;
    #         # set_trace()
    #
    #         d_pos = np.linalg.norm(grip_pos - goal_pos, axis=-1)
    #         d_vel = abs( grip_velp - goal_vel ) ###################
    #
    #         numerator = np.diagonal(grip_velp_vec.dot(goal_dir_vec.transpose()))
    #         denominator = np.linalg.norm(grip_velp_vec, axis=1) * np.linalg.norm(goal_dir_vec, axis=1)
    #         diff_angle = np.arccos(numerator / denominator) * (180 / np.pi)
    #
    #         index = np.array([i for i in range(n)])
    #         # Idx = index[(d_pos <= thre_pos) & (diff_angle <= thre_angle) & (d_vel <= thre_vel)]  # if d_pos<=thre_pos and d_vel<=thre_vel:
    #         Idx = index[(d_pos <= thre_pos)]
    #         reward[Idx] += 100
    #         dones[Idx] = True
    #
    #         Idx = index[dones==0]
    #         reward[Idx] += - d_pos[Idx]
    #         # reward[Idx] += - 10*d_vel[Idx]
    #         #
    #         # Idx = index[diff_angle <= 100]  # if d_pos<=thre_pos and d_vel<=thre_vel:
    #         # reward[Idx] += 1
    #         # Idx = index[diff_angle <= 50]  # if d_pos<=thre_pos and d_vel<=thre_vel:
    #         # reward[Idx] += 2
    #         # Idx = index[diff_angle <= 10]  # if d_pos<=thre_pos and d_vel<=thre_vel:
    #         # reward[Idx] += 5
    #         # Idx = index[diff_angle <= 5]  # if d_pos<=thre_pos and d_vel<=thre_vel:
    #         # reward[Idx] += 10
    #
    #         return reward, dones
    #
    #     else:  # for the real reward when interacting with the environment.
    #         m = len(observations)
    #         assert m == 10
    #         reward = 0
    #
    #         grip_pos = observations[0:3]
    #         grip_velp_vec = observations[-5:-2]
    #         grip_velp = np.linalg.norm(grip_velp_vec)
    #         gripper_state = observations[3:5]
    #         gripper_vel = observations[-2:]
    #
    #         goal_pos = goal[:3]
    #         goal_dir_vec = goal[3:6]
    #         goal_vel = goal[-1]
    #
    #         d_pos = np.linalg.norm(grip_pos - goal_pos, axis=-1)
    #         d_vel = abs(grip_velp - goal_vel)
    #         diff_angle = np.arccos(grip_velp_vec.dot(goal_dir_vec) / (np.linalg.norm(grip_velp_vec) * np.linalg.norm(goal_dir_vec))) * (180 / np.pi)
    #
    #         # from ipdb import set_trace;
    #         # set_trace()
    #
    #         done = False
    #         # if d_pos <= thre_pos and diff_angle <= thre_angle and d_vel <= thre_vel:
    #         if d_pos <= thre_pos :
    #             reward += 100
    #             done = True
    #         else:
    #             reward += - d_pos
    #             # reward += - 10*d_vel
    #         #
    #         #     if diff_angle <= 100:
    #         #         reward += 1
    #         #     if diff_angle <= 50:
    #         #         reward += 2
    #         #     if diff_angle <= 10:
    #         #         reward += 5
    #         #     if diff_angle <= 5:
    #         #         reward += 10
    #         #
    #         #     done = False
    #
    #         return reward, done

    # Added by Ray get_reward() and get_score()
    # ----------------------------
    # def get_reward(self, observations, starting_state, goal,
    #                actions):  ###### FetchReach, V8, grip -> to goal or to object, dense
    #
    #     # if not self.has_object:
    #     #     ag_index = 0
    #     # else:
    #     #     ag_index = 3
    #
    #     thre_pos = self.distance_threshold  # 0.05
    #     thre_vel = self.velocity_threshold  # 0.005
    #     thre_angle = self.angle_threshold  # 5.0
    #
    #     if np.ndim(observations) == 2:  # for the planner to select actions
    #         n, m = observations.shape
    #         assert m == 10
    #         reward = np.zeros(n)
    #         dones = np.zeros(n)
    #
    #         grip_pos = observations[:, 0:3]
    #         grip_velp = observations[:, -5:-2]
    #         gripper_state = observations[:, 3:5]
    #         gripper_vel = observations[:, -2:]
    #
    #         # np.arccos(x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))) * (180 / np.pi)
    #
    #         goal_pos = goal[:, :3]
    #         goal_velp = goal[:, 3:]
    #         d_pos = np.linalg.norm(grip_pos - goal_pos, axis=-1)
    #         d_vel = np.linalg.norm(grip_velp - goal_velp, axis=-1)
    #
    #         # from ipdb import set_trace;
    #         # set_trace()
    #
    #         numerator = np.diagonal(grip_velp.dot(goal_velp.transpose()))
    #         denominator = np.linalg.norm(grip_velp, axis=1) * np.linalg.norm(goal_velp, axis=1)
    #         angle = np.arccos(numerator / denominator) * (180 / np.pi)
    #
    #         index = np.array([i for i in range(n)])
    #
    #         Idx = index[d_pos <= thre_pos]  # if d_pos<=thre_pos and d_vel<=thre_vel:
    #         reward[Idx] += 10
    #         reward[Idx] += -(100 * d_vel[Idx] + 0.01 * angle[Idx])
    #         dones[Idx] = False
    #
    #         Idx = index[(d_pos <= thre_pos) & (angle <= thre_angle) & (
    #                     d_vel <= thre_vel)]  # if d_pos<=thre_pos and d_vel<=thre_vel:
    #         reward[Idx] += 100
    #         dones[Idx] = True
    #
    #         Idx = index[d_pos > thre_pos]  # if d_pos<=thre_pos and d_vel<=thre_vel:
    #         reward[Idx] = -d_pos[Idx]
    #         dones[Idx] = False
    #
    #         return reward, dones
    #
    #     else:  # for the real reward when interacting with the environment.
    #         m = len(observations)
    #         assert m == 10
    #         reward = 0
    #
    #         grip_pos = observations[0:3]
    #         grip_velp = observations[-5:-2]
    #         gripper_state = observations[3:5]
    #         gripper_vel = observations[-2:]
    #
    #         goal_pos = goal[:3]
    #         goal_velp = goal[3:]
    #
    #         d_pos = np.linalg.norm(grip_pos - goal_pos, axis=-1)
    #         d_vel = np.linalg.norm(grip_velp - goal_velp, axis=-1)
    #
    #         if d_pos <= thre_pos:
    #             reward += 10
    #             angle = np.arccos(
    #                 grip_velp.dot(goal_velp) / (np.linalg.norm(grip_velp) * np.linalg.norm(goal_velp))) * (
    #                                 180 / np.pi)
    #             reward += -(100 * d_vel + 0.01 * angle)
    #             done = False
    #             if angle <= thre_angle and d_vel <= thre_vel:
    #                 reward += 100
    #                 done = True
    #         else:
    #             reward = -d_pos
    #             done = False
    #         return reward, done

    # # Added by Ray get_reward() and get_score()
    # # ----------------------------
    # def get_reward(self, observations, starting_state, goal, actions):  ######  FetchPush. V7, grip -> to goal or to object, dense
    #
    #     # if not self.has_object:
    #     #     ag_index = 0
    #     # else:
    #     #     ag_index = 3
    #
    #     thre_pos = self.distance_threshold*2  # 0.05
    #     thre_vel = self.velocity_threshold  # 0.005
    #     thre_angle = self.angle_threshold  # 5.0
    #
    #     if np.ndim(observations) == 2:  # for the planner to select actions
    #         n, m = observations.shape
    #         assert m == 10
    #         reward = np.zeros(n)
    #         dones = np.zeros(n)
    #
    #         grip_pos = observations[:, 0:3]
    #         grip_velp = observations[:, -5:-2]
    #         gripper_state = observations[:, 3:5]
    #         gripper_vel = observations[:, -2:]
    #
    #         # np.arccos(x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))) * (180 / np.pi)
    #
    #         goal_pos = goal[:,:3]
    #         goal_velp = goal[:,3:]
    #         d_pos = np.linalg.norm(grip_pos - goal_pos, axis=-1)
    #         d_vel = np.linalg.norm(grip_velp - goal_velp, axis=-1)
    #
    #         # from ipdb import set_trace;
    #         # set_trace()
    #
    #         numerator = np.diagonal(grip_velp.dot(goal_velp.transpose()))
    #         denominator = np.linalg.norm(grip_velp, axis=1) * np.linalg.norm(goal_velp, axis=1)
    #         angle = np.arccos(numerator / denominator) * (180 / np.pi)
    #
    #
    #         index = np.array([i for i in range(n)])
    #
    #         Idx = index[d_pos <= thre_pos]  # if d_pos<=thre_pos and d_vel<=thre_vel:
    #         reward[Idx] += 10
    #         reward[Idx] += -(100*d_vel[Idx]+0.5*angle[Idx])
    #         dones[Idx] = False
    #
    #         Idx = index[(d_pos <= thre_pos) & (angle <= thre_angle) & (d_vel <= thre_vel)]  # if d_pos<=thre_pos and d_vel<=thre_vel:
    #         reward[Idx] += 100
    #         dones[Idx] = True
    #
    #         Idx = index[d_pos > thre_pos]  # if d_pos<=thre_pos and d_vel<=thre_vel:
    #         reward[Idx] = -d_pos[Idx]
    #         dones[Idx] = False
    #
    #         return reward, dones
    #
    #     else:  # for the real reward when interacting with the environment.
    #         m = len(observations)
    #         assert m == 10
    #         reward = 0
    #
    #         grip_pos = observations[0:3]
    #         grip_velp = observations[-5:-2]
    #         gripper_state = observations[3:5]
    #         gripper_vel = observations[-2:]
    #
    #         goal_pos = goal[:3]
    #         goal_velp = goal[3:]
    #
    #         d_pos = np.linalg.norm(grip_pos - goal_pos, axis=-1)
    #         d_vel = np.linalg.norm(grip_velp - goal_velp, axis=-1)
    #
    #
    #         if d_pos <= thre_pos:
    #             reward += 10
    #             angle = np.arccos(grip_velp.dot(goal_velp) / (np.linalg.norm(grip_velp) * np.linalg.norm(goal_velp))) * (180 / np.pi)
    #             reward += -(100*d_vel+0.5*angle)
    #             done = False
    #             if angle <= thre_angle and d_vel <= thre_vel:
    #                 reward += 100
    #                 done = True
    #         else:
    #             reward = -d_pos
    #             done = False
    #         return reward, done

    # Added by Ray get_reward() and get_score()
    # ----------------------------
    # def get_reward(self, observations, starting_state, goal, actions):  ###### V6, grip -> to goal or to object, dense
    #
    #     # from ipdb import set_trace;
    #     # set_trace()
    #
    #     # if not self.has_object:
    #     #     ag_index = 0
    #     # else:
    #     #     ag_index = 3
    #
    #     starting_grip_pos = starting_state[0:3]
    #     starting_grip_velp = starting_state[-5:-2]
    #     starting_gripper_state = starting_state[3:5]
    #     starting_gripper_vel = starting_state[-2:]
    #
    #     if np.ndim(observations) == 2:  # for the planner to select actions
    #         n, m = observations.shape
    #         assert m == 10
    #         reward = np.zeros(n)
    #         dones = np.zeros(n)
    #
    #         grip_pos = observations[:, 0:3]
    #         grip_velp = observations[:, -5:-2]
    #         gripper_state = observations[:, 3:5]
    #         gripper_vel = observations[:, -2:]
    #
    #         # np.arccos(x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))) * (180 / np.pi)
    #         P_max = np.linalg.norm(starting_grip_pos - goal[0,:3], axis=-1)
    #         V_max = np.linalg.norm(starting_grip_velp - goal[0,3:], axis=-1)
    #
    #         d_pos = np.linalg.norm(grip_pos - goal[:,:3], axis=-1)
    #         d_vel = np.linalg.norm(grip_velp - goal[:,3:], axis=-1)
    #
    #         d_pos_ = d_pos/P_max    # normalization, to make the pos and vel the same important.
    #         d_vel_ = d_vel/V_max
    #
    #         thre_pos = self.distance_threshold #0.05
    #         thre_vel = self.velocity_threshold # 0.005
    #
    #         index = np.array([i for i in range(n)])
    #         Idx = index[(d_pos <= thre_pos) & (d_vel <= thre_vel)] # if d_pos<=thre_pos and d_vel<=thre_vel:
    #         # reward[Idx] += 100
    #         dones[Idx] = True
    #
    #         Idx = index[dones==False] # else
    #         reward[Idx] = -(d_pos_[Idx] + d_vel_[Idx])
    #         dones[Idx] = False
    #
    #         return reward, dones
    #
    #
    #     else:  # for the real reward when interacting with the environment.
    #         m = len(observations)
    #         assert m == 10
    #
    #         grip_pos = observations[0:3]
    #         grip_velp = observations[-5:-2]
    #         gripper_state = observations[3:5]
    #         gripper_vel = observations[-2:]
    #
    #         P_max = np.linalg.norm(starting_grip_pos - goal[:3], axis=-1)
    #         V_max = np.linalg.norm(starting_grip_velp - goal[3:], axis=-1)
    #
    #         d_pos = np.linalg.norm(grip_pos - goal[:3], axis=-1)
    #         d_vel = np.linalg.norm(grip_velp - goal[3:], axis=-1)
    #
    #         d_pos_ = d_pos/P_max    # normalization, to make the pos and vel the same important.
    #         d_vel_ = d_vel/V_max
    #
    #         thre_pos = self.distance_threshold #0.05
    #         thre_vel = self.velocity_threshold
    #
    #         if d_pos<=thre_pos and d_vel<=thre_vel:
    #             reward = 0
    #             done = True
    #         else:
    #             reward = -(d_pos_ + d_vel_)
    #             done = False
    #         return reward, done

    # # Added by Ray get_reward() and get_score()
    # # ----------------------------
    # def get_reward(self, observations, goal, actions):    ###### V5, grip -> to goal or to object, dense
    #
    #     # from ipdb import set_trace;
    #     # set_trace()
    #
    #     # if not self.has_object:
    #     #     ag_index = 0
    #     # else:
    #     #     ag_index = 3
    #
    #     if np.ndim(observations)==2:       # for the planner to select actions
    #         n,m = observations.shape
    #         assert m == 10
    #
    #         grip_pos = observations[:,0:3]
    #         grip_velp = observations[:,-5:-2]
    #
    #         gripper_state = observations[:,3:5]
    #         gripper_vel = observations[:,-2:]
    #
    #         achieved_goal = np.concatenate((grip_pos,grip_velp),axis=1)
    #
    #         reward = self.compute_reward(achieved_goal, goal)
    #         dones = np.zeros((observations.shape)[0])  # this task is never terminated
    #
    #
    #     else:      # for the real reward when interacting with the environment.
    #         m = len(observations)
    #         assert m == 10
    #
    #         grip_pos = observations[0:3]
    #         grip_velp = observations[-5:-2]
    #
    #         gripper_state = observations[3:5]
    #         gripper_vel = observations[-2:]
    #
    #         achieved_goal = np.concatenate((grip_pos, grip_velp))
    #
    #         reward = self.compute_reward(achieved_goal, goal)
    #         dones = 0  # this task is never terminated
    #
    #     return reward, dones

    def get_score(self, obs_dict):
        if not self.has_object:
            ag = obs_dict['observation'][:3]
        else:
            ag = obs_dict['observation'][3:6]

        g = obs_dict['desired_goal']
        return -1.0*np.linalg.norm(g - ag)


    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt  # position velocity
            object_velr = self.sim.data.get_site_xvelr('object0') * dt  # rotation velocity
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
