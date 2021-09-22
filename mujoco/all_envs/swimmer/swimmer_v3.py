import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import pdb
import os
import torch
import random

DEFAULT_CAMERA_CONFIG = {}


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='swimmer.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-4,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, ('%s/assets/'+xml_file) % dir_path, 4)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        xy_position_before = self.sim.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.sim.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = self._forward_reward_weight * x_velocity

        ctrl_cost = self.control_cost(np.clip(action, -1, 1))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            'reward_fwd': forward_reward,
            'reward_ctrl': -ctrl_cost,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation = np.concatenate([position, velocity]).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def set_observation(self, observation):
        self.set_state(observation[0:len(self.init_qpos)], observation[len(self.init_qpos):len(self.init_qpos)+len(self.init_qvel)])

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class SwimmerOursEnv(SwimmerEnv):
    def __init__(self,
                 xml_file='swimmer.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-4,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):

        self.vae=None
        #self.global_gt_env = SwimmerEnv(exclude_current_positions_from_observation=False)
        #self.local_gt_env = SwimmerEnv(exclude_current_positions_from_observation=False)
        self.tracked_gt_states = []
        self.tracked_robot_states = []
        #self.local_verify_rew_list = []
        #self.global_verify_rew_list = []

        super(SwimmerOursEnv, self).__init__(xml_file, forward_reward_weight, ctrl_cost_weight, reset_noise_scale, exclude_current_positions_from_observation)

    def set_vae(self, vae, device):
        self.vae = vae
        self.device = device

    def step(self, action):
        if self.vae is not None:
            position = self.sim.data.qpos.flat.copy()
            velocity = self.sim.data.qvel.flat.copy()
            observations = self._get_obs()
            with torch.no_grad():
                observations = torch.from_numpy(observations).unsqueeze(0).float()
                local_expected_next_state = self.vae(observations).cpu().numpy()[0] #torch.normal(action_mean, action_std).cpu().numpy()[0]

            with torch.no_grad():
                observations = torch.from_numpy(self.tracked_gt_states[-1]).unsqueeze(0).float()
                global_expected_next_state = self.vae(observations).cpu().numpy()[0]
            self.tracked_gt_states.append(global_expected_next_state)
            #expected_next_state = np.concatenate([expected_next_state[0:7], expected_next_state[7+self.leg_num*2:13+self.leg_num*2]], axis=0)

        self.do_simulation(action, self.frame_skip)

        if self.vae is not None:
            position = self.sim.data.qpos.flat.copy()
            velocity = self.sim.data.qvel.flat.copy()

            joint_state_after = np.concatenate((position, velocity))
            self.tracked_robot_states.append(joint_state_after)
            local_reward = np.exp(-np.linalg.norm(joint_state_after-local_expected_next_state, ord=2, axis=0)/2.)
            #global_reward = 30-np.mean(np.linalg.norm(np.array(self.tracked_robot_states)-np.array(self.tracked_gt_states), ord=2, axis=1))
            global_reward = np.exp(-np.linalg.norm(np.array(self.tracked_robot_states[-1])-np.array(self.tracked_gt_states[-1]), ord=2, axis=0)/2.)
            reward = 0*local_reward + global_reward
        else:
            reward = 0.

        done = False
        observation = self._get_obs()
        info = {
        }
        return observation, reward, done, info

    def reset_model(self):
        #print('optimal rew: ', np.mean(self.global_verify_rew_list), ' local rew: ', np.mean(self.local_verify_rew_list), ' rew len: ', len(self.global_verify_rew_list))
        self.tracked_gt_states = []
        self.tracked_robot_states = []
        #self.local_verify_rew_list = []
        #self.global_verify_rew_list = []

        #self.global_gt_env.reset()
        #self.local_gt_env.reset()
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)
        #self.global_gt_env.set_state(qpos, qvel)
        #self.local_gt_env.set_state(qpos, qvel)
        observation = self._get_obs()
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        observation1 = np.concatenate((position, velocity))
        self.tracked_gt_states.append(observation1)
        self.tracked_robot_states.append(observation1)
        return observation



class SwimmerOursGTEnv(SwimmerEnv):
    def __init__(self,
                 xml_file='swimmer.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-4,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):

        self.policy=None
        self.global_gt_env = SwimmerEnv(exclude_current_positions_from_observation=False)
        self.local_gt_env = SwimmerEnv(exclude_current_positions_from_observation=False)
        self.tracked_gt_states = []
        self.tracked_robot_states = []
        self.local_verify_rew_list = []
        self.global_verify_rew_list = []

        super(SwimmerOursGTEnv, self).__init__(xml_file, forward_reward_weight, ctrl_cost_weight, reset_noise_scale, exclude_current_positions_from_observation)

    def set_policy(self, policy, device):
        self.policy = policy
        self.device = device

    def step(self, action):
        if self.policy is not None:
            position = self.sim.data.qpos.flat.copy()
            velocity = self.sim.data.qvel.flat.copy()
            observations = self._get_obs()

            self.local_gt_env.set_state(self.sim.data.qpos, self.sim.data.qvel)
            with torch.no_grad():
                observations = torch.from_numpy(observations[2:]).unsqueeze(0).float()
                action_mean, _, action_std = self.policy(observations)
                expected_action = action_mean.cpu().numpy()[0] #torch.normal(action_mean, action_std).cpu().numpy()[0]
            local_expected_next_state, verify_rew, _, _ = self.local_gt_env.step(expected_action)
            self.local_verify_rew_list.append(verify_rew)

            with torch.no_grad():
                observations = torch.from_numpy(self.tracked_gt_states[-1][2:]).unsqueeze(0).float()
                action_mean, _, action_std = self.policy(observations)
                expected_action = action_mean.cpu().numpy()[0] #torch.normal(action_mean, action_std).cpu().numpy()[0]
            global_expected_next_state, verify_rew, _, _ = self.global_gt_env.step(expected_action)
            self.global_verify_rew_list.append(verify_rew)
            self.tracked_gt_states.append(global_expected_next_state)
            #expected_next_state = np.concatenate([expected_next_state[0:7], expected_next_state[7+self.leg_num*2:13+self.leg_num*2]], axis=0)

        self.do_simulation(action, self.frame_skip)

        if self.policy is not None:
            position = self.sim.data.qpos.flat.copy()
            velocity = self.sim.data.qvel.flat.copy()

            joint_state_after = np.concatenate((position, velocity))
            self.tracked_robot_states.append(joint_state_after)
            local_reward = np.exp(-np.linalg.norm(joint_state_after-local_expected_next_state, ord=2, axis=0)/2.)
            #global_reward = 30-np.mean(np.linalg.norm(np.array(self.tracked_robot_states)-np.array(self.tracked_gt_states), ord=2, axis=1))
            global_reward = np.exp(-np.linalg.norm(np.array(self.tracked_robot_states[-1])-np.array(self.tracked_gt_states[-1]), ord=2, axis=0)/2.)
            reward = local_reward + global_reward
        else:
            reward = 0.

        done = False
        observation = self._get_obs()
        info = {
        }
        return observation, reward, done, info

    def reset_model(self):
        print('optimal rew: ', np.mean(self.global_verify_rew_list), ' local rew: ', np.mean(self.local_verify_rew_list), ' rew len: ', len(self.global_verify_rew_list))
        self.tracked_gt_states = []
        self.tracked_robot_states = []
        self.local_verify_rew_list = []
        self.global_verify_rew_list = []

        self.global_gt_env.reset()
        self.local_gt_env.reset()
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)
        self.global_gt_env.set_state(qpos, qvel)
        self.local_gt_env.set_state(qpos, qvel)
        observation = self._get_obs()
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        observation1 = np.concatenate((position, velocity))
        self.tracked_gt_states.append(observation1)
        self.tracked_robot_states.append(observation1)
        return observation


class SwimmerFeasibilityEnv(SwimmerEnv):
    def __init__(self,
                 xml_file='swimmer.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-4,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True, demos=[]):
        self.demos = demos
        self.step_ = 0
        self.current_traj = None
        super(SwimmerFeasibilityEnv, self).__init__(xml_file, forward_reward_weight, ctrl_cost_weight, reset_noise_scale,
                 exclude_current_positions_from_observation)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        if self.current_traj is None:
            reward = 0
        else:
            reward = -np.linalg.norm(self.current_traj[self.step_+1] - observation, ord=2)

        self.step_ += 1
        if self.current_traj is not None and self.step_ >= len(self.current_traj)-1:
            done = True
        else:
            done = False
        info = {
        }

        return observation, reward, done, info

    def reset_model(self):
        self.step_ = 0
        self.current_traj = self.demos[random.sample(range(len(self.demos)), 1)[0]]
        qpos = self.current_traj[0][0:len(self.init_qpos)]
        qvel = self.current_traj[0][len(self.init_qpos):len(self.init_qvel)+len(self.init_qpos)]
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def set_initial_state(self, demo):
        self.step_ = 0
        self.current_traj = demo
        qpos = self.current_traj[0][0:len(self.init_qpos)]
        qvel = self.current_traj[0][len(self.init_qpos):len(self.init_qvel)+len(self.init_qpos)]
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

