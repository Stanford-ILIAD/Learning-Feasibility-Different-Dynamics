import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_panda.panda_bullet.panda import Panda, DisabledPanda, FeasibilityPanda, RealPanda
from gym_panda.panda_bullet.objects import YCBObject, InteractiveObj, RBOObject
import os
import numpy as np
import pybullet as p
import pybullet_data
import copy
import pickle

class PandaEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, panda_type=Panda):
    # create simulation (GUI)
    self.urdfRootPath = pybullet_data.getDataPath()
    p.connect(p.DIRECT)
    #p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    # set up camera
    self._set_camera()

    # load some scene objects
    p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
    p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

    # example YCB object
    obj1 = YCBObject('large_box')
    obj1.load()
    p.resetBasePositionAndOrientation(obj1.body_id, [0.56, 0., 0.1], [0, 0, 0, 1])

    self.panda = panda_type()
    self.arm_id = self.panda.panda
    self.obj_id = obj1.body_id

    self.n = 3
    self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(self.n, ), dtype=np.float32)
    self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(3,), dtype=np.float32)

  def reset(self):
    self.panda.reset()
    self.reward = 0
    self.reachgoal = False
    self.maxlength=4000
    self.length=0
    return self.panda.state

  def resetobj(self):
    print( self.panda.state['ee_position'])
    p.resetBasePositionAndOrientation(self.obj_id, self.panda.state['ee_position']+np.array([0,0.002,0.05]), [0, 0, 0, 1])
    return self.panda.state

  def close(self):
    p.disconnect()
  
  def seed(self, seed=None):
        self.panda.seed(seed)
        return [seed]

  def step(self, action, verbose=False):
    # get current state
    state = self.panda.state

    self.panda.step(dposition=action)

    # take simulation step
    p.stepSimulation()

    # return next_state, reward, done, info
    next_state = self.panda.state
    self.reward -=1
    state = next_state['ee_position']
    #collision
    self.reachgoal = np.linalg.norm(np.array([state[0] - 0.81, state[1],state[2] - 0.1])) < 0.05
    done =False
    if(self.reachgoal):
      self.reward+=5000
      done = True
    closest_point = p.getClosestPoints(self.arm_id, self.obj_id, 100)
    close_points = [[point[5], point[6]] for point in closest_point]
    min_distance = 100
    for point_pair in close_points:
      dist = (point_pair[0][0]-point_pair[1][0])**2 + (point_pair[0][1]-point_pair[1][1])**2 + (point_pair[0][2]-point_pair[1][2])**2
      if dist < min_distance:
        min_distance = dist
      if verbose:
        print(next_state['ee_position'], min_distance)
      if min_distance < 0.0001:
        self.reward -= 10000
        done = True
        break
    info = {}
    return next_state, self.reward, done, info

  def render(self, mode='human', close=False):
    (width, height, pxl, depth, segmentation) = p.getCameraImage(width=self.camera_width,
                                                                    height=self.camera_height,
                                                                    viewMatrix=self.view_matrix,
                                                                    projectionMatrix=self.proj_matrix)
    rgb_array = np.array(pxl, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _set_camera(self):
    self.camera_width = 256
    self.camera_height = 256
    p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=0, cameraPitch=-15,
                                    cameraTargetPosition=[0.5, -0.2, 0.0])
    self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0],
                                                            distance=1.0,
                                                            yaw=90,
                                                            pitch=-50,
                                                            roll=0,
                                                            upAxisIndex=2)
    self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                    aspect=float(self.camera_width) / self.camera_height,
                                                    nearVal=0.1,
                                                    farVal=100.0)

class DisabledPandaEnv(PandaEnv):

    def __init__(self, panda_type=DisabledPanda):
        super(DisabledPandaEnv, self).__init__(panda_type=panda_type)
        self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32)

    def step(self, action,verbose=False):
        action = action.squeeze()
        action1 = np.array([0., 0., 0.])
        action1[0] = action[0]
        action1[2] = action[2]   #change to 1 for rl, 2 for collect demons
        next_state = self.panda.state
        return super(DisabledPandaEnv, self).step(action1, verbose)


class FeasibilityPandaEnv(PandaEnv):

  def __init__(self, panda_type=FeasibilityPanda):
    super(FeasibilityPandaEnv, self).__init__(panda_type=panda_type)
    self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32)
    self.gt_data1 =  pickle.load(open('..\\data\\dis5.pkl', 'rb'))
    self.gt_data2 =  pickle.load(open('..\\data\\normal48.pkl', 'rb'))
    self.time_step = 0
    self.reward = 0
    self.eps_len = 8000
  
  def _random_select(self, idx=None, version=None):
        if idx is None:
            if version == "dis":
              self.gt_data = self.gt_data1
            else:
              self.gt_data = self.gt_data2
            self.gt_num = np.random.choice(self.gt_data.shape[0])
        else:
            self.gt_num = idx
            if version == "dis":
              self.gt_data = self.gt_data1
            else:
              self.gt_data = self.gt_data2
        self.gt = self.gt_data[self.gt_num][:][:]
        pos = self.gt[0][27:30]
        jointposition = np.concatenate((self.gt[0][:9],np.array([0.03,0.03])),axis=None)
        self.panda._reset_robot(jointposition)
        self.eps_len = self.gt.shape[0]
  
    
  def reset(self,idx=None, version = None):
    self.panda.reset()
    self._random_select(idx, version)
    self.time_step = 0
    return self.panda.state['ee_position']
    return state
  

  def step(self, action,verbose=False):
    action = action.squeeze()
    action1 = np.array([0., 0., 0.])
    action1[0] = action[0]
    action1[2] = action[1]
    super(FeasibilityPandaEnv, self).step(action1, verbose)
    state = self.panda.state['ee_position']
    self.time_step += 1
    done = (self.time_step >= self.eps_len - 1)
    dis = np.linalg.norm(state - self.gt[self.time_step][27:30])
    reward = -dis
    info = {}
    self.prev_state = copy.deepcopy(state)
    full_state = self.panda.state['ee_position']
    info['dis'] = dis
        
    return full_state, reward, done, info

      
class RealPandaEnv(PandaEnv):

  def __init__(self, panda_type=RealPanda):
    self.urdfRootPath = pybullet_data.getDataPath()
    p.connect(p.DIRECT)
    #p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    self._set_camera()

    p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
    p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

    obj1 = YCBObject('czj_book_stand')
    obj1.load()
    p.resetBasePositionAndOrientation(obj1.body_id, [0.42, 0., 0.0], [0, 0, 0, 1])
    obj2 = YCBObject('czj_book_stand')
    obj2.load()
    p.resetBasePositionAndOrientation(obj2.body_id, [0.68, 0., 0.0], [0, 0, 0, 1])
    obj3 = YCBObject('czj_book_stand')
    obj3.load()
    p.resetBasePositionAndOrientation(obj3.body_id, [1.0, 0., 0.0], [0, 0, 0, 1])
    obj4 = YCBObject('czj_book_stand_book')
    obj4.load()
    p.resetBasePositionAndOrientation(obj4.body_id, [0.55, 0, 0], [0, 0, 0, 1])
    obj5 = YCBObject('czj_book_stand_book_2')
    obj5.load()
    p.resetBasePositionAndOrientation(obj5.body_id, [0.55, 0, 0], [0, 0, 0, 1])
    self.panda = panda_type()
    self.arm_id = self.panda.panda
    self.obj_id = obj5.body_id
    self.obj_id2 = obj1.body_id

    self.n = 3
    self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32)
    self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(3,), dtype=np.float32)
    self.point1 = [0.15,0,0.87] #imitationExpert
    self.point2 = [0.16,0,0.17] #ralExpert
    self.point3 = [0.15,0,0.87] #sailExpert
    self.point4 = [0.8,0,0.1] #ourExpert
    self.point = self.point4

  def reset(self):
    print("real reset")
    self.panda.reset()
    self.resetobj()
    print( self.panda.state['ee_position'])
    self.reward = 0
    self.reachgoal = False
    self.maxlength=8000
    self.length=0
    return self.panda.state
  
  def resetobj(self):
    print( self.panda.state['ee_position'])
    p.resetBasePositionAndOrientation(self.obj_id, self.panda.state['ee_position'], [0, 0, 0, 1])
    return self.panda.state
  
  def step(self, action, verbose=False):
    state = self.panda.state
    self.panda.step(dposition=action)
    p.stepSimulation()
    next_state = self.panda.state
    self.reward -=1
    state = next_state['ee_position']
    self.reachgoal = np.linalg.norm(np.array([state[0] - self.point[0], state[1],state[2] - self.point[2]])) < 0.05
    done =False
    if(self.reachgoal):
      self.reward+=5000
      print("reach")
      done = True
    closest_point = p.getClosestPoints(self.arm_id, self.obj_id2, 100)
    close_points = [[point[5], point[6]] for point in closest_point]
    min_distance = 100
    for point_pair in close_points:
      dist = (point_pair[0][0]-point_pair[1][0])**2 + (point_pair[0][1]-point_pair[1][1])**2 + (point_pair[0][2]-point_pair[1][2])**2
      if dist < min_distance:
        min_distance = dist
      if verbose:
        print(next_state['ee_position'], min_distance)
      if min_distance < 0.0001:
        self.reward -= 10000
        done = True
        break
    self.length+=1
    if(self.length==self.maxlength):
      self.reward -=5000
      done = True
    info = {}
    return next_state, self.reward, done, info

  def _set_camera(self):
    self.camera_width = 256
    self.camera_height = 256
    p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=0, cameraPitch=-15,
                                    cameraTargetPosition=[0.5, -0.2, 0.0])
    self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0],
                                                            distance=1.0,
                                                            yaw=90,
                                                            pitch=-50,
                                                            roll=0,
                                                            upAxisIndex=2)
    self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                    aspect=float(self.camera_width) / self.camera_height,
                                                    nearVal=0.1,
                                                    farVal=100.0)

  


      
      