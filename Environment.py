import pybullet as p
import pybullet_data
import numpy as np
import gym
from Utils import *

# OpenAI Gym Environment
class BipedalRobotEnv(gym.Env):
    def __init__(self, Parameters=[0.2, 1], GUI=True, f=240):

        super(BipedalRobotEnv, self).__init__()
        
        self.maxV = 600
        self.dT = 1 / f
        self.servoMax = 10 * (np.pi/3) / 0.16
        
        self.stance = Parameters[0]
        self.cycleTime = Parameters[1]
        self.cycle = self.stance/2

        self.param = np.array([self.cycle, self.stance, self.cycleTime])
        
        # Configuring PyBullet
        if GUI:
            p.connect(p.GUI)
            print('GUI Mode: ON')
        else:
            p.connect(p.DIRECT)
            print('GUI Mode: OFF')

        p.resetSimulation()
        p.setTimeStep(self.dT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.Plane = p.loadURDF('plane.urdf')
        p.setGravity(0, 0, -9.81)
        self.Robot = p.loadURDF(r"/URDF/urdf/URDF.urdf")
        p.changeDynamics(self.Plane, -1,
                         lateralFriction=0.6,
                         spinningFriction=0.1,
                         rollingFriction=0.1)#,
        #                 linearDamping=5,
        #                 angularDamping=0.2)
        

        # ------------------------------------------------

        # Configuring Action Space
        self.actionHigh = np.array([7 * np.pi / 36, 0,
                              7 * np.pi / 36, 0], np.float32)

        self.actionLow = np.array([-np.pi / 15, -11 * np.pi / 30,
                               -np.pi / 15, -11 * np.pi / 30], np.float32)

        self.nA = 7

        # ------------------------------------------------

        # Configuring General Parameters
        ih = p.getBasePositionAndOrientation(self.Robot)[0][-1]
        self.iPos = np.array([0, 0, ih], np.float32)
        self.iOri = np.array([0, 0, 0, 1], np.float32)
        p.resetBasePositionAndOrientation(self.Robot, self.iPos, self.iOri)

        self.Ticks = 0
        self.Position = self.iPos
        self.Orientation = p.getEulerFromQuaternion(self.iOri)
        self.BaseLinVelocity = np.zeros(3)
        self.BaseAngVelocity = np.zeros(3)
        
        self.Joints = np.zeros(self.nA)
        self.JointsVelocity = np.zeros(self.nA)
        self.JointsTorque = np.zeros(self.nA)
        
        self.S = np.concatenate((self.param, self.Orientation[:2], self.BaseAngVelocity[:2]))
        
    def reset(self):
        ori = np.zeros(3)
        av = np.zeros(3)
        JointValue = np.zeros(self.nA)
        
        lv = np.zeros(3)
        self.iOri = p.getQuaternionFromEuler(ori)
        p.resetBasePositionAndOrientation(self.Robot, self.iPos, self.iOri)
        
        p.resetBaseVelocity(self.Robot, lv, av)
        for ii in range(self.nA):
            p.resetJointState(self.Robot, ii, JointValue[ii])

        self.Ticks = 0
        self.Position = self.iPos
        self.Orientation = p.getEulerFromQuaternion(self.iOri)
        self.BaseLinVelocity = np.array(p.getBaseVelocity(self.Robot)[0])
        self.BaseAngVelocity = np.array(p.getBaseVelocity(self.Robot)[1])

        for ii in range(self.nA):
            self.Joints[ii] = p.getJointState(self.Robot, ii)[0]
            self.JointsVelocity[ii] = p.getJointState(self.Robot, ii)[1]
            self.JointsTorque[ii] = p.getJointState(self.Robot, ii)[-1]
            
        self.cycle = self.stance/2
        self.param = np.array([self.cycle, self.stance, self.cycleTime])
        
        self.S = np.concatenate((self.param, self.Orientation[:2], self.BaseAngVelocity[:2]))
        return self.S

    def step(self, a):
        aRL = a[:4] * (self.actionHigh * (a[:4]>0) - self.actionLow * (a[:4]<0))
        aRL = aRL.clip(self.Joints[:4] - self.servoMax * self.dT, self.Joints[:4] + self.servoMax * self.dT)
        aNMB = a[4:].clip(-self.maxV, self.maxV)
        
        p.setJointMotorControlArray(self.Robot, np.arange(4), p.POSITION_CONTROL, aRL)
        p.setJointMotorControl2(self.Robot, 4, p.POSITION_CONTROL, targetVelocity=aNMB[1], positionGain=0,
                                velocityGain=1)
        p.setJointMotorControl2(self.Robot, 5, p.POSITION_CONTROL, targetVelocity=aNMB[1], positionGain=0,
                                velocityGain=1)
        p.setJointMotorControl2(self.Robot, 6, p.POSITION_CONTROL, targetVelocity=aNMB[0], positionGain=0,
                                velocityGain=1)
        p.stepSimulation()
        
        self.Ticks += 1
        self.Position = np.array(p.getBasePositionAndOrientation(self.Robot)[0])
        self.Orientation = np.array(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.Robot)[1]))
        self.BaseLinVelocity = np.array(p.getBaseVelocity(self.Robot)[0])
        self.BaseAngVelocity = np.array(p.getBaseVelocity(self.Robot)[1])

        for ii in range(self.nA):
            self.Joints[ii] = p.getJointState(self.Robot, ii)[0]
            self.JointsVelocity[ii] = p.getJointState(self.Robot, ii)[1]
            self.JointsTorque[ii] = p.getJointState(self.Robot, ii)[-1]
                    
        self.cycle += self.dT / self.cycleTime
        self.param = np.array([self.cycle, self.stance, self.cycleTime])

        self.S = np.concatenate((self.param, self.Orientation[:2], self.BaseAngVelocity[:2]))        
        r = self.WalkingReward()
        
        if self.isDone():
            r -= 1e4
            d = True
            
        elif self.Ticks >= (480*self.cycleTime): d = True
        else: d = False

        return self.S, r, d
    
    
    def isDone(self):
        if abs(self.Orientation[0]) > np.pi/4: return True
        if abs(self.Orientation[1]) > np.pi/6: return True
        if abs(self.Orientation[2]) > np.pi/6: return True
        if self.Position[0] < -0.05 : return True
        if abs(self.Position[1]) > 0.1: return True
        else: return False
        
    def WalkingReward(self):
        r = np.zeros(5)
        r[0] = 20*self.BaseLinVelocity[0]
        r[1] = -np.abs(self.Position - self.iPos)[1]
        r[2] = -np.abs(self.Position - self.iPos)[2]
        r[3] = -3*np.abs(self.BaseAngVelocity).sum()
        r[4] = 30*self.Position[0]
        return r.sum()