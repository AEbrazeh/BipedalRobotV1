import numpy as np
import torch
from Utils import *
from TD3 import *
from ReplayBuffer import *

class WalkerAgent:
    def __init__(self, stateDim, actionDim, shortMemorySize=6, longMemorySize=1000000, lr=1e-4, alpha=0.8, beta=1, f=240, name=''):
        self.actionDim = actionDim
        self.actionDimRLC = actionDim - 2
        self.actionDimNMBC = 2
        
        self.stateDim = stateDim
        self.stateDimRLC = stateDim - 4
        self.stateDimNMBC = 4
        
        self.longMemorySize = longMemorySize
        self.isTrain = True
        
        self.lr = lr
        self.f = f
        
        self.rlController = TD3_Agent(lr, self.stateDimRLC, self.actionDimRLC, 1/f, actorHiddenDim=25, criticHiddenDim=256, expFactor=0.01, dir=name)
        self.nmbController = iLinearController(1/f, shortMemorySize, self.actionDimNMBC, alpha, beta)
                
        self.repBuffer = ReplayBuffer(longMemorySize, self.stateDimRLC, self.actionDimRLC, name)
    
    def updateMemory(self, state, action, reward, state_, done):
        self.repBuffer.Store(state[:, :self.stateDimRLC], action[:self.actionDimRLC], reward, state_[:self.stateDimRLC], done)
        self.nmbController.updateHistory(state[:, self.stateDimRLC:])
    
    def takeAction(self, state, RW=True):
        actionRLC = self.rlController.takeAction(state[:, :self.stateDimRLC], self.isTrain)
        actionNMBC = self.nmbController.takeAction()
        
        action = np.concatenate((actionRLC, actionNMBC * RW), axis=1).flatten()
        return action
    
    def resetMemory(self):
        self.nmbController.resetHistory()
        
    def trainMode(self):
        self.isTrain = True
        self.rlController.trainMode()
        
    def testMode(self):
        self.isTrain = False
        self.rlController.testMode()
        
    def save(self):
        self.rlController.save()
    
    def load(self, name):
        self.rlController.load(name)
    