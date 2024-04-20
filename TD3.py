import torch
import torch.nn as nn
import numpy as np
from Utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DoubleQNetwork(nn.Module):
    def __init__(self, stateDim, actionDim, hidden_dim=128, dir='TD3-Model', name=''):
        super(DoubleQNetwork, self).__init__()
        self.Net1 = CriticNet(stateDim+actionDim, hidden_dim, name+'QNetwork1', dir)
        self.Net2 = CriticNet(stateDim+actionDim, hidden_dim, name+'QNetwork2', dir)
    
    def forward(self, state, action):
        input = torch.cat((state, action), dim=1)
        Q1 = self.Net1(input)
        Q2 = self.Net2(input)
        
        return Q1, Q2
        
    def save(self):
        self.Net1.save()
        self.Net2.save()
        
    def load(self, dir):
        self.Net1.load(dir)
        self.Net2.load(dir)
        
    def freeze(self):
        self.Net1.freeze()
        self.Net2.freeze()
        
        
class MuNetwork(nn.Module):
    def __init__(self, stateDim, actionDim, hidden_dim=128, dir='TD3-Model', name='', dt=1/240):
        super(MuNetwork, self).__init__()
        self.Net = ActorNet(stateDim, actionDim, hidden_dim, name+'MuNetwork', dir, dt)
    
    def forward(self, state, noise=0):
        action = self.Net(state)
        with torch.no_grad():
            if noise > 0.0:
                addedNoise = (torch.rand_like(action)*2 - 1) * noise
                action = action + addedNoise
        action = torch.clamp(action, -1, 1)
        return action
    
    def save(self): self.Net.save()
    def load(self, dir): self.Net.load(dir)
    def freeze(self): self.Net.freeze()
    
class TD3_Agent:
    def __init__(self, lr, stateDim, actionDim, dt=1/240, gamma=0.99, tau=1e-2, expFactor=0.25, expDecay = 0.999, actorHiddenDim=128, criticHiddenDim=512, dir='TD3-Model'):
        
        self.gamma = gamma
        self.expFactor = expFactor
        self.tau = tau
        self.expDecay = expDecay
        self.Loss = nn.MSELoss()
        
        # Defining Main and Target Twin Q Networks
        self.Main_QNet = DoubleQNetwork(stateDim, actionDim, criticHiddenDim, name='Main_', dir=dir).to(device)
        self.Target_QNet = DoubleQNetwork(stateDim, actionDim, criticHiddenDim, name='Target_', dir=dir).to(device)
        
        # Overwriting the parameters of Main Network on the Target Net and freezing the Target Net
        self.Target_QNet.load_state_dict(self.Main_QNet.state_dict())
        self.Target_QNet.freeze()
        
        # Defining Main and Target Mu Networks
        self.Main_MuNet = MuNetwork(stateDim, actionDim, actorHiddenDim, name='Main_', dir=dir, dt=dt).to(device)
        self.Target_MuNet = MuNetwork(stateDim, actionDim, actorHiddenDim, name='Target_', dir=dir, dt=dt).to(device)
        
        # Overwriting the parameters of Main Network on the Target Net and freezing the Target Net
        self.Target_MuNet.load_state_dict(self.Main_MuNet.state_dict())
        self.Target_MuNet.freeze()
        
        # Defining optimizers
        self.Optimizer_Q = torch.optim.Adam(self.Main_QNet.parameters(), lr)
        self.Optimizer_Mu = torch.optim.Adam(self.Main_MuNet.parameters(), lr)
        
    def updateQNet(self, state, action, reward, state_, done):
        TargetMu = self.Target_MuNet.forward(state_, self.expFactor)
        TQ1, TQ2 = self.Target_QNet(state_, TargetMu)
        TargetQ = reward + self.gamma * torch.min(TQ1, TQ2) * (1 - done)
        
        MQ1, MQ2 = self.Main_QNet(state, action)
        self.Optimizer_Q.zero_grad(set_to_none=True)
        LossQ1 = self.Loss(MQ1, TargetQ)
        LossQ2 = self.Loss(MQ2, TargetQ)
        Loss = LossQ1 + LossQ2
        
        Loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.Main_QNet.parameters(), 1e-2, 1)
        self.Optimizer_Q.step()
    
    def updateMuNet(self, state):
        self.Optimizer_Mu.zero_grad(set_to_none=True)
        Mu = self.Main_MuNet(state, 0)
        Q1, _ = self.Main_QNet(state, Mu)
        Loss = -Q1.mean()
        Loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.Main_MuNet.parameters(), 1, 1)
        self.Optimizer_Mu.step()
        
    @torch.no_grad()
    def updateTarget(self):
        for target, main in zip(self.Target_QNet.parameters(), self.Main_QNet.parameters()):
            target.data.copy_(self.tau * main.data + (1 - self.tau) * target.data)
            
        for target, main in zip(self.Target_MuNet.parameters(), self.Main_MuNet.parameters()):
            target.data.copy_(self.tau * main.data + (1 - self.tau) * target.data)
    
    def takeAction(self, state, exp = False):
        state = torch.cuda.FloatTensor(state)

        if exp:
            a = self.Main_MuNet(state, self.expFactor)
        else:
            a = self.Main_MuNet(state, 0)
            
        a = a.detach().cpu().numpy()
        return a
    
    def updateExpRate(self):
        eF = self.expFactor * self.expDecay
        self.expFactor = max(1e-6, eF)
    
    def updateModel(self, state, action, reward, state_, done, update=False):
        state = torch.cuda.FloatTensor(state)
        action = torch.cuda.FloatTensor(action)
        reward = torch.cuda.FloatTensor(reward).unsqueeze(-1)
        state_ = torch.cuda.FloatTensor(state_)
        done = torch.cuda.FloatTensor(done).unsqueeze(-1)
        
        self.updateQNet(state, action, reward, state_, done)
        if update:
            self.updateMuNet(state)
            self.updateTarget()
        
    def trainMode(self):
        self.Main_MuNet.train()
        self.Main_QNet.train()
        
    def testMode(self):
        self.Main_MuNet.eval()
        self.Main_QNet.eval()
        
    def save(self):
        self.Main_MuNet.save()
        self.Main_QNet.save()
        
    def load(self, dir):
        self.Main_MuNet.load(dir)
        self.Main_QNet.load(dir)
        self.Target_MuNet.load_state_dict(self.Main_MuNet.state_dict())
        self.Target_QNet.load_state_dict(self.Main_QNet.state_dict())
