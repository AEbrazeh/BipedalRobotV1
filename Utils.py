from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
import os

def Decimal(x):
    return x - torch.floor(x)

### ! Actor Network ! ###

class ActorNet(nn.Module):
    def __init__(self, inputDim, outputDim, hiddenDim, name, dir, dt):
        super(ActorNet, self).__init__()
        self.name = name
        self.dir = dir
        self.checkpoint_file = os.path.join(self.dir, self.name+'.pth')
        
        self.dt = dt
        self.n = nn.Parameter(1 + torch.arange(hiddenDim), requires_grad=False)
        self.swingHip = nn.Parameter(torch.randn(hiddenDim), requires_grad=True)
        self.stanceHip = nn.Parameter(torch.randn(hiddenDim), requires_grad=True)
        self.swingKnee = nn.Parameter(torch.randn(hiddenDim), requires_grad=True)
        
        self.maxHip = nn.Parameter(torch.randn(hiddenDim), requires_grad=True)
        self.minHip = nn.Parameter(torch.randn(hiddenDim), requires_grad=True)
        self.minKnee = nn.Parameter(torch.randn(hiddenDim), requires_grad=True)
        
             
    def forward(self, state):
        cR = Decimal(state[:, 0]).unsqueeze(-1)
        cL = Decimal(state[:, 0] + 0.5).unsqueeze(-1)

        c = torch.cat((cR, cL), dim=1)
        sT = state[:, 1]#.clamp(0.51, 0.99).unsqueeze(-1)
        cT = state[:, 2].clamp(0.50, None).unsqueeze(-1)
        
        e = 2 * (1 - sT) * (18*cT + 1) / (20 * cT)
                
        hipMax = (self.maxHip * torch.sin((2*self.n-1)*e*np.pi/2)**2 / (2*self.n-1)).sum(dim=-1, keepdim=True)/(self.maxHip / (2*self.n-1)).sum()
        hipMin = -(self.minHip * torch.sin((2*self.n-1)*e*np.pi/2)**2 / (2*self.n-1)).sum(dim=-1, keepdim=True)/(self.minHip / (2*self.n-1)).sum()
        kneeMin = -(self.minKnee * torch.sin((2*self.n-1)*e*np.pi/2)**2 / (2*self.n-1)).sum(dim=-1, keepdim=True)/(self.minKnee / (2*self.n-1)).sum()
        
        tSwing = ((c - sT)/(1-sT)).unsqueeze(-1)
        tStance = ((sT-c)/sT).unsqueeze(-1)

        hip = (c>=sT) * (self.swingHip * torch.sin((2*self.n-1)*tSwing*np.pi/2)**2 / (2*self.n-1)).sum(dim=-1)/(self.swingHip / (2*self.n-1)).sum() + (c<sT) * (self.stanceHip * torch.sin((2*self.n-1)*tStance*np.pi/2)**2 / (2*self.n-1)).sum(dim=-1)/(self.stanceHip/(2*self.n-1)).sum()
        knee = (c>=sT) * (self.swingKnee * torch.sin(self.n*tSwing*np.pi)**2 / self.n).sum(dim=-1)/(self.swingKnee / self.n).sum()
        
        hip = (((hipMax - hipMin) * hip + hipMin)).unsqueeze(-1)
        knee = (kneeMin * knee).unsqueeze(-1)
        
        output = torch.cat((hip, knee), dim=2).reshape(-1, 4)
        return output
    
    def save(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load(self, dir):
        checkpoint_file = os.path.join(dir, self.name+'.pth')
        self.load_state_dict(torch.load(checkpoint_file))
        
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

### ! Critic Network ! ###

class CriticNet(nn.Module):
    def __init__(self, inputDim, hiddenDim, name, dir):
        super(CriticNet, self).__init__()
        self.name = name
        self.dir = dir
        self.checkpoint_file = os.path.join(self.dir, self.name+'.pth')
        
        self.Network = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.LeakyReLU(0.1),
            nn.Linear(hiddenDim, hiddenDim),
            nn.LeakyReLU(0.1),
            nn.Linear(hiddenDim, hiddenDim),
            nn.LeakyReLU(0.1),
            nn.Linear(hiddenDim, hiddenDim),
            nn.LeakyReLU(0.1),
            nn.Linear(hiddenDim, hiddenDim),
            nn.LeakyReLU(0.1),
            nn.Linear(hiddenDim, 1)
        )
        
    def forward(self, x):
        Q = self.Network(x)
        return Q
        
    def save(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load(self, dir):
        checkpoint_file = os.path.join(dir, self.name+'.pth')
        self.load_state_dict(torch.load(checkpoint_file))
        
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


### ! Numerical Integration Functions ! ###

def I1(x, dx):
    ans = np.zeros_like(x)
    for ii in range(len(x)):
        ans[ii] = np.trapz(x[:ii], dx=dx, axis=0)
    return ans

def I2(x, dx):
    return I1(I1(x, dx), dx)
    
### ! Second Order iPID Controller ! ###

class iLinearController:
    def __init__(self, dt, n=6, m=2, alpha=0.8, beta=[1, 1, 1], gamma = 1000):
        self.X = np.zeros((n, m))
        self.Xd = np.zeros((1, m))
        self.Xi = np.zeros((1, m))
        self.A = np.zeros((n, m))
        self.F = np.zeros((1, m))
        self.dt = dt
        
        self.T = (np.arange(0, n)) * dt
        self.T = self.T.reshape(-1, 1).repeat(m, axis=1)
        
        self.alpha = alpha            
        self.beta = beta
        self.gamma = gamma

    def resetHistory(self):
        self.X *= 0
        self.Xd *= 0
        self.Xi *= 0
        self.A *= 0
        self.F *= 0
        
    def updateHistory(self, s):        
        self.X = np.roll(self.X, -1, 0)
        self.X[-1, 0] = s[:, 0]
        self.X[-1, 1] = s[:, 1]
        self.Xd = s[:,2:]
        self.Xi = self.Xi + self.dt * (self.X[-1] + self.X[-2])/2

    def takeAction(self):
        self.F = (6 * (I1(self.T * self.X, self.dt)[-1] 
                       - self.alpha * I2(self.T * self.A, self.dt)[-1]
                       - I2(self.Xd, self.dt)[-1]) / (self.T[-1])**3)
        
        errorP = self.X[-1]
        errorI = self.Xi
        errorD = self.Xd
        
        action = -(self.F + self.gamma * (errorP * self.beta[0] + errorI * self.beta[1] + errorD * self.beta[2])) / self.alpha
        
        self.A = np.roll(self.A, -1, 0)
        self.A[-1] = action
        
        return action