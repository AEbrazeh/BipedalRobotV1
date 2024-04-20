import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, nS, nA, dir):
        self.MemorySize = max_size
        self.MemoryHead = 0
        self.StateDim = nS
        self.ActionDim = nA
        
        self.StateMemory = np.zeros((self.MemorySize, nS))
        self.ActionMemory = np.zeros((self.MemorySize, nA))
        self.RewardMemory = np.zeros(self.MemorySize)
        self.DoneMemory = np.zeros(self.MemorySize)
        self.StatePrimeMemory = np.zeros((self.MemorySize, nS))
        
        self.dir = dir

    def Store(self, s, a, r, sp, d):
        i = self.MemoryHead % self.MemorySize
        self.StateMemory[i] = s
        self.ActionMemory[i] = a
        self.RewardMemory[i] = r
        self.StatePrimeMemory[i] = sp
        self.DoneMemory[i] = d
        self.MemoryHead += 1

    def Sample(self, batchSize):
        length = min(self.MemorySize, self.MemoryHead)
        batch = np.random.choice(length, batchSize)
        s = self.StateMemory[batch].astype(float)
        a = self.ActionMemory[batch].astype(float)
        r = self.RewardMemory[batch].astype(float)
        sp = self.StatePrimeMemory[batch].astype(float)
        d = self.DoneMemory[batch]
        
        return s, a, r, sp, d