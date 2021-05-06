"""
Tyler Gester 2021
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code adapted and improved from code examples provided by Udacity 
"""

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from NeuralNet import Net
from ReplayBuffer import Buffer
from ReplayBuffer import Buffer2

TAU = 3.5e-3              # for soft update of target parameters TAU 4e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    
class agent():
    """Initialize an agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            gamma(float): discount factor
            learningRate(float): learning rate of NN
            updateTargetRate(int): how often to update target NN
            tau(float): soft update factor (how much to update target NN)
            bufferBatch(int): learning batch size
            bufferSize(int): how many records to save
    """
    def __init__(self, state_size, action_size, seed=0, gamma = 0.99, learningRate = .00065, updateTargetRate = 4, tau = .0035, bufferBatch = 64, bufferSize = 10000):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.bufferBatch = bufferBatch
        self.minBatch = 25
        self.updateRate = updateTargetRate
        self.updateCount = 0
        self.tau = tau
        
        # Replay buffer
        #self.buffer = Buffer(action_size, bufferSize, BufferBatch, seed)
        self.buffer = Buffer2(bufferSize, seed)

        # Q-Networks
        self.trainingNN = Net(state_size, action_size, seed).to(device)
        self.targetNN = Net(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.trainingNN.parameters(), lr=learningRate)
    
    """takes new data in and learns if possible.
        
    Params
    ======
        state (array_like): current state
        action (int): action taken
        reward (int): reward from current action
        next_state(array_like): state after action is taken
        done(bool): task complete 
    """
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay buffer
        self.buffer.push(state, action, reward, next_state, done)
        
        # update every updateRate time steps.
        self.updateCount = (self.updateCount + 1)
        if self.updateCount == self.updateRate:
            self.updateCount = 0
            # If enough samples are available in buffer, get random subset and learn
            if len(self.buffer) > self.minBatch:
                i = self.bufferBatch if len(self.buffer) > self.bufferBatch else len(self.buffer)//2
                experiences = self.buffer.get_random(i)
                self.__learn(experiences, self.gamma)
                
    """Returns the Epsilon-greedy action according to the current learned Q policy.
        
    Params
    ======
        state (array_like): current state
        eps (float): epsilon, for epsilon-greedy action selection
    """
    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.trainingNN.eval()
        with torch.no_grad():
            action_values = self.trainingNN(state)
        self.trainingNN.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    #PRIVATE train the training network and update target if needed.
    def __learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.targetNN(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.trainingNN(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.__updateTarget(self.trainingNN, self.targetNN, self.tau)                    

    #PRIVATE update the target network using 
    # θ_target = τ*θ_training + (1 - τ)*θ_target
    def __updateTarget(self, training_NN, target_NN, tau):
        for target_param, training_param in zip(target_NN.parameters(), training_NN.parameters()):
            target_param.data.copy_(tau*training_param.data + (1.0-tau)*target_param.data)