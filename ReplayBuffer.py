import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Transition = namedtuple("Event",
                        ("state", "action", "reward", "next_state", "done"))

class Buffer2:
    """Initialize a ReplayBuffer object.

    Params
    ======
        capacity (int): maximum size of buffer
        seed (int): random seed
    """
    def __init__(self, capacity, seed):
        self.buffer = deque([],maxlen=capacity)
        self.seed = random.seed(seed)

    
    """Add a new event to buffer."""
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    """Randomly get events"""
    def get_random(self, size):
        experiences = random.sample(self.buffer, size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)

"""Fixed-size replay buffer """
class Buffer:
    """Initialize a ReplayBuffer object.

    Params
    ======
        action_size (int): dimension of each action
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        seed (int): random seed
    """
    def __init__(self, action_size, buffer_size, batch_size, seed = 0):
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    """Add a new event to buffer."""
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
    
    """Randomly get events"""
    def get_random(self, size):
        experiences = random.sample(self.buffer, size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)