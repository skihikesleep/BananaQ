import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Net(nn.Module):
    """Initialize Nurnal Net.
        
        Params
        ======
            input_size (int): dimension of each state
            output_size (int): dimension of each action
            seed (int): random seed
            fc1_units(float): size of first layer
            fc2_units(float): size of second layer
            fc3_units(int): size of third layer
    """
    def __init__(self, input_size, output_size, seed, fc1_units=30, fc2_units=15, fc3_units=10):
        super(Net, self).__init__()
        #use manual seed for repeatability 
        self.seed = torch.manual_seed(seed)
        #TODO look into something more complex for better learning mmaybe: Conv2d ect
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, output_size)

    def forward(self, dataIn):
        #basic NN
        x = F.relu(self.fc1(dataIn))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)