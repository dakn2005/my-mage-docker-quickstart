import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(84, 32) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(32, 16) 
        self.fc4 = nn.Linear(16, 4)  

    def forward(self, x):
        x = self.relu1(self.fc1(x)) #torch.relu(self.fc1(x))  
        x = self.relu2(self.fc2(x)) #torch.relu(self.fc2(x))  
        # x = torch.relu(self.fc3(x))  
        x = self.fc4(x)
        return x
    
    def predict(self, x):
        return self.forward(x)
    
    def convert_predict(self, x):
        x = torch.from_numpy(x).float()
        p = self.forward(x)
        return p.detach().numpy()