#Module containing the PyTorch neural networks


#template by SuperDataScience www.superdatascience.com and
#Deep Learning Wizard on www.udemy.com/practical-deep-learning-with-pytorch


import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#1 Hidden Layer (11200-40-40-5)
class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, nb_action)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

#3 Hidden Layer (11200-500-500-300-100-5)
class Network3(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network3, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 300)
        self.fc4 = nn.Linear(300, 100)
        self.fc5 = nn.Linear(100, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        q_values = self.fc5(x)
        return q_values

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn():
    
    def __init__(self, input_size, nb_action, gamma, alpha, neurons=1):
        self.gamma = gamma
        self.reward_window = []
        self.neurons = neurons

        if self.neurons == 1:
            self.model = Network(input_size, nb_action)
        else:
            self.model = Network3(input_size, nb_action)

        self.memory = ReplayMemory(10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = alpha)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.count = 0

    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*50.0)
        try :
            action = probs.multinomial()
        except:
            #error only triggered few times by 3NN due to function design
            return random.choice([0,1,2,3,4])

        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal, action):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        if len(self.memory.memory) > 7:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(7)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, str(self.neurons) + 'nn.pth')
        self.count += 1
        if self.count == 1000:
            print self.model.fc1.weight.data
    
    def load(self):
        if os.path.isfile(str(self.neurons) + 'nn.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load(str(self.neurons) + 'nn.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
