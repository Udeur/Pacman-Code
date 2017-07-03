
import numpy as np
import random
import util
import time
import sys
import GUI
import DQN_Network as DQN #KLASSE in der DQN erzeugt wurde

# Pacman game
from pacman import Directions
from game import Agent
import game

params = {
    # Model backups
    'load_file': None,
    'save_file': None,
    'save_interval' : 10000,

    # Training parameters
    'train_start': 5000,    # Episodes before training starts
    'batch_size': 32,       # Replay memory batch size
    'mem_size': 100000,     # Replay memory size

    'discount': 0.95,       # Discount rate (gamma value)
    'lr': .0002,            # Learning reate
    'rms_decay': 0.99,      # RMS Prop decay
    'rms_eps': 1e-6,        # RMS Prop epsilon

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,             # Epsilon start value
    'eps_final': 0.1,       # Epsilon end value
    'eps_step': 10000       # Epsilon steps between start and end (linear)
}

class DQLAgent(game.agent):
 
 def _init_(self, gsd): #übergebe GameStateData Objekt

     # Load parameters from user-given arguments
     self.params = params
     self.params['width'] = gsd.layout.width
     self.params['height'] = gsd.layout.height
     self.params['num_training'] = GUI.rounds.get()

 def getMove(self,state):

     # Exploit / Explore
     if np.random.rand() > self.params['eps']:
         # Exploit action
         DQN_class = DQN.init() #Erzeuge neue Klasse
         self.Q.pred = DQN_class.predict(state) #Schauen, dass DQN Klasse die unteren Parameter hat, ansonsten anpassen
         self.Q_global.append(max(self.Q_pred))
         a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

         if len(a_winner) > 1:
             move = self.get_direction(
                 a_winner[np.random.randint(0, len(a_winner))][0])
         else:
             move = self.get_direction(
                 a_winner[0][0])
     else:
         # Random:
         move = self.get_direction(np.random.randint(0, 4))

     # Save last_action
     self.last_action = self.get_value(move)

     return move

 def get_value(self, direction):
         if direction == Directions.NORTH:
             return 0.
         elif direction == Directions.EAST:
             return 1.
         elif direction == Directions.SOUTH:
             return 2.
         else:
             return 3.

 def get_direction(self, value):
         if value == 0.:
             return Directions.NORTH
         elif value == 1.:
             return Directions.EAST
         elif value == 2.:
             return Directions.SOUTH
         else:
             return Directions.WEST

 def observation_step(self, state):
     if self.last_action is not None:
         # Process current experience state
         self.last_state = np.copy(self.current_state)
         self.current_state = self.getStateMatrices(state)

         # Process current experience reward
         self.current_score = state.getScore()
         reward = self.current_score - self.last_score
         self.last_score = self.current_score

         if reward > 20:
             self.last_reward = 50.  # Eat ghost   (Yum! Yum!)
         elif reward > 0:
             self.last_reward = 10.  # Eat food    (Yum!)
         elif reward < -10:
             self.last_reward = -500.  # Get eaten   (Ouch!) -500
             self.won = False
         elif reward < 0:
             self.last_reward = -1.  # Punish time (Pff..)

         if (self.terminal and self.won):
             self.last_reward = 100.
         self.ep_rew += self.last_reward

         # Store last experience into memory
         experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
         self.replay_mem.append(experience)
         if len(self.replay_mem) > self.params['mem_size']:
             self.replay_mem.popleft()

         # Save model
         if (params['save_file']):
             if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                 self.qnet.save_ckpt(
                     'saves/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                 print('Model saved')

         #Train
         self.train()

    # Next
     self.local_cnt += 1
     self.frame += 1
     self.params['eps'] = max(self.params['eps_final'], 1.00 - float(self.cnt) / float(self.params['eps_step']))

 def train(self):
     # Train
     if (self.local_cnt > self.params['train_start']):
         batch = random.sample(self.replay_mem, self.params['batch_size'])
         batch_s = []  # States (s)
         batch_r = []  # Rewards (r)
         batch_a = []  # Actions (a)
         batch_n = []  # Next states (s')
         batch_t = []  # Terminal state (t)

         for i in batch:
             batch_s.append(i[0])
             batch_r.append(i[1])
             batch_a.append(i[2])
             batch_n.append(i[3])
             batch_t.append(i[4])
         batch_s = np.array(batch_s)
         batch_r = np.array(batch_r)
         batch_a = self.get_onehot(np.array(batch_a))
         batch_n = np.array(batch_n)
         batch_t = np.array(batch_t)

         self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)

 def get_onehot(self, actions):
     """ Create list of vectors with 1 values at index of action in list """
     actions_onehot = np.zeros((self.params['batch_size'], 4))
     for i in range(len(actions)):
         actions_onehot[i][int(actions[i])] = 1
     return actions_onehot

 def final(self, state):
     self.ep_rew += self.last_reward

     # Do observation
     self.terminal = True
     self.observation_step(state)

 def getAction(self,state):
     return


 def register_initial_state(self, state):
     return

 def mergeStateMatrices(self, stateMatrices):
     """ Merge state matrices to one state tensor """
     stateMatrices = np.swapaxes(stateMatrices, 0, 2)
     total = np.zeros((7, 7))
     for i in range(len(stateMatrices)):
         total += (i + 1) * stateMatrices[i] / 6

     return total


 def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width))
            matrix.dtype = int

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width))
            matrix.dtype = int

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width))
            matrix.dtype = int

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height 
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        observation = np.swapaxes(observation, 0, 2)

        return observation
  
