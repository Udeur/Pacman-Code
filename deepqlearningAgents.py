
#!/usr/bin/env python
# coding: utf8
import os, sys
import numpy as np
import random
import util
import time


# Pacman game
#from pacman import Directions
#from pacman import *

from game import Directions, Actions
import game
from game import Agent
from util import manhattanDistance

#import featureExtractors
import util
import heapq
import numpy as np


# Replay memory
from collections import deque

# Neural nets
import tensorflow as tf
from DeepQ import *


params = {
    # Model backups
    'load_file': None,
    'save_file': None,
    'save_interval' : 100,

    # Training parameters
    'train_start': 500,    # Episodes before training starts
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


class deepqlearningAgents(game.Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        # Load parameters from user-given arguments
        self.params = params
        self.params['num_training'] = args['numTraining']

        self.qnet = DQN(self.params)

        # Q and cost
        self.Q_global = []
        self.cost_disp = 0     

        # Stats
        self.cnt = self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()


    def getMove(self, state):
        # Exploit / Explore
        if np.random.rand() > self.params['eps']:
            # Exploit action
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y3,


                feed_dict = {self.qnet.x: np.reshape(self.current_state, (1,10, 5)),
                             self.qnet.q_t: np.zeros(1),
                             self.qnet.actions: np.zeros((1, 5)),
                             self.qnet.terminals: np.zeros(1),
                             self.qnet.rewards: np.zeros(1)})[0]

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

            self.current_state = self.getStatesFeatureMatrix(state)

            # Process current experience reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 50.    # Eat ghost   (Yum! Yum!)
            elif reward > 0:
                self.last_reward = 10.    # Eat food    (Yum!)
            elif reward < -10:
                self.last_reward = -500.  # Get eaten   (Ouch!) -500
                self.won = False
            elif reward < 0:
                self.last_reward = -1.    # Punish time (Pff..)

            
            if(self.terminal and self.won):
                self.last_reward = 100.
            self.ep_rew += self.last_reward

            # Store last experience into memory 
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            #if(params['save_file']):
            #    if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
            #        self.qnet.save_ckpt('saves/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
            #        print('Model saved')

            # Train
            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = max(self.params['eps_final'], 1.00 - float(self.cnt)/ float(self.params['eps_step']))


    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)

        return state

    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Print stats
        #log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(self.params['height'])+'-x-'+str(self.params['num_training'])+'.log','a')
        #log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
        #                  (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        #log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        #sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
        #                 (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        #sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        #sys.stdout.flush()

    def train(self):
        # Train
        if (self.local_cnt > self.params['train_start']):
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = [] # States (s)
            batch_r = [] # Rewards (r)
            batch_a = [] # Actions (a)
            batch_n = [] # Next states (s')
            batch_t = [] # Terminal state (t)

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
        actions_onehot = np.zeros((self.params['batch_size'], 5))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def closestFood(self, pos, food, walls):
   
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None

    def distanceToCapsule(self, pos, capsules):
        for c in capsules:
            distance = [manhattanDistance(pos, c)]
            return min(distance)

    def getStatesFeatureMatrix(self, state):
        pacmanPostion = state.getPacmanPosition()
        capsules = state.getCapsules()
        distanceCapsule = self.distanceToCapsule(pacmanPostion, capsules)
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        ghostStates = state.getGhostStates()
        
        #Unterteile die Geister 
        scared_ghosts = filter(lambda g: g.scaredTimer > 0, ghostStates)
        normal_ghosts = filter(lambda g: g.scaredTimer == 0, ghostStates)
        
        actions = state.getLegalActions()
        x, y = state.getPacmanPosition()
        
        #Matrizen generieren
        matrix = np.zeros(dtype= float, shape = (10,5))
        foodMatrix = np.zeros(dtype= float, shape = 5)
        distanceToClosestFoodMatrix = np.zeros(dtype= float, shape = 5)
        scaredGhost1StepAwayMatrix = np.zeros(dtype= float, shape = 5)
        ghost1StepAwayMatrix = np.zeros(dtype= float, shape = 5)

        x1, y1 = Actions.directionToVector(actions[0])
        next_x1, next_y1 = int(x + x1), int(y + y1)
        if food[next_x1][next_y1]:
            foodMatrix[0] = 1.0
        distanceFood = self.closestFood((next_x1, next_y1), food, walls)
        if distanceFood is not None:
            distanceToClosestFoodMatrix[0] = float(distanceFood) / (walls.width * walls.height)
        if len(scared_ghosts) > 0:
                scaredGhost1StepAwayMatrix[0] = sum((next_x1, next_y1) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        if len(normal_ghosts) > 0:
             ghost1StepAwayMatrix[0] = sum((x, y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)           
          
        if len(actions) > 1:
            x2, y2 = Actions.directionToVector(actions[1])
            next_x2, next_y2 = int(x + x2), int(y + y2)
            if food[next_x2][next_y2]:
                foodMatrix[1] = 1.0
            distanceFood = self.closestFood((next_x2, next_y2), food, walls)
            if distanceFood is not None:
                distanceToClosestFoodMatrix[1] = float(distanceFood) / (walls.width * walls.height)
            if len(scared_ghosts) > 0:
                scaredGhost1StepAwayMatrix[1] = sum((next_x2, next_y2) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
            if len(normal_ghosts) > 0:
             ghost1StepAwayMatrix[1] = sum((x, y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)    

        if len(actions) > 2:
            x3, y3 = Actions.directionToVector(actions[2])
            next_x3, next_y3 = int(x + x3), int(y + y3) 
            if food[next_x3][next_y3]:
                foodMatrix[2] = 1.0
            distanceFood = self.closestFood((next_x3, next_y3), food, walls)
            if distanceFood is not None:
                distanceToClosestFoodMatrix[2] = float(distanceFood) / (walls.width * walls.height)
            if len(scared_ghosts) > 0:
                scaredGhost1StepAwayMatrix[2] = sum((next_x3, next_y3) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
            if len(normal_ghosts) > 0:
                ghost1StepAwayMatrix[2] = sum((x, y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)    

        if len(actions) > 3:
            x4, y4 = Actions.directionToVector(actions[3])
            next_x4, next_y4 = int(x + x4), int(y + y4)
            if food[next_x4][next_y4]:
                foodMatrix[3] = 1.0
            distanceFood = self.closestFood((next_x4, next_y4), food, walls)
            if distanceFood is not None:
                distanceToClosestFoodMatrix[3] = float(distanceFood) / (walls.width * walls.height)
            if len(scared_ghosts) > 0:
                scaredGhost1StepAwayMatrix[3] = sum((next_x4, next_y4) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
            if len(normal_ghosts) > 0:
                ghost1StepAwayMatrix[3] = sum((x, y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)    


        if len(actions) > 4:
            x5, y5 = Actions.directionToVector(actions[4])
            next_x5, next_y5 = int(x + x5), int(y + y5)
            if food[next_x5][next_y5]:
                foodMatrix[4] = 1.0
            distanceFood = self.closestFood((next_x5, next_y5), food, walls)
            if distanceFood is not None:
                distanceToClosestFoodMatrix[4] = float(distanceFood) / (walls.width * walls.height)
            if len(scared_ghosts) > 0:
                scaredGhost1StepAwayMatrix[4] = sum((next_x4, next_y4) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
            if len(normal_ghosts) > 0:
                 ghost1StepAwayMatrix[4] = sum((x, y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)    

        if len(normal_ghosts) > 0:
            
            distanceToClosestGhost = float (min([manhattanDistance(pacmanPostion, g.getPosition()) for g in normal_ghosts]) / (walls.width * walls.height))
            matrix[0] = distanceToClosestGhost
            
            if len(normal_ghosts) > 1:
                distanceToSecondClosestGhost = float (heapq.nsmallest(2, [manhattanDistance(pacmanPostion, g.getPosition()) for g in normal_ghosts])[-1] / (walls.width * walls.height))
                matrix[1] = distanceToSecondClosestGhost

            if distanceCapsule is not None:
                DistanceToCapsule = float (distanceCapsule) / ((walls.width * walls.height)^2)
                matrix[2] = DistanceToCapsule

            if len(state.getLegalActions()) < 4:
                tunnel = 1.0
                matrix[3] = tunnel

        if len(scared_ghosts) > 0:
            distanceToClosestScaredGhost =float (min([manhattanDistance(pacmanPostion, g.getPosition()) for g in scared_ghosts]) / (walls.width * walls.height))
            matrix[4] = distanceToClosestScaredGhost
        
        matrix[5] = foodMatrix
        matrix[6] = distanceToClosestFoodMatrix
        matrix[7] = scaredGhost1StepAwayMatrix
        matrix[8] = ghost1StepAwayMatrix
        matrix[9] = 1.0
        
        return matrix

    def registerInitialState(self, state): # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStatesFeatureMatrix(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)

        # Stop moving when not legal
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP
        return move
  
