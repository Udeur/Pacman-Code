#Module containing PyTorch Agent using Convolutional Neural Network
#Not integrated in TorchAgents because agent throws failure


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np
import torch
from torch import nn


from torchdqn import Dqn, CNNModel

import random,util,math


class TorchAgentCNN(ReinforcementAgent):

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.edge = 40
        self.index2Action = [Directions.NORTH, Directions.EAST , Directions.SOUTH, Directions.WEST, Directions.STOP]
        self.dqn = Dqn(np.zeros(7 * self.edge * self.edge).reshape(7, self.edge * self.edge), len(self.index2Action), self.discount, self.alpha)
        self.last_reward = 0
        self.scores = []
        #self.dqn.load()
        self.action = 0



    def getAction(self, state):

        self.lastState = state
        legalActions = self.getLegalActions(state)

        if util.flipCoin(self.epsilon):
            new_state = torch.from_numpy(self.getInputs(state))
            self.action = self.dqn.select_action(new_state)
        else:
            self.action = random.choice(self.index2Action)
            self.action = self.index2Action.index(self.action)

        if self.index2Action[self.action] in legalActions:
            return self.index2Action[self.action]
        else:
            return Directions.STOP

    def update(self, state, action, nextState, reward):
        last_signal = self.getInputs(state)
        self.dqn.update(reward, last_signal, self.action)



    def getInputs(self, state):
        size = (state.data.layout.width, state.data.layout.height)
        edge = self.edge

        #40x40 Matrix of walls
        lay_walls = state.getWalls()
        walls = self.updateLayout(np.ones(edge*edge).reshape(edge, edge), lay_walls,size[0], size[1], 1)

        # 40x40 Matrix of food
        lay_food = state.getFood()
        food = self.updateLayout(np.zeros(edge*edge).reshape(edge, edge),lay_food,size[0], size[1],1)

        # 40x40 Matrix of agents
        pacman,legalActions, ghosts, scarredGhosts = self.updateAgentLayout(state,edge,self.getLegalActions(state),1)

        # 40x40 Matrix of power pellets
        pos_capsules = state.getCapsules()
        capsules = np.zeros(edge*edge).reshape(edge, edge)

        for pos in pos_capsules:
            capsules[pos[0], pos[1]] = 1

        matrix = [walls, food, pacman, legalActions, ghosts, scarredGhosts, capsules]
        #matrix = np.concatenate((walls, food, pacman, legalActions,ghosts,scarredGhosts,capsules), axis=0)
        matrix = np.array(matrix)

        return matrix

    def updateLayout(self, layout, spec, width, height, value, comp=True):
        for x in range(width):
            for y in range(height):
                if spec[x][y] == comp:
                    layout[x, y] = value
        return layout

    def updateAgentLayout(self, state, edge,actions, value):
        pacman = np.zeros(edge * edge).reshape(edge, edge)
        ghosts = np.zeros(edge * edge).reshape(edge, edge)
        scarredGhosts = np.zeros(edge * edge).reshape(edge, edge)
        legalActions = np.zeros(edge * edge).reshape(edge, edge)

        for agent in state.data.agentStates:
            x,y = agent.getPosition()
            x = int(x)
            y = int(y)
            if agent.isPacman:
                pacman[x, y] = value
                for action in actions:
                    if action == 'North':
                        legalActions[x, y + 1] = value
                    elif action == 'South':
                        legalActions[x, y - 1] = value
                    elif action == 'West':
                        legalActions[x-1, y] = value
                    elif action == 'East':
                        legalActions[x + 1, y] = value
                    elif action == 'Stop':
                        legalActions[x, y] = value
            else:
                if agent.scaredTimer == 0:
                    ghosts[x, y] = value
                else:
                    scarredGhosts[x, y] = value
        return pacman,legalActions, ghosts, scarredGhosts


    def final(self, state):
        ReinforcementAgent.final(self, state)
        self.dqn.save()
