# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from importExport import TxtFile

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.values = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        maxQ = float('-inf')
        for action in self.getLegalActions(state):
            maxQ = max(maxQ, self.getQValue(state, action))
        return maxQ if maxQ != float('-inf') else 0.0


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        if len(self.getLegalActions(state)) == 0:
            return None

        bestQ = self.computeValueFromQValues(state)
        bestActions = []
        for action in self.getLegalActions(state):
            if bestQ == self.getQValue(state, action):
                bestActions.append(action)

        return random.choice(bestActions)


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        oldValue = self.values[(state, action)]
        newValue = reward + (self.discount * self.computeValueFromQValues(nextState))

        self.values[(state, action)] = (1 - self.alpha) * oldValue + self.alpha * newValue


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)

        sum = 0
        for feature, value in features.iteritems():
            #print "Weights[feature", self.weights[feature]
            #print "value", value
            sum += self.weights[feature] * value
        return sum

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        newValue = reward + self.discount * self.computeValueFromQValues(nextState)
        oldValue = self.getQValue(state, action)
        difference = newValue - oldValue

        features = self.featExtractor.getFeatures(state, action)
        for feature, value in features.iteritems():
          self.weights[feature] += self.alpha * difference * features[feature]
    
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"

            "Exportieren"
           # self.file.emptyFile()
            for key in self.weights.keys():
                self.file.writeToFile(str(self.weights[key]) + '|')
                state = key[0]
                self.file.writeToFile(str(state.data.food) + '|')
                self.file.writeToFile(str(state.data.capsules) + '|')
                agentStates = state.data.agentStates
                for agentState in agentStates:
                    self.file.writeToFile(str(agentState.start) + '!')
                    self.file.writeToFile(str(agentState.configuration) + '!')
                    self.file.writeToFile(str(agentState.isPacman) + '!')
                    self.file.writeToFile(str(agentState.scaredTimer) + '!')
                    self.file.writeToFile(str(agentState.numCarrying) + '!')
                    self.file.writeToFile(str(agentState.numReturned) + '!')
                    self.file.writeToFile('&')
                self.file.writeToFile(str(state.data.layout) + '|')
                self.file.writeToFile(str(state.data._eaten) + '|')
                self.file.writeToFile(str(state.data.score) + '|')
                self.file.writeToFile(str(key[1]) + '|')
                self.file.writeToFile("?")
            pass