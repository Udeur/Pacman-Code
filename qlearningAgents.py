
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from importExport import TxtFile

import random,util,math

class QLearningAgent(ReinforcementAgent):

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.values = util.Counter()


    def getQValue(self, state, action):
        return self.values[(state, action)]


    def computeValueFromQValues(self, state):
        maxQ = float('-inf')
        for action in self.getLegalActions(state):
            maxQ = max(maxQ, self.getQValue(state, action))
        return maxQ if maxQ != float('-inf') else 0.0


    def computeActionFromQValues(self, state):
        if len(self.getLegalActions(state)) == 0:
            return None

        bestQ = self.computeValueFromQValues(state)
        bestActions = []
        for action in self.getLegalActions(state):
            if bestQ == self.getQValue(state, action):
                bestActions.append(action)

        return random.choice(bestActions)


    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        action = None

        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        oldValue = self.values[(state, action)]
        newValue = reward + (self.discount * self.computeValueFromQValues(nextState))

        self.values[(state, action)] = (1 - self.alpha) * oldValue + self.alpha * newValue


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):

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
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.weights = {'DistanceToClostestGhost': 126.60456396130368, 'ClosestFood': -152.99544981031994, 'Bias': 405.421355420597, 'DistanceToClosestScaredGhost': 12.555335840061781, 'DistanceToClostest2Ghost': -59.59943731917117, 'ScaredGhost1StepAway': 1578.2182700200874, 'Food': 234.5011359525439, 'DistanceToCapsule': -140.53879235841436, 'Tunnel': -33.683396107762476, 'Ghost1StepAway': -3780.616407969431}

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(state, action)

        sum = 0
        for feature, value in features.iteritems():
            #print "Weights[feature", self.weights[feature]
            #print "value", value
            sum += self.weights[feature] * value
        return sum

    def update(self, state, action, nextState, reward):
        newValue = reward + self.discount * self.computeValueFromQValues(nextState)
        oldValue = self.getQValue(state, action)
        difference = newValue - oldValue

        features = self.featExtractor.getFeatures(state, action)
        for feature, value in features.iteritems():
          self.weights[feature] += self.alpha * difference * features[feature]
    
    def final(self, state):
        PacmanQAgent.final(self, state)

