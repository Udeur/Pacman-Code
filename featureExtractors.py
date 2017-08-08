# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
from util import manhattanDistance
from pacman import PacmanRules
import util
import heapq
import numpy as np

class FeatureExtractor:
    def getFeatures(self, state, action):
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
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

#added
def distanceToCapsule(pos, capsules):
    for c in capsules:
        distance = [manhattanDistance(pos, c)]
        return min(distance)
#added
def distancToGhost(pos, ghosts):
    for g in ghosts:
        distance = [manhattanDistance(pos, g)]
        return min(distance)

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

#added
class BestExtractor(FeatureExtractor):
    
    def getFeatures(self, state, action):

        #Grundinformationen
        pacmanPostion = state.getPacmanPosition()
        capsules = state.getCapsules()
        distanceCapsule = distanceToCapsule(pacmanPostion, capsules)
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        ghostStates = state.getGhostStates()
      
        #Unterteile die Geister 
        scared_ghosts = filter(lambda g: g.scaredTimer > 0, ghostStates)
        normal_ghosts = filter(lambda g: g.scaredTimer == 0, ghostStates)

        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        distanceFood = closestFood((next_x, next_y), food, walls)

        
        #Initialisiere Counter
        features = util.Counter()
        features["Bias"] = 1.0

        if len(scared_ghosts) > 0:
           
            distanceToClostestScaredGhost = min([manhattanDistance(pacmanPostion, g.getPosition()) for g in scared_ghosts])
            features["DistanceToClosestScaredGhost"] = float (distanceToClostestScaredGhost) / (walls.width * walls.height)
            
            features["ScaredGhost1StepAway"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
            if food[next_x][next_y]:
                features["Food"] = 1.0
            if distanceFood is not None:
                features["ClosestFood"] = float(distanceFood) / (walls.width * walls.height)
    
        if len(normal_ghosts) > 0:
            
            distanceToClosestGhost = min([manhattanDistance(pacmanPostion, g.getPosition()) for g in normal_ghosts])
            
            features["Ghost1StepAway"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
            features["DistanceToClostestGhost"] = float (distanceToClosestGhost) / (walls.width * walls.height)
          
            if len(normal_ghosts) > 1:
                distanceToSecondClosestGhost = heapq.nsmallest(2, [manhattanDistance(pacmanPostion, g.getPosition()) for g in normal_ghosts])[-1]
                features["DistanceToClostest2Ghost"] = float (distanceToSecondClosestGhost) / (walls.width * walls.height)
                #features["DistanceCombination"] = float ((distanceToClosestGhost*distanceToSecondClosestGhost) / (walls.width * walls.height)**2)
                #print float (distanceToClosestGhost*distanceToSecondClosestGhost)
                #print features["DistanceCombination"]

            if distanceCapsule is not None:
                features["DistanceToCapsule"] = float (distanceCapsule) / ((walls.width * walls.height)^2)

            if not features["Ghost1StepAway"] and food[next_x][next_y]:
                features["Food"] = 1.0
                
            if distanceFood is not None:
                features["ClosestFood"] = float(distanceFood) / (walls.width * walls.height)

            if len(PacmanRules.getLegalActions(state)) < 4:
                features["Tunnel"] = 1.0

        features.divideAll(10.0)
        #print features
        return features

#added
class BetterExtractor(FeatureExtractor):
    
   def getFeatures(self, state, action):
        pacmanPostion = state.getPacmanPosition()
        capsules = state.getCapsules()
        distanceCapsule = distanceToCapsule(pacmanPostion, capsules)
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        ghostStates = state.getGhostStates()
        distanceToGhost = distancToGhost(pacmanPostion, ghosts)
        #Location after Pacman takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        distanceFood = closestFood((next_x, next_y), food, walls)
       
        features = util.Counter()
       
        features["Bias"] = 1.0

        for ghost in ghostStates:
            if ghost.scaredTimer > 0:
                features["DistanceToClostestGhost"] = float (distanceToGhost) / (walls.width * walls.height)
                features["ScaredGhost1StepAway"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
                if food[next_x][next_y]:
                    features["Food"] = 1.0
                if distanceFood is not None:
                    features["ClosestFood"] = float(distanceFood) / (walls.width * walls.height)
            else:
                features["Ghost1StepAway"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

                if distanceCapsule is not None:
                    features["DistanceToCapsule"] = 1.0 / (float (distanceCapsule))
                
                if not features["Ghost1StepAway"] and food[next_x][next_y]:
                    features["Food"] = 1.0
                
                if distanceFood is not None:
                    features["ClosestFood"] = float(distanceFood) / (walls.width * walls.height)

                if len(PacmanRules.getLegalActions(state)) < 4:
                    features["Tunnel"] =  1.0

        features.divideAll(10.0)
        return features





