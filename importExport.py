#Module containing text file class
# class can be used to write any information in a textfile
# class can be used to import exact information of Berkley state instance
# Is used for evalution. Not needed for execution

import os, util
from game import GameStateData
from pacman import GameState
from game import AgentState


class TxtFile:
    FILENAME = "exportedFeatures.txt"

    def __init__(self):
        self.realpath = os.path.realpath(__file__)
        self.weights = util.Counter()
        self.dirpath = os.path.dirname(self.realpath)

    def emptyFile(self):
        open(self.dirpath + "/" + self.FILENAME, 'w').close()

    def writeToFile(self, txt):
        file = open(self.dirpath + "/" + self.FILENAME, "a")
        file.write(txt)

        file.close()

    def readFromFile(self):
        file = open(self.dirpath + "/" + self.FILENAME, "r")
        s = file.read()
        file.close()
        return s

    def getWeights(self):
        s = self.readFromFile()
        state = GameState()
        statedata = GameStateData()
        i = j = 0
        k = 0
        while i < len(s) > j:
            curLetter = s[j:j+1]
            if s[i:i+1] == "?":
                i +=1
            if curLetter == "|":
                if k == 0:
                    weight = s[i:j-1]
                if k == 1:
                    statedata.food = s[i:j-1]
                elif k == 2:
                    statedata.capsules = s[i:j-1]
                elif k == 3:
                    l = 0
                    while curLetter != '&' and j <= len(s):
                        if curLetter == "!":
                            if l == 0:
                                start = s[i:j-1]
                            elif l == 1:
                                configurations = s[i:j-1]
                            elif l == 2:
                                isPacman = s[i:j-1]
                                agentstate = AgentState(start,isPacman)
                                agentstate.configuration = configurations
                            elif l == 3:
                                agentstate.scaredTimer = s[i:j-1]
                            elif l == 4:
                                agentstate.numCarrying = s[i:j-1]
                            elif l == 5:
                                agentstate.numReturned = s[i:j-1]
                            elif l == 6:
                                statedata.agentStates = agentstate
                                l = -1
                            l += 1
                            j = j + 1
                            i = j
                        j += 1
                        curLetter = s[j:j + 1]
                elif k == 4:
                    statedata.layout = s[i:j-1]
                elif k == 5:
                    statedata._eaten = s[i:j-1]
                elif k == 6:
                    statedata.score = s[i:j-1]
                elif k == 7:
                    action = s[i:j - 1]
                elif k == 8:
                    state.data = statedata
                    self.weights[(state, action)] = weight
                    k = -1
                k += 1
                j = j + 1
                i = j
            j += 1
        return self.weights


       # self.food = prevState.food.shallowCopy()
        #self.capsules = prevState.capsules[:]
       # self.agentStates = self.copyAgentStates(prevState.agentStates)
       # self.layout = prevState.layout
       # self._eaten = prevState._eaten
       # self.score = prevState.score

        #    self.start = startConfiguration
        #    self.configuration = startConfiguration
        #    self.isPacman = isPacman
        #    self.scaredTimer = 0
        #    self.numCarrying = 0
        #    self.numReturned = 0
