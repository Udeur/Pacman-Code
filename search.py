# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    start = problem.getStartState()
    explored = set()
    #LIFO-Prinzip
    helper = util.Stack()
    helper.push((start, []))
    while not helper.isEmpty():
    	helpState, helpMove = helper.pop()
    	if helpState in explored:
    		continue;

    	explored.add(helpState)

    	if problem.isGoalState(helpState):
    		return helpMove

    	for state, direction, cost in problem.getSuccessors(helpState):
    		helper.push((state, helpMove+[direction],))


def breadthFirstSearch(problem):

    start = problem.getStartState()
    explored = set()
    #FIFO-Prinzip
    helper = util.Queue()
    helper.push((start, []))
    while not helper.isEmpty():
    	helpState, helpMove = helper.pop()
    	if helpState in explored:
    		continue;

    	explored.add(helpState)

    	if problem.isGoalState(helpState):
    	    return helpMove

    	for state, direction, cost in problem.getSuccessors(helpState):
    		helper.push((state, helpMove + [direction]))

    util.raiseNotDefined()


def uniformCostSearch(problem):
    "Search the node of least total cost first. "

    start = problem.getStartState()
    explored = set()
    helper = util.PriorityQueue()
    # push(State, Direction/Move, Cost) aus problem.getSuccessors(...)
    helper.push((start, []), 0)
    while not helper.isEmpty():
    	helpState, helpMove = helper.pop()
    	if helpState in explored:
    		continue;

    	explored.add(helpState)

    	if problem.isGoalState(helpState):
    		return helpMove

    	for state, direction, cost in problem.getSuccessors(helpState):
    		helper.push((state, helpMove + [direction]), cost)
    return []
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    start = problem.getStartState()
    explored = set()
    helper = util.PriorityQueue()
    # push(State, Direction/Move, Cost) aus problem.getSuccessors(...)
    helper.push((start, []), 0)
    while not helper.isEmpty():
    	helpState, helpMove = helper.pop()
    	if helpState in explored:
    		continue;

    	explored.add(helpState)

    	if problem.isGoalState(helpState):
    		return helpMove

    	for state, direction, cost in problem.getSuccessors(helpState):
    		variable = heuristic(state, problem)
    		helper.push((state, helpMove + [direction]), cost + variable)
    return []

    util.raiseNotDefined()
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
