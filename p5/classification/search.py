# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from game import Directions, Actions
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    start = problem.getStartState()
    
    from util import Stack
    fringe = Stack()
    fringe.push([(start, 'Start', 1)])
    
    result = None
    
    while result == None:
      currentPath = fringe.pop()
      currentState = currentPath[-1]
      
      if problem.isGoalState(currentState[0]):
        result = currentPath
        continue
      
      succ = problem.getSuccessors(currentState[0])
      
      for state in succ:
        if state[0] not in map(lambda x: x[0], currentPath):
          new_list = currentPath[:]
          new_list.append(state)
          fringe.push(new_list)
      
    steps = []
    for state in result[1:]:
      steps.append(state[1])

    return steps  
      

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    
    from util import Queue
    fringe = Queue()
    fringe.push([(start, 'Start', 1)])
    
    result = None
    expanded = [problem.getStartState()]
    
    while result == None:
      currentPath = fringe.pop()
      currentState = currentPath[-1]
      
      if problem.isGoalState(currentState[0]):
        result = currentPath
        continue
      
      succ = problem.getSuccessors(currentState[0])
      
      for state in succ:
        if state[0] not in expanded:
          expanded.append(state[0])
          new_list = currentPath[:]
          new_list.append(state)
          fringe.push(new_list)
    
    steps = []
    for state in result[1:]:
      steps.append(state[1])
      
    return steps  

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    
    from util import PriorityQueue
    fringe = PriorityQueue()
    fringe.push([((start, 'Start', 1), 0)], 0)
    
    result = None
    expanded = {}
    
    while result == None:
      currentPath = fringe.pop()
      currentState = currentPath[-1]
      currentCost = currentState[1]
      
      #print currentPath
      
      if problem.isGoalState(currentState[0][0]):
        result = currentPath
        continue
      
      if currentState[0][0] in expanded:
        succ = expanded[currentState[0][0]] 
      else: 
        succ = problem.getSuccessors(currentState[0][0])
        expanded[currentState[0][0]] = succ
      
      for state in succ:
        if state[0] not in expanded:      
          new_cost = currentCost + state[2]
          new_list = currentPath[:]
          new_list.append([state, new_cost])
          fringe.push(new_list, new_cost)
    
    steps = []
    for state in result[1:]:
      steps.append(state[0][1])
      
    return steps 

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    
    from util import PriorityQueue
    fringe = PriorityQueue()
    fringe.push([((start, 'Start', 1), 0)], 0)
    
    result = None
    expanded = {}
    
    while result == None:
      currentPath = fringe.pop()
      currentState = currentPath[-1]
      currentCost = currentState[1]
      
      #print currentPath
      
      if problem.isGoalState(currentState[0][0]):
        result = currentPath
        continue
      
      if currentState[0][0] in expanded.keys():
        #print "Using cached successors for "
        #print currentState[0][0]
        succ = expanded[currentState[0][0]]
      else: 
        #print "Expanding node "
        #print currentState[0][0]
        succ = problem.getSuccessors(currentState[0][0])
        expanded[currentState[0][0]] = succ
      
      for state in succ:
        if state[0] not in expanded.keys():      
          new_cost = currentCost + state[2]
          new_list = currentPath[:]
          new_list.append([state, new_cost])
          fringe.push(new_list, new_cost + heuristic(state[0], problem))
    
    steps = []
    for state in result[1:]:
      steps.append(state[0][1])
      
    return steps 

class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
  
def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = [int(point1[0]), int(point1[1])]
    x2, y2 = [int(point2[0]), int(point2[1])]
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(aStarSearch(prob, manhattanHeuristic))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
