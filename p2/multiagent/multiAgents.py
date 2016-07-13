# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newCaps = successorGameState.getCapsules() #.asList()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"
        """
        print "###########"
        print newPos
        print newFood.asList()"""
        ghostStates = map(lambda x : (abs(x.getPosition()[0] - newPos[0]) + abs(x.getPosition()[1] - newPos[1]), x.scaredTimer ), newGhostStates)
        """
        print ghostStates
        print newScaredTimes
        """
        evaluation = successorGameState.getScore()
        #print dir(currentGameState)
        
        
        if newPos == currentGameState.getPacmanPosition():
          evaluation -= 500
          
        #if newPos in newCaps:
        #  evaluation += 1000
        
        for state in ghostStates:
          if state[0] == 0 and state[1] == 0:
            evaluation = -999999
          if state[0] == 1 and state[1] == 0:
            evaluation = -500
        
        totFoodDist = 0
        for foodPos in newFood:
          totFoodDist -= (abs(foodPos[0] - newPos[0]) + abs(foodPos[1] - newPos[1]))**0.5
        
        
        
        """
        liveGhosts = list()
        for ghost in newGhostStates:
          #print dir(ghost)
          if not ghost.scaredTimer > 0:
            liveGhosts.append(ghost.getPosition())

        totGhostDist = 0
        for ghostPos in liveGhosts:
          #print ghostPos
          totGhostDist += (abs(ghostPos[0] - newPos[0]) + abs(ghostPos[1] - newPos[1]))**0.5
        """   
        #print totFoodDist
            
          #reduce(lambda x, y : (abs(x[0] - newPos[0]) + abs(x[1] - newPos[1]))**0.5 + (abs(y[0] - newPos[0]) + abs(y[1] - newPos[1]))**0.5, newFood)
        if len(newFood) > 0:
          evaluation += totFoodDist / len(newFood)
          #evaluation += (totGhostDist / len(ghostStates)) / 2
        else:
          evaluation += 999999
        
        #print evaluation;
        return evaluation;

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        #print dir(gameState)
        
        pactions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float("-inf")
        for action in pactions:
          v = self.getValue(gameState.generateSuccessor(0, action), 0, 0)
          #print v
          if v > bestValue:
            bestAction = action
            bestValue = v
            
        #print "Best Value: " + str(bestValue) 
        return bestAction
          
    
    def getValue(self, state, prevAgentIndex, depth):
        #TERMINAL TEST
        #print (prevAgentIndex, depth)
        agentIndex = (prevAgentIndex + 1) % state.getNumAgents()
        legalActions = state.getLegalActions(agentIndex)
        #print agentIndex
        #print legalActions
        
        if depth == self.depth or len(legalActions) == 0: # and prevAgentIndex + 1 == state.getNumAgents():
          score = self.evaluationFunction(state)
          return score       

        bestValue = 0
        
        if agentIndex == 0:
          "Do max value, increment depth"
          bestValue = float("-inf")
          for action in legalActions:
            v = self.getValue(state.generateSuccessor(agentIndex, action), agentIndex, depth)
            #print "Depth " + str(depth) + " max value: " + str(v) + " for action " + action
            if v > bestValue:
              bestValue = v         
        else:
          "Do min value"
          bestValue = float("inf")
          
          #only increment depth if this is the last agent
          if (agentIndex + 1) == state.getNumAgents():
            depth += 1
            
          for action in legalActions:
            v = self.getValue(state.generateSuccessor(agentIndex, action), agentIndex, depth)
            #print "Depth " + str(depth) + " min value: " + str(v) + " for action " + action
            if v < bestValue:
              bestValue = v
        
        #print bestValue  
        return bestValue
        
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    
    a = float("-inf")
    B = float("inf")
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
                
        pactions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float("-inf")
        a = float("-inf")
        B = float("inf")
        for action in pactions:
          v = self.getABValue(gameState.generateSuccessor(0, action), 0, 0, a, B)
          #print v
          if v > bestValue:
            bestAction = action
            bestValue = v
          a = max(a, bestValue)
            
        #print "Best Value: " + str(bestValue) 
        return bestAction
          
    
    def getABValue(self, state, prevAgentIndex, depth, a, B):
        #TERMINAL TEST
        #print (prevAgentIndex, depth)
        agentIndex = (prevAgentIndex + 1) % state.getNumAgents()
        legalActions = state.getLegalActions(agentIndex)
        #print agentIndex
        
        if depth == self.depth or len(legalActions) == 0: # and prevAgentIndex + 1 == state.getNumAgents():
          score = self.evaluationFunction(state)
          return score       

        bestValue = 0
        
        if agentIndex == 0:
          "Do max value, increment depth"
          bestValue = float("-inf")
          for action in legalActions:
            bestValue = max(self.getABValue(state.generateSuccessor(agentIndex, action), agentIndex, depth, a, B), bestValue)
            #alpha-beta
            if bestValue > B:
              #print "PRUNE MAX!"
              return bestValue
            a = max(a, bestValue) 
                    
        else:
          "Do min value"
          bestValue = float("inf")
          #only increment depth if this is the last agent
          if (agentIndex + 1) == state.getNumAgents():
            depth += 1
            
          for action in legalActions:
            bestValue = min(self.getABValue(state.generateSuccessor(agentIndex, action), agentIndex, depth, a, B), bestValue)
            #alpha-beta
            if bestValue < a:
              #print "PRUNE MIN!"
              return bestValue
            B = min(B, bestValue)
              
        #print bestValue  
        return bestValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
                #print dir(gameState)
        
        pactions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float("-inf")
        for action in pactions:
          v = self.getValue(gameState.generateSuccessor(0, action), 0, 0)
          #print v
          if v > bestValue:
            bestAction = action
            bestValue = v
            
        #print "Best Value: " + str(bestValue) 
        return bestAction
      
    def getValue(self, state, prevAgentIndex, depth):
        #TERMINAL TEST
        #print (prevAgentIndex, depth)
        agentIndex = (prevAgentIndex + 1) % state.getNumAgents()
        legalActions = state.getLegalActions(agentIndex)
        #print agentIndex
        #print legalActions
        
        if depth == self.depth or len(legalActions) == 0: # and prevAgentIndex + 1 == state.getNumAgents():
          score = self.evaluationFunction(state)
          return score       

        bestValue = 0.0
        
        if agentIndex == 0:
          "Do max value, increment depth"
          bestValue = float("-inf")
          for action in legalActions:
            v = self.getValue(state.generateSuccessor(agentIndex, action), agentIndex, depth)
            #print "Depth " + str(depth) + " max value: " + str(v) + " for action " + action
            if v > bestValue:
              bestValue = v         
        else:
          "Do expected value"
          
          #only increment depth if this is the last agent
          if (agentIndex + 1) == state.getNumAgents():
            depth += 1
            
          for action in legalActions:
            bestValue += self.getValue(state.generateSuccessor(agentIndex, action), agentIndex, depth) * 1.0
            #print "Depth " + str(depth) + " min value: " + str(v) + " for action " + action
          if len(legalActions) > 0:
            bestValue /= len(legalActions)
        
        #print bestValue  
        return bestValue

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    ghostStates = map(lambda x : (abs(x.getPosition()[0] - newPos[0]) + abs(x.getPosition()[1] - newPos[1]), x.scaredTimer ), newGhostStates)
    evaluation = currentGameState.getScore()
    
    for state in ghostStates:
      if state[0] == 0 and state[1] == 0:
        return -999999
      if state[0] == 1 and state[1] == 0:
        evaluation -= 1000
      if state[0] < state[1] and state[1] > 0:
        evaluation += 50

    
    totFoodDist = 0
    for foodPos in newFood:
      totFoodDist -= (abs(foodPos[0] - newPos[0]) + abs(foodPos[1] - newPos[1]))**0.55
    
    evaluation -= len(newFood) * 12
    
    if len(newFood) > 0:
      evaluation += totFoodDist / len(newFood)
      #evaluation += (totGhostDist / len(ghostStates)) / 2
    else:
      evaluation += 999999
    
    return evaluation;

# Abbreviation
better = betterEvaluationFunction

