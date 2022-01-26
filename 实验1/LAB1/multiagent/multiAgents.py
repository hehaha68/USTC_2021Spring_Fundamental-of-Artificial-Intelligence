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

import sys
sys.path.insert(1, '..')
import myImpl

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
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

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        """
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        """

        return better(childGameState)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    if "grading" in sys.modules:
        return currentGameState.getScore()
    else:
        return better(currentGameState)

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

    def getPrevAction(self, state):
        return state._prevAction

class MyGameState():
    def __init__(self, gameState, evalFn, currentAgent=0, prevAction=None):
        self.__gameState = gameState
        self.__evalFn = evalFn
        self.__currentAgent = currentAgent
        self.__prevAction = prevAction

    def isTerminated(self):
        return self.__gameState.isWin() or self.__gameState.isLose()

    def isMe(self):
        return self.__currentAgent == 0

    def getChildren(self):
        return MyGameState.__ChildrenIterator(self.__gameState, self.__gameState.getLegalActions(self.__currentAgent), self.__currentAgent, self.__evalFn)

    def evaluateScore(self):
        return self.__evalFn(self.__gameState)

    def _getPrevAction(self):
        return self.__prevAction

    def __str__(self):
        return self.__gameState.__str__()

    class __ChildrenIterator():
        def __init__(self, state, actions, agent, evalFn):
            self.__state = state
            self.__actions = actions
            self.__agent = agent
            self.__evalFn = evalFn
            self.__index = 0

        def __iter__(self):
            return self

        def __next__(self):
            try:
                action = self.__actions[self.__index]
                nextAgent = (self.__agent + 1) % self.__state.getNumAgents()
                result = MyGameState(self.__state.getNextState(self.__agent, action), self.__evalFn, nextAgent, action)
            except IndexError:
                raise StopIteration
            
            self.__index += 1
            return result
    

class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        myAgent = myImpl.MyMinimaxAgent(self.depth)
        myState = MyGameState(gameState, self.evaluationFunction)

        return myAgent.getNextState(myState)._getPrevAction()

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        myAgent = myImpl.MyAlphaBetaAgent(self.depth)
        myState = MyGameState(gameState, self.evaluationFunction)

        return myAgent.getNextState(myState)._getPrevAction()

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = 0

    foodDistances = [manhattanDistance(newPos, foodPosition) for foodPosition in newFood.asList()]
    foodDistances += [manhattanDistance(newPos, capsulePosition) for capsulePosition in currentGameState.getCapsules()]
    explored = 1 if random.random() > 0.2 else -1
    score -= 0 if len(foodDistances) == 0 else min(foodDistances) * explored + 99 * len(foodDistances)

    for i, ghostPosition in enumerate(currentGameState.getGhostPositions()):
        if manhattanDistance(newPos, ghostPosition) <= 1 and newScaredTimes[i] <= 1:
            score -= 9999 * (len(foodDistances) + 1)

    return currentGameState.getScore() + score

# Abbreviation
better = betterEvaluationFunction
