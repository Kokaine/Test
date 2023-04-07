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


from email.errors import UndecodableBytesDefect
from util import manhattanDistance
from game import Directions
import random, util
from math import sqrt, log

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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # NOTE: this is an incomplete function, just showing how to get current state of the Env and Agent.

        return successorGameState.getScore()

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
      Your minimax agent (question 1)
    """

    def getAction(self, gameState):

        def maxleaf(gameState,mdepth):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if mdepth==self.depth:
                return self.evaluationFunction(gameState)
            maxvalue = -10000
            actions = gameState.getLegalActions(0)#pacman行动
            for newAction in actions:
                child = gameState.generateSuccessor(0,newAction)
                maxvalue = max (maxvalue , minleaf(child,mdepth,1))
            return maxvalue
        
        def minleaf(gameState,mdepth, agentIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if mdepth==self.depth*2:
                return self.evaluationFunction(gameState)
            minvalue = 10000
            actions = gameState.getLegalActions(agentIndex)#ghost行动
            for newAction in actions:
                if agentIndex==gameState.getNumAgents()-1:
                    child = gameState.generateSuccessor(agentIndex,newAction)
                    minvalue = min (minvalue,maxleaf(child,mdepth+1))
                else:
                    child = gameState.generateSuccessor(agentIndex,newAction)
                    minvalue = min (minvalue,minleaf(child,mdepth,agentIndex+1))
            return minvalue

        bestScore = -10000
        chooseAction=''
        actions = gameState.getLegalActions(0)
        for newAction in actions:
            score = minleaf(gameState.generateSuccessor(0,newAction),0,1)
            if score > bestScore:
                chooseAction = newAction
                bestScore = score
        return chooseAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxleaf(gameState,mdepth,alpha,beta):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if mdepth==self.depth:
                return self.evaluationFunction(gameState)
            maxvalue = -10000
            nowAlpha=alpha
            actions = gameState.getLegalActions(0)#pacman行动
            for newAction in actions:
                child = gameState.generateSuccessor(0,newAction)
                maxvalue = max (maxvalue , minleaf(child,mdepth,1,nowAlpha,beta))
                if maxvalue>beta:
                    return maxvalue
                nowAlpha=max(maxvalue,nowAlpha)
            return maxvalue
        
        def minleaf(gameState,mdepth, agentIndex,alpha,beta):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if mdepth==self.depth*2:
                return self.evaluationFunction(gameState)
            minvalue = 10000
            nowBeta=beta
            actions = gameState.getLegalActions(agentIndex)#ghost行动
            for newAction in actions:
                if agentIndex==gameState.getNumAgents()-1:
                    child = gameState.generateSuccessor(agentIndex,newAction)
                    minvalue = min (minvalue,maxleaf(child,mdepth+1,alpha,nowBeta))
                    if minvalue < alpha:
                        return minvalue
                    nowBeta = min(minvalue,nowBeta)
                else:
                    child = gameState.generateSuccessor(agentIndex,newAction)
                    minvalue = min (minvalue,minleaf(child,mdepth,agentIndex+1,alpha,nowBeta))
                    if minvalue < alpha:
                        return minvalue
                    nowBeta = min(minvalue,nowBeta)
            return minvalue

        bestScore = -10000
        alpha=-10000
        beta=10000
        chooseAction=''
        actions = gameState.getLegalActions(0)
        for newAction in actions:
            score = minleaf(gameState.generateSuccessor(0,newAction),0,1,alpha,beta)
            if score > bestScore:
                chooseAction = newAction
                bestScore = score
            if score>beta:
                return chooseAction
            alpha = max(score,alpha)
        return chooseAction
        util.raiseNotDefined()

class MCTSAgent(MultiAgentSearchAgent):
    """
      Your MCTS agent with Monte Carlo Tree Search (question 3)
    """

    def getAction(self, gameState):

        class Nodes:
            '''
            We have provided node structure that you might need in MCTS tree.
            '''
            def __init__(self, data):
                self.north = None
                self.east = None
                self.west = None
                self.south = None
                self.stop = None
                self.parent = None
                self.statevalue = data[0]
                self.numerator = data[1]
                self.denominator = data[2]
                self.Search_Times=0
                self.Children_Num=0
                self.Children_Max_Num=0
                self.Whether_Expanded=0
                self.Children=[]#存储孩子节点在树中的编号
                self.path=[]
                self.value=0

        data = [gameState, 0, 1]
        cgstree = Nodes(data) 

        def Selection(cgs, cgstree):

            "*** YOUR CODE HERE ***"

            util.raiseNotDefined()

        def Expansion(cgs, cgstree):
            "*** YOUR CODE HERE ***"
            util.raiseNotDefined()

        def Simulation(cgs, cgstree):
            "*** YOUR CODE HERE ***"
            util.raiseNotDefined()

        def Backpropagation(cgstree, WinorLose):
            "*** YOUR CODE HERE ***"
            util.raiseNotDefined()

        def HeuristicFunction(currentGameState):
            "*** YOUR CODE HERE ***"
            return 0
            util.raiseNotDefined()

        "*** YOUR CODE HERE ***"
        def ghostmove(State,agtidx):#鬼走仍用上
            def maxleaf(State,mdepth,alpha,beta):
                if State.isWin() or State.isLose():
                    return self.evaluationFunction(State)
                if mdepth==self.depth:
                    return self.evaluationFunction(State)
                maxvalue = -10000
                nowAlpha=alpha
                actions = State.getLegalActions(0)#pacman行动
                for newAction in actions:
                    child = State.generateSuccessor(0,newAction)
                    maxvalue = max (maxvalue , minleaf(child,mdepth,1,nowAlpha,beta))
                    if maxvalue>beta:
                        return maxvalue
                    nowAlpha=max(maxvalue,nowAlpha)
                return maxvalue
        
            def minleaf(State,mdepth, agentIndex,alpha,beta):
                if State.isWin() or State.isLose():
                    return self.evaluationFunction(State)
                if mdepth==self.depth*2:
                    return self.evaluationFunction(State)
                minvalue = 10000
                nowBeta=beta
                actions = State.getLegalActions(agentIndex)#ghost行动
                for newAction in actions:
                    if agentIndex==State.getNumAgents()-1:
                        child = State.generateSuccessor(agentIndex,newAction)
                        minvalue = min (minvalue,maxleaf(child,mdepth+1,alpha,nowBeta))
                        if minvalue < alpha:
                            return minvalue
                        nowBeta = min(minvalue,nowBeta)
                    else:
                        child = State.generateSuccessor(agentIndex,newAction)
                        minvalue = min (minvalue,minleaf(child,mdepth,agentIndex+1,alpha,nowBeta))
                        if minvalue < alpha:
                           return minvalue
                        nowBeta = min(minvalue,nowBeta)
                return minvalue

            bestScore = -10000
            alpha=-10000
            beta=10000
            chooseAction=''
            actions = State.getLegalActions(agtidx)
            for newAction in actions:
                score = maxleaf(State.generateSuccessor(agtidx,newAction),0,alpha,beta)
                if score < bestScore:
                    chooseAction = newAction
                    bestScore = score
                if score<alpha:
                    return chooseAction
                beta = min(score,beta)
            return chooseAction


        Node = [Nodes(data) for _ in range(100000)]
        UCB_Constant=2
        Node_Nums=0
        agent_num=gameState.getNumAgents()

        def UCB_Search(now,state,Node_Nums):
            choose=0
            if Node[now].Whether_Expanded==1:
                max_ucb_value=-1000
                i=0
                while i<Node[now].Children_Max_Num:
                    i+=1
                    ucb_value=Node[Node[now].Children[i]].value/Node[Node[now].Children[i]].Search_Times + UCB_Constant * sqrt(log(Node[now].Search_Times) / Node[Node[now].Children[i]].Search_Times)
                    if ucb_value>max_ucb_value:
                        max_ucb_value=ucb_value
                        choose=Node[now].Children[i]
                return choose
            else:
                actions=state.getLegalActions(0)
                rest=len(actions)-Node[now].Children_Num
                Node[now].Children_Num+=1
                if rest==1:
                    Node[now].Whether_Expanded=1
                    Node[now].Children_Max_Num=Node[now].Children_Num
                if rest==0:
                    return -1
                else:
                    Node_Nums+=1
                    Node[now].Children.append(Node_Nums)
                    #Node[now].Children[Node[now].Children_Num] = Node_Nums
                    Node[Node_Nums].path=actions[Node[now].Children_Num-1]
                    return Node_Nums

        import time
        timeout=60
        depth=10
        start=time.time()
        while time.time()-start<timeout:
            index=0
            ucb=0
            initial=UCB_Search(0,gameState,Node_Nums)
            i=0
            nstate=gameState
            while i<10:
                i+=1
                ucb=initial
                SearchNode=[]
                index=1
                SearchNode.append(ucb)
                nstate = gameState.generateSuccessor(0,Node[ucb].path)
                j=0
                while j<agent_num-1:
                    j+=1
                    p=ghostmove(nstate,j)
                    nstate=nstate.generateSuccessor(j,p)
                while index<=depth:
                    ucb=UCB_Search(ucb,nstate,Node_Nums)
                    nstate = nstate.generateSuccessor(0,Node[ucb].path)
                    w=0
                    while w<agent_num:
                        w+=1
                        p=ghostmove(nstate,j)
                        nstate=nstate.generateSuccessor(w,p)
                    if nstate.isWin() or nstate.isLose():
                        value=self.evaluationFunction(nstate)+HeuristicFunction(nstate)
                        break
                    if ucb==-1:
                        break
                    index+=1
                    SearchNode.append(ucb)
                value=self.evaluationFunction(nstate)+HeuristicFunction(nstate)
                p=index
                while p>0:
                    Node[SearchNode[p]].Search_Times+=1
                    Node[SearchNode[p]].value += value
                Node[0].Search_Times+=1
        maxvalue = -1000
        best_node = 0
        i=1
        while i <= Node[0].Children_Num:
            i+=1
            if Node[Node[0].Children[i]].Search_Times != 0:
                if Node[Node[0].Children[i]].value / Node[Node[0].Children[i]].Search_Times > maxvalue:
                    maxvalue = Node[Node[0].Children[i]].Value / Node[Node[0].Children[i]].Search_Times
                    best_node = Node[0].Children[i]
        return Node[best_node].path
        util.raiseNotDefined()
