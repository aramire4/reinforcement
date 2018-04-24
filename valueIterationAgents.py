# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(iterations):
            oldVal = self.values.copy()
            for state in mdp.getStates():
                if(mdp.isTerminal(state)):
                    self.values[state] = 0
                    continue
                maxu = None
                for action in mdp.getPossibleActions(state):
                    eu = 0
                    for(sp, prob) in mdp.getTransitionStatesAndProbs(state, action):
                        r = mdp.getReward(state, action, sp)
                        r+= self.discount * oldVal[sp]
                        eu += prob * r
                    if (maxu is None or eu > maxu):
                        maxu = eu
                self.values[state] = maxu
            """
            for s in mdp.getStates():
                if(mdp.isTerminal(s)):
                    self.values[s] = 0
                    continue
                maxu = None
                for a in mdp.getPossibleActions(s):
                    eu = 0
                    for(sp, p) in mdp.getTransitionStatesAndProbs(s, a):
                        r = mdp.getReward(s, a, sp)

                        r+= self.discount * oldVal[sp]
                        eu += p * r
                    if (maxu is None or eu > maxu):
                        maxu = eu
                self.values[s] = maxu
            """
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qVal = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for trans in transitions:
            qVal += trans[1]*(self.mdp.getReward(state, action, trans[0]) + self.discount*self.values[trans[0]])
        return qVal
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            None
        else:
            bstVal = -99999999999
            bstAction = None
            allActions = self.mdp.getPossibleActions(state)
            for action in allActions:
                val = self.computeQValueFromValues(state, action)
                if val > bstVal:
                    bstAction = action
                    bstVal = val
            return bstAction
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
