## State Machine Stuff ##


## Transitions ##
class Transition:
    def __init__(self, toState):
        self.toState = toState

    def Execute(self):
        pass

## States ##
class State:
    pass

## Finite State Machine ##
class FSM:
    def __init__(self, char):
        self.char = char
        self.states = {}
        self.transitions = {}
        self.curState = None
        self.prevState = None
        self.cureTrans = None

    def SetState(self, stateName):
        self.prevState = self.curState
        self.curState = self.states[stateName]  # uses name of state to find state instance in dictionary

    def Transition(self, transName):
        self.curTrans = self.transitions[transName] # uses name of transition to find transition instance in dictionary

    def Execute(self):
        if self.curTrans != None:
            self.curState.Exit()
            self.curTrans.Execute()
            self.SetState(self.curTrans.toState)
            self.curTrans = None
        if self.curState != self.prevState:
            self.curState.Enter()
            self.prevState = self.curState
        self.curState.Execute()
        

