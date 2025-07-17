## State Machine Stuff ##


## Transitions ##
class Transition:
    def __init__(self, toState):
        self.toState = toState

    def Execute(self):
        print("Transitioning!")

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
        self.curTrans = None

    def SetState(self, stateName):
        self.prevState = self.curState
        self.curState = self.states[stateName]  # uses name of state to find state instance in dictionary

    def Transition(self, transName):
        self.curTrans = self.transitions[transName] # uses name of transition to find transition instance in dictionary

    def Execute(self):
        if self.curTrans != None:
            if self.curState != None:
                print("Exit")
                self.curState.Exit()
            #print("Execute")
            self.curTrans.Execute()
            self.SetState(self.curTrans.toState)
            self.curTrans = None
        if self.curState != self.prevState:
            print("Enter")
            self.curState.Enter()
            self.prevState = self.curState
        self.curState.Execute()
        

class HS_Model:
    def __init__(self):
        self.FSM = FSM(self)