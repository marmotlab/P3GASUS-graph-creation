import numpy as np
import matplotlib.pyplot as plt
import copy
import networkx as nx
import networkx.algorithms.isomorphism as iso
import lacam

from mapUtil import *

import time
import os

class Position:
    def __init__(self) -> None:
        self.robotDict = dict()   # robot->list of tasks

    def __repr__(self) -> str:
        asd = "\n"
        for i in self.__dir__():
            if not i.startswith('__'):
                asd+=i
                asd+=": "
                asd+=str(getattr(self,i))
                asd+=", "

        return asd

class Task:
    __actionDict__ = {0:np.array([0,0]), 1:np.array([1,0]), 2:np.array([0,1]), 3:np.array([-1,0]), 4:np.array([0,-1])}

    def __init__(self, tid, rid, start, action, time) -> None:
        self.taskID = tid
        self.robotID = rid
        self.action = action
        self.startPos = np.array(start)
        self.goalPos = np.array(self.__actionDict__[action]+start)
        self.time = time

    def __repr__(self) -> str:
        asd = "\n"
        for i in self.__dir__():
            if not i.startswith('__'):
                asd+=i
                asd+=": "
                asd+=str(getattr(self,i))
                asd+=", "

        return asd
  

class ADGraph:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.taskList = dict()
        self.robotList = []

    def fileWrite(self, path):
        with open(path+"/"+(type(self)).__name__+"_Graph.txt", "w") as f:
            for i in sorted(self.taskList):
                t = []
                for j in self.graph.out_edges(i):
                    t.append(j[1])
                f.write(str(i)+":"+str(sorted(t))+"\n")

        with open(path+"/"+(type(self)).__name__+"_TaskList.txt", "w") as f:
            for i in sorted(self.taskList):
                f.write(self.taskList[i].__repr__())

        

class OriginalADG(ADGraph):
    def __init__(self, taskList, startPositions):
        super().__init__()
        taskList = np.array(taskList)
        startPositions = np.array(startPositions)
        
        ## Create Graph Skeleton and type1 dependencies
        currentPositions = copy.deepcopy(startPositions)
        numRobots = len(startPositions)

        tId = 1
        for rid in range(numRobots):
            prevTask = None
            for i, task in enumerate(taskList[rid, :]):
                t = Task(tId, rid, currentPositions[rid], task, i)
                # print(t)
                self.taskList[tId] = t
                tId+=1
                currentPositions[rid] = t.goalPos
                self.graph.add_node(t.taskID)

                if(prevTask is None):
                    self.robotList.append(t)
                else:
                    self.graph.add_edge(prevTask.taskID, t.taskID)

                prevTask = t

        ## Create type2 dependencies
        for rid in range(numRobots):
            firstTid = self.robotList[rid].taskID
            for taskID in range(firstTid, firstTid+len(taskList[1])):
                task = self.taskList[taskID]
                
                for rid_ in range(numRobots):
                    if(rid != rid_):
                        # print(rid, rid_)
                        firstTid_ = self.robotList[rid_].taskID
                        for taskID_ in range(firstTid_, firstTid_+len(taskList[1])):
                            task_ = self.taskList[taskID_]
                            # print(task.startPos, task_.goalPos)
                            if np.array_equal(task.startPos, task_.goalPos) and task.time<=task_.time:
                                # print(task.taskID, task_.taskID)
                                self.graph.add_edge(task.taskID, task_.taskID)
                                break
                            

class SAGE(ADGraph):
    def __init__(self, taskList, startPositions):
        super().__init__()

        taskList = np.array(taskList)
        startPositions = np.array(startPositions)

        positions = {}
        currentPositions = copy.deepcopy(startPositions)
        numRobots = len(taskList)
        tId = 1
        for rid in range(numRobots):
            prevTask = None
            for i, task in enumerate(taskList[rid,:]):
                
                t = Task(tId, rid, currentPositions[rid], task, i)
                # print(t)
                self.taskList[tId] = t
                tId+=1
                currentPositions[rid] = t.goalPos
                self.graph.add_node(t.taskID)

                if(tuple(t.goalPos) not in positions):
                    positions[tuple(t.goalPos)] = Position()
                # positions[tuple(t.goalPos)].taskList.append(t)
                if(rid not in positions[tuple(t.goalPos)].robotDict):
                    positions[tuple(t.goalPos)].robotDict[rid] = list()
                positions[tuple(t.goalPos)].robotDict[rid].append(t.taskID)
                
                if(prevTask is None):
                    self.robotList.append(t)
                else:
                    self.graph.add_edge(prevTask.taskID, t.taskID)

                prevTask = t

        taskQueue = [i.taskID for i in self.robotList]
        

        previousTime = -1
        tasksToClear = []

        while len(taskQueue)!=0:
            

            tID = taskQueue.pop(0)
            t = self.taskList[tID]

            if(t.time>previousTime):
                previousTime+=1
                # Remove t from position.robotDict
                for ttc in tasksToClear:
                    t__ = self.taskList[ttc]
                    rDict = positions[tuple(t__.goalPos)].robotDict
                    rDict[t__.robotID].pop(0)
                    if(len(rDict[t__.robotID])==0):
                        rDict.pop(t__.robotID)
                tasksToClear = []

            tasksToClear.append(tID)

            if(tID+1) in self.taskList and self.taskList[tID+1].time!=0:
                taskQueue.append(tID+1)

            st = tuple(t.startPos)
            if st in positions:
                
                for rid_ in positions[st].robotDict.keys():
                    if(t.robotID!=rid_):
                        self.graph.add_edge(tID, self.taskList[positions[st].robotDict[rid_][0]].taskID)
        
            

class FORTED(ADGraph):
    def __init__(self, taskList, startPositions):
        super().__init__()

        taskList = np.array(taskList)
        startPositions = np.array(startPositions)
        tasksPerRobot = len(taskList[0])
        positions = {}
        currentPositions = copy.deepcopy(startPositions)
        numRobots = len(taskList)

        for time in range(0, tasksPerRobot):
            for rid in range(numRobots):
                
                tid = rid*tasksPerRobot+time+1
                prevTask = tid-1 if tid-1 in self.taskList else None

                t =  Task(tid, rid, currentPositions[rid], taskList[rid, time], time)
                self.taskList[tid] = t
                currentPositions[rid] = t.goalPos
                self.graph.add_node(t.taskID)

                if(prevTask is None):
                    self.robotList.append(t)
                else:
                    self.graph.add_edge(prevTask, t.taskID)

                if(tuple(t.startPos) not in positions):
                    positions[tuple(t.startPos)] = t.taskID
                else:
                    if positions[tuple(t.startPos)] != t.taskID-1:
                        self.graph.add_edge(positions[tuple(t.startPos)], t.taskID-1)
                    positions[tuple(t.startPos)] = t.taskID
        
        for rid in range(numRobots):
            
            tid = (rid+1)*tasksPerRobot
            t = self.taskList[tid]
            if(tuple(t.goalPos) in positions and t.action!=0):
                self.graph.add_edge(positions[tuple(t.goalPos)], t.taskID)

class MAGE(ADGraph):
    def __init__(self, taskList, startPositions, baseADG=FORTED):
        super().__init__()
        self.baseADG = baseADG(taskList, startPositions)
        self.graph = self.baseADG.graph
        self.taskList = self.baseADG.taskList
        self.robotList = self.baseADG.robotList
        
        dp = np.zeros((len(self.taskList)+2, len(self.taskList)+2), dtype=bool)
        for t in self.robotList:
            self.reduceGraph(t.taskID, dp)

    def reduceGraph(self, root, dp):
        if(dp[root][root])==0:    
            dp[root][root]=1
            children = [i[1] for i in self.graph.out_edges(root)]
        
        
            if(root+1) in children:
                dp[root] = np.logical_or(self.reduceGraph(root+1, dp), dp[root])
                children.remove(root+1)
                 
            children = sorted(children, key=lambda x: self.taskList[x].time)
            while(len(children)!=0):
                if(dp[root][children[0]]==1):
                    self.graph.remove_edge(root, children[0])
                else:
                    dp[root] = np.logical_or(self.reduceGraph(children[0], dp), dp[root])
                children.pop(0)             
        return dp[root]
    
def getActionFromPos(currentPos, nextPos):
    temp = np.subtract(nextPos,currentPos)
    if np.array_equal(temp, np.array([1,0])):
        return 1
    elif np.array_equal(temp, np.array([0,1])):
        return 2
    elif np.array_equal(temp,np.array([-1,0])):
        return 3
    elif np.array_equal(temp, np.array([0,-1])):
        return 4
    elif np.array_equal(temp, np.array([0,0])):
        return 0
    else:
        return -1
    
def verifyPaths(paths):
    NUM_AGENTS = len(paths)
    idx = max(len(i) for i in paths)

    PATHS = np.zeros((NUM_AGENTS, idx, 2))

    for i in range(NUM_AGENTS):
        for j in range(len(paths[i])):
            PATHS[i,j] = paths[i][j]
        for k in range(len(paths[i]), idx):
            PATHS[i,k] = paths[i][j]

    for i in range(idx):
        if(len(np.unique(PATHS[:,i,:], axis=0))!=NUM_AGENTS):
            return False
    
    return True


def getWorldSize(NUM_AGENTS = 40, minFreeCellPercentToMaintain = 30):
    cellsNeeded = int(np.ceil(np.sqrt(100*NUM_AGENTS/(100-minFreeCellPercentToMaintain))))
    # cellsNeeded = max(8, cellsNeeded)
    return cellsNeeded, 1-NUM_AGENTS/(cellsNeeded*cellsNeeded)

def testTime(method, ACTIONS, STARTS, subMethod = FORTED):
    start = time.time()
    if(method == MAGE):
        exGraph = method(ACTIONS, STARTS, baseADG=subMethod)
    else:
        exGraph = method(ACTIONS, STARTS,)
    end = time.time()
    
    return len(exGraph.graph.edges)-len(ACTIONS[0])*(len(exGraph.robotList)-1), end-start


def oneTestCase(NUM_AGENTS=40, minFreeCellPercentToMaintain=30):
    STARTS = []
    GOALS = []

    cellsNeeded, freeSpacePercent = getWorldSize(NUM_AGENTS, minFreeCellPercentToMaintain)
    world = np.zeros((cellsNeeded, cellsNeeded), dtype=int)

    tempMap = np.copy(world)
    for i in range(NUM_AGENTS):
        STARTS.append(getFreeCell(tempMap))
        tempMap[STARTS[-1]] = 2

    tempMap = np.copy(world)
    for i in range(NUM_AGENTS):
        
        GOALS.append(getFreeCell(tempMap))
        tempMap[GOALS[-1]] = 3

    world = world.tolist()
    paths = lacam.solve(world, STARTS, GOALS, 5.0)

    if(paths is None):
        raise Exception("No Solution")


    ACTIONS = []

    for i in range(NUM_AGENTS):
        action = []
        for j in range(len(paths[i])-1):
            action.append(getActionFromPos(paths[i][j], paths[i][j+1]))
        ACTIONS.append(action)

    idx = max(len(i) for i in ACTIONS)

    temp = np.zeros((NUM_AGENTS, idx))
    for i in range(len(ACTIONS)):
        for j in range(len(ACTIONS[i])):
            temp[i,j] = ACTIONS[i][j]

    ACTIONS = temp

    return ACTIONS, STARTS, freeSpacePercent