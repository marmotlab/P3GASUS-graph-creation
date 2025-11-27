import numpy as np
import matplotlib.pyplot as plt
import copy
import networkx as nx
import networkx.algorithms.isomorphism as iso

from sklearn.neighbors import KDTree
import json
import csv

import time
import os 

import sys
sys.setrecursionlimit(600000)

class ContinuousTask:

    def __init__(self, tid, rid, start, goal, time) -> None:
        self.taskID = tid
        self.robotID = rid
        self.startPos = np.array(start)
        self.goalPos = np.array(goal)
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
    
class ContinuousExecutionGraph:
    
    def __init__(self, positions = None) -> None:
        assert positions is not None

        self.graph = nx.DiGraph()
        self.taskList = dict()
        self.robotList = []

        self.THRESH = 1e8
        
        for i in range(positions.shape[1]):
            for r in range(positions.shape[0]):
                for r_ in range(r+1, positions.shape[0]):
                    self.THRESH = min(self.THRESH, self.getDistance(positions[r,i], positions[r_,i]))
    
    def checkCollision(self, a, b):
        return self.getDistance(a,b)<self.THRESH

    def getDistance(self, a,b):
        a = np.array(a)
        b = np.array(b)

        if(np.array_equal([-2,-2], a) or np.array_equal([-2,-2], b)):
            return 1e8

        return np.linalg.norm(a-b, 2)

    def fileWrite(self, path):
        
        # print()
        nx.write_adjlist(self.graph,path+"/"+(type(self)).__name__+"_Graph.txt")

        with open(path+"/"+(type(self)).__name__+"_TaskList.txt", "w") as f:
            for i in self.taskList:
                f.write(self.taskList[i].__repr__())

class OriginalADG(ContinuousExecutionGraph):
    
    def __init__(self, allPositions) -> None:
        super().__init__(allPositions)
        ## Create Graph Skeleton and type1 dependencies
        
        numRobots = allPositions.shape[0]
        tId = 1
        for rid in range(numRobots):
            prevTask = None
            for i, task in enumerate(allPositions[rid, :-1]):
                t = ContinuousTask(tId, rid, task, allPositions[rid,i+1], i)
                # print(t)
                self.taskList[tId] = t
                tId+=1
                self.graph.add_node(t.taskID)
                if(prevTask is None):
                    self.robotList.append(t)
                else:
                    self.graph.add_edge(prevTask.taskID, t.taskID)

                prevTask = t

        ## Create type2 dependencies

        for rid in range(numRobots):
            firstTid = self.robotList[rid].taskID
            for taskID in range(firstTid, firstTid+allPositions.shape[1]-1):
                task = self.taskList[taskID]
                if(task.startPos[0]==-2 and task.startPos[1]==-2):
                    break
                for rid_ in range(numRobots):
                    if(rid != rid_):
                        # print(rid, rid_)
                        firstTid_ = self.robotList[rid_].taskID
                        for taskID_ in range(firstTid_, firstTid_+allPositions.shape[1]-1):
                            task_ = self.taskList[taskID_]
                            # print(task.startPos, task_.goalPos)
                            if self.checkCollision(task.startPos, task_.goalPos) and task.time<=task_.time:
                                self.graph.add_edge(task.taskID, task_.taskID)
                                break

class SAGE(ContinuousExecutionGraph):
    def __init__(self, allPositions=None) -> None:
        super().__init__(allPositions)
        numRobots = allPositions.shape[0]
        tId = 1

        positions = []

        for rid in range(numRobots):
            prevTask = None
            for i, task in enumerate(allPositions[rid,:-1]):
                t = ContinuousTask(tId, rid, task, allPositions[rid,i+1], i)

                self.taskList[tId] = t
                tId+=1
                self.graph.add_node(t.taskID)

                positions.append(t.goalPos)

                if(prevTask is None):
                    self.robotList.append(t)
                else:
                    self.graph.add_edge(prevTask.taskID, t.taskID)

                prevTask = t

        tree = KDTree(positions)  

        taskQueue = [i.taskID for i in self.robotList]

        while len(taskQueue)!=0:
            tID = taskQueue.pop(0)
            t = self.taskList[tID]

            if t.startPos[0]==-2 and t.startPos[1]==-2:
                continue

            if(tID+1) in self.taskList and self.taskList[tID+1].time!=0:
                taskQueue.append(tID+1)
            
            possibeDependencies = tree.query_radius([t.startPos], r=self.THRESH)[0]
            possibeDependencies = sorted(possibeDependencies)
            dependentRobots = [t.robotID]

            for tID__ in possibeDependencies:
                tID_ = tID__+1
                t_ = self.taskList[tID_]
                if(t_.robotID not in dependentRobots and t.time<=t_.time):
                    if(self.checkCollision(t.startPos, t_.goalPos)):
                        self.graph.add_edge(tID, tID_)
                        dependentRobots.append(t_.robotID)


class MAGE(SAGE):
    def __init__(self, allPositions=None, filename="temp.dat") -> None:
        super().__init__(allPositions)
        
        if((len(self.taskList)+2)>50000):
            assert filename is not None
            dp = np.memmap(filename, dtype='bool', mode='w+', shape=(len(self.taskList)+2, len(self.taskList)+2))
        else:
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

class Multi_KDTree_SAGE(ContinuousExecutionGraph):
    def __init__(self, allPositions=None) -> None:
        super().__init__(allPositions)

        numRobots = allPositions.shape[0]
        tId = 1

        trees = []

        positions = []

        for rid in range(numRobots):
            prevTask = None
            for i, task in enumerate(allPositions[rid,:-1]):
                t = ContinuousTask(tId, rid, task, allPositions[rid,i+1], i)

                self.taskList[tId] = t
                tId+=1
                self.graph.add_node(t.taskID)

                positions.append(t.goalPos)

                if(prevTask is None):
                    self.robotList.append(t)
                else:
                    self.graph.add_edge(prevTask.taskID, t.taskID)

                prevTask = t

            trees.append(KDTree(positions,leaf_size=4))
            positions = []

        taskQueue = [i.taskID for i in self.robotList]

        while len(taskQueue)!=0:

            tID = taskQueue.pop(0)
            t = self.taskList[tID]

            if t.startPos[0]==-2 and t.startPos[1]==-2:
                continue

            if(tID+1) in self.taskList and self.taskList[tID+1].time!=0:
                taskQueue.append(tID+1)
            
            for rID_ in range(numRobots):
                if rID_==t.robotID:
                    continue
                
                possibeDependencies = trees[rID_].query_radius([t.startPos], r=self.THRESH)[0]
                possibeDependencies = sorted(possibeDependencies)
                for i in possibeDependencies:
                    if i>=t.time:
                        tID_ = self.robotList[rID_].taskID+i
                        self.graph.add_edge(tID, tID_)
                        break

#Helper Functions

def testTime(method, allPos, fname="temp.dat"):
    start = time.time()
    if(method is MAGE):
        exGraph = method(allPos, fname)
    else:
        exGraph = method(allPos)
    end = time.time()
    
    return len(exGraph.graph.edges)-len(allPos[0])*(len(exGraph.robotList)-1), end-start

def jsonToNpy(data, NUM_AGENTS):
    temp = []
    for i in range(NUM_AGENTS):
        temp_ = []
        for j in range(len(data['agent'+str(i)])):
            temp_.append(data['agent'+str(i)][j]["position"])
        temp.append(temp_)
    
    positions = np.full((len(temp), max([len(i) for i in temp]),2), -2.0)
    for idx, _ in np.ndenumerate(positions):
        if(idx[1]<len(temp[idx[0]])):
            positions[idx] = temp[idx[0]][idx[1]][idx[2]]
    return positions