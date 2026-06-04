import numpy as np
import matplotlib.pyplot as plt
import copy
import networkx as nx
import networkx.algorithms.isomorphism as iso

from sklearn.neighbors import KDTree
from rtree import index as rtree_index
import json
import csv

import time
import os 

import sys
sys.setrecursionlimit(600000)

class AABBTree:
    """2D AABB tree for bounding-box intersection queries.

    Each entry is stored as (id, (min_x, min_y, max_x, max_y)).
    Build once with build(), then query many times with query().
    """

    __slots__ = ("bbox", "left", "right", "entry_id")

    def __init__(self):
        self.bbox = None
        self.left = None
        self.right = None
        self.entry_id = None  # set only on leaf nodes

    @classmethod
    def build(cls, entries):
        """entries: list of (id, (min_x, min_y, max_x, max_y))"""
        if not entries:
            return None
        node = cls()
        node.bbox = (
            min(e[1][0] for e in entries),
            min(e[1][1] for e in entries),
            max(e[1][2] for e in entries),
            max(e[1][3] for e in entries),
        )
        if len(entries) == 1:
            node.entry_id = entries[0][0]
            return node
        # split along the axis with the widest spread of bbox centres
        cx = [(e[1][0] + e[1][2]) * 0.5 for e in entries]
        cy = [(e[1][1] + e[1][3]) * 0.5 for e in entries]
        axis = 0 if (max(cx) - min(cx)) >= (max(cy) - min(cy)) else 1
        entries_sorted = sorted(entries, key=lambda e: (e[1][0] + e[1][2]) if axis == 0 else (e[1][1] + e[1][3]))
        mid = len(entries_sorted) // 2
        node.left  = cls.build(entries_sorted[:mid])
        node.right = cls.build(entries_sorted[mid:])
        return node

    def query(self, bbox):
        """Return list of entry ids whose bounding boxes intersect bbox."""
        results = []
        self._query(bbox, results)
        return results

    def _query(self, bbox, results):
        b = self.bbox
        q = bbox
        if b[0] > q[2] or b[2] < q[0] or b[1] > q[3] or b[3] < q[1]:
            return
        if self.entry_id is not None:
            results.append(self.entry_id)
            return
        if self.left:
            self.left._query(bbox, results)
        if self.right:
            self.right._query(bbox, results)


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
    
    def __init__(self, positions = None, radii=None) -> None:
        assert positions is not None

        self.graph = nx.DiGraph()
        self.taskList = dict()
        self.robotList = []

        self.radii = None if radii is None else np.array(radii, dtype=float)
        self.maxRadius = None if self.radii is None else float(np.max(self.radii))

        # calculate threshold the old way if radius is None
        self.THRESH = 1e8
        if self.radii is None:
            for i in range(positions.shape[1]):
                for r in range(positions.shape[0]):
                    for r_ in range(r+1, positions.shape[0]):
                        self.THRESH = min(self.THRESH, self.getDistance(positions[r,i], positions[r_,i]))
        else:
            self.THRESH = None
    
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
    def __init__(self, allPositions=None, radii=None) -> None:
        super().__init__(allPositions, radii=radii)
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
            
            if self.radii is not None:
                threshold = float(self.radii[t.robotID] + self.maxRadius)
            else:
                threshold = self.THRESH

            possibeDependencies = tree.query_radius([t.startPos], r=threshold)[0]
            possibeDependencies = sorted(possibeDependencies)
            dependentRobots = [t.robotID]

            for tID__ in possibeDependencies:
                tID_ = tID__+1
                t_ = self.taskList[tID_]
                if(t_.robotID not in dependentRobots and t.time<=t_.time):
                    if(self.checkCollision(t.startPos, t_.goalPos) if self.radii is None else self.getDistance(t.startPos, t_.goalPos) <= threshold):
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
    def __init__(self, allPositions=None, radii=None) -> None:
        super().__init__(allPositions, radii=radii)

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
                
                if self.radii is not None:
                    threshold = float(self.radii[t.robotID] + self.radii[rID_])
                else:
                    threshold = self.THRESH

                possibeDependencies = trees[rID_].query_radius([t.startPos], r=threshold)[0]
                possibeDependencies = sorted(possibeDependencies)
                for i in possibeDependencies:
                    if i>=t.time:
                        tID_ = self.robotList[rID_].taskID+i
                        if self.radii is None or self.getDistance(t.startPos, self.taskList[tID_].goalPos) <= threshold:
                            self.graph.add_edge(tID, tID_)
                            break

class AABB_SAGE(ContinuousExecutionGraph):
    def __init__(self, allPositions=None, radii=None) -> None:
        super().__init__(allPositions, radii=radii)

        numRobots = allPositions.shape[0]
        tId = 1
        entries = []

        for rid in range(numRobots):
            prevTask = None
            for i, task in enumerate(allPositions[rid, :-1]):
                t = ContinuousTask(tId, rid, task, allPositions[rid, i+1], i)
                self.taskList[tId] = t
                tId += 1
                self.graph.add_node(t.taskID)

                if prevTask is None:
                    self.robotList.append(t)
                else:
                    self.graph.add_edge(prevTask.taskID, t.taskID)

                prevTask = t

                # Expand goalPos bbox by robot's own radius so intersection with
                # a query box expanded by r_i gives the conservative AABB check
                # for distance(start, goal) <= r_i + r_j
                r = float(self.radii[rid]) if self.radii is not None else 0.0
                gx, gy = float(t.goalPos[0]), float(t.goalPos[1])
                entries.append((t.taskID, (gx - r, gy - r, gx + r, gy + r)))

        tree = AABBTree.build(entries)

        taskQueue = [i.taskID for i in self.robotList]

        while taskQueue:
            tID = taskQueue.pop(0)
            t = self.taskList[tID]

            if t.startPos[0] == -2 and t.startPos[1] == -2:
                continue

            if (tID + 1) in self.taskList and self.taskList[tID + 1].time != 0:
                taskQueue.append(tID + 1)

            r_i = float(self.radii[t.robotID]) if self.radii is not None else 0.0
            sx, sy = float(t.startPos[0]), float(t.startPos[1])

            candidates = sorted(tree.query((sx - r_i, sy - r_i, sx + r_i, sy + r_i)))

            dependentRobots = [t.robotID]

            for tID_ in candidates:
                t_ = self.taskList[tID_]
                if t_.robotID in dependentRobots or t.time > t_.time:
                    continue

                threshold = float(self.radii[t.robotID] + self.radii[t_.robotID]) if self.radii is not None else self.THRESH
                if self.getDistance(t.startPos, t_.goalPos) <= threshold:
                    self.graph.add_edge(tID, tID_)
                    dependentRobots.append(t_.robotID)


class RTree_SAGE(ContinuousExecutionGraph):
    def __init__(self, allPositions=None, radii=None) -> None:
        super().__init__(allPositions, radii=radii)

        numRobots = allPositions.shape[0]
        tId = 1

        p = rtree_index.Property()
        p.dimension = 2
        spatial_index = rtree_index.Index(properties=p)

        for rid in range(numRobots):
            prevTask = None
            for i, task in enumerate(allPositions[rid, :-1]):
                t = ContinuousTask(tId, rid, task, allPositions[rid, i+1], i)
                self.taskList[tId] = t
                tId += 1
                self.graph.add_node(t.taskID)

                if prevTask is None:
                    self.robotList.append(t)
                else:
                    self.graph.add_edge(prevTask.taskID, t.taskID)

                prevTask = t

                r = float(self.radii[rid]) if self.radii is not None else 0.0
                gx, gy = float(t.goalPos[0]), float(t.goalPos[1])
                spatial_index.insert(t.taskID, (gx - r, gy - r, gx + r, gy + r))

        taskQueue = [i.taskID for i in self.robotList]

        while taskQueue:
            tID = taskQueue.pop(0)
            t = self.taskList[tID]

            if t.startPos[0] == -2 and t.startPos[1] == -2:
                continue

            if (tID + 1) in self.taskList and self.taskList[tID + 1].time != 0:
                taskQueue.append(tID + 1)

            r_i = float(self.radii[t.robotID]) if self.radii is not None else 0.0
            sx, sy = float(t.startPos[0]), float(t.startPos[1])

            candidates = sorted(spatial_index.intersection((sx - r_i, sy - r_i, sx + r_i, sy + r_i)))

            dependentRobots = [t.robotID]

            for tID_ in candidates:
                t_ = self.taskList[tID_]
                if t_.robotID in dependentRobots or t.time > t_.time:
                    continue

                threshold = float(self.radii[t.robotID] + self.radii[t_.robotID]) if self.radii is not None else self.THRESH
                if self.getDistance(t.startPos, t_.goalPos) <= threshold:
                    self.graph.add_edge(tID, tID_)
                    dependentRobots.append(t_.robotID)


#Helper Functions

def testTime(method, allPos, allConf=None, fname="temp.dat"):
    start = time.time()
    if(method is MAGE):
        exGraph = method(allPos, fname)
    elif method in (SAGE, Multi_KDTree_SAGE, AABB_SAGE, RTree_SAGE):
        exGraph = method(allPos, radii=allConf)
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

def jsonToRadii(data, NUM_AGENTS):
    radii = []
    for i in range(NUM_AGENTS):
        key = 'agent'+str(i)
        value = data[key]
        if isinstance(value, list):
            value = value[0]
        radii.append(float(value))
    return np.array(radii)