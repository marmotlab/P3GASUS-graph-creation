from discreteUtil import *
import csv
import ray
import os

numCPUcore = 16
listOfMethods = [OriginalADG, SAGE, FORTED, MAGE]

File = "Results/discrete.csv"
if not os.path.exists(File):
    with open(File, "w") as f:
        writer = csv.writer(f)
        writer.writerows([["NUM_AGENTS", "PercentSpaceEmpty", "|", "ADG_Time", "SAGE_TIME", "FORTED_TIME", "MAGE_TIME", "|", "ADG_Comms", "SAGE_Comms", "FORTED_Comms", "MAGE_Comms"]])

repeatLen = 10

@ray.remote
def getOneEpData(NUM_AGENTS):
    timeStore = np.zeros(len(listOfMethods))
    commsStore = np.zeros(len(listOfMethods))
    

    ACTIONS,STARTS, freeCellsPercent = oneTestCase(NUM_AGENTS, minFreeCellPercentToMaintain=40)
    
    for idx, val in enumerate(listOfMethods):
        commsLen, timeTaken = testTime(val, ACTIONS, STARTS)
        timeStore[idx] = timeTaken    
        commsStore[idx] = commsLen

    return timeStore, commsStore

ray.init()
for NUM_AGENTS in range(10,50, 10):
    succ = 0
    timeStore = np.zeros((repeatLen, len(listOfMethods)))
    commsStore = np.zeros((repeatLen, len(listOfMethods)))
    
    while succ<repeatLen:
        results = ray.get([getOneEpData.remote(NUM_AGENTS) for _ in range(numCPUcore)])
        for i in results:
            if not(succ<repeatLen):
                break
            if i is not None:
                timeTaken, commsLen = i
                timeStore[succ] = timeTaken    
                commsStore[succ] = commsLen
                succ+=1
                # print(f"NUM_AGENTS: {NUM_AGENTS}, succ: {succ}, timeTaken: {timeTaken}, commsLen: {commsLen}")

    with open(File, "+a") as f:
        writer = csv.writer(f)
        writer.writerows([[NUM_AGENTS, getWorldSize(NUM_AGENTS)[1], "|", np.mean(timeStore[:,0]), np.mean(timeStore[:,1]), np.mean(timeStore[:,2]), np.mean(timeStore[:,3]), "|", np.mean(commsStore[:,0]), np.mean(commsStore[:,1]), np.mean(commsStore[:,2]), np.mean(commsStore[:,3])]])
