from continuousUtil import *
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
import os
import glob

folder_path = 'Results/ContRawData'

# Get a list of all .dat files in the folder
dat_files = glob.glob(os.path.join(folder_path, '*.dat'))

# Delete each .dat file
for file in dat_files:
    os.remove(file)
    print(f"Deleted {file}")


listOfMethods = [OriginalADG, SAGE, MAGE]

@ray.remote
def getOneEpData(NUM_AGENTS, FPS, index):
    if(index>=100):
        return
    filepath = "Results/ContRawData/"+str(NUM_AGENTS)+"_MAGE+"+str(FPS)

    print(NUM_AGENTS, FPS, index)
    if(index>=100):
        return None
    timeStore = np.zeros(len(listOfMethods))
    commsStore = np.zeros(len(listOfMethods))
    
    Fname = "scenario/paths/agents"+str(NUM_AGENTS)+"_fps"+str(FPS)

    with open(Fname+'/{}.json'.format(index), 'r') as f:
        try:
            data = json.load(f)
        except:
            print(index)
            return None
    allPos=jsonToNpy(data,NUM_AGENTS)

    for idx, val in enumerate(listOfMethods):
        commsLen, timeTaken = testTime(val, allPos, filepath+"_"+str(index)+".dat")
        timeStore[idx] = timeTaken    
        commsStore[idx] = commsLen
        
    with open(filepath+".csv", "+a") as f:
            writer = csv.writer(f)
            writer.writerows([[NUM_AGENTS, FPS, index, allPos.shape[1], "|", timeStore[0], "|", commsStore[0]]])

    try:
        os.remove(filepath+"_"+str(index)+".dat")
    except:
        pass

ray.init()
pairsLeft = [(10,10), (10,20), (10,25), (10,50), (10,100), (20,10), (20,20), (20,25), (20,50)]
allPairs = []
for (NUM_AGENTS, FPS) in pairsLeft:
    filepath = "Results/ContRawData/"+str(NUM_AGENTS)+"_MAGE+"+str(FPS)+".csv"
    existing_numbers = set()
    
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                number = int(row[2])
                existing_numbers.add(number)
            except ValueError:
                continue

    missing_numbers = [num for num in range(100) if num not in existing_numbers]
    
    for num in missing_numbers:
        allPairs.append((NUM_AGENTS, FPS, num))
np.random.shuffle(allPairs)
ray.get([getOneEpData.remote(*allPairs[i]) for i in range(len(allPairs))])
print("FINISHED")