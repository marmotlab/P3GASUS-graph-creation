from continuousUtil import *
import argparse
import csv
import glob
import os

os.environ["RAY_DEDUP_LOGS"] = "0"

import ray


def loadRadii(NUM_AGENTS, FPS):
    radii_path = f"Continuous Scenario Paths/{NUM_AGENTS}Agents_{FPS}fps_conf.json"
    
    if os.path.exists(radii_path):
        with open(radii_path, 'r') as f:
            try:
                data = json.load(f)
            except:
                print(f"Error loading radii JSON: {radii_path}")
                return None
        return jsonToRadii(data, NUM_AGENTS)
    
    print(f"Warning: no radii config found for {NUM_AGENTS} agents at {FPS} fps")
    return None

# Get a list of all .dat files in the folder
dat_files = glob.glob(os.path.join(folder_path, '*.dat'))


def parseMethodNames(value):
    if value.lower() == "all":
        return None
    return [name.strip() for name in value.split(",") if name.strip()]


def parsePairs(value):
    if value.lower() == "default":
        return DEFAULT_PAIRS

    pairs = []
    for item in value.split(","):
        if not item.strip():
            continue
        numAgents, fps = item.split(":")
        pairs.append((int(numAgents), int(fps)))

    if len(pairs) == 0:
        raise ValueError("No continuous scenario pairs selected")

    return pairs


def buildHeader(methodSpecs):
    return [
        "NUM_AGENTS",
        "FPS",
        "Index",
        "PathLength",
        "|",
        *[f"{implementation}_{name}_Time" for implementation, name, _, _ in methodSpecs],
        "|",
        *[f"{implementation}_{name}_Comms" for implementation, name, _, _ in methodSpecs],
    ]


def ensureCsvHeader(filePath, header):
    os.makedirs(os.path.dirname(filePath), exist_ok=True)

    if not os.path.exists(filePath) or os.path.getsize(filePath) == 0:
        with open(filePath, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        return

    with open(filePath, "r") as f:
        reader = csv.reader(f)
        existingHeader = next(reader, [])

    if existingHeader != header:
        raise RuntimeError(
            f"{filePath} uses a different result schema. Use a new --output-folder "
            "or move/remove the old CSV before running."
        )


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["python", "cpp", "both"], default="both")
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods, or all. Options: OriginalADG,SAGE,MAGE",
    )
    parser.add_argument("--output-folder", default="Results/ContRawData")
    parser.add_argument("--scenario-root", default="scenario/paths")
    parser.add_argument(
        "--pairs",
        default="default",
        help="default, or comma-separated NUM_AGENTS:FPS pairs such as 10:10,20:50",
    )
    parser.add_argument("--max-index", type=int, default=100)
    parser.add_argument("--num-cpu-core", type=int, default=None)
    parser.add_argument("--keep-dat", action="store_true")
    return parser.parse_args()


def cleanDatFiles(folderPath):
    datFiles = glob.glob(os.path.join(folderPath, "*.dat"))
    for filePath in datFiles:
        os.remove(filePath)
        print(f"Deleted {filePath}")


@ray.remote
def getOneEpData(
    NUM_AGENTS,
    FPS,
    index,
    implementationMode,
    methodNames,
    outputFolder,
    scenarioRoot,
    keepDat,
):
    selectedMethodSpecs = getContinuousMethodSpecs(implementationMode, methodNames)
    timeStore = np.zeros(len(selectedMethodSpecs))
    commsStore = np.zeros(len(selectedMethodSpecs))

    filepath = os.path.join(outputFolder, str(NUM_AGENTS)+"_MAGE+"+str(FPS))
    print(NUM_AGENTS, FPS, index)

    scenarioFolder = os.path.join(
        scenarioRoot,
        "agents"+str(NUM_AGENTS)+"_fps"+str(FPS),
    )

    scenarioFile = os.path.join(scenarioFolder, "{}.json".format(index))
    try:
        with open(scenarioFile, "r") as f:
            data = json.load(f)
        except:
            print(index)
            return None
    allPos=jsonToNpy(data,NUM_AGENTS)
    allRadii = loadRadii(NUM_AGENTS, FPS)

    for idx, val in enumerate(listOfMethods):
        commsLen, timeTaken = testTime(val, allPos, allRadii, filepath+"_"+str(index)+".dat")
        timeStore[idx] = timeTaken    
        commsStore[idx] = commsLen

    with open(filepath+".csv", "+a") as f:
        writer = csv.writer(f)
        writer.writerow([
            NUM_AGENTS,
            FPS,
            index,
            allPos.shape[1],
            "|",
            *timeStore,
            "|",
            *commsStore,
        ])

    if not keepDat:
        for implementation, name, _, _ in selectedMethodSpecs:
            if implementation != "Python":
                continue
            try:
                os.remove(filepath+"_"+name+"_"+str(index)+".dat")
            except FileNotFoundError:
                pass


def main():
    args = parseArgs()
    selectedMethodNames = parseMethodNames(args.methods)
    methodSpecs = getContinuousMethodSpecs(args.implementation, selectedMethodNames)
    header = buildHeader(methodSpecs)
    os.makedirs(args.output_folder, exist_ok=True)

    if not args.keep_dat:
        cleanDatFiles(args.output_folder)

    ray.init()
    pairsLeft = parsePairs(args.pairs)
    allPairs = []
    for (NUM_AGENTS, FPS) in pairsLeft:
        filepath = os.path.join(args.output_folder, str(NUM_AGENTS)+"_MAGE+"+str(FPS)+".csv")
        existingNumbers = set()
        ensureCsvHeader(filepath, header)

        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    number = int(row[2])
                    existingNumbers.add(number)
                except ValueError:
                    continue

        missingNumbers = [num for num in range(args.max_index) if num not in existingNumbers]

        for num in missingNumbers:
            allPairs.append((NUM_AGENTS, FPS, num))

    np.random.shuffle(allPairs)

    if len(allPairs) == 0:
        print("No missing scenarios to run")
    elif args.num_cpu_core is None:
        ray.get([
            getOneEpData.remote(
                *pair,
                args.implementation,
                selectedMethodNames,
                args.output_folder,
                args.scenario_root,
                args.keep_dat,
            )
            for pair in allPairs
        ])
    else:
        batchSize = min(len(allPairs), args.num_cpu_core)
        for start in range(0, len(allPairs), batchSize):
            batch = allPairs[start:start+batchSize]
            ray.get([
                getOneEpData.remote(
                    *pair,
                    args.implementation,
                    selectedMethodNames,
                    args.output_folder,
                    args.scenario_root,
                    args.keep_dat,
                )
                for pair in batch
            ])

    print("FINISHED")


if __name__ == "__main__":
    main()
