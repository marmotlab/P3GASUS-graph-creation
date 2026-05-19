from discreteUtil import *
import argparse
import csv
import os

import ray


def parseMethodNames(value):
    if value.lower() == "all":
        return None
    return [name.strip() for name in value.split(",") if name.strip()]


def buildHeader(methodSpecs):
    return [
        "NUM_AGENTS",
        "PercentSpaceEmpty",
        "|",
        *[f"{implementation}_{name}_Time" for implementation, name, _, _ in methodSpecs],
        "|",
        *[f"{implementation}_{name}_Comms" for implementation, name, _, _ in methodSpecs],
    ]


def ensureCsvHeader(filePath, header):
    outputDir = os.path.dirname(filePath)
    if outputDir:
        os.makedirs(outputDir, exist_ok=True)

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
            f"{filePath} uses a different result schema. Use a new --output path "
            "or move/remove the old CSV before running."
        )


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["python", "cpp", "both"], default="both")
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods, or all. Options: OriginalADG,SAGE,FORTED,MAGE_FORTED,MAGE_SAGE",
    )
    parser.add_argument("--output", default="Results/discrete.csv")
    parser.add_argument("--num-cpu-core", type=int, default=16)
    parser.add_argument("--repeat-len", type=int, default=10)
    parser.add_argument("--agent-start", type=int, default=10)
    parser.add_argument("--agent-stop", type=int, default=50)
    parser.add_argument("--agent-step", type=int, default=10)
    parser.add_argument("--min-free-cell-percent", type=int, default=40)
    return parser.parse_args()


@ray.remote
def getOneEpData(NUM_AGENTS, minFreeCellPercentToMaintain, implementationMode, methodNames):
    selectedMethodSpecs = getDiscreteMethodSpecs(implementationMode, methodNames)
    timeStore = np.zeros(len(selectedMethodSpecs))
    commsStore = np.zeros(len(selectedMethodSpecs))

    try:
        ACTIONS, STARTS, freeCellsPercent = oneTestCase(
            NUM_AGENTS,
            minFreeCellPercentToMaintain=minFreeCellPercentToMaintain,
        )
    except Exception as exc:
        print(f"Skipping failed testcase for {NUM_AGENTS} agents: {exc}")
        return None

    for idx, (implementation, _, method, kwargs) in enumerate(selectedMethodSpecs):
        commsLen, timeTaken = testDiscreteMethod(
            implementation,
            method,
            ACTIONS,
            STARTS,
            **kwargs,
        )
        timeStore[idx] = timeTaken
        commsStore[idx] = commsLen

    return timeStore, commsStore, freeCellsPercent


def main():
    args = parseArgs()
    selectedMethodNames = parseMethodNames(args.methods)
    methodSpecs = getDiscreteMethodSpecs(args.implementation, selectedMethodNames)
    ensureCsvHeader(args.output, buildHeader(methodSpecs))

    ray.init()
    for NUM_AGENTS in range(args.agent_start, args.agent_stop, args.agent_step):
        succ = 0
        timeStore = np.zeros((args.repeat_len, len(methodSpecs)))
        commsStore = np.zeros((args.repeat_len, len(methodSpecs)))
        freeCellsStore = np.zeros(args.repeat_len)

        while succ < args.repeat_len:
            results = ray.get([
                getOneEpData.remote(
                    NUM_AGENTS,
                    args.min_free_cell_percent,
                    args.implementation,
                    selectedMethodNames,
                )
                for _ in range(args.num_cpu_core)
            ])

            for result in results:
                if not (succ < args.repeat_len):
                    break
                if result is not None:
                    timeTaken, commsLen, freeCellsPercent = result
                    timeStore[succ] = timeTaken
                    commsStore[succ] = commsLen
                    freeCellsStore[succ] = freeCellsPercent
                    succ += 1

        with open(args.output, "+a") as f:
            writer = csv.writer(f)
            writer.writerow([
                NUM_AGENTS,
                np.mean(freeCellsStore),
                "|",
                *np.mean(timeStore, axis=0),
                "|",
                *np.mean(commsStore, axis=0),
            ])


if __name__ == "__main__":
    main()
