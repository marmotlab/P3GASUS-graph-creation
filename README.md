# P3GASUS (Graph Construction)

This repository provides the graph construction methods introduced in the paper **“P3GASUS: Pre-Planned Path Execution Graphs for Multi-Agent Systems at Ultra-Large Scale.”**

The core implementations are located in:

- `discreteUtil.py` — for discrete-space scenarios  
- `continuousUtil.py` — for continuous-space scenarios  

Example usage is demonstrated in the accompanying Jupyter notebooks:

- `discrete.ipynb`  
- `continuous.ipynb`

The `results/` folder contains detailed test outputs generated using:

- `generateDiscreteResults.py`  
- `generateContResults.py`

---

## Discrete Case

The discrete implementation relies on **lacam3’s Python bindings**, generated from the repository:

➡️ <https://github.com/Kei18/lacam3/tree/pybind>

These bindings work with **Python 3.11**. For other Python versions, you may need to regenerate the bindings.

All methods implemented in `discreteUtil.py` — namely `OriginalADG`, `SAGE`, `MAGE`, and `FORTED` — inherit from the `ExecutionGraph` class.

Each `ExecutionGraph` instance contains:

- `taskList`  
- `graph` (a NetworkX directed graph)  
- `robotList`

See the paper for full algorithmic details.

---

### Helper Functions

#### `oneTestCase(numAgents, freeSpacePercent)`

Generates an open, obstacle-free world populated with agents while enforcing the requested amount of free space.

- `freeSpacePercent` ranges from **1–100**.
- Returns:
  - `startPositions`
  - `actions` (all agents, all timesteps)
  - `freeSpaceUsed`

All experiments in the paper use **≥ 30% free space** to ensure lacam3 can generate paths reliably.

**Action encoding:**

- `0`: stay still  
- `1`: move East  
- `2`: move South  
- `3`: move West  
- `4`: move North  

---

#### `testTime(MethodClass, actions, starts, filename='temp.dat')`

Runs the specified graph-construction method.

- `filename` — path to a `.dat` file for storing the DP matrix if the task list becomes extremely large (typically > 50,000 tasks).

Returns:

- `graph.edges`  
- computation time  

---

### Example Usage

```python
from discreteUtil import *

actions, starts, free = oneTestCase(100, 50)
testTime(OriginalADG, actions, starts)
```

To write a graph to disk:

``` python
exGraph = FORTED(actions, starts)
exGraph.fileWrite("Debug/")
```

The constructed execution graph is available as:

``` python
exGraph.graph
```

## Continuous Case
All methods implemented here — `OriginalADG`, `SAGE`, and `MAGE` — inherit from the common `ExecutionGraph` class.

Each `ExecutionGraph` instance contains:

- `taskList`
- `graph` (a NetworkX directed graph)
- `robotList`

The folder **`Continuous Scenario Paths/`** contains sample path data in JSON format generated from [**MetaDrive**](https://github.com/metadriverse/metadrive)

---

## Helper Functions

### `jsonToNpy(data, NUM_AGENTS)`
Converts path data loaded from JSON format into a NumPy matrix suitable for further processing.

### `testTime(val, allPos)`
Runs the specified continuous-space graph construction method.

- `val` — the method class (`OriginalADG`, `SAGE`, or `MAGE`)  
- `allPos` — the positions matrix returned from `jsonToNpy`

Returns:

- communication length  
- computation time  

---

## Example Usage

```python
from continuousUtil import *

listOfMethods = [OriginalADG, SAGE, MAGE]

with open("Continuous Scenario Paths/10Agents_10fps", 'r') as f:
    try:
        data = json.load(f)
    except:
        print("Error loading JSON")

allPos = jsonToNpy(data, NUM_AGENTS=10)

for val in listOfMethods:
    commsLen, timeTaken = testTime(val, allPos)
    print(f"{val.__name__}: Time Taken - {timeTaken}, Comms Length - {commsLen}")
```

To write a graph to disk:

``` python
exGraph = SAGE(allPos)
exGraph.fileWrite("Debug/")
```

The constructed execution graph is available as:

``` python
exGraph.graph
```