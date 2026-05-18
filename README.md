# P3GASUS (Graph Construction)

This repository provides the graph construction methods introduced in the paper **“P3GASUS: Pre-Planned Path Execution Graphs for Multi-Agent Systems at Ultra-Large Scale.”**

The core implementations are located in:

- `discreteUtil.py` — for discrete-space scenarios  
- `continuousUtil.py` — for continuous-space scenarios  
- `binding/` — C++/pybind11 backends for faster graph construction

Example usage is demonstrated in the accompanying Jupyter notebooks:

- `discrete.ipynb`  
- `continuous.ipynb`

The `Results/` folder contains detailed test outputs generated using:

- `generateDiscreteResults.py`  
- `generateContResults.py`

---

## C++ Python Bindings

The repository includes pybind11 bindings for both graph-construction domains:

- `p3gasus_discrete_cpp` — discrete `OriginalADG`, `SAGE`, `FORTED`, and `MAGE`
- `p3gasus_continuous_cpp` — continuous `OriginalADG`, `SAGE`, and `MAGE`

Build the bindings from the `binding/` directory:

```bash
cd binding
python setup.py build_ext --inplace
```

The notebooks use the Python 3.11 `py11` environment. To build for that environment:

```bash
cd binding
conda run -n py11 python setup.py build_ext --inplace
```

The bindings can then be imported from the repository root:

```python
import sys
sys.path.insert(0, "binding")

import p3gasus_discrete_cpp as dcpp
import p3gasus_continuous_cpp as ccpp
```

Both binding modules expose a lightweight graph API:

- `edges()`
- `num_edges()`
- `num_nodes()`
- `nodes()`
- `out_neighbors(node)`
- `in_neighbors(node)`
- `has_edge(u, v)`
- `task_list()`
- `robot_list()`
- `count_type2_edges()`
- `file_write(path)`

The continuous binding also exposes `threshold()`.

Smoke/parity tests are available:

```bash
python binding/test_p3gasus_cpp.py
python binding/test_p3gasus_continuous_cpp.py
```

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

### Discrete C++ Binding

The discrete C++ binding mirrors the Python graph constructors:

```python
import numpy as np
import sys

from discreteUtil import oneTestCase, testTime, OriginalADG, SAGE, FORTED, MAGE

sys.path.insert(0, "binding")
import p3gasus_discrete_cpp as p3cpp

actions, starts, free = oneTestCase(100, 50)

# oneTestCase returns actions as a float NumPy array, so cast before C++ calls.
actions_cpp = np.asarray(actions, dtype=np.int64)
starts_cpp = np.asarray(starts, dtype=np.int64)

graph = p3cpp.FORTED(actions_cpp, starts_cpp)
print(graph.num_edges())
graph.file_write("Debug/")
```

`MAGE` accepts a base graph type:

```python
mage_forted = p3cpp.MAGE(actions_cpp, starts_cpp, p3cpp.BaseADGType.BASE_FORTED)
mage_sage = p3cpp.MAGE(actions_cpp, starts_cpp, p3cpp.BaseADGType.BASE_SAGE)
mage_original = p3cpp.MAGE(actions_cpp, starts_cpp, p3cpp.BaseADGType.BASE_ORIGINAL)
```

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

The `discrete.ipynb` notebook contains a single comparison table cell that reports method name, edge count, and runtime for both Python and C++ implementations.

## Continuous Case

All methods implemented here — `OriginalADG`, `SAGE`, and `MAGE` — inherit from the common `ContinuousExecutionGraph` class.

Each `ContinuousExecutionGraph` instance contains:

- `taskList`
- `graph` (a NetworkX directed graph)
- `robotList`

The folder **`Continuous Scenario Paths/`** contains sample path data in JSON format generated from [**MetaDrive**](https://github.com/metadriverse/metadrive)

---

### Continuous C++ Binding

The continuous C++ binding accepts the same position tensor produced by `jsonToNpy`:

```python
import sys

from continuousUtil import jsonToNpy

sys.path.insert(0, "binding")
import p3gasus_continuous_cpp as p3cpp

allPos = jsonToNpy(data, NUM_AGENTS=10)

graph = p3cpp.SAGE(allPos)
print(graph.num_edges(), graph.threshold())
graph.file_write("Debug/")
```

The continuous binding includes:

```python
p3cpp.OriginalADG(allPos)
p3cpp.SAGE(allPos)
p3cpp.MAGE(allPos)
```

The `continuous.ipynb` notebook contains a single comparison table cell that reports method name, edge count, and runtime for both Python and C++ implementations.

### Helper Functions

#### `jsonToNpy(data, NUM_AGENTS)`
Converts path data loaded from JSON format into a NumPy matrix suitable for further processing.

#### `testTime(val, allPos)`
Runs the specified continuous-space graph construction method.

- `val` — the method class (`OriginalADG`, `SAGE`, or `MAGE`)  
- `allPos` — the positions matrix returned from `jsonToNpy`

Returns:

- communication length  
- computation time  

---

### Example Usage

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
