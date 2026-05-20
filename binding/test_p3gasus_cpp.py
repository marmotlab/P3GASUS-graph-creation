"""Smoke/parity tests for the discrete p3gasus C++ pybind module.

Run from the repository root:

    python binding/test_p3gasus_cpp.py

The lacam/cv2-backed map generation helpers are not needed here, so this file
stubs those imports before loading discreteUtil.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "binding"))

# discreteUtil imports these for scenario generation, but the graph classes do
# not need them for deterministic parity tests.
sys.modules.setdefault("lacam", types.SimpleNamespace(solve=lambda *args, **kwargs: None))
sys.modules.setdefault("mapUtil", types.SimpleNamespace())
sys.modules.setdefault("matplotlib", types.SimpleNamespace(pyplot=types.SimpleNamespace()))
sys.modules.setdefault("matplotlib.pyplot", types.SimpleNamespace())

import p3gasus_discrete_cpp as cpp  # noqa: E402
from discreteUtil import FORTED, MAGE, OriginalADG, SAGE  # noqa: E402


CASES = [
    ("swap-and-wait", [[1, 0, 3], [3, 0, 1]], [[0, 0], [1, 0]]),
    (
        "three-agent-cycle",
        [[1, 2, 0, 3], [2, 3, 4, 0], [0, 1, 4, 3]],
        [[0, 0], [1, 0], [0, 1]],
    ),
    (
        "mixed",
        [[0, 1, 1, 4, 3], [2, 0, 3, 1, 4], [4, 4, 1, 0, 2]],
        [[2, 2], [3, 3], [1, 4]],
    ),
]


def py_edges(graph):
    return {tuple(edge) for edge in graph.graph.edges}


def cpp_edges(graph):
    return {tuple(edge) for edge in graph.edges()}


def py_in_neighbors(graph):
    return {
        node: sorted(source for source, _ in graph.graph.in_edges(node))
        for node in graph.graph.nodes
    }


def cpp_in_neighbors(graph):
    return {node: sorted(graph.in_neighbors(node)) for node in graph.nodes()}


def assert_same_graph(name, py_ctor, cpp_ctor, actions, starts):
    py_graph = py_ctor(actions, starts)
    cpp_graph = cpp_ctor(actions, starts)

    assert cpp_graph.num_nodes() == len(py_graph.graph.nodes), f"{name}: node count"
    assert cpp_edges(cpp_graph) == py_edges(py_graph), (
        f"{name}: edge mismatch",
        sorted(cpp_edges(cpp_graph) ^ py_edges(py_graph)),
    )
    assert cpp_graph.num_edges() == len(py_graph.graph.edges), f"{name}: edge count"
    assert len(cpp_graph.task_list()) == len(py_graph.taskList), f"{name}: task_list"
    assert len(cpp_graph.robot_list()) == len(py_graph.robotList), f"{name}: robot_list"
    assert cpp_in_neighbors(cpp_graph) == py_in_neighbors(py_graph), f"{name}: in_neighbors"


def test_parity_with_python_implementations():
    for case_name, actions, starts in CASES:
        assert_same_graph(
            f"{case_name}:OriginalADG", OriginalADG, cpp.OriginalADG, actions, starts
        )
        assert_same_graph(f"{case_name}:SAGE", SAGE, cpp.SAGE, actions, starts)
        assert_same_graph(f"{case_name}:FORTED", FORTED, cpp.FORTED, actions, starts)
        assert_same_graph(
            f"{case_name}:MAGE/FORTED",
            lambda a, s: MAGE(a, s, baseADG=FORTED),
            lambda a, s: cpp.MAGE(a, s, cpp.BaseADGType.BASE_FORTED),
            actions,
            starts,
        )
        assert_same_graph(
            f"{case_name}:MAGE/SAGE",
            lambda a, s: MAGE(a, s, baseADG=SAGE),
            lambda a, s: cpp.MAGE(a, s, cpp.BaseADGType.BASE_SAGE),
            actions,
            starts,
        )
        assert_same_graph(
            f"{case_name}:MAGE/OriginalADG",
            lambda a, s: MAGE(a, s, baseADG=OriginalADG),
            lambda a, s: cpp.MAGE(a, s, cpp.BaseADGType.BASE_ORIGINAL),
            actions,
            starts,
        )
        print(f"{case_name}: parity ok")


def test_mage_accepts_explicit_filename():
    actions = np.array(CASES[0][1], dtype=np.int64)
    starts = np.array(CASES[0][2], dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmpdir:
        graph = cpp.MAGE(
            actions,
            starts,
            cpp.BaseADGType.BASE_FORTED,
            os.path.join(tmpdir, "mage_dp.dat"),
        )
        assert graph.num_nodes() == 6
        assert cpp_edges(graph) == py_edges(MAGE(actions, starts, baseADG=FORTED))

    print("MAGE explicit filename ok")


def test_binding_api_smoke():
    actions = np.array(CASES[0][1], dtype=np.int64)
    starts = np.array(CASES[0][2], dtype=np.int64)
    graph = cpp.FORTED(actions, starts)

    assert graph.num_nodes() == 6
    assert graph.num_edges() == 8
    assert graph.has_edge(1, 2)
    assert 2 in graph.out_neighbors(1)
    assert 1 in graph.in_neighbors(2)
    assert cpp.get_action_from_pos((0, 0), (0, -1)) == 4

    with tempfile.TemporaryDirectory() as tmpdir:
        graph.file_write(tmpdir)
        names = sorted(os.listdir(tmpdir))
        assert names == ["FORTED_Graph.txt", "FORTED_TaskList.txt"], names

        cpp.OriginalADG(actions, starts).file_write(tmpdir)
        names = sorted(name for name in os.listdir(tmpdir) if name.startswith("OriginalADG"))
        assert names == ["OriginalADG_Graph.txt", "OriginalADG_TaskList.txt"], names

    print("binding API smoke ok")


if __name__ == "__main__":
    test_parity_with_python_implementations()
    test_mage_accepts_explicit_filename()
    test_binding_api_smoke()
