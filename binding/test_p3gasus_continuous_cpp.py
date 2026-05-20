"""Smoke/parity tests for the continuous p3gasus C++ pybind module.

Run from the repository root:

    python binding/test_p3gasus_continuous_cpp.py
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

sys.modules.setdefault("matplotlib", types.SimpleNamespace(pyplot=types.SimpleNamespace()))
sys.modules.setdefault("matplotlib.pyplot", types.SimpleNamespace())

import p3gasus_continuous_cpp as cpp  # noqa: E402
from continuousUtil import MAGE, OriginalADG, SAGE  # noqa: E402


CASES = [
    (
        "crossing",
        np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                [[3.0, 0.2], [2.0, 0.2], [1.0, 0.2], [0.0, 0.2]],
            ],
            dtype=float,
        ),
    ),
    (
        "three-agent-with-padding",
        np.array(
            [
                [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
                [[1.5, 0.1], [1.0, 0.1], [0.5, 0.1], [0.0, 0.1]],
                [[3.0, 3.0], [3.1, 3.0], [-2.0, -2.0], [-2.0, -2.0]],
            ],
            dtype=float,
        ),
    ),
    (
        "offset-grid",
        np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                [[0.2, 1.0], [0.2, 0.0], [1.2, 0.0]],
                [[2.0, 0.0], [2.0, 1.0], [1.0, 1.2]],
            ],
            dtype=float,
        ),
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


def assert_same_graph(name, py_ctor, cpp_ctor, positions):
    py_graph = py_ctor(positions)
    cpp_graph = cpp_ctor(positions)

    assert cpp_graph.num_nodes() == len(py_graph.graph.nodes), f"{name}: node count"
    assert cpp_edges(cpp_graph) == py_edges(py_graph), (
        f"{name}: edge mismatch",
        sorted(cpp_edges(cpp_graph) ^ py_edges(py_graph)),
    )
    assert cpp_graph.num_edges() == len(py_graph.graph.edges), f"{name}: edge count"
    assert len(cpp_graph.task_list()) == len(py_graph.taskList), f"{name}: task_list"
    assert len(cpp_graph.robot_list()) == len(py_graph.robotList), f"{name}: robot_list"
    assert np.isclose(cpp_graph.threshold(), py_graph.THRESH), f"{name}: threshold"
    assert cpp_in_neighbors(cpp_graph) == py_in_neighbors(py_graph), f"{name}: in_neighbors"


def test_parity_with_python_implementations():
    for case_name, positions in CASES:
        assert_same_graph(f"{case_name}:OriginalADG", OriginalADG, cpp.OriginalADG, positions)
        assert_same_graph(f"{case_name}:SAGE", SAGE, cpp.SAGE, positions)
        assert_same_graph(f"{case_name}:MAGE", MAGE, cpp.MAGE, positions)
        print(f"{case_name}: parity ok")


def test_mage_accepts_explicit_filename():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph = cpp.MAGE(CASES[0][1], os.path.join(tmpdir, "mage_dp.dat"))
        assert graph.num_nodes() == 6
        assert cpp_edges(graph) == py_edges(MAGE(CASES[0][1]))

    print("continuous MAGE explicit filename ok")


def test_binding_api_smoke():
    graph = cpp.SAGE(CASES[0][1])
    assert graph.num_nodes() == 6
    assert graph.num_edges() >= 4
    assert graph.has_edge(1, 2)
    assert 2 in graph.out_neighbors(1)
    assert 1 in graph.in_neighbors(2)
    assert len(graph.task_list()) == 6
    assert len(graph.robot_list()) == 2

    with tempfile.TemporaryDirectory() as tmpdir:
        graph.file_write(tmpdir)
        names = sorted(os.listdir(tmpdir))
        assert names == ["SAGE_Graph.txt", "SAGE_TaskList.txt"], names

    print("continuous binding API smoke ok")


if __name__ == "__main__":
    test_parity_with_python_implementations()
    test_mage_accepts_explicit_filename()
    test_binding_api_smoke()
