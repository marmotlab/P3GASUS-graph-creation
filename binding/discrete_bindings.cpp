#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "p3gasus_discrete.hpp"

namespace py = pybind11;

template <typename T>
const char *graphClassName();

template <>
const char *graphClassName<OriginalADG>() { return "OriginalADG"; }

template <>
const char *graphClassName<SAGE>() { return "SAGE"; }

template <>
const char *graphClassName<FORTED>() { return "FORTED"; }

// ---------------------------------------------------------------------------
// Helper: convert Python 2-D list/array of actions + list of [x,y] starts
//         into C++ types.  Accepts either Python lists or numpy int arrays.
// ---------------------------------------------------------------------------
static std::vector<std::vector<int>>
pyActionsToVec(py::object taskActions)
{
    std::vector<std::vector<int>> result;
    // Works for list-of-lists and numpy 2-D arrays
    py::list outer = taskActions.cast<py::list>();
    for (auto row : outer)
    {
        std::vector<int> r;
        for (auto v : row.cast<py::list>())
            r.push_back(v.cast<int>());
        result.push_back(std::move(r));
    }
    return result;
}

static std::vector<Pos2D>
pyStartsToVec(py::object starts)
{
    std::vector<Pos2D> result;
    py::list lst = starts.cast<py::list>();
    for (auto item : lst)
    {
        py::list xy = item.cast<py::list>();
        result.push_back({xy[0].cast<int>(), xy[1].cast<int>()});
    }
    return result;
}

// ---------------------------------------------------------------------------
// Thin wrapper so pybind11 can own the object
// ---------------------------------------------------------------------------
template <typename T>
struct GraphWrapper
{
    T graph;

    GraphWrapper(py::object taskActions, py::object startPositions)
        : graph(pyActionsToVec(taskActions), pyStartsToVec(startPositions)) {}

    // edges as list of (u,v) tuples
    std::vector<std::pair<int, int>> edges() const { return graph.graph.edges(); }
    std::size_t numEdges() const { return graph.graph.numEdges(); }
    std::size_t numNodes() const { return graph.graph.numNodes(); }

    // out-neighbors of a node
    std::vector<int> outNeighbors(int n) const
    {
        const auto &s = graph.graph.outNeighbors(n);
        return std::vector<int>(s.begin(), s.end());
    }
    std::vector<int> inNeighbors(int n) const
    {
        const auto &s = graph.graph.inNeighbors(n);
        return std::vector<int>(s.begin(), s.end());
    }

    bool hasEdge(int u, int v) const { return graph.graph.hasEdge(u, v); }
    std::vector<int> nodes() const { return graph.graph.nodes(); }

    // taskList as Python dict {id: dict-of-fields}
    py::dict taskListPy() const
    {
        py::dict d;
        for (auto &[tid, t] : graph.taskList)
        {
            py::dict td;
            td["taskID"] = t.taskID;
            td["robotID"] = t.robotID;
            td["action"] = t.action;
            td["time"] = t.time;
            td["startPos"] = py::make_tuple(t.startPos.x, t.startPos.y);
            td["goalPos"] = py::make_tuple(t.goalPos.x, t.goalPos.y);
            d[py::int_(tid)] = td;
        }
        return d;
    }

    // robotList (first task per robot) as list of dicts
    py::list robotListPy() const
    {
        py::list l;
        for (auto &t : graph.robotList)
        {
            py::dict td;
            td["taskID"] = t.taskID;
            td["robotID"] = t.robotID;
            td["action"] = t.action;
            td["time"] = t.time;
            td["startPos"] = py::make_tuple(t.startPos.x, t.startPos.y);
            td["goalPos"] = py::make_tuple(t.goalPos.x, t.goalPos.y);
            l.append(td);
        }
        return l;
    }

    long long countType2Edges() const { return graph.countType2Edges(); }

    void fileWrite(const std::string &path) const
    {
        graph.fileWrite(path, graphClassName<T>());
    }

    std::string repr() const
    {
        return std::string(graphClassName<T>()) +
               " nodes=" + std::to_string(numNodes()) +
               " edges=" + std::to_string(numEdges());
    }
};

// ---------------------------------------------------------------------------
// MAGE wrapper (extra baseType arg)
// ---------------------------------------------------------------------------
struct MAGEWrapper
{
    MAGE graph;

    MAGEWrapper(py::object taskActions, py::object startPositions,
                int baseType = 0, const std::string &filename = "temp.dat")
        : graph(pyActionsToVec(taskActions), pyStartsToVec(startPositions),
                static_cast<MAGE::BaseType>(baseType), filename) {}

    std::vector<std::pair<int, int>> edges() const { return graph.graph.edges(); }
    std::size_t numEdges() const { return graph.graph.numEdges(); }
    std::size_t numNodes() const { return graph.graph.numNodes(); }
    std::vector<int> outNeighbors(int n) const
    {
        const auto &s = graph.graph.outNeighbors(n);
        return {s.begin(), s.end()};
    }
    std::vector<int> inNeighbors(int n) const
    {
        const auto &s = graph.graph.inNeighbors(n);
        return {s.begin(), s.end()};
    }
    bool hasEdge(int u, int v) const { return graph.graph.hasEdge(u, v); }
    std::vector<int> nodes() const { return graph.graph.nodes(); }
    long long countType2Edges() const { return graph.countType2Edges(); }
    py::dict taskListPy() const
    {
        py::dict d;
        for (auto &[tid, t] : graph.taskList)
        {
            py::dict td;
            td["taskID"] = t.taskID;
            td["robotID"] = t.robotID;
            td["action"] = t.action;
            td["time"] = t.time;
            td["startPos"] = py::make_tuple(t.startPos.x, t.startPos.y);
            td["goalPos"] = py::make_tuple(t.goalPos.x, t.goalPos.y);
            d[py::int_(tid)] = td;
        }
        return d;
    }
    py::list robotListPy() const
    {
        py::list l;
        for (auto &t : graph.robotList)
        {
            py::dict td;
            td["taskID"] = t.taskID;
            td["robotID"] = t.robotID;
            td["action"] = t.action;
            td["time"] = t.time;
            td["startPos"] = py::make_tuple(t.startPos.x, t.startPos.y);
            td["goalPos"] = py::make_tuple(t.goalPos.x, t.goalPos.y);
            l.append(td);
        }
        return l;
    }
    void fileWrite(const std::string &path) const
    {
        graph.fileWrite(path, "MAGE");
    }
    std::string repr() const
    {
        return "MAGE nodes=" + std::to_string(numNodes()) +
               " edges=" + std::to_string(numEdges());
    }
};

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(p3gasus_discrete_cpp, m)
{
    m.doc() = "C++ ADG implementation: OriginalADG, SAGE, FORTED, MAGE";

    // --- OriginalADG ---
    py::class_<GraphWrapper<OriginalADG>>(m, "OriginalADG")
        .def(py::init<py::object, py::object>(),
             py::arg("taskActions"), py::arg("startPositions"),
             "Build original Action Dependency Graph.\n"
             "taskActions: list-of-lists [numRobots][numTasks], values 0-4\n"
             "startPositions: list of [x,y] for each robot")
        .def("edges", &GraphWrapper<OriginalADG>::edges)
        .def("num_edges", &GraphWrapper<OriginalADG>::numEdges)
        .def("num_nodes", &GraphWrapper<OriginalADG>::numNodes)
        .def("out_neighbors", &GraphWrapper<OriginalADG>::outNeighbors)
        .def("in_neighbors", &GraphWrapper<OriginalADG>::inNeighbors)
        .def("has_edge", &GraphWrapper<OriginalADG>::hasEdge)
        .def("nodes", &GraphWrapper<OriginalADG>::nodes)
        .def("task_list", &GraphWrapper<OriginalADG>::taskListPy)
        .def("robot_list", &GraphWrapper<OriginalADG>::robotListPy)
        .def("count_type2_edges", &GraphWrapper<OriginalADG>::countType2Edges)
        .def("file_write", &GraphWrapper<OriginalADG>::fileWrite)
        .def("__repr__", &GraphWrapper<OriginalADG>::repr);

    // --- SAGE ---
    py::class_<GraphWrapper<SAGE>>(m, "SAGE")
        .def(py::init<py::object, py::object>(),
             py::arg("taskActions"), py::arg("startPositions"))
        .def("edges", &GraphWrapper<SAGE>::edges)
        .def("num_edges", &GraphWrapper<SAGE>::numEdges)
        .def("num_nodes", &GraphWrapper<SAGE>::numNodes)
        .def("out_neighbors", &GraphWrapper<SAGE>::outNeighbors)
        .def("in_neighbors", &GraphWrapper<SAGE>::inNeighbors)
        .def("has_edge", &GraphWrapper<SAGE>::hasEdge)
        .def("nodes", &GraphWrapper<SAGE>::nodes)
        .def("task_list", &GraphWrapper<SAGE>::taskListPy)
        .def("robot_list", &GraphWrapper<SAGE>::robotListPy)
        .def("count_type2_edges", &GraphWrapper<SAGE>::countType2Edges)
        .def("file_write", &GraphWrapper<SAGE>::fileWrite)
        .def("__repr__", &GraphWrapper<SAGE>::repr);

    // --- FORTED ---
    py::class_<GraphWrapper<FORTED>>(m, "FORTED")
        .def(py::init<py::object, py::object>(),
             py::arg("taskActions"), py::arg("startPositions"))
        .def("edges", &GraphWrapper<FORTED>::edges)
        .def("num_edges", &GraphWrapper<FORTED>::numEdges)
        .def("num_nodes", &GraphWrapper<FORTED>::numNodes)
        .def("out_neighbors", &GraphWrapper<FORTED>::outNeighbors)
        .def("in_neighbors", &GraphWrapper<FORTED>::inNeighbors)
        .def("has_edge", &GraphWrapper<FORTED>::hasEdge)
        .def("nodes", &GraphWrapper<FORTED>::nodes)
        .def("task_list", &GraphWrapper<FORTED>::taskListPy)
        .def("robot_list", &GraphWrapper<FORTED>::robotListPy)
        .def("count_type2_edges", &GraphWrapper<FORTED>::countType2Edges)
        .def("file_write", &GraphWrapper<FORTED>::fileWrite)
        .def("__repr__", &GraphWrapper<FORTED>::repr);

    // --- MAGE ---
    py::enum_<MAGE::BaseType>(m, "BaseADGType")
        .value("BASE_FORTED", MAGE::BASE_FORTED)
        .value("BASE_SAGE", MAGE::BASE_SAGE)
        .value("BASE_ORIGINAL", MAGE::BASE_ORIGINAL);

    py::class_<MAGEWrapper>(m, "MAGE")
        .def(py::init<py::object, py::object, int, const std::string &>(),
             py::arg("taskActions"), py::arg("startPositions"),
             py::arg("base_type") = 0,
             py::arg("filename") = "temp.dat",
             "MAGE with transitive reduction.\n"
             "base_type: 0=FORTED (default), 1=SAGE, 2=OriginalADG\n"
             "filename: backing file used when the DP matrix exceeds 50000 rows")
        .def("edges", &MAGEWrapper::edges)
        .def("num_edges", &MAGEWrapper::numEdges)
        .def("num_nodes", &MAGEWrapper::numNodes)
        .def("out_neighbors", &MAGEWrapper::outNeighbors)
        .def("in_neighbors", &MAGEWrapper::inNeighbors)
        .def("has_edge", &MAGEWrapper::hasEdge)
        .def("nodes", &MAGEWrapper::nodes)
        .def("task_list", &MAGEWrapper::taskListPy)
        .def("robot_list", &MAGEWrapper::robotListPy)
        .def("count_type2_edges", &MAGEWrapper::countType2Edges)
        .def("file_write", &MAGEWrapper::fileWrite)
        .def("__repr__", &MAGEWrapper::repr);

    // --- Utility ---
    m.def("get_action_from_pos", [](py::tuple cur, py::tuple nxt)
          {
              Pos2D c{cur[0].cast<int>(), cur[1].cast<int>()};
              Pos2D n{nxt[0].cast<int>(), nxt[1].cast<int>()};
              return getActionFromPos(c, n); }, "Return action int (0-4) from current to next position");
}
