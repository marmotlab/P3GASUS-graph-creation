#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <type_traits>

#include "p3gasus_continuous.hpp"

namespace py = pybind11;
namespace pc = p3gasus_continuous;

static pc::PositionMatrix pyPositionsToVec(py::object positionsObj)
{
    py::array_t<double, py::array::c_style | py::array::forcecast> positions(positionsObj);
    if (positions.ndim() != 3 || positions.shape(2) != 2)
        throw std::invalid_argument("positions must have shape [numRobots, numSteps, 2]");

    const auto arr = positions.unchecked<3>();
    pc::PositionMatrix result(
        static_cast<std::size_t>(arr.shape(0)),
        std::vector<pc::Pos2F>(static_cast<std::size_t>(arr.shape(1))));

    for (py::ssize_t r = 0; r < arr.shape(0); ++r)
        for (py::ssize_t t = 0; t < arr.shape(1); ++t)
            result[static_cast<std::size_t>(r)][static_cast<std::size_t>(t)] = {arr(r, t, 0), arr(r, t, 1)};

    return result;
}

template <typename T>
struct ContinuousGraphWrapper
{
    T graph;

    explicit ContinuousGraphWrapper(py::object positions)
        : graph(pyPositionsToVec(positions)) {}

    ContinuousGraphWrapper(py::object positions, const std::string &filename)
        : graph(makeGraph(positions, filename)) {}

    std::vector<std::pair<int, int>> edges() const { return graph.graph.edges(); }
    std::size_t numEdges() const { return graph.graph.numEdges(); }
    std::size_t numNodes() const { return graph.graph.numNodes(); }
    double threshold() const { return graph.THRESH; }
    long long countType2Edges() const { return graph.countType2Edges(); }
    bool hasEdge(int u, int v) const { return graph.graph.hasEdge(u, v); }
    std::vector<int> nodes() const { return graph.graph.nodes(); }

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

    py::dict taskListPy() const
    {
        py::dict d;
        for (const auto &[tid, task] : graph.taskList)
        {
            py::dict td;
            td["taskID"] = task.taskID;
            td["robotID"] = task.robotID;
            td["time"] = task.time;
            td["startPos"] = py::make_tuple(task.startPos.x, task.startPos.y);
            td["goalPos"] = py::make_tuple(task.goalPos.x, task.goalPos.y);
            d[py::int_(tid)] = td;
        }
        return d;
    }

    py::list robotListPy() const
    {
        py::list l;
        for (const auto &task : graph.robotList)
        {
            py::dict td;
            td["taskID"] = task.taskID;
            td["robotID"] = task.robotID;
            td["time"] = task.time;
            td["startPos"] = py::make_tuple(task.startPos.x, task.startPos.y);
            td["goalPos"] = py::make_tuple(task.goalPos.x, task.goalPos.y);
            l.append(td);
        }
        return l;
    }

    void fileWrite(const std::string &path, const std::string &name) const
    {
        graph.fileWrite(path, name);
    }

    std::string repr(const std::string &name) const
    {
        return name + " nodes=" + std::to_string(numNodes()) +
               " edges=" + std::to_string(numEdges()) +
               " threshold=" + std::to_string(threshold());
    }

private:
    static T makeGraph(py::object positions, const std::string &filename)
    {
        if constexpr (std::is_same_v<T, pc::MAGE>)
            return T(pyPositionsToVec(positions), filename);
        else
            return T(pyPositionsToVec(positions));
    }
};

PYBIND11_MODULE(p3gasus_continuous_cpp, m)
{
    m.doc() = "C++ continuous-space ADG implementation: OriginalADG, SAGE, MAGE";

    py::class_<ContinuousGraphWrapper<pc::OriginalADG>>(m, "OriginalADG")
        .def(py::init<py::object>(), py::arg("positions"))
        .def("edges", &ContinuousGraphWrapper<pc::OriginalADG>::edges)
        .def("num_edges", &ContinuousGraphWrapper<pc::OriginalADG>::numEdges)
        .def("num_nodes", &ContinuousGraphWrapper<pc::OriginalADG>::numNodes)
        .def("threshold", &ContinuousGraphWrapper<pc::OriginalADG>::threshold)
        .def("count_type2_edges", &ContinuousGraphWrapper<pc::OriginalADG>::countType2Edges)
        .def("out_neighbors", &ContinuousGraphWrapper<pc::OriginalADG>::outNeighbors)
        .def("in_neighbors", &ContinuousGraphWrapper<pc::OriginalADG>::inNeighbors)
        .def("has_edge", &ContinuousGraphWrapper<pc::OriginalADG>::hasEdge)
        .def("nodes", &ContinuousGraphWrapper<pc::OriginalADG>::nodes)
        .def("task_list", &ContinuousGraphWrapper<pc::OriginalADG>::taskListPy)
        .def("robot_list", &ContinuousGraphWrapper<pc::OriginalADG>::robotListPy)
        .def("file_write", [](const ContinuousGraphWrapper<pc::OriginalADG> &self, const std::string &path)
             { self.fileWrite(path, "OriginalADG"); })
        .def("__repr__", [](const ContinuousGraphWrapper<pc::OriginalADG> &self)
             { return self.repr("OriginalADG"); });

    py::class_<ContinuousGraphWrapper<pc::SAGE>>(m, "SAGE")
        .def(py::init<py::object>(), py::arg("positions"))
        .def("edges", &ContinuousGraphWrapper<pc::SAGE>::edges)
        .def("num_edges", &ContinuousGraphWrapper<pc::SAGE>::numEdges)
        .def("num_nodes", &ContinuousGraphWrapper<pc::SAGE>::numNodes)
        .def("threshold", &ContinuousGraphWrapper<pc::SAGE>::threshold)
        .def("count_type2_edges", &ContinuousGraphWrapper<pc::SAGE>::countType2Edges)
        .def("out_neighbors", &ContinuousGraphWrapper<pc::SAGE>::outNeighbors)
        .def("in_neighbors", &ContinuousGraphWrapper<pc::SAGE>::inNeighbors)
        .def("has_edge", &ContinuousGraphWrapper<pc::SAGE>::hasEdge)
        .def("nodes", &ContinuousGraphWrapper<pc::SAGE>::nodes)
        .def("task_list", &ContinuousGraphWrapper<pc::SAGE>::taskListPy)
        .def("robot_list", &ContinuousGraphWrapper<pc::SAGE>::robotListPy)
        .def("file_write", [](const ContinuousGraphWrapper<pc::SAGE> &self, const std::string &path)
             { self.fileWrite(path, "SAGE"); })
        .def("__repr__", [](const ContinuousGraphWrapper<pc::SAGE> &self)
             { return self.repr("SAGE"); });

    py::class_<ContinuousGraphWrapper<pc::MAGE>>(m, "MAGE")
        .def(py::init<py::object, const std::string &>(),
             py::arg("positions"), py::arg("filename") = "temp.dat")
        .def("edges", &ContinuousGraphWrapper<pc::MAGE>::edges)
        .def("num_edges", &ContinuousGraphWrapper<pc::MAGE>::numEdges)
        .def("num_nodes", &ContinuousGraphWrapper<pc::MAGE>::numNodes)
        .def("threshold", &ContinuousGraphWrapper<pc::MAGE>::threshold)
        .def("count_type2_edges", &ContinuousGraphWrapper<pc::MAGE>::countType2Edges)
        .def("out_neighbors", &ContinuousGraphWrapper<pc::MAGE>::outNeighbors)
        .def("in_neighbors", &ContinuousGraphWrapper<pc::MAGE>::inNeighbors)
        .def("has_edge", &ContinuousGraphWrapper<pc::MAGE>::hasEdge)
        .def("nodes", &ContinuousGraphWrapper<pc::MAGE>::nodes)
        .def("task_list", &ContinuousGraphWrapper<pc::MAGE>::taskListPy)
        .def("robot_list", &ContinuousGraphWrapper<pc::MAGE>::robotListPy)
        .def("file_write", [](const ContinuousGraphWrapper<pc::MAGE> &self, const std::string &path)
             { self.fileWrite(path, "MAGE"); })
        .def("__repr__", [](const ContinuousGraphWrapper<pc::MAGE> &self)
             { return self.repr("MAGE"); });
}
