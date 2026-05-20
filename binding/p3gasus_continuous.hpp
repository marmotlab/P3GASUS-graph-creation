#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mage_dp_matrix.hpp"

namespace p3gasus_continuous
{

struct Pos2F
{
    double x = -2.0;
    double y = -2.0;
};

inline bool isPadding(const Pos2F &p)
{
    return p.x == -2.0 && p.y == -2.0;
}

inline double distance(const Pos2F &a, const Pos2F &b)
{
    if (isPadding(a) || isPadding(b))
        return 1e8;
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

struct ContinuousTask
{
    int taskID = 0;
    int robotID = 0;
    Pos2F startPos;
    Pos2F goalPos;
    int time = 0;

    ContinuousTask() = default;
    ContinuousTask(int tid, int rid, Pos2F start, Pos2F goal, int t)
        : taskID(tid), robotID(rid), startPos(start), goalPos(goal), time(t) {}

    std::string repr() const
    {
        std::ostringstream s;
        s << "ContinuousTask{id=" << taskID
          << " rid=" << robotID
          << " t=" << time
          << " start=(" << startPos.x << "," << startPos.y << ")"
          << " goal=(" << goalPos.x << "," << goalPos.y << ")}";
        return s.str();
    }
};

class DiGraph
{
public:
    void addNode(int n)
    {
        adj_.try_emplace(n);
        radj_.try_emplace(n);
    }

    void addEdge(int u, int v)
    {
        adj_[u].insert(v);
        radj_[v].insert(u);
        adj_.try_emplace(v);
        radj_.try_emplace(u);
    }

    bool removeEdge(int u, int v)
    {
        auto it = adj_.find(u);
        if (it == adj_.end())
            return false;
        const bool removed = it->second.erase(v) > 0;
        auto rit = radj_.find(v);
        if (rit != radj_.end())
            rit->second.erase(u);
        return removed;
    }

    bool hasEdge(int u, int v) const
    {
        auto it = adj_.find(u);
        return it != adj_.end() && it->second.count(v);
    }

    const std::unordered_set<int> &outNeighbors(int n) const
    {
        static const std::unordered_set<int> empty;
        auto it = adj_.find(n);
        return it == adj_.end() ? empty : it->second;
    }

    const std::unordered_set<int> &inNeighbors(int n) const
    {
        static const std::unordered_set<int> empty;
        auto it = radj_.find(n);
        return it == radj_.end() ? empty : it->second;
    }

    std::vector<int> nodes() const
    {
        std::vector<int> v;
        v.reserve(adj_.size());
        for (const auto &[n, _] : adj_)
            v.push_back(n);
        return v;
    }

    std::vector<std::pair<int, int>> edges() const
    {
        std::vector<std::pair<int, int>> e;
        for (const auto &[u, nbrs] : adj_)
            for (int v : nbrs)
                e.emplace_back(u, v);
        return e;
    }

    std::size_t numEdges() const
    {
        std::size_t count = 0;
        for (const auto &[_, nbrs] : adj_)
            count += nbrs.size();
        return count;
    }

    std::size_t numNodes() const { return adj_.size(); }

private:
    std::unordered_map<int, std::unordered_set<int>> adj_;
    std::unordered_map<int, std::unordered_set<int>> radj_;
};

using PositionMatrix = std::vector<std::vector<Pos2F>>;

class ContinuousExecutionGraph
{
public:
    DiGraph graph;
    std::unordered_map<int, ContinuousTask> taskList;
    std::vector<ContinuousTask> robotList;
    double THRESH = 1e8;

    explicit ContinuousExecutionGraph(const PositionMatrix &positions)
    {
        if (positions.empty() || positions.front().size() < 2)
            throw std::invalid_argument("positions must have shape [numRobots][numSteps>=2][2]");

        const int numRobots = static_cast<int>(positions.size());
        const int numSteps = static_cast<int>(positions.front().size());
        for (const auto &robotPositions : positions)
            if (static_cast<int>(robotPositions.size()) != numSteps)
                throw std::invalid_argument("all robots must have the same number of positions");

        for (int i = 0; i < numSteps; ++i)
            for (int r = 0; r < numRobots; ++r)
                for (int r2 = r + 1; r2 < numRobots; ++r2)
                    THRESH = std::min(THRESH, distance(positions[r][i], positions[r2][i]));
    }

    bool checkCollision(const Pos2F &a, const Pos2F &b) const
    {
        return distance(a, b) < THRESH;
    }

    long long countType2Edges() const
    {
        const long long total = static_cast<long long>(graph.numEdges());
        const long long type1 = static_cast<long long>(taskList.size()) -
                                static_cast<long long>(robotList.size());
        return total - type1;
    }

    void fileWrite(const std::string &path, const std::string &className) const
    {
        {
            std::ofstream f(path + "/" + className + "_Graph.txt");
            std::vector<int> keys;
            for (const auto &[k, _] : taskList)
                keys.push_back(k);
            std::sort(keys.begin(), keys.end());
            for (int k : keys)
            {
                std::vector<int> nbrs(graph.outNeighbors(k).begin(), graph.outNeighbors(k).end());
                std::sort(nbrs.begin(), nbrs.end());
                f << k;
                for (int n : nbrs)
                    f << " " << n;
                f << "\n";
            }
        }
        {
            std::ofstream f(path + "/" + className + "_TaskList.txt");
            std::vector<int> keys;
            for (const auto &[k, _] : taskList)
                keys.push_back(k);
            std::sort(keys.begin(), keys.end());
            for (int k : keys)
                f << taskList.at(k).repr() << "\n";
        }
    }
};

class OriginalADG : public ContinuousExecutionGraph
{
public:
    explicit OriginalADG(const PositionMatrix &allPositions)
        : ContinuousExecutionGraph(allPositions)
    {
        const int numRobots = static_cast<int>(allPositions.size());
        const int tasksPerRobot = static_cast<int>(allPositions.front().size()) - 1;

        int tId = 1;
        for (int rid = 0; rid < numRobots; ++rid)
        {
            int prevTaskID = -1;
            for (int i = 0; i < tasksPerRobot; ++i)
            {
                ContinuousTask t(tId, rid, allPositions[rid][i], allPositions[rid][i + 1], i);
                taskList[tId] = t;
                graph.addNode(tId);
                if (prevTaskID == -1)
                    robotList.push_back(t);
                else
                    graph.addEdge(prevTaskID, tId);
                prevTaskID = tId++;
            }
        }

        for (int rid = 0; rid < numRobots; ++rid)
        {
            const int firstTid = robotList[rid].taskID;
            for (int taskID = firstTid; taskID < firstTid + tasksPerRobot; ++taskID)
            {
                const ContinuousTask &task = taskList.at(taskID);
                if (isPadding(task.startPos))
                    break;
                for (int rid2 = 0; rid2 < numRobots; ++rid2)
                {
                    if (rid == rid2)
                        continue;
                    const int firstTid2 = robotList[rid2].taskID;
                    for (int taskID2 = firstTid2; taskID2 < firstTid2 + tasksPerRobot; ++taskID2)
                    {
                        const ContinuousTask &task2 = taskList.at(taskID2);
                        if (checkCollision(task.startPos, task2.goalPos) && task.time <= task2.time)
                        {
                            graph.addEdge(task.taskID, task2.taskID);
                            break;
                        }
                    }
                }
            }
        }
    }
};

class SAGE : public ContinuousExecutionGraph
{
public:
    explicit SAGE(const PositionMatrix &allPositions)
        : ContinuousExecutionGraph(allPositions)
    {
        build(allPositions);
    }

protected:
    void build(const PositionMatrix &allPositions)
    {
        const int numRobots = static_cast<int>(allPositions.size());
        const int tasksPerRobot = static_cast<int>(allPositions.front().size()) - 1;
        std::vector<Pos2F> goalPositions;
        goalPositions.reserve(numRobots * tasksPerRobot);

        int tId = 1;
        for (int rid = 0; rid < numRobots; ++rid)
        {
            int prevTaskID = -1;
            for (int i = 0; i < tasksPerRobot; ++i)
            {
                ContinuousTask t(tId, rid, allPositions[rid][i], allPositions[rid][i + 1], i);
                taskList[tId] = t;
                graph.addNode(tId);
                goalPositions.push_back(t.goalPos);
                if (prevTaskID == -1)
                    robotList.push_back(t);
                else
                    graph.addEdge(prevTaskID, tId);
                prevTaskID = tId++;
            }
        }

        std::vector<int> taskQueue;
        taskQueue.reserve(robotList.size());
        for (const auto &task : robotList)
            taskQueue.push_back(task.taskID);

        std::size_t qi = 0;
        while (qi < taskQueue.size())
        {
            const int tID = taskQueue[qi++];
            const ContinuousTask &t = taskList.at(tID);

            if (isPadding(t.startPos))
                continue;

            if (taskList.count(tID + 1) && taskList.at(tID + 1).time != 0)
                taskQueue.push_back(tID + 1);

            std::vector<int> possibleDependencies;
            for (std::size_t i = 0; i < goalPositions.size(); ++i)
                if (distance(t.startPos, goalPositions[i]) <= THRESH)
                    possibleDependencies.push_back(static_cast<int>(i));

            std::sort(possibleDependencies.begin(), possibleDependencies.end());
            std::vector<int> dependentRobots{t.robotID};

            for (int positionIndex : possibleDependencies)
            {
                const int candidateTaskID = positionIndex + 1;
                const ContinuousTask &candidate = taskList.at(candidateTaskID);
                if (std::find(dependentRobots.begin(), dependentRobots.end(), candidate.robotID) ==
                        dependentRobots.end() &&
                    t.time <= candidate.time)
                {
                    if (checkCollision(t.startPos, candidate.goalPos))
                    {
                        graph.addEdge(tID, candidateTaskID);
                        dependentRobots.push_back(candidate.robotID);
                    }
                }
            }
        }
    }
};

class MAGE : public SAGE
{
public:
    explicit MAGE(const PositionMatrix &allPositions, const std::string &filename = "temp.dat")
        : SAGE(allPositions)
    {
        const int n = static_cast<int>(taskList.size()) + 2;
        dp_.reset(static_cast<std::size_t>(n), filename);
        for (const auto &task : robotList)
            reduceGraph(task.taskID);
    }

private:
    MageDpMatrix dp_;

    void reduceGraph(int root)
    {
        if (dp_.at(root, root))
            return;
        dp_.at(root, root) = 1;

        std::vector<int> children(graph.outNeighbors(root).begin(), graph.outNeighbors(root).end());

        auto seqIt = std::find(children.begin(), children.end(), root + 1);
        if (seqIt != children.end())
        {
            children.erase(seqIt);
            reduceGraph(root + 1);
            uint8_t *rootDp = dp_.row(root);
            const uint8_t *childDp = dp_.row(root + 1);
            for (std::size_t i = 0; i < dp_.size(); ++i)
                rootDp[i] = rootDp[i] | childDp[i];
        }

        std::sort(children.begin(), children.end(),
                  [this](int a, int b)
                  {
                      return taskList.at(a).time < taskList.at(b).time;
                  });

        while (!children.empty())
        {
            const int child = children.front();
            children.erase(children.begin());
            if (dp_.at(root, child))
            {
                graph.removeEdge(root, child);
            }
            else
            {
                reduceGraph(child);
                uint8_t *rootDp = dp_.row(root);
                const uint8_t *childDp = dp_.row(child);
                for (std::size_t i = 0; i < dp_.size(); ++i)
                    rootDp[i] = rootDp[i] | childDp[i];
            }
        }
    }
};

} // namespace p3gasus_continuous
