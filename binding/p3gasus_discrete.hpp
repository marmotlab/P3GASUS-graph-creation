#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <array>
#include <tuple>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <memory>
#include <functional>
#include "mage_dp_matrix.hpp"

// ---------------------------------------------------------------------------
// Pos2D  – lightweight 2-D integer coordinate
// ---------------------------------------------------------------------------
struct Pos2D
{
    int x = 0, y = 0;
    bool operator==(const Pos2D &o) const { return x == o.x && y == o.y; }
    Pos2D operator+(const Pos2D &o) const { return {x + o.x, y + o.y}; }
    Pos2D operator-(const Pos2D &o) const { return {x - o.x, y - o.y}; }
};

struct Pos2DHash
{
    std::size_t operator()(const Pos2D &p) const
    {
        // Cantor-style hash, good for moderate coordinates
        std::size_t h1 = std::hash<int>{}(p.x);
        std::size_t h2 = std::hash<int>{}(p.y);
        return h1 ^ (h2 * 2654435761ULL);
    }
};

// ---------------------------------------------------------------------------
// Action helpers
// ---------------------------------------------------------------------------
inline Pos2D actionDelta(int action)
{
    // 0=stay, 1=+x, 2=+y, 3=-x, 4=-y
    static const Pos2D deltas[5] = {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    if (action < 0 || action > 4)
        throw std::out_of_range("Invalid action");
    return deltas[action];
}

inline int getActionFromPos(const Pos2D &cur, const Pos2D &nxt)
{
    Pos2D d = nxt - cur;
    if (d.x == 1 && d.y == 0)
        return 1;
    if (d.x == 0 && d.y == 1)
        return 2;
    if (d.x == -1 && d.y == 0)
        return 3;
    if (d.x == 0 && d.y == -1)
        return 4;
    if (d.x == 0 && d.y == 0)
        return 0;
    return -1;
}

// ---------------------------------------------------------------------------
// Task
// ---------------------------------------------------------------------------
struct Task
{
    int taskID = 0;
    int robotID = 0;
    int action = 0;
    int time = 0;
    Pos2D startPos;
    Pos2D goalPos;

    Task() = default;
    Task(int tid, int rid, Pos2D start, int act, int t)
        : taskID(tid), robotID(rid), action(act), time(t),
          startPos(start), goalPos(start + actionDelta(act)) {}

    std::string repr() const
    {
        std::ostringstream s;
        s << "Task{id=" << taskID << " rid=" << robotID
          << " act=" << action << " t=" << time
          << " start=(" << startPos.x << "," << startPos.y << ")"
          << " goal=(" << goalPos.x << "," << goalPos.y << ")}";
        return s.str();
    }
};

// ---------------------------------------------------------------------------
// Minimal directed graph over integer node IDs
// ---------------------------------------------------------------------------
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
        for (auto &[n, _] : adj_)
            v.push_back(n);
        return v;
    }
    // Returns list of (u,v) edges
    std::vector<std::pair<int, int>> edges() const
    {
        std::vector<std::pair<int, int>> e;
        for (auto &[u, nbrs] : adj_)
            for (int v : nbrs)
                e.emplace_back(u, v);
        return e;
    }
    std::size_t numEdges() const
    {
        std::size_t cnt = 0;
        for (auto &[_, nbrs] : adj_)
            cnt += nbrs.size();
        return cnt;
    }
    std::size_t numNodes() const { return adj_.size(); }

private:
    std::unordered_map<int, std::unordered_set<int>> adj_;
    std::unordered_map<int, std::unordered_set<int>> radj_;
};

// ---------------------------------------------------------------------------
// Position helper (mirrors Python Position class)
// ---------------------------------------------------------------------------
struct Position
{
    // robotID -> ordered list of taskIDs
    std::unordered_map<int, std::vector<int>> robotDict;
};

// ---------------------------------------------------------------------------
// Base ExecutionGraph
// ---------------------------------------------------------------------------
class ExecutionGraph
{
public:
    DiGraph graph;
    std::unordered_map<int, Task> taskList;
    std::vector<Task> robotList; // first task per robot

    virtual ~ExecutionGraph() = default;

    void fileWrite(const std::string &path, const std::string &className) const
    {
        // Graph file
        {
            std::ofstream f(path + "/" + className + "_Graph.txt");
            std::vector<int> keys;
            for (auto &[k, _] : taskList)
                keys.push_back(k);
            std::sort(keys.begin(), keys.end());
            for (int k : keys)
            {
                auto &nbrs = graph.outNeighbors(k);
                std::vector<int> sorted_nbrs(nbrs.begin(), nbrs.end());
                std::sort(sorted_nbrs.begin(), sorted_nbrs.end());
                f << k << ":[";
                for (std::size_t i = 0; i < sorted_nbrs.size(); ++i)
                    f << sorted_nbrs[i] << (i + 1 < sorted_nbrs.size() ? ", " : "");
                f << "]\n";
            }
        }
        // TaskList file
        {
            std::ofstream f(path + "/" + className + "_TaskList.txt");
            std::vector<int> keys;
            for (auto &[k, _] : taskList)
                keys.push_back(k);
            std::sort(keys.begin(), keys.end());
            for (int k : keys)
                f << taskList.at(k).repr() << "\n";
        }
    }

    // Convenience: number of type-2 (cross-robot) edges
    // = total edges - (numTasks - numRobots)  [type-1 edges = sequential per robot]
    long long countType2Edges() const
    {
        long long total = (long long)graph.numEdges();
        long long type1 = (long long)taskList.size() - (long long)robotList.size();
        return total - type1;
    }
};

// ---------------------------------------------------------------------------
// OriginalADG
// ---------------------------------------------------------------------------
class OriginalADG : public ExecutionGraph
{
public:
    // taskActions: [numRobots][numTasks] action values (0..4)
    // startPositions: [numRobots] initial (x,y)
    OriginalADG(const std::vector<std::vector<int>> &taskActions,
                const std::vector<Pos2D> &startPositions)
    {
        int numRobots = (int)startPositions.size();
        int numTasks = (int)taskActions[0].size();
        std::vector<Pos2D> currentPositions = startPositions;

        int tId = 1;
        // Phase 1: skeleton + type-1 edges
        for (int rid = 0; rid < numRobots; ++rid)
        {
            int prevTaskID = -1;
            for (int i = 0; i < numTasks; ++i)
            {
                Task t(tId, rid, currentPositions[rid], taskActions[rid][i], i);
                taskList[tId] = t;
                currentPositions[rid] = t.goalPos;
                graph.addNode(tId);
                if (prevTaskID == -1)
                    robotList.push_back(t);
                else
                    graph.addEdge(prevTaskID, tId);
                prevTaskID = tId++;
            }
        }

        // Phase 2: type-2 edges
        for (int rid = 0; rid < numRobots; ++rid)
        {
            int firstTid = robotList[rid].taskID;
            for (int taskID = firstTid; taskID < firstTid + numTasks; ++taskID)
            {
                const Task &task = taskList[taskID];
                for (int rid_ = 0; rid_ < numRobots; ++rid_)
                {
                    if (rid == rid_)
                        continue;
                    int firstTid_ = robotList[rid_].taskID;
                    for (int taskID_ = firstTid_; taskID_ < firstTid_ + numTasks; ++taskID_)
                    {
                        const Task &task_ = taskList[taskID_];
                        if (task.startPos == task_.goalPos && task.time <= task_.time)
                        {
                            graph.addEdge(task.taskID, task_.taskID);
                            break;
                        }
                    }
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// SAGE
// ---------------------------------------------------------------------------
class SAGE : public ExecutionGraph
{
public:
    SAGE(const std::vector<std::vector<int>> &taskActions,
         const std::vector<Pos2D> &startPositions)
    {
        int numRobots = (int)taskActions.size();
        int numTasks = (int)taskActions[0].size();
        std::vector<Pos2D> currentPositions = startPositions;

        std::unordered_map<Pos2D, Position, Pos2DHash> positions;

        int tId = 1;
        for (int rid = 0; rid < numRobots; ++rid)
        {
            int prevTaskID = -1;
            for (int i = 0; i < numTasks; ++i)
            {
                Task t(tId, rid, currentPositions[rid], taskActions[rid][i], i);
                taskList[tId] = t;
                currentPositions[rid] = t.goalPos;
                graph.addNode(tId);

                positions[t.goalPos].robotDict[rid].push_back(tId);

                if (prevTaskID == -1)
                    robotList.push_back(t);
                else
                    graph.addEdge(prevTaskID, tId);
                prevTaskID = tId++;
            }
        }

        // BFS-style sweep matching Python implementation
        std::vector<int> taskQueue;
        taskQueue.reserve(numRobots);
        for (auto &rt : robotList)
            taskQueue.push_back(rt.taskID);

        int previousTime = -1;
        std::vector<int> tasksToClear;

        std::size_t qi = 0;
        while (qi < taskQueue.size())
        {
            int tID = taskQueue[qi++];
            const Task &t = taskList[tID];

            if (t.time > previousTime)
            {
                ++previousTime;
                // Pop front of each robot's list for cleared tasks
                for (int ttc : tasksToClear)
                {
                    const Task &t__ = taskList[ttc];
                    auto &rDict = positions[t__.goalPos].robotDict;
                    auto it = rDict.find(t__.robotID);
                    if (it != rDict.end())
                    {
                        it->second.erase(it->second.begin());
                        if (it->second.empty())
                            rDict.erase(it);
                    }
                }
                tasksToClear.clear();
            }
            tasksToClear.push_back(tID);

            // Enqueue next task for this robot if it exists
            if (taskList.count(tID + 1) && taskList.at(tID + 1).time != 0)
                taskQueue.push_back(tID + 1);

            // Type-2 edges via startPos occupancy
            auto pit = positions.find(t.startPos);
            if (pit != positions.end())
            {
                for (auto &[rid_, tidList] : pit->second.robotDict)
                {
                    if (t.robotID != rid_)
                    {
                        graph.addEdge(tID, taskList.at(tidList.front()).taskID);
                    }
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// FORTED
// ---------------------------------------------------------------------------
class FORTED : public ExecutionGraph
{
public:
    FORTED(const std::vector<std::vector<int>> &taskActions,
           const std::vector<Pos2D> &startPositions)
    {
        int numRobots = (int)taskActions.size();
        int tasksPerRobot = (int)taskActions[0].size();
        std::vector<Pos2D> currentPositions = startPositions;

        std::unordered_map<Pos2D, int, Pos2DHash> positions; // pos -> last taskID there

        for (int time = 0; time < tasksPerRobot; ++time)
        {
            for (int rid = 0; rid < numRobots; ++rid)
            {
                int tid = rid * tasksPerRobot + time + 1;
                int prevTid = (taskList.count(tid - 1)) ? tid - 1 : -1;

                Task t(tid, rid, currentPositions[rid], taskActions[rid][time], time);
                taskList[tid] = t;
                currentPositions[rid] = t.goalPos;
                graph.addNode(tid);

                if (prevTid == -1)
                    robotList.push_back(t);
                else
                    graph.addEdge(prevTid, tid);

                auto pit = positions.find(t.startPos);
                if (pit == positions.end())
                {
                    positions[t.startPos] = tid;
                }
                else
                {
                    if (pit->second != tid - 1)
                        graph.addEdge(pit->second, tid - 1);
                    pit->second = tid;
                }
            }
        }

        // Final goalPos sweep
        for (int rid = 0; rid < numRobots; ++rid)
        {
            int tid = (rid + 1) * tasksPerRobot;
            const Task &t = taskList.at(tid);
            auto pit = positions.find(t.goalPos);
            if (pit != positions.end() && t.action != 0)
                graph.addEdge(pit->second, tid);
        }
    }
};

// ---------------------------------------------------------------------------
// MAGE – transitive-reduction wrapper
// ---------------------------------------------------------------------------
class MAGE : public ExecutionGraph
{
public:
    // baseADGType: 0=FORTED (default), 1=SAGE, 2=OriginalADG
    enum BaseType
    {
        BASE_FORTED = 0,
        BASE_SAGE = 1,
        BASE_ORIGINAL = 2
    };

    MAGE(const std::vector<std::vector<int>> &taskActions,
         const std::vector<Pos2D> &startPositions,
         BaseType baseType = BASE_FORTED,
         const std::string &filename = "temp.dat")
    {
        // Build base ADG
        std::unique_ptr<ExecutionGraph> base;
        switch (baseType)
        {
        case BASE_SAGE:
            base = std::make_unique<SAGE>(taskActions, startPositions);
            break;
        case BASE_ORIGINAL:
            base = std::make_unique<OriginalADG>(taskActions, startPositions);
            break;
        default:
            base = std::make_unique<FORTED>(taskActions, startPositions);
            break;
        }
        graph = std::move(base->graph);
        taskList = std::move(base->taskList);
        robotList = std::move(base->robotList);

        int N = (int)taskList.size() + 2;
        // dp[u][v] = reachable(u,v) already computed
        dp_.reset(static_cast<std::size_t>(N), filename);

        for (auto &rt : robotList)
            reduceGraph(rt.taskID);
    }

private:
    MageDpMatrix dp_;

    // Fills dp_[root] with all nodes reachable from root.
    void reduceGraph(int root)
    {
        if (dp_.at(root, root))
            return;
        dp_.at(root, root) = 1;

        // Collect children sorted by time, but process root+1 first
        std::vector<int> children;
        for (int v : graph.outNeighbors(root))
            children.push_back(v);

        // Process sequential successor first
        auto it1 = std::find(children.begin(), children.end(), root + 1);
        if (it1 != children.end())
        {
            children.erase(it1);
            reduceGraph(root + 1);
            uint8_t *root_dp = dp_.row(root);
            const uint8_t *child_dp = dp_.row(root + 1);
            for (std::size_t i = 0; i < dp_.size(); ++i)
                root_dp[i] = root_dp[i] | child_dp[i];
        }

        // Sort remaining children by time
        std::sort(children.begin(), children.end(),
                  [this](int a, int b)
                  {
                      return taskList.at(a).time < taskList.at(b).time;
                  });

        while (!children.empty())
        {
            int child = children.front();
            children.erase(children.begin());
            if (dp_.at(root, child))
            {
                graph.removeEdge(root, child);
            }
            else
            {
                reduceGraph(child);
                uint8_t *root_dp = dp_.row(root);
                const uint8_t *child_dp = dp_.row(child);
                for (std::size_t i = 0; i < dp_.size(); ++i)
                    root_dp[i] = root_dp[i] | child_dp[i];
            }
        }
    }
};
