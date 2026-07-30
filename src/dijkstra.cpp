#include "ai_module.h"
#include "graph_state.h"

#include <algorithm>
#include <limits>
#include <vector>

static bool find_point_range_in_edge(
  const EdgeChain& edge, int from_point, int to_point, size_t& out_begin, size_t& out_end) {
  for (size_t i = 0; i < edge.point_ids.size(); ++i) {
    if (edge.point_ids[i] != from_point) continue;
    for (size_t j = i; j < edge.point_ids.size(); ++j) {
      if (edge.point_ids[j] == to_point) {
        out_begin = i;
        out_end = j;
        return true;
      }
    }
  }
  return false;
}

static bool is_edge_subpath_enabled_locked(
  const GraphState& state, const EdgeChain& edge, size_t begin_idx, size_t end_idx) {
  if (begin_idx > end_idx || end_idx >= edge.point_ids.size()) return false;
  for (size_t i = begin_idx; i < end_idx; ++i) {
    if (!is_seg_enabled_locked(state, edge.seg_ids[i])) return false;
  }
  return true;
}

static bool collect_edge_subpath_locked(
  const GraphState& state, const EdgeChain& edge, int from_point, int to_point,
  std::vector<int>& out_points, double& out_time) {
  size_t begin_idx = 0;
  size_t end_idx = 0;
  if (!find_point_range_in_edge(edge, from_point, to_point, begin_idx, end_idx)) return false;
  if (!is_edge_subpath_enabled_locked(state, edge, begin_idx, end_idx)) return false;

  out_points.assign(edge.point_ids.begin() + begin_idx, edge.point_ids.begin() + end_idx + 1);
  out_time = edge_subpath_weight_locked(edge, begin_idx, end_idx);
  return true;
}

static bool shortest_edge_path_locked(
  GraphState& state, int start_point, int goal_point,
  std::vector<const EdgeChain*>& out_edges, double& out_time) {
  out_edges.clear();
  out_time = 0.0;

  if (start_point == goal_point) return true;

  auto itStartVertex = state.point_index_by_id.find(start_point);
  auto itGoalVertex = state.point_index_by_id.find(goal_point);
  if (itStartVertex == state.point_index_by_id.end() || itGoalVertex == state.point_index_by_id.end()) {
    return false;
  }

  const int start_vertex = itStartVertex->second;
  const int goal_vertex = itGoalVertex->second;
  const int vertex_count = (int)state.point_id_by_index.size();
  if (start_vertex < 0 || goal_vertex < 0 || start_vertex >= vertex_count || goal_vertex >= vertex_count) {
    return false;
  }
  auto heap_less = [](const RouteHeapItem& a, const RouteHeapItem& b) { return a.dist > b.dist; };

  RoutingScratch& scratch = state.routing_scratch;
  scratch.dist.assign(vertex_count, std::numeric_limits<double>::infinity());
  scratch.prev_vertex.assign(vertex_count, -1);
  scratch.prev_edge.assign(vertex_count, nullptr);
  scratch.heap.clear();

  scratch.dist[start_vertex] = 0.0;
  scratch.heap.push_back(RouteHeapItem{start_vertex, 0.0});
  std::push_heap(scratch.heap.begin(), scratch.heap.end(), heap_less);

  while (!scratch.heap.empty()) {
    std::pop_heap(scratch.heap.begin(), scratch.heap.end(), heap_less);
    RouteHeapItem cur = scratch.heap.back();
    scratch.heap.pop_back();

    if (cur.vertex < 0 || cur.vertex >= vertex_count) continue;
    if (cur.dist > scratch.dist[cur.vertex]) continue;
    if (cur.vertex == goal_vertex) break;
    if (cur.vertex >= (int)state.edge_out_by_vertex.size()) continue;

    const std::vector<EdgeChain*>& out_edges_from_vertex = state.edge_out_by_vertex[cur.vertex];
    for (const EdgeChain* edge_ptr : out_edges_from_vertex) {
      if (!edge_ptr) continue;
      const EdgeChain& edge = *edge_ptr;
      const int next_vertex = edge.end_vertex;
      if (next_vertex < 0 || next_vertex >= vertex_count) continue;

      const double w = edge_effective_weight_locked(state, edge);
      if (w == std::numeric_limits<double>::infinity()) continue;

      const double nd = cur.dist + w;
      if (nd < scratch.dist[next_vertex]) {
        scratch.dist[next_vertex] = nd;
        scratch.prev_vertex[next_vertex] = cur.vertex;
        scratch.prev_edge[next_vertex] = &edge;
        scratch.heap.push_back(RouteHeapItem{next_vertex, nd});
        std::push_heap(scratch.heap.begin(), scratch.heap.end(), heap_less);
      }
    }
  }

  if (scratch.dist[goal_vertex] == std::numeric_limits<double>::infinity()) return false;
  out_time = scratch.dist[goal_vertex];

  scratch.rev_edges.clear();
  int vertex = goal_vertex;
  while (vertex != start_vertex) {
    if (vertex < 0 || vertex >= vertex_count) return false;
    const EdgeChain* edge = scratch.prev_edge[vertex];
    const int parent = scratch.prev_vertex[vertex];
    if (!edge || parent < 0) return false;
    scratch.rev_edges.push_back(edge);
    vertex = parent;
  }

  out_edges.assign(scratch.rev_edges.rbegin(), scratch.rev_edges.rend());
  return true;
}
bool dijkstra_shortest_path(int start, int goal, std::vector<int>& out_path, double& out_time) {
  std::lock_guard<std::mutex> lk(g_mtx);
  out_path.clear();
  out_time = 0.0;

  if (!g_state.has_topology) {
    LogFile("[Dijkstra] fail: no topology start=%d goal=%d", start, goal);
    return false;
  }
  if (start == goal) {
    out_path.push_back(start);
    return true;
  }

  auto itStartEdgeId = g_state.start_edge_id_by_point.find(start);
  auto itGoalEdgeId = g_state.goal_edge_id_by_point.find(goal);
  if (itStartEdgeId == g_state.start_edge_id_by_point.end() ||
      itGoalEdgeId == g_state.goal_edge_id_by_point.end()) {
    LogFile("[Dijkstra] fail: edge mapping miss start=%d goal=%d has_start=%d has_goal=%d",
            start, goal,
            itStartEdgeId != g_state.start_edge_id_by_point.end() ? 1 : 0,
            itGoalEdgeId != g_state.goal_edge_id_by_point.end() ? 1 : 0);
    return false;
  }

  auto itStartEdge = g_state.edge_by_id.find(itStartEdgeId->second);
  auto itGoalEdge = g_state.edge_by_id.find(itGoalEdgeId->second);
  if (itStartEdge == g_state.edge_by_id.end() || itGoalEdge == g_state.edge_by_id.end()) {
    LogFile("[Dijkstra] fail: edge lookup miss start=%d goal=%d start_edge_id=%d goal_edge_id=%d",
            start, goal, itStartEdgeId->second, itGoalEdgeId->second);
    return false;
  }

  const EdgeChain& start_edge = itStartEdge->second;
  const EdgeChain& goal_edge = itGoalEdge->second;

  if (start_edge.edge_id == goal_edge.edge_id) {
    const bool ok = collect_edge_subpath_locked(g_state, start_edge, start, goal, out_path, out_time);
    if (ok) return true;
  }

  RoutingScratch& scratch = g_state.routing_scratch;
  scratch.prefix_path.clear();
  scratch.suffix_path.clear();
  scratch.middle_edges.clear();
  double prefix_time = 0.0;
  double suffix_time = 0.0;
  if (!collect_edge_subpath_locked(
        g_state, start_edge, start, start_edge.end_point, scratch.prefix_path, prefix_time)) {
    LogFile("[Dijkstra] fail: prefix miss start=%d goal=%d start_edge_id=%d edge_start=%d edge_end=%d",
            start, goal, start_edge.edge_id, start_edge.start_point, start_edge.end_point);
    return false;
  }
  if (!collect_edge_subpath_locked(
        g_state, goal_edge, goal_edge.start_point, goal, scratch.suffix_path, suffix_time)) {
    LogFile("[Dijkstra] fail: suffix miss start=%d goal=%d goal_edge_id=%d edge_start=%d edge_end=%d",
            start, goal, goal_edge.edge_id, goal_edge.start_point, goal_edge.end_point);
    return false;
  }

  double middle_time = 0.0;
  if (!shortest_edge_path_locked(
        g_state, start_edge.end_point, goal_edge.start_point, scratch.middle_edges, middle_time)) {
    LogFile("[Dijkstra] fail: middle path miss start=%d goal=%d from=%d to=%d start_edge_id=%d goal_edge_id=%d",
            start, goal, start_edge.end_point, goal_edge.start_point,
            start_edge.edge_id, goal_edge.edge_id);
    return false;
  }

  append_points_no_dup(out_path, scratch.prefix_path);
  for (const EdgeChain* edge : scratch.middle_edges) {
    if (!edge) {
      LogFile("[Dijkstra] fail: middle edge missing start=%d goal=%d", start, goal);
      return false;
    }
    append_points_no_dup(out_path, edge->point_ids);
  }
  append_points_no_dup(out_path, scratch.suffix_path);
  out_time = prefix_time + middle_time + suffix_time;
  if (out_path.empty()) {
    LogFile("[Dijkstra] fail: empty path after compose start=%d goal=%d", start, goal);
    return false;
  }
  return true;
}

void dijkstra_enable_segment(int seg_id) {
  std::lock_guard<std::mutex> lk(g_mtx);
  g_state.disabled_seg.erase(seg_id);
}

void dijkstra_disable_segment(int seg_id) {
  std::lock_guard<std::mutex> lk(g_mtx);
  g_state.disabled_seg.insert(seg_id);
}

bool dijkstra_has_topology() {
  std::lock_guard<std::mutex> lk(g_mtx);
  return g_state.has_topology;
}

void dijkstra_clear() {
#if USE_LEARN
  stop_edge_learning_worker();
  g_last_message_by_vehicle.clear();
  g_edge_progress_by_vehicle.clear();
#endif
  std::lock_guard<std::mutex> lk(g_mtx);
  g_state = GraphState{};
}
