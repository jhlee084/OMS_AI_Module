#include "ai_module.h"
#include "graph_state.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

struct VehicleRouteInfo {
  bool mapped = false;
  bool tail_enabled = false;
  int edge_id = 0;
  int source_vertex = -1;
  int segment_index = -1;
  double segment_fraction = 0.0;
  double position_time = 0.0;
  double tail_time = 0.0;
};

struct OrderRouteInfo {
  bool mapped = false;
  bool head_enabled = false;
  int edge_id = 0;
  int target_vertex = -1;
  int segment_index = -1;
  double segment_fraction = 0.0;
  double position_time = 0.0;
  double head_time = 0.0;
};

struct DistanceScratch {
  std::vector<double> dist;
  std::vector<RouteHeapItem> heap;
  std::vector<unsigned char> target;
};

bool segment_range_enabled(
  const GraphState& state, const EdgeChain& edge, int begin_segment, int end_segment) {
  if (begin_segment < 0 || end_segment < begin_segment ||
      end_segment > static_cast<int>(edge.seg_ids.size())) {
    return false;
  }
  for (int i = begin_segment; i < end_segment; ++i) {
    if (!is_seg_enabled_locked(state, edge.seg_ids[static_cast<size_t>(i)])) return false;
  }
  return true;
}

bool offset_fraction(
  const GraphState& state, int segment_id, int32_t offset, double& out_fraction) {
  if (offset < 0) return false;
  const auto itSegment = state.seg_by_id.find(segment_id);
  if (itSegment == state.seg_by_id.end() || itSegment->second.offset_length <= 0.0) {
    // A zero offset is still well-defined when old topology data has no length.
    if (offset == 0) {
      out_fraction = 0.0;
      return true;
    }
    return false;
  }
  out_fraction = std::min(
    1.0, static_cast<double>(offset) / itSegment->second.offset_length);
  return true;
}

VehicleRouteInfo make_vehicle_info(
  const GraphState& state, int segment_id, int32_t offset) {
  VehicleRouteInfo info;
  const auto itEdgeId = state.seg_to_edge_id.find(segment_id);
  const auto itSegmentIndex = state.seg_index_in_edge.find(segment_id);
  if (itEdgeId == state.seg_to_edge_id.end() || itSegmentIndex == state.seg_index_in_edge.end()) {
    return info;
  }

  const auto itEdge = state.edge_by_id.find(itEdgeId->second);
  if (itEdge == state.edge_by_id.end()) return info;
  const EdgeChain& edge = itEdge->second;
  const int segment_index = itSegmentIndex->second;
  if (segment_index < 0 || segment_index >= static_cast<int>(edge.seg_ids.size()) ||
      edge.prefix_w.size() != edge.seg_ids.size() + 1 || edge.end_vertex < 0) {
    return info;
  }

  double fraction = 0.0;
  if (!offset_fraction(state, segment_id, offset, fraction)) return info;
  const double segment_time =
    edge.prefix_w[static_cast<size_t>(segment_index + 1)] -
    edge.prefix_w[static_cast<size_t>(segment_index)];

  info.mapped = true;
  info.edge_id = edge.edge_id;
  info.source_vertex = edge.end_vertex;
  info.segment_index = segment_index;
  info.segment_fraction = fraction;
  info.position_time = edge.prefix_w[static_cast<size_t>(segment_index)]
    + segment_time * fraction;
  info.tail_time = edge.prefix_w.back() - info.position_time;
  const int tail_begin = fraction >= 1.0 ? segment_index + 1 : segment_index;
  info.tail_enabled = segment_range_enabled(
    state, edge, tail_begin, static_cast<int>(edge.seg_ids.size()));
  return info;
}

VehicleRouteInfo make_vehicle_point_info(
  const GraphState& state, int point_id) {
  VehicleRouteInfo info;

  // A branch point is a real compressed-graph vertex. Start directly at the
  // vertex so Dijkstra can consider every outgoing edge.
  const auto itOutgoing = state.out_seg_ids_by_point.find(point_id);
  const auto itVertex = state.point_index_by_id.find(point_id);
  if (itOutgoing != state.out_seg_ids_by_point.end() &&
      itOutgoing->second.size() > 1 &&
      itVertex != state.point_index_by_id.end()) {
    info.mapped = true;
    info.tail_enabled = true;
    info.source_vertex = itVertex->second;
    info.tail_time = 0.0;
    return info;
  }

  const auto itEdgeId = state.start_edge_id_by_point.find(point_id);
  if (itEdgeId == state.start_edge_id_by_point.end()) return info;

  const auto itEdge = state.edge_by_id.find(itEdgeId->second);
  if (itEdge == state.edge_by_id.end()) return info;
  const EdgeChain& edge = itEdge->second;
  if (edge.seg_ids.empty() || edge.prefix_w.size() != edge.seg_ids.size() + 1 ||
      edge.end_vertex < 0) {
    return info;
  }

  size_t point_index = edge.point_ids.size();
  for (size_t i = 0; i < edge.point_ids.size(); ++i) {
    if (edge.point_ids[i] == point_id) {
      point_index = i;
      break;
    }
  }
  if (point_index >= edge.point_ids.size() ||
      point_index >= edge.prefix_w.size()) {
    return info;
  }

  info.mapped = true;
  info.edge_id = edge.edge_id;
  info.source_vertex = edge.end_vertex;
  info.position_time = edge.prefix_w[point_index];
  info.tail_time = edge.prefix_w.back() - info.position_time;
  if (point_index == edge.seg_ids.size()) {
    info.segment_index = static_cast<int>(edge.seg_ids.size()) - 1;
    info.segment_fraction = 1.0;
  } else {
    info.segment_index = static_cast<int>(point_index);
    info.segment_fraction = 0.0;
  }
  info.tail_enabled = segment_range_enabled(
    state, edge, static_cast<int>(point_index),
    static_cast<int>(edge.seg_ids.size()));
  return info;
}

OrderRouteInfo make_order_info(
  const GraphState& state, int segment_id, int32_t destination_offset) {
  OrderRouteInfo info;
  const auto itEdgeId = state.seg_to_edge_id.find(segment_id);
  const auto itSegmentIndex = state.seg_index_in_edge.find(segment_id);
  if (itEdgeId == state.seg_to_edge_id.end() || itSegmentIndex == state.seg_index_in_edge.end()) {
    return info;
  }

  const auto itEdge = state.edge_by_id.find(itEdgeId->second);
  if (itEdge == state.edge_by_id.end()) return info;
  const EdgeChain& edge = itEdge->second;
  const int segment_index = itSegmentIndex->second;
  if (segment_index < 0 || segment_index >= static_cast<int>(edge.seg_ids.size()) ||
      edge.prefix_w.size() != edge.seg_ids.size() + 1 || edge.start_vertex < 0) {
    return info;
  }

  double fraction = 0.0;
  if (!offset_fraction(state, segment_id, destination_offset, fraction)) return info;
  const double segment_time =
    edge.prefix_w[static_cast<size_t>(segment_index + 1)] -
    edge.prefix_w[static_cast<size_t>(segment_index)];

  info.mapped = true;
  info.edge_id = edge.edge_id;
  info.target_vertex = edge.start_vertex;
  info.segment_index = segment_index;
  info.segment_fraction = fraction;
  info.position_time = edge.prefix_w[static_cast<size_t>(segment_index)]
    + segment_time * fraction;
  info.head_time = info.position_time;
  const int head_end = segment_index + (fraction > 0.0 ? 1 : 0);
  info.head_enabled = segment_range_enabled(state, edge, 0, head_end);
  return info;
}

OrderRouteInfo make_order_point_info(
  const GraphState& state, int point_id) {
  OrderRouteInfo info;
  const auto itEdgeId = state.goal_edge_id_by_point.find(point_id);
  if (itEdgeId == state.goal_edge_id_by_point.end()) return info;

  const auto itEdge = state.edge_by_id.find(itEdgeId->second);
  if (itEdge == state.edge_by_id.end()) return info;
  const EdgeChain& edge = itEdge->second;
  if (edge.seg_ids.empty() || edge.prefix_w.size() != edge.seg_ids.size() + 1 ||
      edge.start_vertex < 0) {
    return info;
  }

  size_t point_index = edge.point_ids.size();
  for (size_t i = 0; i < edge.point_ids.size(); ++i) {
    if (edge.point_ids[i] == point_id) {
      point_index = i;
      break;
    }
  }
  if (point_index >= edge.point_ids.size() ||
      point_index >= edge.prefix_w.size()) {
    return info;
  }

  info.mapped = true;
  info.edge_id = edge.edge_id;
  info.target_vertex = edge.start_vertex;
  info.position_time = edge.prefix_w[point_index];
  info.head_time = info.position_time;
  if (point_index == edge.seg_ids.size()) {
    info.segment_index = static_cast<int>(edge.seg_ids.size()) - 1;
    info.segment_fraction = 1.0;
  } else {
    info.segment_index = static_cast<int>(point_index);
    info.segment_fraction = 0.0;
  }
  info.head_enabled = segment_range_enabled(
    state, edge, 0, static_cast<int>(point_index));
  return info;
}

bool direct_same_edge_time(
  const GraphState& state,
  const VehicleRouteInfo& vehicle,
  const OrderRouteInfo& order,
  double& out_time) {
  if (!vehicle.mapped || !order.mapped || vehicle.edge_id != order.edge_id ||
      vehicle.position_time > order.position_time) {
    return false;
  }
  const auto itEdge = state.edge_by_id.find(vehicle.edge_id);
  if (itEdge == state.edge_by_id.end()) return false;
  const EdgeChain& edge = itEdge->second;
  const int begin_segment = vehicle.segment_index
    + (vehicle.segment_fraction >= 1.0 ? 1 : 0);
  const int end_segment = order.segment_index
    + (order.segment_fraction > 0.0 ? 1 : 0);
  if (!segment_range_enabled(state, edge, begin_segment, end_segment)) {
    return false;
  }
  out_time = order.position_time - vehicle.position_time;
  return out_time >= 0.0 && std::isfinite(out_time);
}

void calculate_distances(
  const GraphState& state,
  int source_vertex,
  bool reverse,
  const std::vector<int>& target_vertices,
  DistanceScratch& scratch) {
  const int vertex_count = static_cast<int>(state.point_id_by_index.size());
  scratch.dist.assign(static_cast<size_t>(vertex_count), kInfinity);
  scratch.heap.clear();
  scratch.target.assign(static_cast<size_t>(vertex_count), 0);
  if (source_vertex < 0 || source_vertex >= vertex_count) return;

  int remaining_targets = 0;
  for (int vertex : target_vertices) {
    if (vertex < 0 || vertex >= vertex_count || scratch.target[static_cast<size_t>(vertex)]) continue;
    scratch.target[static_cast<size_t>(vertex)] = 1;
    ++remaining_targets;
  }
  if (remaining_targets == 0) return;

  auto heap_less = [](const RouteHeapItem& a, const RouteHeapItem& b) {
    return a.dist > b.dist;
  };
  scratch.dist[static_cast<size_t>(source_vertex)] = 0.0;
  scratch.heap.push_back(RouteHeapItem{source_vertex, 0.0});

  while (!scratch.heap.empty() && remaining_targets > 0) {
    std::pop_heap(scratch.heap.begin(), scratch.heap.end(), heap_less);
    const RouteHeapItem cur = scratch.heap.back();
    scratch.heap.pop_back();
    if (cur.vertex < 0 || cur.vertex >= vertex_count ||
        cur.dist > scratch.dist[static_cast<size_t>(cur.vertex)]) {
      continue;
    }

    if (scratch.target[static_cast<size_t>(cur.vertex)]) {
      scratch.target[static_cast<size_t>(cur.vertex)] = 0;
      --remaining_targets;
      if (remaining_targets == 0) break;
    }

    const auto& adjacent = reverse
      ? state.edge_in_by_vertex[static_cast<size_t>(cur.vertex)]
      : state.edge_out_by_vertex[static_cast<size_t>(cur.vertex)];
    for (const EdgeChain* edge : adjacent) {
      if (!edge) continue;
      const int next_vertex = reverse ? edge->start_vertex : edge->end_vertex;
      if (next_vertex < 0 || next_vertex >= vertex_count) continue;
      const double edge_time = edge_effective_weight_locked(state, *edge);
      if (!std::isfinite(edge_time) || edge_time < 0.0) continue;
      const double next_time = cur.dist + edge_time;
      if (next_time < scratch.dist[static_cast<size_t>(next_vertex)]) {
        scratch.dist[static_cast<size_t>(next_vertex)] = next_time;
        scratch.heap.push_back(RouteHeapItem{next_vertex, next_time});
        std::push_heap(scratch.heap.begin(), scratch.heap.end(), heap_less);
      }
    }
  }
}

template <typename CostAt>
std::vector<int> hungarian_minimize(int row_count, int column_count, CostAt cost_at) {
  const long double infinity = std::numeric_limits<long double>::infinity();
  std::vector<long double> u(static_cast<size_t>(row_count) + 1, 0.0L);
  std::vector<long double> v(static_cast<size_t>(column_count) + 1, 0.0L);
  std::vector<int> p(static_cast<size_t>(column_count) + 1, 0);
  std::vector<int> way(static_cast<size_t>(column_count) + 1, 0);

  for (int row = 1; row <= row_count; ++row) {
    p[0] = row;
    int column0 = 0;
    std::vector<long double> min_value(static_cast<size_t>(column_count) + 1, infinity);
    std::vector<unsigned char> used(static_cast<size_t>(column_count) + 1, 0);
    do {
      used[static_cast<size_t>(column0)] = 1;
      const int row0 = p[static_cast<size_t>(column0)];
      long double delta = infinity;
      int column1 = 0;
      for (int column = 1; column <= column_count; ++column) {
        if (used[static_cast<size_t>(column)]) continue;
        const long double current = cost_at(row0 - 1, column - 1)
          - u[static_cast<size_t>(row0)] - v[static_cast<size_t>(column)];
        if (current < min_value[static_cast<size_t>(column)]) {
          min_value[static_cast<size_t>(column)] = current;
          way[static_cast<size_t>(column)] = column0;
        }
        if (min_value[static_cast<size_t>(column)] < delta) {
          delta = min_value[static_cast<size_t>(column)];
          column1 = column;
        }
      }
      for (int column = 0; column <= column_count; ++column) {
        if (used[static_cast<size_t>(column)]) {
          u[static_cast<size_t>(p[static_cast<size_t>(column)])] += delta;
          v[static_cast<size_t>(column)] -= delta;
        } else {
          min_value[static_cast<size_t>(column)] -= delta;
        }
      }
      column0 = column1;
    } while (p[static_cast<size_t>(column0)] != 0);

    do {
      const int column1 = way[static_cast<size_t>(column0)];
      p[static_cast<size_t>(column0)] = p[static_cast<size_t>(column1)];
      column0 = column1;
    } while (column0 != 0);
  }

  std::vector<int> row_to_column(static_cast<size_t>(row_count), -1);
  for (int column = 1; column <= column_count; ++column) {
    const int row = p[static_cast<size_t>(column)];
    if (row > 0) row_to_column[static_cast<size_t>(row - 1)] = column - 1;
  }
  return row_to_column;
}

}  // namespace

AI_API int __cdecl AI_OptimizeAssignments(
  const AI_ORDER_INPUT* orders,
  int32_t order_count,
  const AI_VEHICLE_INPUT* vehicles,
  int32_t vehicle_count,
  const uint8_t* allowed_matrix,
  float order_change_threshold_seconds,
  AI_ASSIGNMENT_RESULT* out_results,
  int32_t* out_result_count) {
  if (!out_result_count) return 0;
  *out_result_count = 0;
  if (order_count < 0 || vehicle_count < 0) return 0;
  if (order_count == 0) return 1;
  if (!orders || !out_results) return 0;
  if (!std::isfinite(order_change_threshold_seconds) ||
      order_change_threshold_seconds < 0.0) return 0;
  const double threshold_seconds =
    static_cast<double>(order_change_threshold_seconds);

  for (int order_index = 0; order_index < order_count; ++order_index) {
    AI_ASSIGNMENT_RESULT& result = out_results[order_index];
    result.order_id = orders[order_index].order_id;
    result.vehicle_id = 0;
    result.expected_time = 0.0;
    result.status = AI_ASSIGNMENT_NO_ALLOWED_VEHICLE;
    result.reserved = 0;
  }

  if (vehicle_count == 0) {
    *out_result_count = order_count;
    return 1;
  }
  if (!vehicles || !allowed_matrix) return 0;

  try {
    std::unordered_set<int32_t> order_ids;
    std::unordered_set<int32_t> vehicle_ids;
    order_ids.reserve(static_cast<size_t>(order_count));
    vehicle_ids.reserve(static_cast<size_t>(vehicle_count));
    for (int i = 0; i < order_count; ++i) {
      if (!order_ids.insert(orders[i].order_id).second) return 0;
    }
    for (int i = 0; i < vehicle_count; ++i) {
      if (!vehicle_ids.insert(vehicles[i].vehicle_id).second) return 0;
    }

    const size_t order_size = static_cast<size_t>(order_count);
    const size_t vehicle_size = static_cast<size_t>(vehicle_count);
    if (vehicle_size != 0 && order_size > std::numeric_limits<size_t>::max() / vehicle_size) {
      return 0;
    }
    const size_t matrix_size = order_size * vehicle_size;
    std::vector<double> pair_times(matrix_size, kInfinity);
    std::vector<unsigned char> order_has_allowed(order_size, 0);
    std::vector<unsigned char> order_has_reachable(order_size, 0);

    for (size_t index = 0; index < matrix_size; ++index) {
      if (allowed_matrix[index] > 1) return 0;
      if (allowed_matrix[index]) {
        order_has_allowed[index / vehicle_size] = 1;
      }
    }

    {
      std::lock_guard<std::mutex> lock(g_mtx);
      if (!g_state.has_topology) return 0;

      std::vector<VehicleRouteInfo> current_vehicle_info(vehicle_size);
      std::vector<VehicleRouteInfo> vehicle_info(vehicle_size);
      std::vector<double> committed_times(vehicle_size, kInfinity);
      std::vector<OrderRouteInfo> order_info(order_size);
      for (int i = 0; i < vehicle_count; ++i) {
        const size_t index = static_cast<size_t>(i);
        current_vehicle_info[index] = make_vehicle_info(
          g_state,
          vehicles[i].current_segment_id,
          vehicles[i].current_offset);
        if (vehicles[i].start_point_id > 0) {
          vehicle_info[index] = make_vehicle_point_info(
            g_state, vehicles[i].start_point_id);
        } else {
          vehicle_info[index] = current_vehicle_info[index];
        }
        if (vehicles[i].start_point_id <= 0) {
          committed_times[index] = 0.0;
        } else {
          const auto itCurrentSegment =
            g_state.seg_by_id.find(vehicles[i].current_segment_id);
          if (vehicles[i].current_offset == 0 &&
              itCurrentSegment != g_state.seg_by_id.end() &&
              itCurrentSegment->second.from == vehicles[i].start_point_id) {
            committed_times[index] = 0.0;
          }
        }
      }
      for (int i = 0; i < order_count; ++i) {
        order_info[static_cast<size_t>(i)] =
          make_order_info(
            g_state,
            orders[i].destination_segment_id,
            orders[i].destination_offset);
      }

      // Calculate each vehicle's mandatory travel from its current position
      // to start_point_id. Direct paths inside one
      // compressed edge are resolved first; the rest are grouped so Dijkstra
      // runs once per unique source or target vertex, whichever is fewer.
      std::vector<OrderRouteInfo> committed_target_info(vehicle_size);
      std::vector<int> unresolved_committed;
      unresolved_committed.reserve(vehicle_size);
      for (int i = 0; i < vehicle_count; ++i) {
        const size_t index = static_cast<size_t>(i);
        if (std::isfinite(committed_times[index])) continue;
        const VehicleRouteInfo& current = current_vehicle_info[index];
        committed_target_info[index] = make_order_point_info(
          g_state, vehicles[i].start_point_id);
        const OrderRouteInfo& target = committed_target_info[index];
        double direct_time = 0.0;
        if (direct_same_edge_time(g_state, current, target, direct_time)) {
          committed_times[index] = direct_time;
        } else if (current.mapped && current.tail_enabled &&
                   target.mapped && target.head_enabled) {
          unresolved_committed.push_back(i);
        }
      }

      if (!unresolved_committed.empty()) {
        std::unordered_map<int, std::vector<int>> committed_by_source;
        std::unordered_map<int, std::vector<int>> committed_by_target;
        for (int vehicle_index : unresolved_committed) {
          const size_t index = static_cast<size_t>(vehicle_index);
          committed_by_source[current_vehicle_info[index].source_vertex]
            .push_back(vehicle_index);
          committed_by_target[committed_target_info[index].target_vertex]
            .push_back(vehicle_index);
        }

        std::vector<int> committed_sources;
        std::vector<int> committed_targets;
        committed_sources.reserve(committed_by_source.size());
        committed_targets.reserve(committed_by_target.size());
        for (const auto& item : committed_by_source)
          committed_sources.push_back(item.first);
        for (const auto& item : committed_by_target)
          committed_targets.push_back(item.first);

        DistanceScratch committed_scratch;
        if (committed_sources.size() <= committed_targets.size()) {
          for (const auto& source_group : committed_by_source) {
            calculate_distances(
              g_state, source_group.first, false, committed_targets,
              committed_scratch);
            for (int vehicle_index : source_group.second) {
              const size_t index = static_cast<size_t>(vehicle_index);
              const VehicleRouteInfo& current = current_vehicle_info[index];
              const OrderRouteInfo& target = committed_target_info[index];
              const double middle =
                committed_scratch.dist[static_cast<size_t>(target.target_vertex)];
              if (std::isfinite(middle)) {
                committed_times[index] =
                  current.tail_time + middle + target.head_time;
              }
            }
          }
        } else {
          for (const auto& target_group : committed_by_target) {
            calculate_distances(
              g_state, target_group.first, true, committed_sources,
              committed_scratch);
            for (int vehicle_index : target_group.second) {
              const size_t index = static_cast<size_t>(vehicle_index);
              const VehicleRouteInfo& current = current_vehicle_info[index];
              const OrderRouteInfo& target = committed_target_info[index];
              const double middle =
                committed_scratch.dist[static_cast<size_t>(current.source_vertex)];
              if (std::isfinite(middle)) {
                committed_times[index] =
                  current.tail_time + middle + target.head_time;
              }
            }
          }
        }
      }

      // Resolve direct travel within the same compressed edge first.
      for (int order_index = 0; order_index < order_count; ++order_index) {
        for (int vehicle_index = 0; vehicle_index < vehicle_count; ++vehicle_index) {
          const size_t matrix_index = static_cast<size_t>(order_index) * vehicle_size
            + static_cast<size_t>(vehicle_index);
          if (!allowed_matrix[matrix_index]) continue;
          const double committed_time =
            committed_times[static_cast<size_t>(vehicle_index)];
          if (!std::isfinite(committed_time)) continue;
          double direct_time = 0.0;
          if (direct_same_edge_time(
                g_state,
                vehicle_info[static_cast<size_t>(vehicle_index)],
                order_info[static_cast<size_t>(order_index)],
                direct_time)) {
            pair_times[matrix_index] = committed_time + direct_time;
          }
        }
      }

      std::vector<unsigned char> active_vehicles(vehicle_size, 0);
      std::vector<unsigned char> active_orders(order_size, 0);
      for (int order_index = 0; order_index < order_count; ++order_index) {
        const OrderRouteInfo& order = order_info[static_cast<size_t>(order_index)];
        if (!order.mapped || !order.head_enabled) continue;
        for (int vehicle_index = 0; vehicle_index < vehicle_count; ++vehicle_index) {
          const size_t matrix_index = static_cast<size_t>(order_index) * vehicle_size
            + static_cast<size_t>(vehicle_index);
          const VehicleRouteInfo& vehicle = vehicle_info[static_cast<size_t>(vehicle_index)];
          if (!allowed_matrix[matrix_index] || std::isfinite(pair_times[matrix_index]) ||
              !std::isfinite(committed_times[static_cast<size_t>(vehicle_index)]) ||
              !vehicle.mapped || !vehicle.tail_enabled) {
            continue;
          }
          active_orders[static_cast<size_t>(order_index)] = 1;
          active_vehicles[static_cast<size_t>(vehicle_index)] = 1;
        }
      }

      std::unordered_map<int, std::vector<int>> vehicles_by_source;
      std::unordered_map<int, std::vector<int>> orders_by_target;
      for (int i = 0; i < vehicle_count; ++i) {
        const VehicleRouteInfo& info = vehicle_info[static_cast<size_t>(i)];
        if (active_vehicles[static_cast<size_t>(i)]) {
          vehicles_by_source[info.source_vertex].push_back(i);
        }
      }
      for (int i = 0; i < order_count; ++i) {
        const OrderRouteInfo& info = order_info[static_cast<size_t>(i)];
        if (active_orders[static_cast<size_t>(i)]) {
          orders_by_target[info.target_vertex].push_back(i);
        }
      }

      std::vector<int> source_vertices;
      std::vector<int> target_vertices;
      source_vertices.reserve(vehicles_by_source.size());
      target_vertices.reserve(orders_by_target.size());
      for (const auto& item : vehicles_by_source) source_vertices.push_back(item.first);
      for (const auto& item : orders_by_target) target_vertices.push_back(item.first);

      DistanceScratch scratch;
      const bool use_forward = source_vertices.size() <= target_vertices.size();
      if (use_forward) {
        for (const auto& source_group : vehicles_by_source) {
          calculate_distances(
            g_state, source_group.first, false, target_vertices, scratch);
          for (int vehicle_index : source_group.second) {
            const VehicleRouteInfo& vehicle = vehicle_info[static_cast<size_t>(vehicle_index)];
            for (int order_index = 0; order_index < order_count; ++order_index) {
              const size_t matrix_index = static_cast<size_t>(order_index) * vehicle_size
                + static_cast<size_t>(vehicle_index);
              if (!allowed_matrix[matrix_index] || std::isfinite(pair_times[matrix_index])) continue;
              const OrderRouteInfo& order = order_info[static_cast<size_t>(order_index)];
              if (!order.mapped || !order.head_enabled || order.target_vertex < 0) continue;
              const double middle_time = scratch.dist[static_cast<size_t>(order.target_vertex)];
              if (!std::isfinite(middle_time)) continue;
              pair_times[matrix_index] =
                committed_times[static_cast<size_t>(vehicle_index)] +
                vehicle.tail_time + middle_time + order.head_time;
            }
          }
        }
      } else {
        for (const auto& target_group : orders_by_target) {
          calculate_distances(
            g_state, target_group.first, true, source_vertices, scratch);
          for (int order_index : target_group.second) {
            const OrderRouteInfo& order = order_info[static_cast<size_t>(order_index)];
            for (int vehicle_index = 0; vehicle_index < vehicle_count; ++vehicle_index) {
              const size_t matrix_index = static_cast<size_t>(order_index) * vehicle_size
                + static_cast<size_t>(vehicle_index);
              if (!allowed_matrix[matrix_index] || std::isfinite(pair_times[matrix_index])) continue;
              const VehicleRouteInfo& vehicle = vehicle_info[static_cast<size_t>(vehicle_index)];
              if (!vehicle.mapped || !vehicle.tail_enabled || vehicle.source_vertex < 0) continue;
              const double middle_time = scratch.dist[static_cast<size_t>(vehicle.source_vertex)];
              if (!std::isfinite(middle_time)) continue;
              pair_times[matrix_index] =
                committed_times[static_cast<size_t>(vehicle_index)] +
                vehicle.tail_time + middle_time + order.head_time;
            }
          }
        }
      }
    }

    double max_real_time = 0.0;
    for (int order_index = 0; order_index < order_count; ++order_index) {
      for (int vehicle_index = 0; vehicle_index < vehicle_count; ++vehicle_index) {
        const size_t matrix_index = static_cast<size_t>(order_index) * vehicle_size
          + static_cast<size_t>(vehicle_index);
        const double time = pair_times[matrix_index];
        if (!std::isfinite(time)) continue;
        order_has_reachable[static_cast<size_t>(order_index)] = 1;
        if (time > max_real_time) max_real_time = time;
      }
    }

    const int row_count = std::min(order_count, vehicle_count);
    const bool order_rows = order_count <= vehicle_count;
    const int real_column_count = order_rows ? vehicle_count : order_count;
    const int column_count = real_column_count + row_count;
    const long double dummy_cost =
      (static_cast<long double>(max_real_time) + 1.0L) *
      (static_cast<long double>(row_count) + 1.0L);
    const long double forbidden_cost = dummy_cost * 2.0L + 1.0L;

    auto cost_at = [&](int row, int column) -> long double {
      if (column >= real_column_count) return dummy_cost;
      const int order_index = order_rows ? row : column;
      const int vehicle_index = order_rows ? column : row;
      const double time = pair_times[
        static_cast<size_t>(order_index) * vehicle_size + static_cast<size_t>(vehicle_index)];
      if (!std::isfinite(time)) return forbidden_cost;
      double assignment_cost = time;
      if (threshold_seconds > 0.0 &&
          orders[order_index].previous_vehicle_id != 0 &&
          orders[order_index].previous_vehicle_id == vehicles[vehicle_index].vehicle_id) {
        assignment_cost = std::max(
          0.0, assignment_cost - threshold_seconds);
      }
      return static_cast<long double>(assignment_cost);
    };

    const std::vector<int> row_to_column =
      hungarian_minimize(row_count, column_count, cost_at);

    for (int order_index = 0; order_index < order_count; ++order_index) {
      AI_ASSIGNMENT_RESULT& result = out_results[order_index];
      if (!order_has_allowed[static_cast<size_t>(order_index)]) {
        result.status = AI_ASSIGNMENT_NO_ALLOWED_VEHICLE;
      } else if (!order_has_reachable[static_cast<size_t>(order_index)]) {
        result.status = AI_ASSIGNMENT_UNREACHABLE;
      } else {
        result.status = AI_ASSIGNMENT_UNASSIGNED;
      }
    }

    for (int row = 0; row < row_count; ++row) {
      const int column = row_to_column[static_cast<size_t>(row)];
      if (column < 0 || column >= real_column_count) continue;
      const int order_index = order_rows ? row : column;
      const int vehicle_index = order_rows ? column : row;
      const double time = pair_times[
        static_cast<size_t>(order_index) * vehicle_size + static_cast<size_t>(vehicle_index)];
      if (!std::isfinite(time)) continue;
      AI_ASSIGNMENT_RESULT& result = out_results[order_index];
      result.vehicle_id = vehicles[vehicle_index].vehicle_id;
      result.expected_time = time;
      result.status = AI_ASSIGNMENT_ASSIGNED;
    }

    *out_result_count = order_count;
    return 1;
  } catch (const std::bad_alloc&) {
    LogFile("[Error] [AIModule] AI_OptimizeAssignments allocation failed.");
    return 0;
  }
}
