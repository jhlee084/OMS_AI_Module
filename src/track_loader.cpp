#include "graph_state.h"
#include "parse_utils.h"

#include <cstdlib>
#include <unordered_map>
#include <unordered_set>

bool build_graph_from_track_json(const std::string& json_text, DijkstraStats* stats) {
#if USE_LEARN
  stop_edge_learning_worker();
  dijkstra_stop_learned_autosave_worker();
#endif

  picojson::value root;
  std::string err = picojson::parse(root, json_text);
  if (!err.empty() || !root.is_object()) return false;

  const auto& obj = root.get_object();
  auto itSeg = obj.find("segments");
  if (itSeg == obj.end()) return false;

  std::unordered_map<int, Seg> seg_by_id;
  std::unordered_map<int, std::vector<int>> out_seg_ids_by_point;
  std::unordered_map<int, std::vector<int>> in_seg_ids_by_point;
  std::unordered_set<int> disabled;
  std::unordered_set<int> all_points;
#if USE_LEARN
  std::unordered_map<uint64_t, int> pair_to_seg_ids;
#endif
  std::vector<int> all_seg_ids;
  int seg_cnt = 0;

  auto process_segment_obj = [&](const picojson::object& so, int fallback_id) {
    int seg_id = fallback_id;

    auto itId = so.find("id");
    if (itId != so.end()) {
      parse_int(itId->second, seg_id);
    } else {
      auto itIdent = so.find("identifier");
      if (itIdent != so.end()) parse_int(itIdent->second, seg_id);
    }

    int sp = 0;
    int ep = 0;
    auto itSp = so.find("start_point");
    auto itEp = so.find("end_point");
    if (itSp == so.end() || itEp == so.end()) return;
    if (!parse_int(itSp->second, sp)) return;
    if (!parse_int(itEp->second, ep)) return;

    auto itUnuse = so.find("unuse");
    const bool is_unuse =
      (itUnuse != so.end() && itUnuse->second.is_bool() && itUnuse->second.get_bool());

    double travel_time = 0.0;
    double length = 0.0;
    double speed = 0.0;
    double speed_ratio = 100.0;
    auto itTT = so.find("travel_time");
    auto itL = so.find("length");
    auto itS = so.find("speed");
    auto itSR = so.find("speed_ratio");
    if (itTT != so.end()) parse_double(itTT->second, travel_time);
    if (itL != so.end()) parse_double(itL->second, length);
    if (itS != so.end()) parse_double(itS->second, speed);
    if (itSR != so.end()) parse_double(itSR->second, speed_ratio);

    if (travel_time <= 0.0) {
      const double eff_speed = speed * (speed_ratio / 100.0);
      if (length > 0.0 && eff_speed > 0.0) {
        travel_time = (length * 1000.0) / eff_speed;
      } else {
        travel_time = 1.0;
      }
    }

    Seg seg;
    seg.from = sp;
    seg.to = ep;
    seg.offset_length = length > 0.0 ? length * 1000.0 : 0.0;
    seg.base_w = travel_time;
    seg.seg_id = seg_id;

    seg_by_id[seg_id] = seg;
    out_seg_ids_by_point[sp].push_back(seg_id);
    in_seg_ids_by_point[ep].push_back(seg_id);
    all_seg_ids.push_back(seg_id);
    all_points.insert(sp);
    all_points.insert(ep);
#if USE_LEARN
    pair_to_seg_ids[seg_pair_key(sp, ep)] = seg_id;
#endif
    seg_cnt++;

    if (is_unuse) disabled.insert(seg_id);
  };

  const picojson::value& segv = itSeg->second;
  if (segv.is_object()) {
    for (const auto& kv : segv.get_object()) {
      int key_id = std::atoi(kv.first.c_str());
      if (!kv.second.is_object()) continue;
      process_segment_obj(kv.second.get_object(), key_id);
    }
  } else if (segv.is_array()) {
    for (const auto& sv : segv.get_array()) {
      if (!sv.is_object()) continue;
      process_segment_obj(sv.get_object(), 0);
    }
  } else {
    return false;
  }

  std::unordered_map<int, EdgeChain> edge_by_id;
  std::unordered_map<int, std::vector<int>> edge_out_ids_by_point;
  std::unordered_map<int, int> start_edge_id_by_point;
  std::unordered_map<int, int> goal_edge_id_by_point;
  std::unordered_map<int, int> internal_edge_id_by_point;
  std::unordered_map<int, std::vector<int>> entering_edge_ids_by_point;
  std::unordered_map<int, std::vector<int>> leaving_edge_ids_by_point;
  std::unordered_map<int, int> seg_to_edge_id;
  std::unordered_map<int, int> seg_index_in_edge;
  std::unordered_set<int> visited_seg_ids;
  int next_edge_id = 1;

  auto out_degree = [&](int point) -> size_t {
    auto it = out_seg_ids_by_point.find(point);
    return (it == out_seg_ids_by_point.end()) ? 0 : it->second.size();
  };
  auto in_degree = [&](int point) -> size_t {
    auto it = in_seg_ids_by_point.find(point);
    return (it == in_seg_ids_by_point.end()) ? 0 : it->second.size();
  };

  auto build_edge_from_seed = [&](int seed_seg_id) {
    if (visited_seg_ids.find(seed_seg_id) != visited_seg_ids.end()) return;

    auto itSeed = seg_by_id.find(seed_seg_id);
    if (itSeed == seg_by_id.end()) return;

    EdgeChain edge;
    edge.edge_id = next_edge_id++;
    edge.start_point = itSeed->second.from;
    edge.point_ids.push_back(edge.start_point);

    int cur_seg_id = seed_seg_id;
    while (true) {
      if (visited_seg_ids.find(cur_seg_id) != visited_seg_ids.end()) break;

      const Seg& seg = seg_by_id[cur_seg_id];
      visited_seg_ids.insert(cur_seg_id);
      seg_to_edge_id[cur_seg_id] = edge.edge_id;
      seg_index_in_edge[cur_seg_id] = (int)edge.seg_ids.size();
      edge.seg_ids.push_back(cur_seg_id);
      edge.point_ids.push_back(seg.to);
      edge.base_w += seg.base_w;
      edge.end_point = seg.to;

      if (out_degree(seg.to) != 1 || in_degree(seg.to) != 1) break;

      const auto& next_ids = out_seg_ids_by_point[seg.to];
      const int next_seg_id = next_ids.front();
      if (visited_seg_ids.find(next_seg_id) != visited_seg_ids.end()) break;
      cur_seg_id = next_seg_id;
    }

    edge_out_ids_by_point[edge.start_point].push_back(edge.edge_id);
    leaving_edge_ids_by_point[edge.start_point].push_back(edge.edge_id);
    entering_edge_ids_by_point[edge.end_point].push_back(edge.edge_id);
    for (size_t i = 1; i + 1 < edge.point_ids.size(); ++i) {
      internal_edge_id_by_point[edge.point_ids[i]] = edge.edge_id;
    }
    edge_by_id[edge.edge_id] = std::move(edge);
  };

  for (int seg_id : all_seg_ids) {
    auto itSegById = seg_by_id.find(seg_id);
    if (itSegById == seg_by_id.end()) continue;
    const Seg& seg = itSegById->second;
    if (out_degree(seg.from) == 1 && in_degree(seg.from) == 1) continue;
    build_edge_from_seed(seg_id);
  }

  for (int seg_id : all_seg_ids) {
    build_edge_from_seed(seg_id);
  }

  std::unordered_map<int, int> point_index_by_id;
  std::vector<int> point_id_by_index;
  point_id_by_index.reserve(all_points.size());
  for (int point : all_points) {
    const int idx = (int)point_id_by_index.size();
    point_index_by_id[point] = idx;
    point_id_by_index.push_back(point);
  }

  std::vector<std::vector<int>> edge_out_ids_by_vertex(point_id_by_index.size());
  for (auto& kv : edge_by_id) {
    EdgeChain& edge = kv.second;
    auto itStartVertex = point_index_by_id.find(edge.start_point);
    auto itEndVertex = point_index_by_id.find(edge.end_point);
    if (itStartVertex == point_index_by_id.end() || itEndVertex == point_index_by_id.end()) continue;
    edge.start_vertex = itStartVertex->second;
    edge.end_vertex = itEndVertex->second;
    edge_out_ids_by_vertex[edge.start_vertex].push_back(edge.edge_id);
  }

  for (int point : all_points) {
    auto itInternal = internal_edge_id_by_point.find(point);
    if (itInternal != internal_edge_id_by_point.end()) {
      start_edge_id_by_point[point] = itInternal->second;
      goal_edge_id_by_point[point] = itInternal->second;
      continue;
    }

    const auto itEntering = entering_edge_ids_by_point.find(point);
    const auto itLeaving = leaving_edge_ids_by_point.find(point);
    const std::vector<int>* entering =
      (itEntering == entering_edge_ids_by_point.end()) ? nullptr : &itEntering->second;
    const std::vector<int>* leaving =
      (itLeaving == leaving_edge_ids_by_point.end()) ? nullptr : &itLeaving->second;

    const bool is_branch_point = out_degree(point) > 1;
    const bool is_merge_point = in_degree(point) > 1;

    if (is_branch_point && entering && !entering->empty()) {
      start_edge_id_by_point[point] = (*entering)[0];
      goal_edge_id_by_point[point] = (*entering)[0];
      continue;
    }

    if (is_merge_point && leaving && !leaving->empty()) {
      start_edge_id_by_point[point] = (*leaving)[0];
      goal_edge_id_by_point[point] = (*leaving)[0];
      continue;
    }

    if (entering && !entering->empty()) {
      start_edge_id_by_point[point] = (*entering)[0];
      goal_edge_id_by_point[point] = (*entering)[0];
      continue;
    }

    if (leaving && !leaving->empty()) {
      start_edge_id_by_point[point] = (*leaving)[0];
      goal_edge_id_by_point[point] = (*leaving)[0];
    }
  }

  {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_state = GraphState{};
    g_state.seg_by_id = std::move(seg_by_id);
    g_state.out_seg_ids_by_point = std::move(out_seg_ids_by_point);
    g_state.in_seg_ids_by_point = std::move(in_seg_ids_by_point);
    g_state.edge_by_id = std::move(edge_by_id);
    g_state.edge_out_ids_by_point = std::move(edge_out_ids_by_point);
    g_state.point_index_by_id = std::move(point_index_by_id);
    g_state.point_id_by_index = std::move(point_id_by_index);
    g_state.edge_out_ids_by_vertex = std::move(edge_out_ids_by_vertex);
    g_state.start_edge_id_by_point = std::move(start_edge_id_by_point);
    g_state.goal_edge_id_by_point = std::move(goal_edge_id_by_point);
    g_state.seg_to_edge_id = std::move(seg_to_edge_id);
    g_state.seg_index_in_edge = std::move(seg_index_in_edge);
    g_state.disabled_seg = std::move(disabled);
#if USE_LEARN
    g_state.pair_to_seg_ids = std::move(pair_to_seg_ids);
    g_state.learned_edge_time_by_vehicle_count.clear();
    g_state.moving_vehicle_count_by_edge.clear();
    g_state.latest_vehicle_timestamp = 0;
    g_last_message_by_vehicle.clear();
    g_edge_progress_by_vehicle.clear();
    g_vehicle_occupancy.clear();
#endif
    g_state.edge_out_by_vertex.assign(g_state.point_id_by_index.size(), std::vector<EdgeChain*>());
    g_state.edge_in_by_vertex.assign(g_state.point_id_by_index.size(), std::vector<EdgeChain*>());
    for (auto& kv : g_state.edge_by_id) {
      EdgeChain& edge = kv.second;

      if (edge.start_vertex >= 0 && edge.start_vertex < (int)g_state.edge_out_by_vertex.size()) {
        g_state.edge_out_by_vertex[edge.start_vertex].push_back(&edge);
      }
      if (edge.end_vertex >= 0 && edge.end_vertex < (int)g_state.edge_in_by_vertex.size()) {
        g_state.edge_in_by_vertex[edge.end_vertex].push_back(&edge);
      }
    }
    refresh_all_edge_weight_cache_locked(g_state);
    g_state.has_topology = true;
  }

  if (stats) {
    stats->segments_loaded = seg_cnt;
    stats->vertices = all_points.size();
  }
  return true;
}
