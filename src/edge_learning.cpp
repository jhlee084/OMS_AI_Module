#include "ai_module.h"
#include "graph_state.h"
#include "parse_utils.h"

#if USE_LEARN

static bool parse_message_object(const picojson::value& v, VehicleDataPoint& out) {
  if (!v.is_object()) return false;
  const auto& obj = v.get_object();

  auto itMode = obj.find("mode");
  auto itMoving = obj.find("moving_state");
  auto itErr = obj.find("error_code");
  auto itPD = obj.find("point_distance");
  auto itPid = obj.find("physical_id");
  auto itLP = obj.find("last_point");
  auto itNP = obj.find("next_point");
  auto itTs = obj.find("timestamp");
  if (itMode == obj.end() || itMoving == obj.end() || itErr == obj.end() || itPD == obj.end() ||
      itPid == obj.end() || itLP == obj.end() || itNP == obj.end() || itTs == obj.end()) {
    return false;
  }

  if (!itMode->second.is_string() || itMode->second.get_string() != "A") return false;
  if (!itMoving->second.is_string() || itMoving->second.get_string() != "M") return false;
  if (!itErr->second.is_string() || !itErr->second.get_string().empty()) return false;

  double point_distance = 0.0;
  if (!parse_double(itPD->second, point_distance)) return false;
  if (point_distance != 0.0) return false;

  if (itPid->second.is_string()) {
    out.physical_id = itPid->second.get_string();
  } else if (itPid->second.is_number()) {
    out.physical_id = std::to_string((int64_t)itPid->second.get_number());
  } else {
    return false;
  }
  if (out.physical_id.empty()) return false;

  if (!parse_int(itLP->second, out.last_point)) return false;
  if (!parse_int(itNP->second, out.next_point)) return false;
  if (!parse_int64(itTs->second, out.timestamp)) return false;
  return true;
}

static bool parse_message_json(const std::string& json_text, picojson::value& out_obj) {
  picojson::value root;
  std::string err = picojson::parse(root, json_text);
  if (!err.empty()) {
    LogFile("[Dijkstra] parse_message_json failed. err=%s", err.c_str());
    return false;
  }
  if (!root.is_object()) {
    LogFile("[Dijkstra] parse_message_json rejected. root_is_object=0");
    return false;
  }
  out_obj = std::move(root);
  return true;
}

static bool lookup_segment_edge_locked(
  int from_point, int to_point, int& out_seg_id, int& out_edge_id,
  int& out_seg_index, int& out_edge_seg_count, double& out_edge_base_w) {
  auto itPair = g_state.pair_to_seg_ids.find(seg_pair_key(from_point, to_point));
  if (itPair == g_state.pair_to_seg_ids.end()) return false;

  out_seg_id = itPair->second;
  auto itSegEdge = g_state.seg_to_edge_id.find(out_seg_id);
  auto itSegIndex = g_state.seg_index_in_edge.find(out_seg_id);
  if (itSegEdge == g_state.seg_to_edge_id.end() || itSegIndex == g_state.seg_index_in_edge.end()) {
    return false;
  }

  out_edge_id = itSegEdge->second;
  out_seg_index = itSegIndex->second;
  auto itEdge = g_state.edge_by_id.find(out_edge_id);
  if (itEdge == g_state.edge_by_id.end()) return false;

  out_edge_seg_count = (int)itEdge->second.seg_ids.size();
  out_edge_base_w = itEdge->second.base_w;
  return out_edge_seg_count > 0;
}

static void add_completed_segment_to_edge_batch(
  const VehicleDataPoint& prev, const VehicleDataPoint& cur, double dt,
  std::unordered_map<int, LearnedAgg>& out_edge_batch) {
  int seg_id = 0;
  int edge_id = 0;
  int seg_index = 0;
  int edge_seg_count = 0;
  double edge_base_w = -1.0;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (!lookup_segment_edge_locked(prev.last_point, prev.next_point, seg_id, edge_id,
                                    seg_index, edge_seg_count, edge_base_w)) {
      LogFile("[Dijkstra] edge learn skipped. physical_id=%s last_point=%d next_point=%d reason=segment_mapping_miss",
              cur.physical_id.c_str(), prev.last_point, prev.next_point);
      return;
    }
  }

  auto& progress = g_edge_progress_by_vehicle[cur.physical_id];
  if (seg_index == 0) {
    progress.edge_id = edge_id;
    progress.next_seg_index = 1;
    progress.elapsed = dt;
  } else if (progress.edge_id == edge_id && progress.next_seg_index == seg_index) {
    progress.next_seg_index++;
    progress.elapsed += dt;
  } else {
    g_edge_progress_by_vehicle.erase(cur.physical_id);
    LogFile("[Dijkstra] edge learn skipped. physical_id=%s seg_id=%d edge_id=%d seg_index=%d reason=edge_sequence_mismatch",
            cur.physical_id.c_str(), seg_id, edge_id, seg_index);
    return;
  }

  LogFile("[Dijkstra] edge progress. physical_id=%s seg_id=%d edge_id=%d seg_index=%d edge_seg_count=%d dt=%.3f elapsed=%.3f base=%.3f",
          cur.physical_id.c_str(), seg_id, edge_id, seg_index, edge_seg_count,
          dt, progress.elapsed, edge_base_w);

  if (progress.next_seg_index >= edge_seg_count) {
    double learn_elapsed = progress.elapsed;
    if (learn_elapsed < edge_base_w) {
      LogFile("[Dijkstra] edge learn clamped. physical_id=%s edge_id=%d base=%.3f elapsed=%.3f learned_elapsed=%.3f reason=edge_elapsed_lt_base",
              cur.physical_id.c_str(), edge_id, edge_base_w, progress.elapsed, edge_base_w);
      learn_elapsed = edge_base_w;
    }

    auto& agg = out_edge_batch[edge_id];
    agg.sum += learn_elapsed;
    agg.count += 1.0;
    const double batch_mean = agg.sum / agg.count;
    LogFile("[Dijkstra] edge batch. physical_id=%s edge_id=%d base=%.3f elapsed=%.3f learned_elapsed=%.3f batch_mean=%.3f batch_count=%.0f",
            cur.physical_id.c_str(), edge_id, edge_base_w, progress.elapsed, learn_elapsed,
            batch_mean, agg.count);
    g_edge_progress_by_vehicle.erase(cur.physical_id);
  }
}

static bool build_edge_learn_batch_from_payloads(
  const std::vector<std::string>& payloads, std::unordered_map<int, LearnedAgg>& out_edge_batch,
  int& accepted_messages) {
  accepted_messages = 0;
  out_edge_batch.clear();

  for (const auto& json_text : payloads) {
    picojson::value raw;
    if (!parse_message_json(json_text, raw)) continue;
    VehicleDataPoint p;
    if (!parse_message_object(raw, p)) continue;
    accepted_messages++;

    auto itPrev = g_last_message_by_vehicle.find(p.physical_id);
    if (itPrev != g_last_message_by_vehicle.end()) {
      const VehicleDataPoint& prev = itPrev->second;
      const int64_t dt = p.timestamp - prev.timestamp;
      if (p.last_point != prev.last_point) {
        if (prev.last_point != prev.next_point && p.last_point == prev.next_point && dt > 0) {
          add_completed_segment_to_edge_batch(prev, p, (double)dt, out_edge_batch);
        } else {
          g_edge_progress_by_vehicle.erase(p.physical_id);
          LogFile("[Dijkstra] edge learn skipped. physical_id=%s prev last_point=%d prev next_point=%d cur last_point=%d cur next_point=%d reason=transition_mismatch",
                  p.physical_id.c_str(), prev.last_point, prev.next_point, p.last_point, p.next_point);
        }
        itPrev->second = p;
      }
    } else {
      g_last_message_by_vehicle[p.physical_id] = p;
      LogFile("[Dijkstra] vehicle anchor created. physical_id=%s anchor=(%d->%d,%lld)",
              p.physical_id.c_str(), p.last_point, p.next_point, (long long)p.timestamp);
    }
  }

  if (accepted_messages <= 0) {
    LogFile("[Dijkstra] build_edge_learn_batch_from_payloads end. no accepted messages. payloads=%d",
            (int)payloads.size());
    return false;
  }
  LogFile("[Dijkstra] build_edge_learn_batch_from_payloads end. edges=%d", (int)out_edge_batch.size());
  return !out_edge_batch.empty();
}

static void commit_edge_learn_batch_locked(
  const std::unordered_map<int, LearnedAgg>& edge_batch, int& learned_edges) {
  learned_edges = 0;

  for (const auto& kv : edge_batch) {
    const int edge_id = kv.first;
    auto itEdge = g_state.edge_by_id.find(edge_id);
    if (itEdge == g_state.edge_by_id.end()) continue;

    merge_learned_agg(g_state.learned_edge_time_by_id[edge_id], kv.second);
    refresh_edge_weight_cache_locked(g_state, edge_id);
    const auto& committed = g_state.learned_edge_time_by_id[edge_id];
    const double committed_mean = (committed.count > 0.0) ? (committed.sum / committed.count) : -1.0;
    const double batch_mean = (kv.second.count > 0.0) ? (kv.second.sum / kv.second.count) : -1.0;

    LogFile("[Dijkstra] edge committed. edge_id=%d start_point=%d end_point=%d seg_count=%d base=%.3f batch_mean=%.3f batch_count=%.0f committed_mean=%.3f committed_count=%.0f",
            edge_id, itEdge->second.start_point, itEdge->second.end_point,
            (int)itEdge->second.seg_ids.size(), itEdge->second.base_w,
            batch_mean, kv.second.count, committed_mean, committed.count);
    learned_edges++;
  }
}

static void learn_worker_loop() {
  LogFile("[Dijkstra] learn_worker_loop started.");
  while (true) {
    std::vector<std::string> payloads;
    {
      std::unique_lock<std::mutex> lk(g_learn_queue_mtx);
      g_learn_cv.wait(lk, []() { return g_learn_stop_requested || !g_learn_queue.empty(); });
      if (g_learn_stop_requested && g_learn_queue.empty()) break;
      while (!g_learn_queue.empty()) {
        payloads.push_back(std::move(g_learn_queue.front()));
        g_learn_queue.pop_front();
      }
    }

    std::unordered_map<int, LearnedAgg> edge_batch;
    int accepted_messages = 0;
    bool is_ok = build_edge_learn_batch_from_payloads(payloads, edge_batch, accepted_messages);
    if (!is_ok) continue;

    int learned_edges = 0;
    {
      std::lock_guard<std::mutex> lk(g_mtx);
      commit_edge_learn_batch_locked(edge_batch, learned_edges);
    }
  }
  LogFile("[Dijkstra] learn_worker_loop stopped.");
}

static void ensure_learn_worker_started() {
  std::lock_guard<std::mutex> lk(g_learn_queue_mtx);
  if (g_learn_worker_running) return;
  g_learn_stop_requested = false;
  g_learn_worker = std::thread(learn_worker_loop);
  g_learn_worker_running = true;
}

void stop_edge_learning_worker() {
  {
    std::lock_guard<std::mutex> lk(g_learn_queue_mtx);
    if (!g_learn_worker_running) return;
    g_learn_stop_requested = true;
    g_learn_queue.clear();
  }
  g_learn_cv.notify_all();
  if (g_learn_worker.joinable()) g_learn_worker.join();
  {
    std::lock_guard<std::mutex> lk(g_learn_queue_mtx);
    g_learn_worker_running = false;
    g_learn_stop_requested = false;
  }
}

bool dijkstra_learn_from_message_json(const std::string& json_text, DijkstraLearnStats* stats) {
  if (stats) {
    stats->accepted_messages = 0;
    stats->learned_segments = 0;
  }
  if (json_text.empty()) return false;

  ensure_learn_worker_started();
  {
    std::lock_guard<std::mutex> lk(g_learn_queue_mtx);
    g_learn_queue.push_back(json_text);
  }
  g_learn_cv.notify_one();
  return true;
}

#else

bool dijkstra_learn_from_message_json(const std::string& json_text, DijkstraLearnStats* stats) {
  (void)json_text;
  if (stats) *stats = DijkstraLearnStats{};
  return false;
}

void stop_edge_learning_worker() {}

#endif