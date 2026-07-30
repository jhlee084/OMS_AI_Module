#include "graph_state.h"

#include <cmath>
#include <limits>

GraphState g_state;
std::mutex g_mtx;

#if USE_LEARN
std::unordered_map<std::string, VehicleDataPoint> g_last_message_by_vehicle;
std::unordered_map<std::string, VehicleEdgeProgress> g_edge_progress_by_vehicle;
std::mutex g_learn_queue_mtx;
std::condition_variable g_learn_cv;
std::deque<std::string> g_learn_queue;
std::thread g_learn_worker;
bool g_learn_worker_running = false;
bool g_learn_stop_requested = false;
#endif

void append_points_no_dup(std::vector<int>& dst, const std::vector<int>& src) {
  if (src.empty()) return;
  size_t begin = 0;
  if (!dst.empty() && dst.back() == src.front()) begin = 1;
  dst.insert(dst.end(), src.begin() + begin, src.end());
}

bool is_seg_enabled_locked(const GraphState& state, int seg_id) {
  return state.disabled_seg.find(seg_id) == state.disabled_seg.end();
}

void refresh_edge_weight_cache_locked(GraphState& state, int edge_id) {
  auto itEdge = state.edge_by_id.find(edge_id);
  if (itEdge == state.edge_by_id.end()) return;

  EdgeChain& edge = itEdge->second;
  double edge_w = edge.base_w;
#if USE_LEARN
  auto itLearn = state.learned_edge_time_by_id.find(edge.edge_id);
  if (itLearn != state.learned_edge_time_by_id.end()) {
    edge_w = blend_learned_time(edge.base_w, &itLearn->second);
  }
#endif

  edge.current_w = edge_w;
  edge.prefix_w.assign(edge.seg_ids.size() + 1, 0.0);
  for (size_t i = 0; i < edge.seg_ids.size(); ++i) {
    const int seg_id = edge.seg_ids[i];
    auto itSeg = state.seg_by_id.find(seg_id);
    double seg_w = 0.0;
    if (itSeg != state.seg_by_id.end()) {
      if (edge.base_w > 0.0 && itSeg->second.base_w > 0.0) {
        seg_w = edge_w * (itSeg->second.base_w / edge.base_w);
      } else if (!edge.seg_ids.empty()) {
        seg_w = edge_w / (double)edge.seg_ids.size();
      }
      itSeg->second.current_w = seg_w;
    }
    edge.prefix_w[i + 1] = edge.prefix_w[i] + seg_w;
  }
}

void refresh_all_edge_weight_cache_locked(GraphState& state) {
  std::vector<int> edge_ids;
  edge_ids.reserve(state.edge_by_id.size());
  for (const auto& kv : state.edge_by_id) edge_ids.push_back(kv.first);
  for (int edge_id : edge_ids) refresh_edge_weight_cache_locked(state, edge_id);
}

double seg_effective_weight_locked(const GraphState& state, int seg_id) {
  auto itSeg = state.seg_by_id.find(seg_id);
  if (itSeg == state.seg_by_id.end()) return 0.0;
  return itSeg->second.current_w;
}

double edge_effective_weight_locked(const GraphState& state, const EdgeChain& edge) {
  if (state.disabled_seg.empty()) return edge.current_w;
  for (int seg_id : edge.seg_ids) {
    if (!is_seg_enabled_locked(state, seg_id)) return std::numeric_limits<double>::infinity();
  }
  return edge.current_w;
}

double edge_subpath_weight_locked(const EdgeChain& edge, size_t begin_idx, size_t end_idx) {
  if (begin_idx > end_idx || end_idx >= edge.point_ids.size()) return 0.0;
  if (edge.prefix_w.size() == edge.seg_ids.size() + 1 && end_idx < edge.prefix_w.size()) {
    return edge.prefix_w[end_idx] - edge.prefix_w[begin_idx];
  }

  double sum = 0.0;
  for (size_t i = begin_idx; i < end_idx && i + 1 < edge.prefix_w.size(); ++i) {
    sum += edge.prefix_w[i + 1] - edge.prefix_w[i];
  }
  return sum;
}

#if USE_LEARN
void merge_learned_agg(LearnedAgg& dst, const LearnedAgg& add) {
  if (add.count <= 0.0) return;

  const double cap = (double)LEARN_MAX_SAMPLES;
  if (cap <= 1.0) {
    dst = add;
    return;
  }

  double taken = 0.0;
  if (dst.count < cap) {
    const double space = cap - dst.count;
    const double take = (add.count < space) ? add.count : space;
    if (take > 0.0) {
      const double ratio = take / add.count;
      dst.sum += add.sum * ratio;
      dst.count += take;
      taken = take;
    }
  }

  const double overflow = add.count - taken;
  if (overflow > 0.0) {
    const double mean_add = add.sum / add.count;
    if (dst.count <= 0.0) {
      dst.sum = mean_add * cap;
      dst.count = cap;
      return;
    }
    double mean_dst = dst.sum / dst.count;
    const double alpha = 1.0 / cap;
    const double alpha_eff = 1.0 - std::pow(1.0 - alpha, overflow);
    mean_dst = mean_dst + alpha_eff * (mean_add - mean_dst);
    dst.sum = mean_dst * cap;
    dst.count = cap;
  }
}

uint64_t seg_pair_key(int sp, int ep) {
  return (uint64_t)(uint32_t)sp << 32 | (uint64_t)(uint32_t)ep;
}

double blend_learned_time(double base_w, const LearnedAgg* agg) {
  if (!agg || agg->count <= 0.0) return base_w;

  double sample_conf = agg->count / (double)LEARN_MAX_SAMPLES;
  if (sample_conf > 1.0) sample_conf = 1.0;

  const double learned_mean = agg->sum / agg->count;
  return base_w * (1.0 - sample_conf) + learned_mean * sample_conf;
}
#endif