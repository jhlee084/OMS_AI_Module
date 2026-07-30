#pragma once

#include "dijkstra.h"
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Seg {
  int from = 0;
  int to = 0;
  double base_w = 0.0;
  double current_w = 0.0;
  int seg_id = 0;
};

struct EdgeChain {
  int edge_id = 0;
  int start_point = 0;
  int end_point = 0;
  int start_vertex = -1;
  int end_vertex = -1;
  double base_w = 0.0;
  double current_w = 0.0;
  std::vector<int> seg_ids;
  std::vector<int> point_ids;
  std::vector<double> prefix_w;
};

struct LearnedAgg {
  double sum = 0.0;
  double count = 0.0;
};

struct VehicleDataPoint {
  std::string physical_id;
  int last_point = 0;
  int next_point = 0;
  int64_t timestamp = 0;
};


#if USE_LEARN
struct VehicleEdgeProgress {
  int edge_id = 0;
  int next_seg_index = 0;
  double elapsed = 0.0;
};
#endif

struct RouteHeapItem {
  int vertex = 0;
  double dist = 0.0;
};

struct RoutingScratch {
  std::vector<double> dist;
  std::vector<int> prev_vertex;
  std::vector<const EdgeChain*> prev_edge;
  std::vector<RouteHeapItem> heap;
  std::vector<const EdgeChain*> rev_edges;
  std::vector<const EdgeChain*> middle_edges;
  std::vector<int> prefix_path;
  std::vector<int> suffix_path;
};
struct GraphState {
  bool has_topology = false;
  std::unordered_map<int, Seg> seg_by_id;
  std::unordered_map<int, std::vector<int>> out_seg_ids_by_point;
  std::unordered_map<int, std::vector<int>> in_seg_ids_by_point;
  std::unordered_map<int, EdgeChain> edge_by_id;
  std::unordered_map<int, std::vector<int>> edge_out_ids_by_point;
  std::unordered_map<int, int> point_index_by_id;
  std::vector<int> point_id_by_index;
  std::vector<std::vector<int>> edge_out_ids_by_vertex;
  std::vector<std::vector<EdgeChain*>> edge_out_by_vertex;
  std::unordered_map<int, int> start_edge_id_by_point;
  std::unordered_map<int, int> goal_edge_id_by_point;
  std::unordered_map<int, int> seg_to_edge_id;
  std::unordered_map<int, int> seg_index_in_edge;
  std::unordered_set<int> disabled_seg;
  RoutingScratch routing_scratch;
#if USE_LEARN
  std::unordered_map<int, LearnedAgg> learned_edge_time_by_id;
  std::unordered_map<uint64_t, int> pair_to_seg_ids;
#endif
};

extern GraphState g_state;
extern std::mutex g_mtx;

#if USE_LEARN
extern std::unordered_map<std::string, VehicleDataPoint> g_last_message_by_vehicle;
extern std::unordered_map<std::string, VehicleEdgeProgress> g_edge_progress_by_vehicle;
extern std::mutex g_learn_queue_mtx;
extern std::condition_variable g_learn_cv;
extern std::deque<std::string> g_learn_queue;
extern std::thread g_learn_worker;
extern bool g_learn_worker_running;
extern bool g_learn_stop_requested;
#endif

void append_points_no_dup(std::vector<int>& dst, const std::vector<int>& src);
bool is_seg_enabled_locked(const GraphState& state, int seg_id);
double seg_effective_weight_locked(const GraphState& state, int seg_id);
double edge_effective_weight_locked(const GraphState& state, const EdgeChain& edge);
double edge_subpath_weight_locked(const EdgeChain& edge, size_t begin_idx, size_t end_idx);
void refresh_edge_weight_cache_locked(GraphState& state, int edge_id);
void refresh_all_edge_weight_cache_locked(GraphState& state);

#if USE_LEARN
void merge_learned_agg(LearnedAgg& dst, const LearnedAgg& add);
uint64_t seg_pair_key(int sp, int ep);
double blend_learned_time(double base_w, const LearnedAgg* agg);
void stop_edge_learning_worker();
#endif
