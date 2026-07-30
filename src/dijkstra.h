#pragma once
#include <string>
#include <unordered_set>
#include <vector>

// Compile-time switch: 1 = use learned segment time, 0 = use track travel_time only.
#ifndef USE_LEARN
#define USE_LEARN 1
#endif

// Max effective sample size per edge/vehicle-count bucket.
// After this cap, new samples are applied as an EWMA with alpha=1/cap.
#ifndef LEARN_MAX_SAMPLES
#define LEARN_MAX_SAMPLES 10
#endif

// Vehicle-count buckets are 0, 1, ..., LEARN_VEHICLE_BUCKET_MAX.
// The last bucket includes LEARN_VEHICLE_BUCKET_MAX or more vehicles.
#ifndef LEARN_VEHICLE_BUCKET_MAX
#define LEARN_VEHICLE_BUCKET_MAX 5
#endif

// A vehicle is removed from live edge occupancy when no newer status has been
// observed for this amount of event time.
#ifndef LEARN_VEHICLE_STALE_MS
#define LEARN_VEHICLE_STALE_MS 10000
#endif

// Dirty learned data is written automatically at most once per interval.
#ifndef LEARN_AUTOSAVE_INTERVAL_MS
#define LEARN_AUTOSAVE_INTERVAL_MS 300000
#endif

struct DijkstraStats {
  int segments_loaded = 0;
  size_t vertices = 0;
};

struct DijkstraLearnStats {
  int accepted_messages = 0;
  int learned_segments = 0;
};

bool build_graph_from_track_json(const std::string& json_text, DijkstraStats* stats = nullptr);
bool dijkstra_learn_from_message_json(const std::string& json_text, DijkstraLearnStats* stats = nullptr);
bool dijkstra_serialize_learned_data(std::string& out_json);
bool dijkstra_restore_learned_data(
  const std::string& json_text, int* restored_buckets = nullptr);
void dijkstra_mark_learned_data_dirty();
void dijkstra_autosave_learned_data_if_due();
void dijkstra_stop_learned_autosave_worker();
bool dijkstra_load_learned_data_file();
bool dijkstra_shortest_path(int start, int goal, std::vector<int>& out_path, double& out_time);

// segment enable/disable helpers
void dijkstra_enable_segment(int seg_id);
void dijkstra_disable_segment(int seg_id);
bool dijkstra_has_topology();
void dijkstra_clear();
