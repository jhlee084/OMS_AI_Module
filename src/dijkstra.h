#pragma once
#include <string>
#include <unordered_set>
#include <vector>

// Compile-time switch: 1 = use learned segment time, 0 = use track travel_time only.
#ifndef USE_LEARN
#define USE_LEARN 1
#endif

// Max effective sample size per segment for learning.
// Older data influence is gradually reduced after this cap.
#ifndef LEARN_MAX_SAMPLES
#define LEARN_MAX_SAMPLES 10
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
bool dijkstra_shortest_path(int start, int goal, std::vector<int>& out_path, double& out_time);

// segment enable/disable helpers
void dijkstra_enable_segment(int seg_id);
void dijkstra_disable_segment(int seg_id);
bool dijkstra_has_topology();
void dijkstra_clear();
