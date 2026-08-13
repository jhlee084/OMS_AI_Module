#include "ai_module.h"
#include "dijkstra.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
static std::mutex g_log_mtx;

void LogFile(const char* fmt, ...)
{
    std::lock_guard<std::mutex> lk(g_log_mtx);

    FILE* fp = nullptr;
    fopen_s(&fp, "OMS_AI_Module.log", "a");
    if (!fp) return;

    va_list args;
    va_start(args, fmt);
    vfprintf(fp, fmt, args);
    fprintf(fp, "\n");
    va_end(args);

    fclose(fp);
}

static void logf(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  std::vfprintf(stdout, fmt, args);
  std::fprintf(stdout, "\n");
  std::fflush(stdout);
  va_end(args);
}

// ---------------- Exported C API ----------------

AI_API int __cdecl AI_InitModule() {
  dijkstra_clear();
  logf("[AIModule] AI_InitModule()");
  return 1;
}

AI_API int __cdecl AI_UninitModule() {
  dijkstra_clear();
  logf("[AIModule] AI_UninitModule()");
  return 1;
}

AI_API int __cdecl AI_PrepareData(int src_opt, int data_type, unsigned char* data, int size) {
  LogFile("[AIModule] data_type=%d. size=%d", data_type, size);
  if (data_type == ai::DTYPE::CONFIG) {
      LogFile("[AIModule] CONFIG is not used. Ignore. data_type=%d", data_type);
    return 0;
  }
  if (!data || size <= 0) {
      LogFile("[Error] [AIModule] invalid prepare data. data=%p size=%d", data, size);
      return 0;
  }
  if (data_type == ai::DTYPE::QLEARNED) {
    if (src_opt != ai::SRC_OPT::SOURCE_MEMORY) {
      LogFile(
        "[Error] [AIModule] QLEARNED supports SOURCE_MEMORY only. src_opt=%d",
        src_opt);
      return 0;
    }

    std::string json_text(
      reinterpret_cast<char*>(data),
      reinterpret_cast<char*>(data) + size);
    int restored_buckets = 0;
    const bool ok =
      dijkstra_restore_learned_data(json_text, &restored_buckets);
    if (!ok) {
      LogFile("[Error] [AIModule] Failed to restore learned data.");
      return 0;
    }
    LogFile(
      "[AIModule] Restored learned data. buckets=%d bytes=%d",
      restored_buckets, size);
    return restored_buckets > 0 ? restored_buckets : 1;
  }

  if (data_type != ai::DTYPE::TOPOLOGY) {
    LogFile("[AIModule] Unsupported data_type=%d", data_type);
    return 0;
  }

  std::string json_text(reinterpret_cast<char*>(data), reinterpret_cast<char*>(data) + size);

  DijkstraStats st;
  bool ok = build_graph_from_track_json(json_text, &st);
  if (ok) {
      logf("[AIModule] Loaded topology. segments=%d, vertices=%zu", st.segments_loaded, st.vertices);
      LogFile("[AIModule] Loaded topology. segments=%d, vertices=%zu", st.segments_loaded, st.vertices);
      dijkstra_load_learned_data_file();
  }
  else {
      logf("[AIModule] Failed to load topology.");
      LogFile("[Error] [AIModule] Failed to load topology.");
  }
  return ok ? st.segments_loaded : 0;
}

AI_API int __cdecl AI_PutMessage(unsigned char* data, int size) {
#if !USE_LEARN
  (void)data;
  (void)size;
  return 1;
#else
  if (!data || size <= 0) return 0;
  std::string json_text(reinterpret_cast<char*>(data), reinterpret_cast<char*>(data) + size);
  DijkstraLearnStats st;
  bool ok = dijkstra_learn_from_message_json(json_text, &st);
  if (!ok) {
    LogFile("[AIModule] AI_PutMessage parse failed.");
    return 0;
  }
  //LogFile("[AIModule] AI_PutMessage data_size=%d", size);
  return 1;
#endif
}

AI_API int __cdecl AI_InitRouter() {
  return dijkstra_has_topology() ? 1 : 0;
}

AI_API int __cdecl AI_Start() { return 1; }
AI_API int __cdecl AI_Stop() { return 1; }
AI_API int __cdecl AI_ShutDown() { return 1; }
AI_API int __cdecl AI_GetLearnedQueueValuesSize() {
  std::string learned_json;
  if (!dijkstra_serialize_learned_data(learned_json) ||
      learned_json.size() > (size_t)INT_MAX) {
    LogFile("[Error] [AIModule] Failed to get learned data size.");
    return 0;
  }
  return (int)learned_json.size();
}

AI_API int __cdecl AI_SaveLearnedQueueValues(
  int src_opt, unsigned char* data, int* size) {
  if (src_opt != ai::SRC_OPT::SOURCE_MEMORY || !size) {
    LogFile(
      "[Error] [AIModule] Invalid learned save request. src_opt=%d size_ptr=%p",
      src_opt, size);
    return 0;
  }

  std::string learned_json;
  if (!dijkstra_serialize_learned_data(learned_json) ||
      learned_json.size() > (size_t)INT_MAX) {
    LogFile("[Error] [AIModule] Failed to serialize learned data.");
    return 0;
  }

  const int required = (int)learned_json.size();
  const int capacity = *size;
  if (!data || capacity < required) {
    *size = required;
    LogFile(
      "[AIModule] Learned save buffer is too small. capacity=%d required=%d",
      capacity, required);
    return 0;
  }

  std::memcpy(data, learned_json.data(), learned_json.size());
  *size = required;
  LogFile("[AIModule] Saved learned data. bytes=%d", required);
  return 1;
}
AI_API int __cdecl AI_Update() { return 1; }

AI_API int __cdecl AI_GetExpectedPath(int cur_point, int dst_point, bool using_init, bool is_mtl,
                                      double* p_etime, int* p_path, int* path_size) {
  (void)using_init; (void)is_mtl;
  if (!p_etime || !p_path || !path_size) {
      LogFile("[Error] [AIModule] p_etime: %f, p_path: %d, path_size: %d", *p_etime, *p_path, *path_size);
      return 0;
  }

  std::vector<int> path;
  double t = 0.0;
  if (!dijkstra_shortest_path(cur_point, dst_point, path, t)) {
    *p_etime = 0.0;
    *path_size = 0;
    LogFile("[Error] [AIModule] Dijkstra Error ..");
    return 0;
  }

  *p_etime = t;
  int cap = *path_size;
  int max_out = (cap > 0) ? cap : 4096;
  int n = (int)path.size();
  if (n > max_out) n = max_out;

  for (int i = 0; i < n; ++i) p_path[i] = path[i];
  *path_size = n;

  //LogFile("[AIModule] Dijkstra OK path_size: %d", *path_size);
  return 1;
}

AI_API int __cdecl AI_EnableSegments(unsigned char* source, int size) { (void)source; (void)size; return 0; }

AI_API int __cdecl AI_EnableSegment(int segment_id) { dijkstra_enable_segment(segment_id); return 1; }

AI_API int __cdecl AI_DisableSegment(int segment_id) { dijkstra_disable_segment(segment_id); return 1; }

AI_API int __cdecl AI_DoRouting(int cur_point, int dst_point, bool using_init,
                                int* p_cur_point, int* p_dst_point, int* p_next_point,
                                double* p_etime, int** p_path, int* path_size) {
  (void)using_init;
  if (p_cur_point) *p_cur_point = cur_point;
  if (p_dst_point) *p_dst_point = dst_point;

  std::vector<int> path;
  double t = 0.0;
  if (!dijkstra_shortest_path(cur_point, dst_point, path, t)) {
    if (p_next_point) *p_next_point = 0;
    if (p_etime) *p_etime = 0.0;
    if (path_size) *path_size = 0;
    return 0;
  }
  if (p_next_point) *p_next_point = (path.size() >= 2) ? path[1] : dst_point;
  if (p_etime) *p_etime = t;

  // Ownership contract unknown: do not allocate.
  if (p_path) *p_path = nullptr;
  if (path_size) *path_size = 0;
  return 1;
}

AI_API int __cdecl AI_GetQueueValue(unsigned char* key1, int size1, unsigned char* key2, int size2, double* p_q1, double* p_q2) {
  (void)key1; (void)size1; (void)key2; (void)size2;
  if (p_q1) *p_q1 = 0.0;
  if (p_q2) *p_q2 = 0.0;
  return 1;
}

AI_API int __cdecl AI_ClearCache() { return 1; }
AI_API int __cdecl AI_AddCache(int cur_vtx, int dst_vtx, int pt, void* decision_info) { (void)cur_vtx;(void)dst_vtx;(void)pt;(void)decision_info; return 0; }
AI_API int __cdecl AI_GetCache(int cur_vtx, int dst_vtx, int* p_pt, void* decision_info) { (void)cur_vtx;(void)dst_vtx;(void)decision_info; if (p_pt) *p_pt = 0; return 0; }

AI_API int __cdecl AI_CalculateP(double q1, double q2, double* p_p1, double* p_p2) {
  if (p_p1) *p_p1 = 0.0;
  if (p_p2) *p_p2 = 0.0;
  double s = q1 + q2;
  if (s > 0) {
    if (p_p1) *p_p1 = q1 / s;
    if (p_p2) *p_p2 = q2 / s;
  }
  return 1;
}

AI_API int __cdecl AI_MakeChoice(int cur_pt, int dst_pt, bool use_init, int* p_pt, void* decision_info) {
  (void)use_init; (void)decision_info;
  std::vector<int> path;
  double t = 0.0;
  if (!dijkstra_shortest_path(cur_pt, dst_pt, path, t)) { if (p_pt) *p_pt = 0; return 0; }
  if (p_pt) *p_pt = (path.size() >= 2) ? path[1] : dst_pt;
  return 1;
}

AI_API int __cdecl AI_DoUpdate() { return 1; }

AI_API int __cdecl AI_getExpectedPath(int start_pt, int dst_pt, double min_etime, int max_choice, bool use_init,
                                      double* p_etime, int* p_path, int* path_size) {
  (void)min_etime; (void)max_choice; (void)use_init;
  return AI_GetExpectedPath(start_pt, dst_pt, false, false, p_etime, p_path, path_size);
}

AI_API int __cdecl AI_GetTime(int start_pt_id, int end_pt_id, double* p_time, bool use_init, double start_offset, double end_offset) {
  (void)use_init;
  if (!p_time) return 0;
  *p_time = 0.0;
  // start_offset and end_offset are elapsed-time values, not positions.
  // The point-to-point route contains the whole first segment, so remove the
  // already travelled time. Add the time from the destination point to its
  // offset location. Invalid time inputs are rejected at the native boundary.
  if (!std::isfinite(start_offset) || !std::isfinite(end_offset) ||
      start_offset < 0.0 || end_offset < 0.0) {
    return 0;
  }
  std::vector<int> path;
  double t = 0.0;
  if (!dijkstra_shortest_path(start_pt_id, end_pt_id, path, t)) return 0;

  const double remaining_time = std::max(0.0, t - start_offset);
  *p_time = remaining_time + end_offset;
  return 1;
}

AI_API int __cdecl AI_GetPath(int start_pt_id, int end_pt_id, int** p_path, int* path_size) {
  (void)start_pt_id; (void)end_pt_id;
  if (p_path) *p_path = nullptr;
  if (path_size) *path_size = 0;
  return 1;
}
