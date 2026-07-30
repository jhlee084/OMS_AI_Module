#include "ai_module.h"
#include "graph_state.h"
#include "parse_utils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>
#include <windows.h>

#if USE_LEARN

namespace {

constexpr const char* kLearnedFormat = "OMS_AI_EDGE_BUCKET_LEARNING";
constexpr int kLearnedVersion = 1;

struct SnapshotEdgeLearning {
  int edge_id = 0;
  std::vector<int> seg_ids;
  std::array<LearnedAgg, LEARN_VEHICLE_BUCKET_MAX + 1> buckets{};
};

struct LearnedSnapshot {
  std::vector<SnapshotEdgeLearning> edges;
  int bucket_count = 0;
};

std::mutex g_autosave_mtx;
std::condition_variable g_autosave_cv;
uint64_t g_dirty_generation = 0;
uint64_t g_saved_generation = 0;
std::chrono::steady_clock::time_point g_last_autosave;
std::thread g_autosave_worker;
bool g_autosave_worker_running = false;
bool g_autosave_stop_requested = false;
bool g_autosave_save_in_progress = false;
bool g_autosave_has_pending_snapshot = false;
uint64_t g_autosave_pending_generation = 0;
LearnedSnapshot g_autosave_pending_snapshot;

struct PersistedEdgeLearning {
  int edge_id = 0;
  std::vector<int> seg_ids;
  std::array<LearnedAgg, LEARN_VEHICLE_BUCKET_MAX + 1> buckets{};
  std::array<bool, LEARN_VEHICLE_BUCKET_MAX + 1> has_bucket{};
};

bool same_segments(const EdgeChain& edge, const std::vector<int>& seg_ids) {
  return edge.seg_ids == seg_ids;
}

int resolve_edge_id_locked(const PersistedEdgeLearning& saved) {
  auto itById = g_state.edge_by_id.find(saved.edge_id);
  if (itById != g_state.edge_by_id.end() &&
      same_segments(itById->second, saved.seg_ids)) {
    return saved.edge_id;
  }

  for (const auto& kv : g_state.edge_by_id) {
    if (same_segments(kv.second, saved.seg_ids)) return kv.first;
  }
  return 0;
}

bool parse_saved_edge(
  const picojson::value& value, PersistedEdgeLearning& out) {
  if (!value.is_object()) return false;
  const auto& obj = value.get_object();

  auto itEdgeId = obj.find("edge_id");
  auto itSegments = obj.find("segments");
  auto itBuckets = obj.find("buckets");
  if (itEdgeId == obj.end() || itSegments == obj.end() ||
      itBuckets == obj.end()) {
    return false;
  }
  if (!parse_int(itEdgeId->second, out.edge_id) ||
      !itSegments->second.is_array() || !itBuckets->second.is_array()) {
    return false;
  }

  for (const auto& segValue : itSegments->second.get_array()) {
    int seg_id = 0;
    if (!parse_int(segValue, seg_id)) return false;
    out.seg_ids.push_back(seg_id);
  }
  if (out.seg_ids.empty()) return false;

  for (const auto& bucketValue : itBuckets->second.get_array()) {
    if (!bucketValue.is_object()) return false;
    const auto& bucketObj = bucketValue.get_object();
    auto itBucket = bucketObj.find("bucket");
    auto itSum = bucketObj.find("sum");
    auto itCount = bucketObj.find("count");
    if (itBucket == bucketObj.end() || itSum == bucketObj.end() ||
        itCount == bucketObj.end()) {
      return false;
    }

    int bucket = 0;
    double sum = 0.0;
    double count = 0.0;
    if (!parse_int(itBucket->second, bucket) ||
        !parse_double(itSum->second, sum) ||
        !parse_double(itCount->second, count)) {
      return false;
    }
    if (bucket < 0 || bucket > LEARN_VEHICLE_BUCKET_MAX ||
        !std::isfinite(sum) || !std::isfinite(count) ||
        sum < 0.0 || count <= 0.0 || out.has_bucket[bucket]) {
      return false;
    }

    const double cap = (double)LEARN_MAX_SAMPLES;
    if (cap > 0.0 && count > cap) {
      sum *= cap / count;
      count = cap;
    }
    out.buckets[bucket].sum = sum;
    out.buckets[bucket].count = count;
    out.has_bucket[bucket] = true;
  }
  return true;
}

std::filesystem::path learned_data_file_path() {
  const DWORD env_size =
    GetEnvironmentVariableW(L"OMS_AI_LEARNED_DATA_PATH", nullptr, 0);
  if (env_size > 1) {
    std::vector<wchar_t> env_value(env_size);
    if (GetEnvironmentVariableW(
          L"OMS_AI_LEARNED_DATA_PATH", env_value.data(), env_size) > 0) {
      return std::filesystem::path(env_value.data());
    }
  }

  HMODULE module = nullptr;
  if (GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&learned_data_file_path), &module)) {
    std::vector<wchar_t> module_path(32768);
    const DWORD length =
      GetModuleFileNameW(module, module_path.data(), (DWORD)module_path.size());
    if (length > 0 && length < module_path.size()) {
      return std::filesystem::path(
               std::wstring(module_path.data(), length))
        .parent_path() / L"OMS_AI_Module.learned.json";
    }
  }
  return std::filesystem::path(L"OMS_AI_Module.learned.json");
}

bool write_file_atomically(
  const std::filesystem::path& path, const std::string& data) {
  std::error_code ec;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return false;
  }

  std::filesystem::path temp = path;
  temp += L".tmp.";
  temp += std::to_wstring(GetCurrentProcessId());
  {
    std::ofstream output(temp, std::ios::binary | std::ios::trunc);
    if (!output) return false;
    output.write(data.data(), (std::streamsize)data.size());
    output.flush();
    if (!output) {
      output.close();
      std::filesystem::remove(temp, ec);
      return false;
    }
  }

  if (!MoveFileExW(
        temp.c_str(), path.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    std::filesystem::remove(temp, ec);
    return false;
  }
  return true;
}

}  // namespace

static LearnedSnapshot capture_learned_snapshot_locked() {
  LearnedSnapshot snapshot;
  std::vector<int> edge_ids;
  edge_ids.reserve(g_state.learned_edge_time_by_vehicle_count.size());
  for (const auto& kv : g_state.learned_edge_time_by_vehicle_count) {
    bool has_samples = false;
    for (const LearnedAgg& agg : kv.second) {
      if (agg.count > 0.0) {
        has_samples = true;
        break;
      }
    }
    if (has_samples && g_state.edge_by_id.find(kv.first) != g_state.edge_by_id.end()) {
      edge_ids.push_back(kv.first);
    }
  }
  std::sort(edge_ids.begin(), edge_ids.end());

  snapshot.edges.reserve(edge_ids.size());
  for (int edge_id : edge_ids) {
    SnapshotEdgeLearning saved;
    saved.edge_id = edge_id;
    saved.seg_ids = g_state.edge_by_id.at(edge_id).seg_ids;
    saved.buckets =
      g_state.learned_edge_time_by_vehicle_count.at(edge_id);
    for (const LearnedAgg& agg : saved.buckets) {
      if (agg.count > 0.0) snapshot.bucket_count++;
    }
    snapshot.edges.push_back(std::move(saved));
  }
  return snapshot;
}

static bool serialize_learned_snapshot(
  const LearnedSnapshot& snapshot, std::string& out_json) {
  std::ostringstream json;
  json.imbue(std::locale::classic());
  json << std::setprecision(17);
  json << "{\"format\":\"" << kLearnedFormat
       << "\",\"version\":" << kLearnedVersion
       << ",\"bucket_max\":" << LEARN_VEHICLE_BUCKET_MAX
       << ",\"max_samples\":" << LEARN_MAX_SAMPLES
       << ",\"edges\":[\n";

  bool first_edge = true;
  for (const SnapshotEdgeLearning& edge : snapshot.edges) {
    if (!first_edge) json << ",\n";
    first_edge = false;
    json << "{\"edge_id\":" << edge.edge_id << ",\"segments\":[";
    for (size_t i = 0; i < edge.seg_ids.size(); ++i) {
      if (i > 0) json << ',';
      json << edge.seg_ids[i];
    }
    json << "],\"buckets\":[";

    bool first_bucket = true;
    for (int bucket = 0; bucket <= LEARN_VEHICLE_BUCKET_MAX; ++bucket) {
      const LearnedAgg& agg = edge.buckets[bucket];
      if (agg.count <= 0.0) continue;
      if (!first_bucket) json << ',';
      first_bucket = false;
      json << "{\"bucket\":" << bucket
           << ",\"sum\":" << std::llround(agg.sum)
           << ",\"count\":" << agg.count << '}';
    }
    json << "]}";
  }
  json << "\n]}\n";
  out_json = json.str();

  LogFile(
    "[Dijkstra] learned data serialized. edges=%d buckets=%d bytes=%d",
    (int)snapshot.edges.size(), snapshot.bucket_count, (int)out_json.size());
  return true;
}

bool dijkstra_serialize_learned_data(std::string& out_json) {
  flush_edge_learning_worker();
  LearnedSnapshot snapshot;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    snapshot = capture_learned_snapshot_locked();
  }
  return serialize_learned_snapshot(snapshot, out_json);
}

bool dijkstra_restore_learned_data(
  const std::string& json_text, int* restored_buckets) {
  if (restored_buckets) *restored_buckets = 0;
  if (json_text.empty()) return false;

  picojson::value root;
  const std::string error = picojson::parse(root, json_text);
  if (!error.empty() || !root.is_object()) {
    LogFile("[Error] [Dijkstra] learned data parse failed. err=%s", error.c_str());
    return false;
  }

  const auto& obj = root.get_object();
  auto itFormat = obj.find("format");
  auto itVersion = obj.find("version");
  auto itEdges = obj.find("edges");
  int version = 0;
  if (itFormat == obj.end() || !itFormat->second.is_string() ||
      itFormat->second.get_string() != kLearnedFormat ||
      itVersion == obj.end() || !parse_int(itVersion->second, version) ||
      version != kLearnedVersion || itEdges == obj.end() ||
      !itEdges->second.is_array()) {
    LogFile("[Error] [Dijkstra] learned data header is invalid.");
    return false;
  }

  std::vector<PersistedEdgeLearning> saved_edges;
  saved_edges.reserve(itEdges->second.get_array().size());
  for (const auto& edgeValue : itEdges->second.get_array()) {
    PersistedEdgeLearning saved;
    if (!parse_saved_edge(edgeValue, saved)) {
      LogFile("[Error] [Dijkstra] learned data edge record is invalid.");
      return false;
    }
    saved_edges.push_back(std::move(saved));
  }

  flush_edge_learning_worker();
  std::lock_guard<std::mutex> lk(g_mtx);
  if (!g_state.has_topology) {
    LogFile("[Error] [Dijkstra] learned data restore requires topology first.");
    return false;
  }

  std::unordered_map<
    int, std::array<LearnedAgg, LEARN_VEHICLE_BUCKET_MAX + 1>> restored;
  int loaded_edges = 0;
  int loaded_buckets = 0;
  int skipped_edges = 0;
  for (const PersistedEdgeLearning& saved : saved_edges) {
    const int edge_id = resolve_edge_id_locked(saved);
    if (edge_id <= 0) {
      skipped_edges++;
      continue;
    }

    bool loaded_edge = false;
    for (int bucket = 0; bucket <= LEARN_VEHICLE_BUCKET_MAX; ++bucket) {
      if (!saved.has_bucket[bucket]) continue;
      merge_learned_agg(restored[edge_id][bucket], saved.buckets[bucket]);
      loaded_buckets++;
      loaded_edge = true;
    }
    if (loaded_edge) loaded_edges++;
  }

  g_state.learned_edge_time_by_vehicle_count = std::move(restored);
  refresh_all_edge_weight_cache_locked(g_state);
  if (restored_buckets) *restored_buckets = loaded_buckets;

  LogFile(
    "[Dijkstra] learned data restored. edges=%d buckets=%d skipped_edges=%d",
    loaded_edges, loaded_buckets, skipped_edges);
  return true;
}

void dijkstra_mark_learned_data_dirty() {
  std::lock_guard<std::mutex> lk(g_autosave_mtx);
  ++g_dirty_generation;
}

static void learned_autosave_worker_loop() {
  while (true) {
    LearnedSnapshot snapshot;
    uint64_t generation = 0;
    {
      std::unique_lock<std::mutex> lk(g_autosave_mtx);
      g_autosave_cv.wait(
        lk, []() {
          return g_autosave_stop_requested ||
                 g_autosave_has_pending_snapshot;
        });
      if (g_autosave_stop_requested &&
          !g_autosave_has_pending_snapshot) {
        break;
      }

      snapshot = std::move(g_autosave_pending_snapshot);
      generation = g_autosave_pending_generation;
      g_autosave_has_pending_snapshot = false;
      g_autosave_save_in_progress = true;
    }

    std::string learned_json;
    const bool serialized =
      serialize_learned_snapshot(snapshot, learned_json);
    const std::filesystem::path path = learned_data_file_path();
    const bool saved =
      serialized && write_file_atomically(path, learned_json);

    {
      std::lock_guard<std::mutex> lk(g_autosave_mtx);
      if (saved) {
        if (generation > g_saved_generation) {
          g_saved_generation = generation;
        }
        g_last_autosave = std::chrono::steady_clock::now();
      }
      g_autosave_save_in_progress = false;
    }
    g_autosave_cv.notify_all();

    if (saved) {
      LogFile(
        "[Dijkstra] learned data autosaved. path=%ls bytes=%d "
        "generation=%llu",
        path.c_str(), (int)learned_json.size(),
        (unsigned long long)generation);
    } else {
      LogFile(
        "[Error] [Dijkstra] learned data autosave failed. path=%ls",
        path.c_str());
    }
  }
}

void dijkstra_autosave_learned_data_if_due() {
  uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lk(g_autosave_mtx);
    if (g_dirty_generation == g_saved_generation) return;
    if (g_autosave_has_pending_snapshot ||
        g_autosave_save_in_progress) {
      return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (g_last_autosave.time_since_epoch().count() > 0 &&
        LEARN_AUTOSAVE_INTERVAL_MS > 0 &&
        std::chrono::duration_cast<std::chrono::milliseconds>(
          now - g_last_autosave).count() < LEARN_AUTOSAVE_INTERVAL_MS) {
      return;
    }
    generation = g_dirty_generation;
  }

  LearnedSnapshot snapshot;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    snapshot = capture_learned_snapshot_locked();
  }

  {
    std::lock_guard<std::mutex> lk(g_autosave_mtx);
    if (!g_autosave_worker_running) {
      g_autosave_stop_requested = false;
      g_autosave_worker =
        std::thread(learned_autosave_worker_loop);
      g_autosave_worker_running = true;
    }
    g_autosave_pending_snapshot = std::move(snapshot);
    g_autosave_pending_generation = generation;
    g_autosave_has_pending_snapshot = true;
  }
  g_autosave_cv.notify_one();
}

void dijkstra_stop_learned_autosave_worker() {
  {
    std::lock_guard<std::mutex> lk(g_autosave_mtx);
    if (!g_autosave_worker_running) return;
    g_autosave_stop_requested = true;
  }
  g_autosave_cv.notify_all();
  if (g_autosave_worker.joinable()) g_autosave_worker.join();

  {
    std::lock_guard<std::mutex> lk(g_autosave_mtx);
    g_autosave_worker_running = false;
    g_autosave_stop_requested = false;
    g_autosave_save_in_progress = false;
    g_autosave_has_pending_snapshot = false;
    g_autosave_pending_generation = 0;
    g_autosave_pending_snapshot = LearnedSnapshot{};
  }
}

bool dijkstra_load_learned_data_file() {
  const std::filesystem::path path = learned_data_file_path();
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::lock_guard<std::mutex> lk(g_autosave_mtx);
    g_dirty_generation = 0;
    g_saved_generation = 0;
    g_last_autosave = {};
    LogFile("[Dijkstra] learned data file not found. path=%ls", path.c_str());
    return false;
  }

  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof()) {
    LogFile(
      "[Error] [Dijkstra] learned data file read failed. path=%ls",
      path.c_str());
    return false;
  }

  int restored_buckets = 0;
  if (!dijkstra_restore_learned_data(contents.str(), &restored_buckets)) {
    LogFile(
      "[Error] [Dijkstra] learned data file restore failed. path=%ls",
      path.c_str());
    return false;
  }

  {
    std::lock_guard<std::mutex> lk(g_autosave_mtx);
    g_dirty_generation = 0;
    g_saved_generation = 0;
    g_last_autosave = std::chrono::steady_clock::now();
  }
  LogFile(
    "[Dijkstra] learned data file loaded. path=%ls buckets=%d bytes=%d",
    path.c_str(), restored_buckets, (int)contents.str().size());
  return true;
}

#else

bool dijkstra_serialize_learned_data(std::string& out_json) {
  out_json.clear();
  return false;
}

bool dijkstra_restore_learned_data(
  const std::string& json_text, int* restored_buckets) {
  (void)json_text;
  if (restored_buckets) *restored_buckets = 0;
  return false;
}

void dijkstra_mark_learned_data_dirty() {}
void dijkstra_autosave_learned_data_if_due() {}
void dijkstra_stop_learned_autosave_worker() {}
bool dijkstra_load_learned_data_file() { return false; }

#endif
