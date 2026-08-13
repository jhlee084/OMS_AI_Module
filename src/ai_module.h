#pragma once
#include <cstdint>

#ifdef _WIN32
  #define AI_API extern "C" __declspec(dllexport)
#else
  #define AI_API extern "C"
#endif

#pragma pack(push, 1)
struct AI_DECISION_INFO {
  int32_t cur_pt;
  int32_t dst_pt;
  int32_t selected_point;
  int32_t opt1_point;
  int32_t opt2_point;
  double q1;
  double q2;
  double p1;
  double p2;
  double toss;
};
#pragma pack(pop)

struct AI_VEHICLE_INPUT {
  int32_t vehicle_id;
  int32_t segment_id;
  int32_t offset;
};

struct AI_ORDER_INPUT {
  int32_t order_id;
  int32_t destination_segment_id;
  int32_t destination_offset;
};

enum AI_ASSIGNMENT_STATUS : int32_t {
  AI_ASSIGNMENT_ASSIGNED = 1,
  AI_ASSIGNMENT_NO_ALLOWED_VEHICLE = 2,
  AI_ASSIGNMENT_UNREACHABLE = 3,
  AI_ASSIGNMENT_UNASSIGNED = 4
};

struct AI_ASSIGNMENT_RESULT {
  int32_t order_id;
  int32_t vehicle_id;
  double expected_time;
  int32_t status;
  int32_t reserved;
};

static_assert(sizeof(AI_VEHICLE_INPUT) == 12, "Unexpected AI_VEHICLE_INPUT layout");
static_assert(sizeof(AI_ORDER_INPUT) == 12, "Unexpected AI_ORDER_INPUT layout");
static_assert(sizeof(AI_ASSIGNMENT_RESULT) == 24, "Unexpected AI_ASSIGNMENT_RESULT layout");

AI_API int __cdecl AI_OptimizeAssignments(
  const AI_ORDER_INPUT* orders,
  int32_t order_count,
  const AI_VEHICLE_INPUT* vehicles,
  int32_t vehicle_count,
  const uint8_t* allowed_matrix,
  AI_ASSIGNMENT_RESULT* out_results,
  int32_t* out_result_count);

namespace ai {
  enum SRC_OPT : int32_t { SOURCE_MEMORY = 1, SOURCE_FILE = 2 };
  enum DTYPE   : int32_t { TOPOLOGY = 1, QDATA = 2, QBUILD = 3, QLEARNED = 4, CONFIG = 5 };
}

void LogFile(const char* fmt, ...);
