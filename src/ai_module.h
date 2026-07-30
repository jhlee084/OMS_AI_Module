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

namespace ai {
  enum SRC_OPT : int32_t { SOURCE_MEMORY = 1, SOURCE_FILE = 2 };
  enum DTYPE   : int32_t { TOPOLOGY = 1, QDATA = 2, QBUILD = 3, QLEARNED = 4, CONFIG = 5 };
}

void LogFile(const char* fmt, ...);
