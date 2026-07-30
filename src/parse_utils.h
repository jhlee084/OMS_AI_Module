#pragma once

#include "picojson.h"
#include <cstdint>
#include <cstdlib>

inline bool parse_int(const picojson::value& v, int& out) {
  if (v.is_number()) {
    out = (int)v.get_number();
    return true;
  }
  if (v.is_string()) {
    out = std::atoi(v.get_string().c_str());
    return true;
  }
  return false;
}

inline bool parse_double(const picojson::value& v, double& out) {
  if (v.is_number()) {
    out = v.get_number();
    return true;
  }
  if (v.is_string()) {
    out = std::atof(v.get_string().c_str());
    return true;
  }
  return false;
}

#if USE_LEARN
inline bool parse_int64(const picojson::value& v, int64_t& out) {
  if (v.is_number()) {
    out = (int64_t)v.get_number();
    return true;
  }
  if (v.is_string()) {
    out = std::atoll(v.get_string().c_str());
    return true;
  }
  return false;
}
#endif
