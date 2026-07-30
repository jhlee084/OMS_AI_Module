// picojson.h - a header-file-only JSON parser (public domain / ISC-like)
// Source: https://github.com/kazuho/picojson (trimmed comments). Included inline for convenience.
#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace picojson {

enum {
  null_type,
  boolean_type,
  number_type,
  string_type,
  array_type,
  object_type
};

class value;

typedef std::vector<value> array;
typedef std::map<std::string, value> object;

class value {
 public:
  typedef object::size_type size_type;

  value() : type_(null_type) {}
  value(std::nullptr_t) : type_(null_type) {}
  value(bool b) : type_(boolean_type) { u_.boolean_ = b; }
  value(double n) : type_(number_type) { u_.number_ = n; }
  value(const std::string& s) : type_(string_type), s_(s) {}
  value(const char* s) : type_(string_type), s_(s) {}
  value(const array& a) : type_(array_type), a_(a) {}
  value(const object& o) : type_(object_type), o_(o) {}

  int get_type() const { return type_; }

  bool is_null() const { return type_ == null_type; }
  bool is_bool() const { return type_ == boolean_type; }
  bool is_number() const { return type_ == number_type; }
  bool is_string() const { return type_ == string_type; }
  bool is_array() const { return type_ == array_type; }
  bool is_object() const { return type_ == object_type; }

  bool get_bool() const { return u_.boolean_; }
  double get_number() const { return u_.number_; }
  const std::string& get_string() const { return s_; }
  const array& get_array() const { return a_; }
  const object& get_object() const { return o_; }

  array& get_array() { return a_; }
  object& get_object() { return o_; }
  std::string& get_string() { return s_; }

  const value& operator[](size_t i) const { return a_[i]; }
  const value& operator[](const std::string& k) const { return o_.find(k)->second; }

  bool contains(const std::string& k) const { return o_.find(k) != o_.end(); }

 private:
  int type_;
  union {
    bool boolean_;
    double number_;
  } u_{};
  std::string s_;
  array a_;
  object o_;
};

class input {
 public:
  explicit input(const std::string& s) : s_(s), i_(0) {}
  int getc() { return i_ < s_.size() ? s_[i_++] : -1; }
  void ungetc() { if (i_) --i_; }
 private:
  const std::string& s_;
  size_t i_;
};

inline void skip_ws(input& in) {
  for (;;) {
    int c = in.getc();
    if (c == -1) return;
    if (!std::isspace(static_cast<unsigned char>(c))) { in.ungetc(); return; }
  }
}

inline bool match(input& in, const char* pat) {
  for (const char* p = pat; *p; ++p) {
    int c = in.getc();
    if (c != *p) return false;
  }
  return true;
}

inline bool parse_string(input& in, std::string& out, std::string& err) {
  out.clear();
  for (;;) {
    int c = in.getc();
    if (c == -1) { err = "unexpected EOF in string"; return false; }
    if (c == '"') return true;
    if (c == '\\') {
      int e = in.getc();
      if (e == -1) { err = "unexpected EOF in escape"; return false; }
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          int code = 0;
          for (int i = 0; i < 4; ++i) {
            int h = in.getc();
            if (h == -1) { err = "unexpected EOF in \\u"; return false; }
            code <<= 4;
            if (h >= '0' && h <= '9') code |= (h - '0');
            else if (h >= 'a' && h <= 'f') code |= (10 + h - 'a');
            else if (h >= 'A' && h <= 'F') code |= (10 + h - 'A');
            else { err = "invalid hex in \\u"; return false; }
          }
          if (code <= 0x7f) out.push_back(static_cast<char>(code));
          else if (code <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | ((code >> 6) & 0x1f)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
          } else {
            out.push_back(static_cast<char>(0xe0 | ((code >> 12) & 0x0f)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
          }
          break;
        }
        default: err = "invalid escape"; return false;
      }
    } else {
      out.push_back(static_cast<char>(c));
    }
  }
}

inline bool parse_value(input& in, value& out, std::string& err);

inline bool parse_array(input& in, array& out, std::string& err) {
  out.clear();
  skip_ws(in);
  int c = in.getc();
  if (c == ']') return true;
  in.ungetc();
  for (;;) {
    value v;
    if (!parse_value(in, v, err)) return false;
    out.push_back(v);
    skip_ws(in);
    c = in.getc();
    if (c == ']') return true;
    if (c != ',') { err = "expected , or ]"; return false; }
  }
}

inline bool parse_object(input& in, object& out, std::string& err) {
  out.clear();
  skip_ws(in);
  int c = in.getc();
  if (c == '}') return true;
  in.ungetc();
  for (;;) {
    skip_ws(in);
    c = in.getc();
    if (c != '"') { err = "expected string key"; return false; }
    std::string key;
    if (!parse_string(in, key, err)) return false;
    skip_ws(in);
    c = in.getc();
    if (c != ':') { err = "expected :"; return false; }
    value v;
    if (!parse_value(in, v, err)) return false;
    out[key] = v;
    skip_ws(in);
    c = in.getc();
    if (c == '}') return true;
    if (c != ',') { err = "expected , or }"; return false; }
  }
}

inline bool parse_number(input& in, double& out, std::string& err) {
  std::string s;
  int c = in.getc();
  if (c == '-') { s.push_back('-'); c = in.getc(); }
  if (c == -1) { err = "unexpected EOF in number"; return false; }
  if (!(c >= '0' && c <= '9')) { err = "invalid number"; return false; }
  if (c == '0') {
    s.push_back('0');
    c = in.getc();
  } else {
    while (c >= '0' && c <= '9') { s.push_back(static_cast<char>(c)); c = in.getc(); }
  }
  if (c == '.') {
    s.push_back('.');
    c = in.getc();
    if (!(c >= '0' && c <= '9')) { err = "invalid number"; return false; }
    while (c >= '0' && c <= '9') { s.push_back(static_cast<char>(c)); c = in.getc(); }
  }
  if (c == 'e' || c == 'E') {
    s.push_back('e');
    c = in.getc();
    if (c == '+' || c == '-') { s.push_back(static_cast<char>(c)); c = in.getc(); }
    if (!(c >= '0' && c <= '9')) { err = "invalid number"; return false; }
    while (c >= '0' && c <= '9') { s.push_back(static_cast<char>(c)); c = in.getc(); }
  }
  if (c != -1) in.ungetc();
  char* endp = nullptr;
  out = std::strtod(s.c_str(), &endp);
  if (!endp || *endp) { err = "invalid number"; return false; }
  return true;
}

inline bool parse_value(input& in, value& out, std::string& err) {
  skip_ws(in);
  int c = in.getc();
  if (c == -1) { err = "unexpected EOF"; return false; }
  switch (c) {
    case 'n': if (!match(in, "ull")) { err = "invalid token"; return false; } out = value(nullptr); return true;
    case 't': if (!match(in, "rue")) { err = "invalid token"; return false; } out = value(true); return true;
    case 'f': if (!match(in, "alse")) { err = "invalid token"; return false; } out = value(false); return true;
    case '"': { std::string s; if (!parse_string(in, s, err)) return false; out = value(s); return true; }
    case '[': { array a; if (!parse_array(in, a, err)) return false; out = value(a); return true; }
    case '{': { object o; if (!parse_object(in, o, err)) return false; out = value(o); return true; }
    default:
      if (c == '-' || (c >= '0' && c <= '9')) {
        in.ungetc();
        double n;
        if (!parse_number(in, n, err)) return false;
        out = value(n);
        return true;
      }
      err = "unexpected character";
      return false;
  }
}

inline std::string parse(value& out, const std::string& s) {
  input in(s);
  std::string err;
  if (!parse_value(in, out, err)) return err;
  skip_ws(in);
  if (in.getc() != -1) return "trailing characters";
  return "";
}

}  // namespace picojson
