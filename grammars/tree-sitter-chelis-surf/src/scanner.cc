#include <tree_sitter/parser.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <system_error>

namespace {

enum TokenType {
  CANONICAL_NUMBER,
  CANONICAL_PATTERN_NUMBER,
  CANONICAL_AXIS_INTEGER,
  CANONICAL_NONZERO_AXIS_INTEGER,
};

bool is_digit(int32_t value) { return value >= '0' && value <= '9'; }

bool consume_digits(TSLexer *lexer, std::string &text) {
  bool consumed = false;
  while (is_digit(lexer->lookahead)) {
    consumed = true;
    text.push_back(static_cast<char>(lexer->lookahead));
    lexer->advance(lexer, false);
  }
  return consumed;
}

std::string canonical_float(double value) {
  char scientific[64];
  const auto result = std::to_chars(
      scientific, scientific + sizeof(scientific), value,
      std::chars_format::scientific);
  if (result.ec != std::errc()) {
    return {};
  }

  const std::string shortest(scientific, result.ptr);
  const std::size_t exponent_pos = shortest.find('e');
  if (exponent_pos == std::string::npos) {
    return {};
  }

  std::string digits;
  for (std::size_t index = 0; index < exponent_pos; ++index) {
    if (shortest[index] != '.') {
      digits.push_back(shortest[index]);
    }
  }

  int exponent = 0;
  const char *exponent_begin = shortest.data() + exponent_pos + 1;
  const char *exponent_end = shortest.data() + shortest.size();
  if (exponent_begin != exponent_end && *exponent_begin == '+') {
    ++exponent_begin;
  }
  const auto exponent_result =
      std::from_chars(exponent_begin, exponent_end, exponent);
  if (exponent_result.ec != std::errc() || exponent_result.ptr != exponent_end) {
    return {};
  }

  const int kk = exponent + 1;
  const int length = static_cast<int>(digits.size());
  const int k = kk - length;
  if (k >= 0 && kk <= 16) {
    return digits + std::string(static_cast<std::size_t>(k), '0') + ".0";
  }
  if (kk > 0 && kk <= 16) {
    return digits.substr(0, static_cast<std::size_t>(kk)) + "." +
           digits.substr(static_cast<std::size_t>(kk));
  }
  if (kk > -5 && kk <= 0) {
    return "0." + std::string(static_cast<std::size_t>(-kk), '0') + digits;
  }

  std::string rendered(1, digits.front());
  if (digits.size() > 1) {
    rendered += "." + digits.substr(1);
  }
  rendered += "e" + std::to_string(exponent);
  return rendered;
}

bool is_canonical_number(const std::string &numeric,
                         const std::string &suffix,
                         bool pattern_context) {
  if (numeric.empty() || (numeric.size() > 1 && numeric.front() == '0' &&
                          numeric[1] != '.')) {
    return false;
  }

  if (pattern_context && !suffix.empty()) {
    return false;
  }

  const bool is_float = numeric.find_first_of(".e") != std::string::npos;
  if (!is_float) {
    if (suffix.empty()) {
      return true;
    }
    return suffix == "i8" || suffix == "i16" || suffix == "i32" ||
           suffix == "i64";
  }

  if (!suffix.empty() && suffix != "f16" && suffix != "bf16" &&
      suffix != "f32" && suffix != "f64") {
    return false;
  }

  double value = 0.0;
  const char *begin = numeric.data();
  const char *end = numeric.data() + numeric.size();
  const auto parsed = std::from_chars(begin, end, value);
  return parsed.ec == std::errc() && parsed.ptr == end &&
         std::isfinite(value) && canonical_float(value) == numeric;
}

bool scan_number(TSLexer *lexer, const bool *valid_symbols) {
  const bool expression_context = valid_symbols[CANONICAL_NUMBER];
  const bool pattern_context = valid_symbols[CANONICAL_PATTERN_NUMBER];
  const bool axis_context = valid_symbols[CANONICAL_AXIS_INTEGER];
  const bool nonzero_axis_context =
      valid_symbols[CANONICAL_NONZERO_AXIS_INTEGER];
  if (!expression_context && !pattern_context && !axis_context &&
      !nonzero_axis_context) {
    return false;
  }

  while (lexer->lookahead == ' ' || lexer->lookahead == '\t' ||
         lexer->lookahead == '\n' || lexer->lookahead == '\r') {
    lexer->advance(lexer, true);
  }
  if (!is_digit(lexer->lookahead)) {
    return false;
  }

  std::string numeric;
  consume_digits(lexer, numeric);

  if (lexer->lookahead == '.') {
    numeric.push_back('.');
    lexer->advance(lexer, false);
    if (!consume_digits(lexer, numeric)) {
      return false;
    }
  }

  if (lexer->lookahead == 'e') {
    numeric.push_back('e');
    lexer->advance(lexer, false);
    if (lexer->lookahead == '-' || lexer->lookahead == '+') {
      numeric.push_back(static_cast<char>(lexer->lookahead));
      lexer->advance(lexer, false);
    }
    if (!consume_digits(lexer, numeric)) {
      return false;
    }
  }

  std::string suffix;
  while ((lexer->lookahead >= 'A' && lexer->lookahead <= 'Z') ||
         (lexer->lookahead >= 'a' && lexer->lookahead <= 'z') ||
         is_digit(lexer->lookahead) || lexer->lookahead == '_') {
    suffix.push_back(static_cast<char>(lexer->lookahead));
    lexer->advance(lexer, false);
  }
  lexer->mark_end(lexer);

  if (!expression_context && !pattern_context &&
      (axis_context || nonzero_axis_context)) {
    const bool canonical_axis = suffix.empty() &&
                                numeric.find_first_of(".e") == std::string::npos &&
                                (numeric == "0" || numeric.front() != '0');
    if (!canonical_axis || (nonzero_axis_context && numeric == "0")) {
      return false;
    }
    lexer->result_symbol = nonzero_axis_context
                               ? CANONICAL_NONZERO_AXIS_INTEGER
                               : CANONICAL_AXIS_INTEGER;
    return true;
  }

  const bool use_pattern = pattern_context && !expression_context;
  if (!is_canonical_number(numeric, suffix, use_pattern)) {
    return false;
  }

  lexer->result_symbol =
      use_pattern ? CANONICAL_PATTERN_NUMBER : CANONICAL_NUMBER;
  return true;
}

}  // namespace

extern "C" {

void *tree_sitter_chelis_surf_external_scanner_create() { return nullptr; }

void tree_sitter_chelis_surf_external_scanner_destroy(void *) {}

unsigned tree_sitter_chelis_surf_external_scanner_serialize(void *, char *) {
  return 0;
}

void tree_sitter_chelis_surf_external_scanner_deserialize(void *, const char *,
                                                          unsigned) {}

bool tree_sitter_chelis_surf_external_scanner_scan(void *, TSLexer *lexer,
                                                   const bool *valid_symbols) {
  return scan_number(lexer, valid_symbols);
}

}  // extern "C"
