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
  CANONICAL_MIN_INTEGER_MAGNITUDE,
  CANONICAL_MIN_PATTERN_MAGNITUDE,
  CANONICAL_STRING,
  CANONICAL_IDENTIFIER,
  CANONICAL_RECORD_FIELD_NAME,
  CANONICAL_RECORD_PATTERN_FIELD_NAME,
  CANONICAL_PIPE_LAMBDA_FN,
};

bool is_digit(int32_t value) { return value >= '0' && value <= '9'; }

bool is_identifier_start(int32_t value) {
  return value == '_' || (value >= 'a' && value <= 'z');
}

bool is_identifier_continue(int32_t value) {
  return is_identifier_start(value) || (value >= 'A' && value <= 'Z') ||
         is_digit(value);
}

std::string consume_identifier(TSLexer *lexer) {
  std::string identifier;
  while (is_identifier_continue(lexer->lookahead)) {
    identifier.push_back(static_cast<char>(lexer->lookahead));
    lexer->advance(lexer, false);
  }
  return identifier;
}

bool is_reserved_identifier(const std::string &identifier) {
  return identifier == "_" || identifier == "unit" || identifier == "def" ||
         identifier == "sig" || identifier == "type" || identifier == "dim" ||
         identifier == "macro" || identifier == "match" || identifier == "with" ||
         identifier == "fn" || identifier == "module" || identifier == "import" ||
         identifier == "if" || identifier == "then" || identifier == "else" ||
         identifier == "grad" || identifier == "vmap" || identifier == "jit" ||
         identifier == "realize" || identifier == "copy" ||
         identifier == "tensor" || identifier == "cast" ||
         identifier == "export" || identifier == "par" || identifier == "do" ||
         identifier == "quote" || identifier == "unquote" ||
         identifier == "splice" || identifier == "true" || identifier == "false";
}

void skip_line_comment(TSLexer *lexer) {
  while (!lexer->eof(lexer) && lexer->lookahead != '\n' &&
         lexer->lookahead != '\r') {
    lexer->advance(lexer, false);
  }
}

bool skip_trivia_for_lookahead(TSLexer *lexer) {
  while (!lexer->eof(lexer)) {
    while (lexer->lookahead == ' ' || lexer->lookahead == '\t' ||
           lexer->lookahead == '\n' || lexer->lookahead == '\r') {
      lexer->advance(lexer, false);
    }
    if (lexer->lookahead == '-') {
      lexer->advance(lexer, false);
      if (lexer->lookahead != '-') {
        return false;
      }
      lexer->advance(lexer, false);
      skip_line_comment(lexer);
      continue;
    }
    if (lexer->lookahead == '{') {
      lexer->advance(lexer, false);
      if (lexer->lookahead != '-') {
        return false;
      }
      lexer->advance(lexer, false);
      unsigned depth = 1;
      while (!lexer->eof(lexer) && depth > 0) {
        if (lexer->lookahead == '{') {
          lexer->advance(lexer, false);
          if (lexer->lookahead == '-') {
            lexer->advance(lexer, false);
            ++depth;
          }
        } else if (lexer->lookahead == '-') {
          lexer->advance(lexer, false);
          if (lexer->lookahead == '}') {
            lexer->advance(lexer, false);
            --depth;
          }
        } else {
          lexer->advance(lexer, false);
        }
      }
      if (depth != 0) {
        return false;
      }
      continue;
    }
    return true;
  }
  return false;
}

bool consume_arrow_after_trivia(TSLexer *lexer) {
  while (!lexer->eof(lexer)) {
    while (lexer->lookahead == ' ' || lexer->lookahead == '\t' ||
           lexer->lookahead == '\n' || lexer->lookahead == '\r') {
      lexer->advance(lexer, false);
    }
    if (lexer->lookahead == '-') {
      lexer->advance(lexer, false);
      if (lexer->lookahead == '>') {
        lexer->advance(lexer, false);
        return true;
      }
      if (lexer->lookahead != '-') {
        return false;
      }
      lexer->advance(lexer, false);
      skip_line_comment(lexer);
      continue;
    }
    if (lexer->lookahead != '{') {
      return false;
    }
    lexer->advance(lexer, false);
    if (lexer->lookahead != '-') {
      return false;
    }
    lexer->advance(lexer, false);
    unsigned depth = 1;
    while (!lexer->eof(lexer) && depth > 0) {
      if (lexer->lookahead == '{') {
        lexer->advance(lexer, false);
        if (lexer->lookahead == '-') {
          lexer->advance(lexer, false);
          ++depth;
        }
      } else if (lexer->lookahead == '-') {
        lexer->advance(lexer, false);
        if (lexer->lookahead == '}') {
          lexer->advance(lexer, false);
          --depth;
        }
      } else {
        lexer->advance(lexer, false);
      }
    }
    if (depth != 0) {
      return false;
    }
  }
  return false;
}

bool scan_identifier(TSLexer *lexer, const bool *valid_symbols) {
  if (!is_identifier_start(lexer->lookahead)) {
    return false;
  }
  const std::string name = consume_identifier(lexer);
  lexer->mark_end(lexer);
  if (is_reserved_identifier(name)) {
    return false;
  }

  const bool expression_field = valid_symbols[CANONICAL_RECORD_FIELD_NAME];
  const bool pattern_field = valid_symbols[CANONICAL_RECORD_PATTERN_FIELD_NAME];
  if (expression_field || pattern_field) {
    bool explicit_field = skip_trivia_for_lookahead(lexer) && lexer->lookahead == ':';
    bool redundant_pun = false;
    if (explicit_field) {
      lexer->advance(lexer, false);
      if (skip_trivia_for_lookahead(lexer) &&
          is_identifier_start(lexer->lookahead)) {
        const std::string value_name = consume_identifier(lexer);
        redundant_pun = value_name == name && skip_trivia_for_lookahead(lexer) &&
                        (lexer->lookahead == ',' || lexer->lookahead == '}');
      }
    }
    if (explicit_field && !redundant_pun) {
      lexer->result_symbol = pattern_field && !expression_field
                                 ? CANONICAL_RECORD_PATTERN_FIELD_NAME
                                 : CANONICAL_RECORD_FIELD_NAME;
      return true;
    }
  }

  if (valid_symbols[CANONICAL_IDENTIFIER]) {
    lexer->result_symbol = CANONICAL_IDENTIFIER;
    return true;
  }
  return false;
}

bool scan_pipe_lambda_fn(TSLexer *lexer) {
  if (lexer->lookahead != 'f') {
    return false;
  }
  lexer->advance(lexer, false);
  if (lexer->lookahead != 'n') {
    return false;
  }
  lexer->advance(lexer, false);
  if (is_identifier_continue(lexer->lookahead)) {
    return false;
  }
  lexer->mark_end(lexer);

  if (!skip_trivia_for_lookahead(lexer) || lexer->lookahead != '(') {
    return true;
  }
  lexer->advance(lexer, false);
  if (!skip_trivia_for_lookahead(lexer) ||
      !(is_identifier_start(lexer->lookahead) ||
        (lexer->lookahead >= 'A' && lexer->lookahead <= 'Z'))) {
    return true;
  }
  const std::string parameter = consume_identifier(lexer);
  if (!skip_trivia_for_lookahead(lexer) || lexer->lookahead != ')') {
    return true;  // typed or multiple parameters are never call-stage aliases
  }
  lexer->advance(lexer, false);
  if (!consume_arrow_after_trivia(lexer)) {
    return true;
  }
  if (!skip_trivia_for_lookahead(lexer) ||
      !(is_identifier_start(lexer->lookahead) ||
        (lexer->lookahead >= 'A' && lexer->lookahead <= 'Z'))) {
    return true;
  }

  std::string callable = consume_identifier(lexer);
  if (!skip_trivia_for_lookahead(lexer)) {
    return true;
  }
  while (lexer->lookahead == '.') {
    lexer->advance(lexer, false);
    if (!skip_trivia_for_lookahead(lexer) ||
        !(is_identifier_start(lexer->lookahead) ||
          (lexer->lookahead >= 'A' && lexer->lookahead <= 'Z'))) {
      return true;
    }
    callable = consume_identifier(lexer);
    if (!skip_trivia_for_lookahead(lexer)) {
      return true;
    }
  }
  if (lexer->lookahead != '(') {
    return true;
  }
  lexer->advance(lexer, false);
  if (!skip_trivia_for_lookahead(lexer) ||
      !(is_identifier_start(lexer->lookahead) ||
        (lexer->lookahead >= 'A' && lexer->lookahead <= 'Z'))) {
    return true;
  }
  const std::string first_argument = consume_identifier(lexer);
  if (first_argument != parameter || !skip_trivia_for_lookahead(lexer)) {
    return true;
  }

  const bool direct_argument_end = lexer->lookahead == ')' || lexer->lookahead == ',';
  const bool transform_alias =
      (callable == "realize" || callable == "copy") && lexer->lookahead == ')';
  const bool cast_alias = callable == "cast" && lexer->lookahead == ',';
  if (transform_alias || cast_alias ||
      (callable != "realize" && callable != "copy" && callable != "cast" &&
       direct_argument_end)) {
    return false;
  }

  lexer->result_symbol = CANONICAL_PIPE_LAMBDA_FN;
  return true;
}

bool consume_digits(TSLexer *lexer, std::string &text) {
  bool consumed = false;
  while (is_digit(lexer->lookahead)) {
    consumed = true;
    text.push_back(static_cast<char>(lexer->lookahead));
    lexer->advance(lexer, false);
  }
  return consumed;
}

bool fits_positive_i64(const std::string &digits) {
  static constexpr char MAX_I64[] = "9223372036854775807";
  const std::size_t max_length = sizeof(MAX_I64) - 1;
  return digits.size() < max_length ||
         (digits.size() == max_length && digits <= MAX_I64);
}

bool is_canonical_control_escape(const std::string &digits) {
  if (digits.empty() || (digits.size() > 1 && digits.front() == '0')) {
    return false;
  }
  uint32_t scalar = 0;
  const auto parsed = std::from_chars(
      digits.data(), digits.data() + digits.size(), scalar, 16);
  if (parsed.ec != std::errc() || parsed.ptr != digits.data() + digits.size()) {
    return false;
  }
  char rendered[8];
  const auto formatted = std::to_chars(rendered, rendered + sizeof(rendered), scalar, 16);
  if (formatted.ec != std::errc() ||
      std::string(rendered, formatted.ptr) != digits) {
    return false;
  }
  const bool unnamed_c0 = scalar >= 1 && scalar <= 0x1f &&
                          scalar != 0x09 && scalar != 0x0a && scalar != 0x0d;
  return unnamed_c0 || (scalar >= 0x7f && scalar <= 0x9f);
}

bool scan_string(TSLexer *lexer) {
  lexer->advance(lexer, false);  // opening quote
  while (!lexer->eof(lexer)) {
    const int32_t lookahead = lexer->lookahead;
    if (lookahead == '"') {
      lexer->advance(lexer, false);
      lexer->mark_end(lexer);
      lexer->result_symbol = CANONICAL_STRING;
      return true;
    }
    if (lookahead == '\\') {
      lexer->advance(lexer, false);
      if (lexer->eof(lexer)) {
        return false;
      }
      if (lexer->lookahead == '\\' || lexer->lookahead == '"' ||
          lexer->lookahead == 'n' || lexer->lookahead == 't' ||
          lexer->lookahead == 'r' || lexer->lookahead == '0') {
        lexer->advance(lexer, false);
        continue;
      }
      if (lexer->lookahead != 'u') {
        return false;
      }
      lexer->advance(lexer, false);
      if (lexer->lookahead != '{') {
        return false;
      }
      lexer->advance(lexer, false);
      std::string digits;
      while (is_digit(lexer->lookahead) ||
             (lexer->lookahead >= 'a' && lexer->lookahead <= 'f')) {
        digits.push_back(static_cast<char>(lexer->lookahead));
        lexer->advance(lexer, false);
      }
      if (lexer->lookahead != '}' || !is_canonical_control_escape(digits)) {
        return false;
      }
      lexer->advance(lexer, false);
      continue;
    }
    if (lookahead < 0x20 || (lookahead >= 0x7f && lookahead <= 0x9f)) {
      return false;
    }
    lexer->advance(lexer, false);
  }
  return false;
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
    if (!fits_positive_i64(numeric)) {
      return false;
    }
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
  const bool min_integer_context =
      valid_symbols[CANONICAL_MIN_INTEGER_MAGNITUDE];
  const bool min_pattern_context =
      valid_symbols[CANONICAL_MIN_PATTERN_MAGNITUDE];
  if (!expression_context && !pattern_context && !axis_context &&
      !nonzero_axis_context && !min_integer_context && !min_pattern_context) {
    return false;
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

  if (numeric == "9223372036854775808") {
    if (min_pattern_context && suffix.empty()) {
      lexer->result_symbol = CANONICAL_MIN_PATTERN_MAGNITUDE;
      return true;
    }
    if (min_integer_context && (suffix.empty() || suffix == "i64")) {
      lexer->result_symbol = CANONICAL_MIN_INTEGER_MAGNITUDE;
      return true;
    }
    return false;
  }

  if (!expression_context && !pattern_context &&
      (axis_context || nonzero_axis_context)) {
    const bool canonical_axis = suffix.empty() &&
                                numeric.find_first_of(".e") == std::string::npos &&
                                (numeric == "0" || numeric.front() != '0') &&
                                fits_positive_i64(numeric);
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
  while (lexer->lookahead == ' ' || lexer->lookahead == '\t' ||
         lexer->lookahead == '\n' || lexer->lookahead == '\r') {
    lexer->advance(lexer, true);
  }
  if (valid_symbols[CANONICAL_STRING] && lexer->lookahead == '"') {
    return scan_string(lexer);
  }
  if (valid_symbols[CANONICAL_PIPE_LAMBDA_FN] && lexer->lookahead == 'f') {
    const bool accepted = scan_pipe_lambda_fn(lexer);
    if (accepted) {
      lexer->result_symbol = CANONICAL_PIPE_LAMBDA_FN;
    }
    return accepted;
  }
  if (is_identifier_start(lexer->lookahead) &&
      (valid_symbols[CANONICAL_IDENTIFIER] ||
       valid_symbols[CANONICAL_RECORD_FIELD_NAME] ||
       valid_symbols[CANONICAL_RECORD_PATTERN_FIELD_NAME])) {
    return scan_identifier(lexer, valid_symbols);
  }
  return scan_number(lexer, valid_symbols);
}

}  // extern "C"
