//! The C and Objective-C leg: clang's front end reads the headers.
//!
//! A seam row is `kind|path|owner`, and the owner is the enclosing
//! *declaration*. Producing it therefore means parsing declarations, and C
//! declaration syntax is not a token vocabulary: aggregates, unions,
//! macro-typed declarators, multi-declarator lists, attributes, K&R
//! definitions, and string literals containing keywords are all forms a
//! hand-written walk mis-modelled in turn. Rather than model that grammar a
//! second time, this module asks the compiler for the AST (`clang -fsyntax-only
//! -Xclang -ast-dump=json`) and reads declarations off it.
//!
//! # Host independence
//!
//! Every header is parsed under a fixed target triple with `-ffreestanding
//! -nostdlibinc` and a committed stub SDK (`sdk-stubs/`) in place of libc, the
//! HIP SDK, the Apple frameworks, and the SIMD intrinsics headers, and clang
//! runs with a scrubbed environment. Linux CI, macOS CI, Devenv, and a
//! workstation therefore see the same preprocessed text and produce the same
//! rows, and the dumps stay small. The stub SDK is deliberately minimal: a
//! runtime header that starts using an SDK symbol the stub does not declare
//! fails the scan until the stub declares it, the same fail-closed discipline
//! the capacity census applies to an unknown type word.
//!
//! # Configurations and conditional arms
//!
//! The compiler reads one preprocessing configuration at a time, so a lane
//! names the closed set of configurations a header is parsed under
//! (`HeaderLane::configurations`), and the header's row set is the union over
//! them. That set is checked, not trusted: every conditional arm in the
//! header that carries code must be parsed by at least one configuration, or
//! the scan fails naming the directive. The check asks the preprocessor
//! itself which arms are live, through a marker pragma planted at the start
//! of every arm, so no directive grammar is modelled here. An arm whose only
//! content is other directives, or a linkage-specification brace, needs no
//! configuration.
//!
//! # What fails closed
//!
//! - `clang` missing, or any diagnostic error, is a `ScanError` carrying the
//!   compiler's own message.
//! - A conditional arm with code that no declared configuration parses.
//! - An `#include` that resolves outside the universe: anything but the
//!   staged copy, the stub SDK, the compiler's own resource headers, and the
//!   published include directory (whose every header is registered).
//! - A declaration kind this reader does not model, so a new form cannot be
//!   dropped silently.
//! - A type spelt with a word no vocabulary classifies (`_Float16`, `__bf16`,
//!   `__int128`, ...), in declaration position or in a cast, compound
//!   literal, or `sizeof` operand.
//! - A seam that no named declaration encloses; a placeholder owner would be a
//!   sink that absorbs every later seam of its kind.
//! - A node missing the JSON fields the reader depends on, so a clang JSON
//!   dialect change cannot look like an empty scan.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

use crate::{
    C_ELEMENT_TYPES, NARROWABLE_FIELDS, ScanError, SeamRow, c_lexical, excerpt, is_descriptor,
};

/// One preprocessing configuration a header is parsed under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Configuration {
    pub name: &'static str,
    /// Preprocessor flags that select the configuration.
    pub flags: &'static [&'static str],
}

/// The front-end lane a registered header is read through: one language and
/// target, and the closed set of configurations whose union is the header's
/// row set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderLane {
    /// clang `-x` language.
    pub language: &'static str,
    /// clang `-target` triple.
    pub target: &'static str,
    /// Additional front-end flags the language needs.
    pub extra: &'static [&'static str],
    /// The configurations whose union is the row set.
    pub configurations: &'static [Configuration],
}

/// The published headers: plain C, read scalar, then with the AVX2 arm and
/// the NEON arm of `chelis_simd.h` selected. The SIMD macros are defined
/// directly because the stub intrinsics headers carry no target-feature
/// requirements, so no second target triple is needed.
pub const PUBLIC_C_LANE: HeaderLane = HeaderLane {
    language: "c",
    target: "x86_64-unknown-linux-gnu",
    extra: &[],
    configurations: &[
        Configuration {
            name: "scalar",
            flags: &[],
        },
        Configuration {
            name: "avx2",
            flags: &["-D__AVX2__=1"],
        },
        Configuration {
            name: "neon",
            flags: &["-D__ARM_NEON=1"],
        },
    ],
};

/// The HIP support header: a host-side driver-API file with no device
/// syntax, whose only conditional beyond `__has_include` is `NDEBUG`.
pub const HIP_LANE: HeaderLane = HeaderLane {
    language: "c",
    target: "x86_64-unknown-linux-gnu",
    extra: &[],
    configurations: &[
        Configuration {
            name: "debug",
            flags: &[],
        },
        Configuration {
            name: "release",
            flags: &["-DNDEBUG"],
        },
    ],
};

/// Objective-C on an x86_64 macOS target. The arm64 target would define
/// `__ARM_NEON` and pull the NEON arm into every dump for no row.
pub const OBJECTIVE_C_LANE: HeaderLane = HeaderLane {
    language: "objective-c",
    target: "x86_64-apple-macosx14.0",
    extra: &["-fobjc-arc", "-fblocks"],
    configurations: &[
        Configuration {
            name: "debug",
            flags: &[],
        },
        Configuration {
            name: "release",
            flags: &["-DNDEBUG"],
        },
    ],
};

/// Which lane reads a registered header path.
pub fn lane_for(path: &str) -> HeaderLane {
    if path.ends_with("chelis_metal_runtime.h") {
        OBJECTIVE_C_LANE
    } else if path.ends_with("chelis_hip_runtime.h") {
        HIP_LANE
    } else {
        PUBLIC_C_LANE
    }
}

/// Objective-C ownership qualifiers and nullability that may sit between a
/// type and its star, plus the C qualifiers.
const QUALIFIER_WORDS: &[&str] = &[
    "_Atomic",
    "_Nonnull",
    "_Null_unspecified",
    "_Nullable",
    "__autoreleasing",
    "__kindof",
    "__restrict",
    "__restrict__",
    "__strong",
    "__unsafe_unretained",
    "__weak",
    "const",
    "long",
    "restrict",
    "signed",
    "unsigned",
    "volatile",
];

/// Objective-C spellings a declared type may contain that no C list names.
const OBJECTIVE_C_TYPE_WORDS: &[&str] = &["BOOL", "Class", "SEL", "id", "instancetype"];

/// Type operators that may appear inside a spelling and name no type of
/// their own.
const TYPE_OPERATOR_WORDS: &[&str] = &[
    "__typeof",
    "__typeof__",
    "__typeof_unqual",
    "__typeof_unqual__",
    "typeof",
    "typeof_unqual",
];

/// Element words the pointer rule accepts in a canonical (desugared) spelling
/// in addition to `C_ELEMENT_TYPES`: `int64_t` desugars to `long`, `int16_t`
/// to `short`.
const CANONICAL_ELEMENT_WORDS: &[&str] = &["int", "long", "short"];

/// The tokens a linkage-specification arm may contain: `extern "C" {` and
/// its closing brace carry no seam and need no C++ configuration.
const LINKAGE_TOKENS: &[&str] = &["extern", "\"C\"", "{", "}"];

const ARM_MARKER: &str = "chelis_inventory_arm";

/// Where the stub SDK and the published include directory live. The binary is
/// built by the oracle inside the checkout it scans, so the crate's own
/// manifest directory is the right anchor for both.
fn support_dirs() -> Result<(PathBuf, PathBuf), ScanError> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let stubs = manifest.join("sdk-stubs");
    let include = manifest.join("../chelis-runtime/include");
    for (label, dir) in [("stub SDK", &stubs), ("published include", &include)] {
        if !dir.is_dir() {
            return Err(ScanError::new(format!(
                "the {label} directory `{}` is missing; the inventory cannot parse a C \
                 header without it",
                dir.display()
            )));
        }
    }
    Ok((stubs, include))
}

/// A clang invocation with the lane's language, target, stub SDK, published
/// include directory, one configuration, and a scrubbed environment.
fn clang_command(
    lane: HeaderLane,
    configuration: Configuration,
    stubs: &Path,
    include: &Path,
) -> Command {
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut command = Command::new("clang");
    command
        .env_clear()
        .env("PATH", path)
        .args(["-w", "-fno-color-diagnostics"])
        .args(["-x", lane.language, "-target", lane.target])
        .args(["-ffreestanding", "-nostdlibinc"])
        .arg("-isystem")
        .arg(stubs)
        .arg("-I")
        .arg(include)
        .args(lane.extra)
        .args(configuration.flags);
    command
}

fn run_clang(mut command: Command, what: &str) -> Result<Vec<u8>, ScanError> {
    let output = command.output().map_err(|error| {
        ScanError::new(format!(
            "the runtime-representation inventory requires `clang` on PATH to parse the \
             registered C and Objective-C headers; none ran: {error}"
        ))
    })?;
    if !output.status.success() {
        return Err(ScanError::new(format!(
            "clang rejected {what} ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(output.stdout)
}

/// The compiler's own header directory, so `stdint.h` and friends are
/// recognised as inside the universe.
fn resource_dir() -> Result<String, ScanError> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut command = Command::new("clang");
    command
        .env_clear()
        .env("PATH", path)
        .arg("-print-resource-dir");
    let stdout = run_clang(command, "`-print-resource-dir`")?;
    let dir = String::from_utf8_lossy(&stdout).trim().to_string();
    if dir.is_empty() {
        return Err(ScanError::new(
            "clang did not report a resource directory; the inventory cannot tell the \
             compiler's own headers from an include outside the universe",
        ));
    }
    Ok(dir)
}

/// Run clang over one staged file under one configuration and return its
/// JSON AST.
fn dump_ast(
    lane: HeaderLane,
    configuration: Configuration,
    stubs: &Path,
    include: &Path,
    file: &Path,
) -> Result<Value, ScanError> {
    let mut command = clang_command(lane, configuration, stubs, include);
    command
        .args(["-fsyntax-only", "-Xclang", "-ast-dump=json"])
        .arg(file);
    let stdout = run_clang(
        command,
        &format!(
            "`{}` under the `{}` configuration",
            file.display(),
            configuration.name
        ),
    )?;
    serde_json::from_slice(&stdout).map_err(|error| {
        ScanError::new(format!(
            "clang's AST dump for `{}` is not the JSON object this reader expects: {error}",
            file.display()
        ))
    })
}

/// The arm ids the preprocessor keeps under one configuration, read back
/// from the marker pragmas the staged copy plants.
fn live_arms(
    lane: HeaderLane,
    configuration: Configuration,
    stubs: &Path,
    include: &Path,
    marked_file: &Path,
) -> Result<BTreeSet<usize>, ScanError> {
    let mut command = clang_command(lane, configuration, stubs, include);
    command.args(["-E", "-P"]).arg(marked_file);
    let stdout = run_clang(
        command,
        &format!(
            "the arm-marked copy of `{}` under the `{}` configuration",
            marked_file.display(),
            configuration.name
        ),
    )?;
    let text = String::from_utf8_lossy(&stdout);
    let prefix = format!("{ARM_MARKER}(");
    let mut live = BTreeSet::new();
    for line in text.lines() {
        let mut rest = line;
        while let Some(start) = rest.find(&prefix) {
            let after = &rest[start + prefix.len()..];
            if let Some(close) = after.find(')')
                && let Ok(id) = after[..close].trim().parse::<usize>()
            {
                live.insert(id);
            }
            rest = after;
        }
    }
    Ok(live)
}

/// Scan one registered header's source text through its lane.
pub fn scan_c_header(path: &str, source: &str) -> Result<Vec<SeamRow>, ScanError> {
    scan_c_source(path, source, lane_for(path))
}

/// Scan source text as if it were the registered header at `path`, under an
/// explicit lane: every configuration is parsed, the rows are the union, and
/// every conditional arm that carries code must have been parsed by one of
/// them.
pub fn scan_c_source(
    path: &str,
    source: &str,
    lane: HeaderLane,
) -> Result<Vec<SeamRow>, ScanError> {
    let file_name = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ScanError::new(format!("`{path}` has no file name")))?;
    let (stubs, include) = support_dirs()?;
    let resource = resource_dir()?;
    let scratch = tempfile::Builder::new()
        .prefix("chelis-repr-inventory-")
        .tempdir()
        .map_err(|error| ScanError::new(format!("cannot create a scratch directory: {error}")))?;
    // A staged copy must not shadow the real header a snippet or a sibling
    // includes by quoted name, so it never carries the registered file name.
    let file = scratch.path().join(format!("staged-{file_name}"));
    std::fs::write(&file, source)
        .map_err(|error| ScanError::new(format!("cannot stage `{path}`: {error}")))?;
    let arms = ConditionalArms::analyze(source);
    let marked_file = scratch.path().join(format!("staged-arms-{file_name}"));
    std::fs::write(&marked_file, &arms.marked_source)
        .map_err(|error| ScanError::new(format!("cannot stage `{path}`: {error}")))?;
    let allowed_prefixes = vec![
        file.to_string_lossy().to_string(),
        stubs.to_string_lossy().to_string(),
        include.to_string_lossy().to_string(),
        resource,
    ];

    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    let mut live = BTreeSet::new();
    for configuration in lane.configurations {
        live.extend(live_arms(
            lane,
            *configuration,
            &stubs,
            &include,
            &marked_file,
        )?);
        let tree = dump_ast(lane, *configuration, &stubs, &include, &file)?;
        let mut names = DeclaredNames::default();
        names.collect(&tree);
        let mut reader = Reader {
            path,
            source,
            file: file.to_string_lossy().to_string(),
            allowed_prefixes: &allowed_prefixes,
            current_file: None,
            names,
            owners: Vec::new(),
            function_depth: 0,
            rows: std::mem::take(&mut rows),
            seen: std::mem::take(&mut seen),
            record_descriptor: false,
            error: None,
        };
        reader.visit_translation_unit(&tree);
        rows = reader.rows;
        seen = reader.seen;
        if let Some(error) = reader.error {
            return Err(error);
        }
    }
    for arm in &arms.arms {
        if arm.has_code && !live.contains(&arm.id) {
            return Err(ScanError::new(format!(
                "`{path}` line {}: the arm opened by `{}` carries code that none of the `{}` \
                 lane's configurations ({}) parses, so its seams would be invisible; add the \
                 configuration to the lane in crates/chelis-repr-inventory/src/c_ast.rs, or \
                 remove the conditional",
                arm.line,
                arm.directive,
                lane.language,
                lane.configurations
                    .iter()
                    .map(|configuration| configuration.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    }
    Ok(rows)
}

/// One arm of a conditional directive, and whether it carries code that a
/// configuration has to parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConditionalArm {
    pub id: usize,
    /// The one-based physical line of the directive that opens the arm.
    pub line: usize,
    pub directive: String,
    pub has_code: bool,
}

/// The conditional structure of a header, read lexically: where each arm
/// opens, what it contains, and a copy of the source with a marker pragma
/// planted at the start of every arm so the preprocessor can report which
/// arms a configuration keeps.
pub struct ConditionalArms {
    pub arms: Vec<ConditionalArm>,
    pub marked_source: String,
}

impl ConditionalArms {
    pub fn analyze(source: &str) -> Self {
        let stripped = c_lexical::strip_c_comments(source);
        let original: Vec<&str> = source.split('\n').collect();
        let physical: Vec<&str> = stripped.split('\n').collect();
        // Logical lines: a backslash continues a directive onto the next
        // physical line. Each logical line remembers its physical span.
        let mut logical: Vec<(usize, usize, String)> = Vec::new();
        let mut index = 0;
        while index < physical.len() {
            let start = index;
            let mut text = String::new();
            loop {
                let line = physical[index];
                if let Some(head) = line.strip_suffix('\\')
                    && index + 1 < physical.len()
                {
                    text.push_str(head);
                    text.push(' ');
                    index += 1;
                    continue;
                }
                text.push_str(line);
                break;
            }
            logical.push((start, index, text));
            index += 1;
        }
        let directive_of = |text: &str| -> Option<String> {
            let trimmed = text.trim_start();
            let rest = trimmed
                .strip_prefix('#')
                .or_else(|| trimmed.strip_prefix("%:"))?;
            Some(rest.trim_start().to_string())
        };
        let is_conditional_open = |directive: &str| {
            ["if", "ifdef", "ifndef"]
                .iter()
                .any(|word| starts_with_word(directive, word))
        };
        let is_conditional_alternative = |directive: &str| {
            starts_with_word(directive, "elif") || starts_with_word(directive, "else")
        };
        let is_conditional_close = |directive: &str| starts_with_word(directive, "endif");

        // Pass one: find every arm's opening logical line and its extent.
        let mut arms: Vec<ConditionalArm> = Vec::new();
        let mut open_arm_by_logical: BTreeMap<usize, usize> = BTreeMap::new();
        let mut spans: Vec<(usize, usize)> = Vec::new(); // arm id -> logical [start, end)
        let mut stack: Vec<usize> = Vec::new(); // open arm ids
        for (logical_index, (first, _, text)) in logical.iter().enumerate() {
            let Some(directive) = directive_of(text) else {
                continue;
            };
            if is_conditional_open(&directive) {
                let id = arms.len();
                arms.push(ConditionalArm {
                    id,
                    line: first + 1,
                    directive: excerpt(&directive),
                    has_code: false,
                });
                spans.push((logical_index + 1, logical.len()));
                open_arm_by_logical.insert(logical_index, id);
                stack.push(id);
            } else if is_conditional_alternative(&directive) {
                if let Some(previous) = stack.pop() {
                    spans[previous].1 = logical_index;
                }
                let id = arms.len();
                arms.push(ConditionalArm {
                    id,
                    line: first + 1,
                    directive: excerpt(&directive),
                    has_code: false,
                });
                spans.push((logical_index + 1, logical.len()));
                open_arm_by_logical.insert(logical_index, id);
                stack.push(id);
            } else if is_conditional_close(&directive)
                && let Some(previous) = stack.pop()
            {
                spans[previous].1 = logical_index;
            }
        }
        // Pass two: an arm carries code when any non-directive logical line in
        // its span has tokens beyond a linkage brace.
        for (id, (start, end)) in spans.iter().enumerate() {
            let has_code = logical[*start..*end].iter().any(|(_, _, text)| {
                if directive_of(text).is_some() {
                    return false;
                }
                let tokens = c_lexical::lex_c_tokens(text);
                !tokens.is_empty()
                    && !tokens
                        .iter()
                        .all(|token| LINKAGE_TOKENS.contains(&token.as_str()))
            });
            arms[id].has_code = has_code;
        }
        // Pass three: plant a marker after every arm-opening directive.
        let mut marked = String::with_capacity(source.len() + arms.len() * 40);
        let mut physical_index = 0;
        for (logical_index, (first, last, _)) in logical.iter().enumerate() {
            debug_assert_eq!(physical_index, *first);
            for line in &original[*first..=*last] {
                marked.push_str(line);
                marked.push('\n');
            }
            physical_index = last + 1;
            if let Some(id) = open_arm_by_logical.get(&logical_index) {
                marked.push_str(&format!("#pragma {ARM_MARKER}({id})\n"));
            }
        }
        Self {
            arms,
            marked_source: marked,
        }
    }
}

fn starts_with_word(text: &str, word: &str) -> bool {
    text.strip_prefix(word)
        .is_some_and(|rest| !rest.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_'))
}

/// Names the translation unit declares, gathered before the row pass so a
/// desugared type can be checked against every tag, typedef, and Objective-C
/// interface the compiler knows, wherever it was declared.
#[derive(Default)]
struct DeclaredNames {
    words: BTreeSet<String>,
    /// `RecordDecl` id -> the typedef that names an anonymous aggregate.
    typedef_of_record: BTreeMap<String, String>,
    /// Typedef name -> the words of the type it names, one level down. The
    /// dumper desugars one level only, so `elem_t *` keeps its alias in both
    /// spellings; the census's `resolve_words` walks the chain instead.
    aliases: BTreeMap<String, Vec<String>>,
}

impl DeclaredNames {
    fn collect(&mut self, node: &Value) {
        let Some(kind) = node.get("kind").and_then(Value::as_str) else {
            return;
        };
        let name = node.get("name").and_then(Value::as_str).unwrap_or_default();
        if !name.is_empty()
            && matches!(
                kind,
                "RecordDecl"
                    | "EnumDecl"
                    | "TypedefDecl"
                    | "ObjCInterfaceDecl"
                    | "ObjCProtocolDecl"
                    | "ObjCTypeParamDecl"
                    | "CXXRecordDecl"
            )
        {
            self.words.insert(name.to_string());
        }
        if kind == "TypedefDecl" && !name.is_empty() {
            for owned in owned_tag_ids(node) {
                self.typedef_of_record
                    .entry(owned)
                    .or_insert_with(|| name.to_string());
            }
            if let Some(target) = desugared_type(node.get("type")) {
                let words = c_lexical::lex_c_tokens(&strip_annotations(target));
                if words.iter().any(|word| word != name) {
                    self.aliases.entry(name.to_string()).or_insert(words);
                }
            }
        }
        for child in children(node) {
            self.collect(child);
        }
    }
}

/// The ids of the anonymous tag declarations a typedef's type owns, found
/// through the `ElaboratedType.ownedTagDecl` link the dumper emits for
/// `typedef struct { ... } name;`.
fn owned_tag_ids(node: &Value) -> Vec<String> {
    let mut ids = Vec::new();
    fn walk(node: &Value, ids: &mut Vec<String>) {
        if let Some(owned) = node.get("ownedTagDecl")
            && let Some(id) = owned.get("id").and_then(Value::as_str)
        {
            ids.push(id.to_string());
        }
        for child in children(node) {
            walk(child, ids);
        }
    }
    for child in children(node) {
        walk(child, &mut ids);
    }
    ids
}

fn children(node: &Value) -> impl Iterator<Item = &Value> {
    node.get("inner")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|child| child.get("kind").is_some())
}

fn field_str<'a>(node: &'a Value, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Value::as_str)
}

/// The written spelling of a type object (`{"qualType": ..}`).
fn qual_type(type_object: Option<&Value>) -> Option<&str> {
    type_object.and_then(|ty| field_str(ty, "qualType"))
}

/// The one-level desugared spelling of a type object, or its written
/// spelling when the dumper printed only that.
fn desugared_type(type_object: Option<&Value>) -> Option<&str> {
    type_object
        .and_then(|ty| field_str(ty, "desugaredQualType"))
        .or_else(|| qual_type(type_object))
}

/// A source position in the file the dumper last named, read the way the
/// dumper prints it: `file` appears only when it differs from the previous
/// printed location, so the reader threads that state through every `loc`,
/// `range.begin`, and `range.end` in print order.
#[derive(Debug, Clone, Copy)]
struct Position {
    offset: usize,
    token_length: usize,
    /// The position is a macro expansion site rather than the seam's own
    /// spelling, so a sample has to show the whole invocation.
    expanded: bool,
}

struct Reader<'a> {
    path: &'a str,
    source: &'a str,
    /// The staged file's path as clang prints it.
    file: String,
    /// Path prefixes a declaration may come from: the staged copy, the stub
    /// SDK, the published include directory, and the compiler's own headers.
    allowed_prefixes: &'a [String],
    current_file: Option<String>,
    names: DeclaredNames,
    owners: Vec<String>,
    /// How many function-like owners enclose the current node. Inside one,
    /// the function owns every statement and local declaration.
    function_depth: usize,
    rows: Vec<SeamRow>,
    seen: BTreeSet<(String, String)>,
    /// Whether the aggregate being read is a tensor descriptor, inherited by
    /// its anonymous members.
    record_descriptor: bool,
    error: Option<ScanError>,
}

impl<'a> Reader<'a> {
    fn fail(&mut self, message: String) {
        if self.error.is_none() {
            self.error = Some(ScanError::new(message));
        }
    }

    /// Advance the dumper's file state through one location object and
    /// return its position when it names a place in the staged file.
    fn read_location(&mut self, location: Option<&Value>) -> Option<Position> {
        let location = location?;
        let mut position = None;
        // The dumper prints a macro location as `spellingLoc` then
        // `expansionLoc`; the expansion is where the header spells the seam.
        for key in ["spellingLoc", "expansionLoc"] {
            if let Some(inner) = location.get(key) {
                let candidate = self.read_plain_location(inner);
                if key == "expansionLoc" {
                    position = candidate.map(|found| Position {
                        expanded: true,
                        ..found
                    });
                }
            }
        }
        if location.get("offset").is_some() {
            position = self.read_plain_location(location);
        }
        position
    }

    fn read_plain_location(&mut self, location: &Value) -> Option<Position> {
        if let Some(file) = field_str(location, "file") {
            self.current_file = Some(file.to_string());
        }
        let offset = location.get("offset").and_then(Value::as_u64)?;
        let token_length = location.get("tokLen").and_then(Value::as_u64).unwrap_or(0);
        (self.current_file.as_deref() == Some(self.file.as_str())).then_some(Position {
            offset: usize::try_from(offset).ok()?,
            token_length: usize::try_from(token_length).ok()?,
            expanded: false,
        })
    }

    /// Read a node's `loc` and `range` in print order and report whether the
    /// declaration itself sits in the staged file.
    fn locate(&mut self, node: &Value) -> (bool, Option<Position>, Option<Position>) {
        let at = self.read_location(node.get("loc"));
        let range = node.get("range");
        let begin = self.read_location(range.and_then(|range| range.get("begin")));
        let end = self.read_location(range.and_then(|range| range.get("end")));
        (at.is_some() || begin.is_some(), begin, end)
    }

    /// The source text of a declaration: from its first token to the end of
    /// its statement or the brace that opens its body.
    fn declaration_sample(&self, begin: Option<Position>) -> Option<String> {
        let start = begin?.offset;
        let text = self.source.get(start..)?;
        let mut depth = 0usize;
        let mut end = text.len();
        for (index, character) in text.char_indices() {
            match character {
                '(' | '[' => depth += 1,
                ')' | ']' => depth = depth.saturating_sub(1),
                ';' | '{' | '}' if depth == 0 => {
                    end = index;
                    break;
                }
                _ => {}
            }
        }
        Some(excerpt(&text[..end]))
    }

    /// The source text of an expression seam: its own tokens when the header
    /// spells them directly, or the whole statement when the seam sits in a
    /// macro argument, so the sample shows the invocation that carries it.
    fn expression_sample(&self, begin: Option<Position>, end: Option<Position>) -> Option<String> {
        let begin = begin?;
        let start = begin.offset;
        if !begin.expanded
            && let Some(end) = end.filter(|end| !end.expanded)
        {
            let finish = end.offset + end.token_length;
            if finish >= start {
                return self.source.get(start..finish).map(excerpt);
            }
        }
        self.declaration_sample(Some(begin))
    }

    fn owner(&self) -> Option<String> {
        (!self.owners.is_empty()).then(|| self.owners.join("::"))
    }

    fn push(&mut self, kind: &str, sample: Option<String>, fallback: &str) {
        let Some(owner) = self.owner() else {
            self.fail(format!(
                "`{}` carries a {kind} seam that no named declaration encloses: `{}`. A \
                 placeholder owner would absorb every later seam of that kind in this file",
                self.path,
                excerpt(fallback)
            ));
            return;
        };
        let sample = sample.unwrap_or_else(|| excerpt(fallback));
        if self.seen.insert((kind.to_string(), owner.clone())) {
            self.rows.push(SeamRow::new(kind, &owner, sample));
        }
    }

    fn with_owner(&mut self, owner: String, function_like: bool, visit: impl FnOnce(&mut Self)) {
        // Inside a function the function owns everything, including a local
        // aggregate's fields and a nested declaration's carriers.
        if self.function_depth > 0 {
            visit(self);
            return;
        }
        self.owners.push(owner);
        if function_like {
            self.function_depth += 1;
        }
        visit(self);
        if function_like {
            self.function_depth -= 1;
        }
        self.owners.pop();
    }

    /// A type spelling's words with every typedef alias resolved through the
    /// translation unit's own typedefs.
    fn resolved_words(&self, spelling: &str) -> Vec<String> {
        c_lexical::resolve_words(
            c_lexical::lex_c_tokens(&strip_annotations(spelling)),
            &self.names.aliases,
        )
    }

    /// Reject a type spelt with a word no vocabulary classifies, wherever the
    /// spelling appears: a declaration, a cast, a compound literal, or a
    /// `sizeof` operand.
    fn check_type_words(&mut self, type_object: Option<&Value>, what: &str) {
        let Some(spelling) = desugared_type(type_object) else {
            return;
        };
        for word in self.resolved_words(spelling) {
            let is_identifier = word
                .chars()
                .next()
                .is_some_and(|character| character.is_ascii_alphabetic() || character == '_');
            if !is_identifier || self.is_known_type_word(&word) {
                continue;
            }
            self.fail(format!(
                "`{}` spells {what} with the unrecognized C type word `{word}` (in \
                 `{spelling}`): classify it in tests/support/c_lexical.rs rather than \
                 routing around it",
                self.path
            ));
            return;
        }
    }

    fn is_known_type_word(&self, word: &str) -> bool {
        c_lexical::NUMERIC_C_TYPES.contains(&word)
            || c_lexical::NON_NUMERIC_C_TYPE_WORDS.contains(&word)
            || C_ELEMENT_TYPES.contains(&word)
            || QUALIFIER_WORDS.contains(&word)
            || OBJECTIVE_C_TYPE_WORDS.contains(&word)
            || TYPE_OPERATOR_WORDS.contains(&word)
            || self.names.words.contains(word)
    }

    /// Does a type govern an element pointer, as written or once every
    /// typedef alias is resolved?
    fn governs_element_pointer(&self, type_object: Option<&Value>) -> bool {
        [qual_type(type_object), desugared_type(type_object)]
            .into_iter()
            .flatten()
            .any(|spelling| governs_a_pointer(&self.resolved_words(spelling)))
    }

    fn visit_translation_unit(&mut self, node: &Value) {
        let kind = field_str(node, "kind").unwrap_or_default();
        if kind != "TranslationUnitDecl" {
            self.fail(format!(
                "clang's AST dump for `{}` does not start at a TranslationUnitDecl (found \
                 `{kind}`)",
                self.path
            ));
            return;
        }
        for child in children(node) {
            self.visit_top_level(child);
        }
    }

    /// A declaration directly in the file, or inside a linkage block.
    fn visit_top_level(&mut self, node: &Value) {
        let kind = field_str(node, "kind").unwrap_or_default();
        if node.get("isImplicit").and_then(Value::as_bool) == Some(true) {
            return;
        }
        let (in_file, begin, end) = self.locate(node);
        if kind == "LinkageSpecDecl" {
            for child in children(node) {
                self.visit_top_level(child);
            }
            return;
        }
        if !in_file {
            // Declarations from the stub SDK, the compiler's own headers, and
            // the published include directory are not this header's surface,
            // but anything else reached through an include is a file the
            // universe does not contain. Their children still advance the
            // dumper's file state, which the reader must follow.
            if let Some(file) = self.current_file.clone()
                && !self
                    .allowed_prefixes
                    .iter()
                    .any(|prefix| file.starts_with(prefix.as_str()))
            {
                self.fail(format!(
                    "`{}` includes `{file}`, which is outside the inventory universe (the \
                     header itself, the stub SDK, the compiler's own headers, and the \
                     published include directory); register the file under an inventory \
                     root, or move the declarations it carries",
                    self.path
                ));
            }
            self.skip_subtree(node);
            return;
        }
        self.visit_declaration(node, kind, begin, end);
    }

    /// Walk a subtree only to thread the dumper's file state through it.
    fn skip_subtree(&mut self, node: &Value) {
        for child in children(node) {
            self.locate(child);
            self.skip_subtree(child);
        }
    }

    fn name_of(&mut self, node: &Value) -> Option<String> {
        match field_str(node, "name") {
            Some(name) if !name.is_empty() => Some(name.to_string()),
            _ => {
                if let Some(id) = field_str(node, "id")
                    && let Some(typedef) = self.names.typedef_of_record.get(id)
                {
                    return Some(typedef.clone());
                }
                None
            }
        }
    }

    /// One declaration whose location the caller has already read.
    fn visit_declaration(
        &mut self,
        node: &Value,
        kind: &str,
        begin: Option<Position>,
        end: Option<Position>,
    ) {
        match kind {
            "RecordDecl" | "CXXRecordDecl" => self.visit_record(node),
            "EnumDecl" => {
                let name = self.name_of(node).unwrap_or_default();
                if name.is_empty() {
                    // An anonymous enum's constants own themselves.
                    for child in children(node) {
                        self.visit_nested(child);
                    }
                } else {
                    self.with_owner(name, false, |reader| {
                        for child in children(node) {
                            reader.visit_nested(child);
                        }
                    });
                }
            }
            "EnumConstantDecl" => {
                let name = field_str(node, "name").unwrap_or_default().to_string();
                self.with_owner(name, false, |reader| {
                    for child in children(node) {
                        reader.visit_nested(child);
                    }
                });
            }
            "FunctionDecl" | "ObjCMethodDecl" => self.visit_function(node, kind, begin),
            "VarDecl" => {
                let name = field_str(node, "name").unwrap_or_default().to_string();
                self.check_type_words(node.get("type"), &format!("`{name}`"));
                let sample = self.declaration_sample(begin);
                let governs = self.governs_element_pointer(node.get("type"));
                self.with_owner(name.clone(), false, |reader| {
                    if governs {
                        reader.push("raw-element-pointer", sample.clone(), &name);
                    }
                    for child in children(node) {
                        reader.visit_nested(child);
                    }
                });
            }
            "TypedefDecl" => {
                let name = field_str(node, "name").unwrap_or_default().to_string();
                self.check_type_words(node.get("type"), &format!("typedef `{name}`"));
                let sample = self.declaration_sample(begin);
                let governs = self.governs_element_pointer(node.get("type"));
                self.with_owner(name.clone(), false, |reader| {
                    if governs {
                        reader.push("raw-element-pointer", sample.clone(), &name);
                    }
                    // A typedef's owned anonymous aggregate is visited through
                    // the RecordDecl that precedes it; its type nodes carry
                    // no seams of their own.
                });
            }
            "FieldDecl" => self.visit_field(node, begin, end),
            "ObjCInterfaceDecl"
            | "ObjCProtocolDecl"
            | "ObjCCategoryDecl"
            | "ObjCImplementationDecl"
            | "ObjCCategoryImplDecl" => {
                let name = field_str(node, "name").unwrap_or_default().to_string();
                self.with_owner(name, false, |reader| {
                    for child in children(node) {
                        reader.visit_nested(child);
                    }
                });
            }
            "ObjCIvarDecl" | "ObjCPropertyDecl" => self.visit_field(node, begin, end),
            "ObjCTypeParamDecl" | "EmptyDecl" | "IndirectFieldDecl" | "ImplicitParamDecl"
            | "LabelDecl" | "BlockDecl" | "ParmVarDecl" => {
                // `IndirectFieldDecl` mirrors an anonymous member's fields the
                // reader already visited inside the anonymous aggregate;
                // parameters are read with their function.
                for child in children(node) {
                    self.visit_nested(child);
                }
            }
            other if other.ends_with("Decl") => {
                self.fail(format!(
                    "`{}` contains a `{other}` declaration the inventory reader does not \
                     model; model it in crates/chelis-repr-inventory/src/c_ast.rs before \
                     the file can be scanned",
                    self.path
                ));
            }
            _ => self.visit_expression(node, kind, begin, end),
        }
    }

    /// A node below a declaration: a nested declaration, a statement, an
    /// expression, a type, or an attribute.
    fn visit_nested(&mut self, node: &Value) {
        let kind = field_str(node, "kind").unwrap_or_default();
        if node.get("isImplicit").and_then(Value::as_bool) == Some(true) {
            self.skip_subtree(node);
            return;
        }
        let (_, begin, end) = self.locate(node);
        self.visit_declaration(node, kind, begin, end);
    }

    fn visit_record(&mut self, node: &Value) {
        let name = self.name_of(node);
        let is_definition = node
            .get("completeDefinition")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !is_definition {
            return;
        }
        // An anonymous aggregate nested in a named one contributes its fields
        // to the enclosing name, which is how the compiler exposes them too.
        let mut inherited = false;
        let record_name = match name {
            Some(name) => name,
            None => match self.owners.last() {
                Some(enclosing) if self.function_depth == 0 => {
                    inherited = true;
                    enclosing.clone()
                }
                _ => {
                    if self.function_depth == 0 {
                        self.fail(format!(
                            "`{}` declares an anonymous aggregate at file scope that no \
                             typedef names; give it a tag or a typedef so its fields can \
                             own their rows",
                            self.path
                        ));
                        return;
                    }
                    String::new()
                }
            },
        };
        let field_names: BTreeSet<String> = collect_field_names(node);
        let descriptor = if inherited {
            self.record_descriptor
        } else {
            is_descriptor(&field_names)
        };
        if self.function_depth > 0 {
            for child in children(node) {
                self.visit_nested(child);
            }
            return;
        }
        let previous = self.owners.clone();
        let previous_descriptor = self.record_descriptor;
        self.record_descriptor = descriptor;
        self.owners = vec![record_name];
        for child in children(node) {
            let kind = field_str(child, "kind").unwrap_or_default();
            let (_, begin, end) = self.locate(child);
            if kind == "FieldDecl" {
                self.visit_record_field(child, descriptor, begin, end);
            } else {
                self.visit_declaration(child, kind, begin, end);
            }
        }
        self.owners = previous;
        self.record_descriptor = previous_descriptor;
    }

    fn visit_record_field(
        &mut self,
        node: &Value,
        descriptor: bool,
        begin: Option<Position>,
        _end: Option<Position>,
    ) {
        let name = field_str(node, "name").unwrap_or_default().to_string();
        if name.is_empty() {
            // An anonymous member's aggregate follows as its own RecordDecl.
            for child in children(node) {
                self.visit_nested(child);
            }
            return;
        }
        self.check_type_words(node.get("type"), &format!("field `{name}`"));
        let sample = self.declaration_sample(begin);
        let rendered = sample.clone().unwrap_or_else(|| {
            format!("{} {name}", qual_type(node.get("type")).unwrap_or_default())
        });
        let governs = self.governs_element_pointer(node.get("type"));
        let narrow = NARROWABLE_FIELDS.contains(&name.as_str())
            && qual_type(node.get("type")).is_some_and(|spelling| {
                c_lexical::lex_c_tokens(spelling)
                    .iter()
                    .any(|word| word == "int" || word == "int32_t")
            });
        let extent = fixed_extent(qual_type(node.get("type")).unwrap_or_default(), &rendered);
        self.owners.push(name.clone());
        if governs {
            self.push("raw-element-pointer", Some(rendered.clone()), &rendered);
        }
        if descriptor {
            self.push("descriptor-field", Some(rendered.clone()), &rendered);
            if narrow {
                self.push("narrow-metadata", Some(rendered.clone()), &rendered);
            }
            if let Some(extent) = extent {
                let sample = format!("{rendered} (extent {extent})");
                self.push("fixed-rank-metadata", Some(sample.clone()), &sample);
            }
        }
        self.owners.pop();
    }

    fn visit_field(&mut self, node: &Value, begin: Option<Position>, end: Option<Position>) {
        // A field reached outside `visit_record` (an Objective-C ivar or
        // property) is read with the non-descriptor rule.
        self.visit_record_field(node, false, begin, end);
    }

    /// A C function or an Objective-C method. A function's type spells its
    /// parameters and result together; a method carries its result under
    /// `returnType` and its parameters as children, so both are read.
    fn visit_function(&mut self, node: &Value, kind: &str, begin: Option<Position>) {
        let name = field_str(node, "name").unwrap_or_default().to_string();
        let is_method = kind == "ObjCMethodDecl";
        let type_object = if is_method {
            node.get("returnType")
        } else {
            node.get("type")
        };
        let what = if is_method {
            format!("the result of method `{name}`")
        } else {
            format!("`{name}`")
        };
        self.check_type_words(type_object, &what);
        let sample = self.declaration_sample(begin);
        let mut governs = self.governs_element_pointer(type_object);
        if is_method {
            governs = governs
                || children(node)
                    .filter(|child| field_str(child, "kind") == Some("ParmVarDecl"))
                    .any(|parameter| self.governs_element_pointer(parameter.get("type")));
        }
        self.with_owner(name.clone(), true, |reader| {
            if governs {
                reader.push("raw-element-pointer", sample.clone(), &name);
            }
            for child in children(node) {
                let kind = field_str(child, "kind").unwrap_or_default();
                let (_, begin, end) = reader.locate(child);
                if kind == "ParmVarDecl" {
                    let parameter = field_str(child, "name").unwrap_or_default().to_string();
                    reader.check_type_words(child.get("type"), &format!("parameter `{parameter}`"));
                    for grandchild in children(child) {
                        reader.visit_nested(grandchild);
                    }
                } else {
                    reader.visit_declaration(child, kind, begin, end);
                }
            }
        });
    }

    /// Statements, expressions, types, and attributes: only three expression
    /// forms are seams, and everything else is walked for what it encloses.
    /// A type spelt in expression position is classified like a declared one.
    fn visit_expression(
        &mut self,
        node: &Value,
        kind: &str,
        begin: Option<Position>,
        end: Option<Position>,
    ) {
        match kind {
            "MemberExpr" if field_str(node, "name") == Some("data") => {
                let sample = self.expression_sample(begin, end);
                self.push("direct-data-access", sample, "->data");
            }
            "UnaryExprOrTypeTraitExpr" => {
                self.check_type_words(node.get("argType"), "a sizeof operand");
                if field_str(node, "name") == Some("sizeof") {
                    let sample = self.expression_sample(begin, end);
                    self.push("width-arithmetic", sample, "sizeof");
                }
            }
            "CStyleCastExpr" | "CompoundLiteralExpr" | "CXXFunctionalCastExpr" => {
                self.check_type_words(node.get("type"), "a cast");
                if self.governs_element_pointer(node.get("type")) {
                    let sample = self.expression_sample(begin, end);
                    self.push(
                        "raw-element-pointer",
                        sample,
                        qual_type(node.get("type")).unwrap_or("element pointer cast"),
                    );
                }
            }
            _ => {}
        }
        for child in children(node) {
            self.visit_nested(child);
        }
    }
}

/// Every field name an aggregate declares, including those of anonymous
/// members, which is what the descriptor test needs.
fn collect_field_names(node: &Value) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    fn walk(node: &Value, names: &mut BTreeSet<String>) {
        for child in children(node) {
            let kind = field_str(child, "kind").unwrap_or_default();
            if kind == "FieldDecl" {
                if let Some(name) = field_str(child, "name")
                    && !name.is_empty()
                {
                    names.insert(name.to_string());
                }
            } else if kind == "RecordDecl" {
                walk(child, names);
            }
        }
    }
    walk(node, &mut names);
    names
}

/// Remove `__attribute__((...))` groups and the dumper's `(unnamed at ...)`,
/// `(anonymous ...)`, and `(lambda at ...)` annotations from a type spelling
/// before its words are classified: a path inside an annotation is not a
/// type word.
fn strip_annotations(spelling: &str) -> String {
    let mut output = String::with_capacity(spelling.len());
    let mut rest = spelling;
    loop {
        let attribute = rest.find("__attribute__");
        let annotation = ["(unnamed", "(anonymous", "(lambda"]
            .iter()
            .filter_map(|marker| rest.find(marker))
            .min();
        let (start, group_start) = match (attribute, annotation) {
            (Some(a), Some(b)) if a <= b => (a, a + "__attribute__".len()),
            (Some(a), None) => (a, a + "__attribute__".len()),
            (_, Some(b)) => (b, b),
            (None, None) => break,
        };
        output.push_str(&rest[..start]);
        let after = &rest[group_start..];
        let mut depth = 0usize;
        let mut consumed = after.len();
        for (index, character) in after.char_indices() {
            match character {
                '(' => depth += 1,
                ')' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        consumed = index + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        output.push(' ');
        rest = &after[consumed..];
    }
    output.push_str(rest);
    output
}

/// Does an element type in this spelling GOVERN a pointer? The star must
/// follow the element type through qualifiers, or through the parenthesis
/// that opens a pointer-to-array or pointer-to-function declarator, so an
/// opaque handle beside an integer return width does not count.
fn governs_a_pointer(words: &[String]) -> bool {
    words.iter().enumerate().any(|(index, word)| {
        if !(C_ELEMENT_TYPES.contains(&word.as_str())
            || CANONICAL_ELEMENT_WORDS.contains(&word.as_str()))
        {
            return false;
        }
        words[index + 1..]
            .iter()
            .take_while(|next| QUALIFIER_WORDS.contains(&next.as_str()) || *next == "(")
            .count()
            .checked_add(index + 1)
            .and_then(|star| words.get(star))
            .is_some_and(|next| next == "*")
    })
}

/// A fixed-rank array is one whose extent is a compile-time rank cap: a
/// `*MAX_DIM` constant in the source spelling, or a literal above one in the
/// compiler's canonical spelling. A multi-dimensional extent is rendered with
/// its dimensions joined by `x`.
fn fixed_extent(qual_type: &str, rendered: &str) -> Option<String> {
    let canonical: Vec<&str> = bracket_groups(qual_type);
    let first = canonical.first()?.trim();
    if first.is_empty() {
        return None;
    }
    let source: Vec<&str> = bracket_groups(rendered);
    let spelled =
        if source.len() == canonical.len() && source.iter().all(|group| !group.trim().is_empty()) {
            source
                .iter()
                .map(|group| group.trim())
                .collect::<Vec<_>>()
                .join("x")
        } else {
            canonical
                .iter()
                .map(|group| group.trim())
                .collect::<Vec<_>>()
                .join("x")
        };
    let is_cap = spelled.contains("MAX_DIM") || first.parse::<u64>().is_ok_and(|extent| extent > 1);
    is_cap.then_some(spelled)
}

/// The contents of every `[...]` group in a spelling, in order.
fn bracket_groups(spelling: &str) -> Vec<&str> {
    let mut groups = Vec::new();
    let mut rest = spelling;
    while let Some(open) = rest.find('[') {
        let Some(close) = rest[open..].find(']') else {
            break;
        };
        groups.push(&rest[open + 1..open + close]);
        rest = &rest[open + close + 1..];
    }
    groups
}
