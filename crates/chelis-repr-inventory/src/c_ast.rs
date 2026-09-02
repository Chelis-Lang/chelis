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
//! HIP SDK, and the Apple frameworks, and clang runs with a scrubbed
//! environment. Linux CI, macOS CI, Devenv, and a workstation therefore see the
//! same preprocessed text and produce the same rows, and the dumps stay small.
//! The stub SDK is deliberately minimal: a runtime header that starts using an
//! SDK symbol the stub does not declare fails the scan until the stub declares
//! it, the same fail-closed discipline the capacity census applies to an
//! unknown type word.
//!
//! # Configuration
//!
//! `chelis_hip_runtime.h`'s only conditional beyond `__has_include` is
//! `NDEBUG`. The lane parses the debug arm (no `-DNDEBUG`), which declares a
//! strict superset of the release arm; `crate::c_ast::release_arm_rows` lets a
//! test prove that claim by execution rather than by prose.
//!
//! # What fails closed
//!
//! - `clang` missing, or any diagnostic error, is a `ScanError` carrying the
//!   compiler's own message.
//! - A declaration kind this reader does not model is a `ScanError`, so a new
//!   form cannot be dropped silently.
//! - A declared type spelt with a word no vocabulary classifies (`_Float16`,
//!   `__bf16`, `__int128`, ...) is a `ScanError` naming the word.
//! - A seam that no named declaration encloses is a `ScanError`; a placeholder
//!   owner would be a sink that absorbs every later seam of its kind.
//! - A node missing the JSON fields the reader depends on is a `ScanError`,
//!   so a clang JSON dialect change cannot look like an empty scan.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

use crate::{
    C_ELEMENT_TYPES, NARROWABLE_FIELDS, ScanError, SeamRow, c_lexical, excerpt, is_descriptor,
};

/// The front-end lane a registered header is read through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderLane {
    /// clang `-x` language.
    pub language: &'static str,
    /// clang `-target` triple.
    pub target: &'static str,
    /// Additional front-end flags the language needs.
    pub extra: &'static [&'static str],
}

/// Plain C on a Linux x86_64 target: the five published headers and the HIP
/// support header (a host-side driver-API file with no device syntax).
pub const C_LANE: HeaderLane = HeaderLane {
    language: "c",
    target: "x86_64-unknown-linux-gnu",
    extra: &[],
};

/// Objective-C on an x86_64 macOS target. The arm64 target would define
/// `__ARM_NEON` and pull `arm_neon.h` into every dump for no row.
pub const OBJECTIVE_C_LANE: HeaderLane = HeaderLane {
    language: "objective-c",
    target: "x86_64-apple-macosx14.0",
    extra: &["-fobjc-arc", "-fblocks"],
};

/// Which lane reads a registered header path.
pub fn lane_for(path: &str) -> HeaderLane {
    if path.ends_with("chelis_metal_runtime.h") {
        OBJECTIVE_C_LANE
    } else {
        C_LANE
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

/// Element words the pointer rule accepts in a canonical (desugared) spelling
/// in addition to `C_ELEMENT_TYPES`: `int64_t` desugars to `long`, `int16_t`
/// to `short`.
const CANONICAL_ELEMENT_WORDS: &[&str] = &["int", "long", "short"];

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

/// Run clang over one file and return its JSON AST.
fn dump_ast(lane: HeaderLane, file: &Path, defines: &[&str]) -> Result<Value, ScanError> {
    let (stubs, include) = support_dirs()?;
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut command = Command::new("clang");
    command
        .env_clear()
        .env("PATH", path)
        .args([
            "-fsyntax-only",
            "-Xclang",
            "-ast-dump=json",
            "-w",
            "-fno-color-diagnostics",
        ])
        .args(["-x", lane.language, "-target", lane.target])
        .args(["-ffreestanding", "-nostdlibinc"])
        .arg("-isystem")
        .arg(&stubs)
        .arg("-I")
        .arg(&include)
        .args(lane.extra)
        .args(defines)
        .arg(file);
    let output = command.output().map_err(|error| {
        ScanError::new(format!(
            "the runtime-representation inventory requires `clang` on PATH to parse the \
             registered C and Objective-C headers; none ran: {error}"
        ))
    })?;
    if !output.status.success() {
        return Err(ScanError::new(format!(
            "clang rejected `{}` ({}): {}",
            file.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    serde_json::from_slice(&output.stdout).map_err(|error| {
        ScanError::new(format!(
            "clang's AST dump for `{}` is not the JSON object this reader expects: {error}",
            file.display()
        ))
    })
}

/// Scan one registered header's source text through its lane.
pub fn scan_c_header(path: &str, source: &str) -> Result<Vec<SeamRow>, ScanError> {
    scan_c_source(path, source, lane_for(path), &[])
}

/// The rows the release arm (`-DNDEBUG`) of a header yields. A test uses this
/// to prove the debug arm the lane pins is a superset.
pub fn release_arm_rows(path: &str, source: &str) -> Result<Vec<SeamRow>, ScanError> {
    scan_c_source(path, source, lane_for(path), &["-DNDEBUG"])
}

/// Scan source text as if it were the registered header at `path`, under an
/// explicit lane and extra preprocessor definitions.
pub fn scan_c_source(
    path: &str,
    source: &str,
    lane: HeaderLane,
    defines: &[&str],
) -> Result<Vec<SeamRow>, ScanError> {
    let file_name = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ScanError::new(format!("`{path}` has no file name")))?;
    let scratch = tempfile::Builder::new()
        .prefix("chelis-repr-inventory-")
        .tempdir()
        .map_err(|error| ScanError::new(format!("cannot create a scratch directory: {error}")))?;
    // A staged copy must not shadow the real header a snippet or a sibling
    // includes by quoted name, so it never carries the registered file name.
    let file = scratch.path().join(format!("staged-{file_name}"));
    std::fs::write(&file, source)
        .map_err(|error| ScanError::new(format!("cannot stage `{path}`: {error}")))?;
    let tree = dump_ast(lane, &file, defines)?;
    let file_text = file.to_string_lossy().to_string();
    let mut names = DeclaredNames::default();
    names.collect(&tree);
    let mut reader = Reader {
        path,
        source,
        file: file_text,
        current_file: None,
        names,
        owners: Vec::new(),
        function_depth: 0,
        rows: Vec::new(),
        seen: BTreeSet::new(),
        record_descriptor: false,
        error: None,
    };
    reader.visit_translation_unit(&tree);
    match reader.error {
        Some(error) => Err(error),
        None => Ok(reader.rows),
    }
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
            if let Some(target) = desugared_type(node) {
                let words = c_lexical::lex_c_tokens(&strip_attributes(target));
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

fn qual_type(node: &Value) -> Option<&str> {
    node.get("type").and_then(|ty| field_str(ty, "qualType"))
}

fn desugared_type(node: &Value) -> Option<&str> {
    node.get("type")
        .and_then(|ty| field_str(ty, "desugaredQualType"))
        .or_else(|| qual_type(node))
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
            c_lexical::lex_c_tokens(&strip_attributes(spelling)),
            &self.names.aliases,
        )
    }

    /// Reject a declared type spelt with a word no vocabulary classifies.
    fn check_type_words(&mut self, node: &Value, what: &str) {
        let Some(spelling) = desugared_type(node) else {
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
                "`{}` declares {what} with the unrecognized C type word `{word}` (in \
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
            || self.names.words.contains(word)
    }

    /// Does a declared type govern an element pointer, as written or once
    /// every typedef alias is resolved?
    fn governs_element_pointer(&self, node: &Value) -> bool {
        [qual_type(node), desugared_type(node)]
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
            // Declarations from the stub SDK and the compiler's own headers
            // are not inventory surface. Their children still advance the
            // dumper's file state, which the reader must follow.
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

    fn name_of(&mut self, node: &Value, kind: &str) -> Option<String> {
        match field_str(node, "name") {
            Some(name) if !name.is_empty() => Some(name.to_string()),
            _ => {
                if let Some(id) = field_str(node, "id")
                    && let Some(typedef) = self.names.typedef_of_record.get(id)
                {
                    return Some(typedef.clone());
                }
                let _ = kind;
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
            "RecordDecl" | "CXXRecordDecl" => self.visit_record(node, begin),
            "EnumDecl" => {
                let name = self.name_of(node, kind).unwrap_or_default();
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
            "FunctionDecl" | "ObjCMethodDecl" => self.visit_function(node, begin),
            "VarDecl" => {
                let name = field_str(node, "name").unwrap_or_default().to_string();
                self.check_type_words(node, &format!("`{name}`"));
                let sample = self.declaration_sample(begin);
                let governs = self.governs_element_pointer(node);
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
                self.check_type_words(node, &format!("typedef `{name}`"));
                let sample = self.declaration_sample(begin);
                let governs = self.governs_element_pointer(node);
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
            "StaticAssertDecl"
            | "FileScopeAsmDecl"
            | "PragmaCommentDecl"
            | "PragmaDetectMismatchDecl"
            | "ImportDecl"
            | "NamespaceDecl"
            | "UsingDecl"
            | "UsingDirectiveDecl"
            | "FunctionTemplateDecl"
            | "ClassTemplateDecl"
            | "TypeAliasDecl" => {
                self.fail(format!(
                    "`{}` contains a `{kind}` declaration the inventory reader does not \
                     model; model it in crates/chelis-repr-inventory/src/c_ast.rs before \
                     the file can be scanned",
                    self.path
                ));
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

    fn visit_record(&mut self, node: &Value, begin: Option<Position>) {
        let name = self.name_of(node, "RecordDecl");
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
        let _ = begin;
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
        end: Option<Position>,
    ) {
        let name = field_str(node, "name").unwrap_or_default().to_string();
        if name.is_empty() {
            // An anonymous member's aggregate follows as its own RecordDecl.
            for child in children(node) {
                self.visit_nested(child);
            }
            return;
        }
        self.check_type_words(node, &format!("field `{name}`"));
        let sample = self.declaration_sample(begin);
        let rendered = sample
            .clone()
            .unwrap_or_else(|| format!("{} {name}", qual_type(node).unwrap_or_default()));
        let owner = match self.owners.last() {
            Some(record) => format!("{record}::{name}"),
            None => name.clone(),
        };
        let governs = self.governs_element_pointer(node);
        let narrow = NARROWABLE_FIELDS.contains(&name.as_str())
            && qual_type(node).is_some_and(|spelling| {
                c_lexical::lex_c_tokens(spelling)
                    .iter()
                    .any(|word| word == "int" || word == "int32_t")
            });
        let extent = fixed_extent(qual_type(node).unwrap_or_default(), &rendered);
        self.owners.push(name.clone());
        let _ = end;
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
        let _ = owner;
        self.owners.pop();
    }

    fn visit_field(&mut self, node: &Value, begin: Option<Position>, end: Option<Position>) {
        // A field reached outside `visit_record` (an Objective-C ivar or
        // property) is read with the non-descriptor rule.
        self.visit_record_field(node, false, begin, end);
    }

    fn visit_function(&mut self, node: &Value, begin: Option<Position>) {
        let name = field_str(node, "name").unwrap_or_default().to_string();
        self.check_type_words(node, &format!("`{name}`"));
        let sample = self.declaration_sample(begin);
        let governs = self.governs_element_pointer(node);
        self.with_owner(name.clone(), true, |reader| {
            if governs {
                reader.push("raw-element-pointer", sample.clone(), &name);
            }
            for child in children(node) {
                let kind = field_str(child, "kind").unwrap_or_default();
                let (_, begin, end) = reader.locate(child);
                if kind == "ParmVarDecl" {
                    let parameter = field_str(child, "name").unwrap_or_default().to_string();
                    reader.check_type_words(child, &format!("parameter `{parameter}`"));
                    // The function's own type already covered the carrier.
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
            "UnaryExprOrTypeTraitExpr" if field_str(node, "name") == Some("sizeof") => {
                let sample = self.expression_sample(begin, end);
                self.push("width-arithmetic", sample, "sizeof");
            }
            "CStyleCastExpr" | "CompoundLiteralExpr" if self.governs_element_pointer(node) => {
                let sample = self.expression_sample(begin, end);
                self.push(
                    "raw-element-pointer",
                    sample,
                    qual_type(node).unwrap_or("element pointer cast"),
                );
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

/// Remove `__attribute__((...))` groups from a type spelling before its words
/// are classified.
fn strip_attributes(spelling: &str) -> String {
    let mut output = String::with_capacity(spelling.len());
    let mut rest = spelling;
    while let Some(start) = rest.find("__attribute__") {
        output.push_str(&rest[..start]);
        let after = &rest[start + "__attribute__".len()..];
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
        rest = &after[consumed..];
    }
    output.push_str(rest);
    output
}

/// Does an element type in this spelling GOVERN a pointer? The star must
/// follow the element type through qualifiers only, so an opaque handle
/// beside an integer return width does not count.
fn governs_a_pointer(words: &[String]) -> bool {
    words.iter().enumerate().any(|(index, word)| {
        if !(C_ELEMENT_TYPES.contains(&word.as_str())
            || CANONICAL_ELEMENT_WORDS.contains(&word.as_str()))
        {
            return false;
        }
        words[index + 1..]
            .iter()
            .take_while(|next| QUALIFIER_WORDS.contains(&next.as_str()))
            .count()
            .checked_add(index + 1)
            .and_then(|star| words.get(star))
            .is_some_and(|next| next == "*")
    })
}

/// A fixed-rank array is one whose extent is a compile-time rank cap: a
/// `*MAX_DIM` constant in the source spelling, or a literal above one in the
/// compiler's canonical spelling.
fn fixed_extent(qual_type: &str, rendered: &str) -> Option<String> {
    let open = qual_type.find('[')?;
    let close = qual_type[open..].find(']')? + open;
    let canonical = qual_type[open + 1..close].trim();
    if canonical.is_empty() {
        return None;
    }
    let source_extent = rendered.find('[').map(|open| {
        rendered[open + 1..]
            .trim_end_matches(']')
            .trim()
            .to_string()
    });
    let spelled = source_extent
        .as_deref()
        .filter(|extent| !extent.is_empty())
        .unwrap_or(canonical);
    let is_cap =
        spelled.contains("MAX_DIM") || canonical.parse::<u64>().is_ok_and(|extent| extent > 1);
    is_cap.then(|| spelled.to_string())
}
