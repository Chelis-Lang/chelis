//! chelis#2624, chelis#2880: a package entry program prints the same text in
//! `chelis eval` and in its compiled C artifact ([05-OBS-6], [05-OBS-7]).
//!
//! The entry module owes exactly the roots it declares. A package build links
//! chelis-std, path dependencies and the package's sibling modules ahead of
//! the entry module; their top-level values and pure nullary definitions are
//! library code the entry calls, never roots of the entry program. Every lane
//! renders a constructor by its declared source name ([05-OP-32]): the reef
//! linker's qualified internal name stays inside the compiler, for a
//! dependency's data type and for the entry module's own data type alike.

mod common;

use assert_cmd::Command;
use common::{gcc_available, link_generated};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

const DRAWLIB: &str = "module Drawlib.Draw
export (sampled, Point, Shape, origin, shift, area)
type Point =
  | Point(f64, f64)
type Shape =
  | Circle(f64)
  | Square(f64)
sampled = to_tensor([1.0f32, 2.0f32])
def origin() -> Point = Point(0.0f64, 0.0f64)
def shift(p: Point) -> Point =
  match p with {
    | Point(x, y) => Point(add(x, 1.0f64), y)
  }
def area(s: Shape) -> f64 =
  match s with {
    | Circle(r) => mul(r, r)
    | Square(side) => mul(side, side)
  }
";

const HELPERS: &str = "module App.Helpers
export (scale, scaled)
scale = 2.0f64
def scaled(x: f64) -> f64 = mul(x, scale)
";

const ENTRY: &str = "module App.Main
import Drawlib.Draw (sampled, Point, Shape, Circle, Square, origin, shift, area)
import App.Helpers (scaled)
import Std.Time (Date, Monday)
import Std.Contracts (normal_cdf_contract_seed)
type Pair =
  | Pair { left: i64, right: i64 }
def main() -> tensor[2, f32] = sampled
moved = shift(origin())
points = [Point(1.0f64, 2.0f64), shift(Point(3.0f64, 4.0f64))]
areas = [area(Circle(1.5f64)), area(Square(scaled(2.0f64)))]
shapes = (Circle(1.0f64), [Square(2.0f64)])
same = eq(shift(origin()), Point(1.0f64, 0.0f64))
differ = eq(Circle(1.0f64), Square(1.0f64))
day = Date { year: 2026i64, month: 10i64, day: 1i64 }
days = [Some(day), None]
weekday = Monday
own = [Pair { left: 1i64, right: 2i64 }]
def own_origin() -> Pair = Pair { left: 0i64, right: 0i64 }
seed = normal_cdf_contract_seed()
";

/// The entry program's roots, in declaration order. Library values reached
/// through it (`sampled`, `origin`, `scale`, chelis-std's
/// `normal_cdf_contract_seed`) are absent; the entry's own pure nullary
/// `own_origin` is present.
const EXPECTED: &str = "main = tensor(shape=[2], data=[1.0, 2.0])
moved = Point(1.0, 0.0)
points = [Point(1.0, 2.0), Point(4.0, 4.0)]
areas = [2.25, 16.0]
shapes.0.0 = 1.0
shapes.1 = [Square(2.0)]
same = true
differ = false
day.year = 2026
day.month = 10
day.day = 1
days = [Some(Date(2026, 10, 1)), None]
weekday = Monday
own = [Pair(1, 2)]
own_origin.left = 0
own_origin.right = 0
seed = 3235848230
";

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

/// `app` (entry `App.Main`, sibling `App.Helpers`) with a path dependency on
/// `drawlib`; chelis-std resolves from the toolchain.
fn package() -> (TempDir, PathBuf) {
    package_with(
        DRAWLIB,
        &[("src/helpers.ch", HELPERS), ("src/main.ch", ENTRY)],
    )
}

/// `app`, holding `app_files`, with a path dependency on `drawlib`, whose
/// `Drawlib.Draw` module is `library`.
fn package_with(library: &str, app_files: &[(&str, &str)]) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let version = env!("CARGO_PKG_VERSION");
    let app = dir.path().join("app");
    write_file(
        &dir.path().join("drawlib/reef.toml"),
        &format!(
            "schema = \"1\"\n\n[package]\nname = \"drawlib\"\nversion = \"0.1.0\"\n\
             compiler = \"={version}\"\nmodule_prefix = \"Drawlib\"\n"
        ),
    );
    write_file(&dir.path().join("drawlib/src/draw.ch"), library);
    write_file(
        &app.join("reef.toml"),
        &format!(
            "schema = \"1\"\n\n[package]\nname = \"app\"\nversion = \"0.1.0\"\n\
             compiler = \"={version}\"\nmodule_prefix = \"App\"\n\n[dependencies]\n\
             drawlib = {{ path = \"../drawlib\" }}\n"
        ),
    );
    for (file, source) in app_files {
        write_file(&app.join(file), source);
    }
    (dir, app)
}

fn chelis(dir: &Path, cwd: &Path, args: &[&str]) -> String {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(cwd)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", dir.join("reef-home"))
        .env("XDG_CACHE_HOME", dir.join("cache"))
        .args(args)
        .output()
        .expect("run chelis");
    let stdout = String::from_utf8(output.stdout).expect("stdout UTF-8");
    assert!(
        output.status.success(),
        "chelis {args:?} failed\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
}

/// Build `entry` to C under `cwd`, link it, run it, and return its stdout.
/// `None` when no host C compiler exists.
fn compiled_stdout(dir: &Path, cwd: &Path, entry: &str, stem: &str) -> Option<String> {
    let out = dir.join(format!("{stem}-out"));
    chelis(
        dir,
        cwd,
        &[
            "build",
            entry,
            "--target",
            "c",
            "--output",
            out.to_str().expect("UTF-8 output dir"),
        ],
    );
    if !gcc_available() {
        eprintln!("skipped the native run of {stem}.c: no host C compiler");
        return None;
    }
    let status = link_generated(&out, &format!("{stem}.c"), stem);
    assert!(status.success(), "link failed: {status}");
    let run = std::process::Command::new(out.join(stem))
        .output()
        .expect("compiled binary runs");
    let stdout = String::from_utf8(run.stdout).expect("compiled stdout UTF-8");
    assert!(
        run.status.success(),
        "compiled {stem} failed: {}\nstdout: {stdout}\nstderr: {}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    Some(stdout)
}

fn labels(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .map(|line| {
            line.split_once(" = ")
                .unwrap_or_else(|| panic!("observation is not `name = value`: {line:?}"))
                .0
        })
        .collect()
}

#[test]
fn package_entry_prints_the_same_roots_and_constructors_in_eval_and_c() {
    let (dir, app) = package();
    let evaluated = chelis(dir.path(), &app, &["eval", "--file", "src/main.ch"]);
    assert_eq!(evaluated, EXPECTED, "eval must print the entry's roots");
    let Some(compiled) = compiled_stdout(dir.path(), &app, "src/main.ch", "main") else {
        return;
    };
    assert_eq!(
        compiled, evaluated,
        "compiled C and eval must print byte-identical observations"
    );
}

#[test]
fn package_entry_owes_no_library_root_and_leaks_no_linker_name() {
    let (dir, app) = package();
    let evaluated = chelis(dir.path(), &app, &["eval", "--file", "src/main.ch"]);
    let Some(compiled) = compiled_stdout(dir.path(), &app, "src/main.ch", "main") else {
        return;
    };
    for (lane, stdout) in [("eval", &evaluated), ("compiled C", &compiled)] {
        let roots = labels(stdout);
        for library_root in [
            "sampled",
            "origin.0",
            "origin.1",
            "scale",
            "normal_cdf_contract_seed",
        ] {
            assert!(
                !roots.contains(&library_root),
                "{lane} must not print the library root `{library_root}`:\n{stdout}"
            );
        }
        assert!(
            !stdout.contains("Pkg__") && !stdout.contains("pkg__"),
            "{lane} must not print a reef linker name:\n{stdout}"
        );
    }
}

#[test]
fn constructors_spelled_with_double_underscores_keep_their_identity() {
    // The stored constructor name is the whole source spelling, so `Foo__Bar`
    // and `Bar` remain two constructors: each match selects its own arm, and
    // structural equality tells them apart. A terminal-segment de-mangle
    // would store both as `Bar`.
    let library = "module Drawlib.Draw
export (Box, Tag, unbox, tag_value)
type Box[a] =
  | Box(a)
type Tag =
  | Old__New(i64)
  | New(i64)
def unbox[a](b: Box[a]) -> a =
  match b with {
    | Box(x) => x
  }
def tag_value(t: Tag) -> i64 =
  match t with {
    | Old__New(x) => x
    | New(y) => add(y, 100i64)
  }
";
    let entry = "module App.Main
import Drawlib.Draw (Box, Tag, Old__New, New, unbox, tag_value)
type T =
  | Foo__Bar(i64)
  | Bar(i64)
def pick(t: T) -> i64 =
  match t with {
    | Foo__Bar(x) => x
    | Bar(y) => add(y, 100i64)
  }
own = [Foo__Bar(1i64), Bar(2i64)]
own_picks = [pick(Foo__Bar(3i64)), pick(Bar(4i64))]
distinct = eq(Bar(2i64), Foo__Bar(2i64))
lib_picks = [tag_value(Old__New(7i64)), tag_value(New(8i64))]
boxes = [Box(1.5f64), Box(2.5f64)]
opened = unbox(Box(9i64))
";
    let (dir, app) = package_with(library, &[("src/main.ch", entry)]);
    let evaluated = chelis(dir.path(), &app, &["eval", "--file", "src/main.ch"]);
    assert_eq!(
        evaluated,
        "own = [Foo__Bar(1), Bar(2)]\nown_picks = [3, 104]\ndistinct = false\n\
         lib_picks = [7, 108]\nboxes = [Box(1.5), Box(2.5)]\nopened = 9\n"
    );
    let Some(compiled) = compiled_stdout(dir.path(), &app, "src/main.ch", "main") else {
        return;
    };
    assert_eq!(compiled, evaluated);
}

#[test]
fn a_file_outside_a_package_still_owes_every_root_it_declares() {
    // Control: the same library-shaped declarations in one loose file are
    // that program's own roots, so both lanes print every one of them.
    let dir = tempdir().expect("tempdir");
    let source = "type Point =
  | Point(f64, f64)
sampled = to_tensor([1.0f32, 2.0f32])
def origin() -> Point = Point(0.0f64, 0.0f64)
def main() -> tensor[2, f32] = sampled
";
    write_file(&dir.path().join("loose.ch"), source);
    let evaluated = chelis(dir.path(), dir.path(), &["eval", "--file", "loose.ch"]);
    assert_eq!(
        evaluated,
        "sampled = tensor(shape=[2], data=[1.0, 2.0])\norigin.0 = 0.0\norigin.1 = 0.0\n\
         main = tensor(shape=[2], data=[1.0, 2.0])\n"
    );
    let Some(compiled) = compiled_stdout(dir.path(), dir.path(), "loose.ch", "loose") else {
        return;
    };
    assert_eq!(compiled, evaluated);
}
