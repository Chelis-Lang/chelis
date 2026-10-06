//! Authored spellings of the names the package linker qualifies, for
//! diagnostics (chelis#3269).
//!
//! The linker rewrites every top-level declaration of a package module to
//! its private `pkg__<package>__<Module>__<name>` (or `Pkg__…`) form before
//! the front end runs, so a checker or macro-expansion message that
//! interpolates a declaration name carries the linked spelling. A
//! diagnostic names what the author wrote instead. [`LinkedSourceNames`]
//! records, for each linked name, the declaration it was assigned to,
//! read from the tables the linker assigns names from rather than recovered
//! by inverting the encoding, so an authored name that contains `__` is
//! shown whole (chelis#2919). [`DiagnosticNames`] spells each recorded
//! name for one program (spec/04-type-system.md §2.5):
//!
//! - a declaration of an entry module, the package module a diagnosed file
//!   is by its location, is spelled by its bare name, as the same source
//!   reads as a standalone file;
//! - any other declaration is spelled qualified by its module path,
//!   `Demo.Util.helper`, the form an author writes a qualified reference in;
//! - when that still spells two linked names alike, which happens only for
//!   equal module paths in two packages, both are qualified by package as
//!   well, `lib/Demo.Util.helper`;
//! - a linked name the linker assigned to more than one declaration (its
//!   encoding is not injective: `My__Shape` in `Demo.Main` and `Shape` in
//!   `Demo.Main.My` link alike) is left as linked, because no one authored
//!   spelling names it.
//!
//! No two linked names are ever spelled alike.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

/// The declaration a linked name was assigned to.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Origin {
    package: String,
    module: String,
    name: String,
}

/// Linked-name to declaration table for one linked program.
///
/// Only names the linker changed are recorded: a lexical program, or a
/// single-file entry whose names the linker keeps, contributes nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LinkedSourceNames {
    origins: BTreeMap<String, BTreeSet<Origin>>,
}

impl LinkedSourceNames {
    /// Record that the linker spells the declaration `name` of `module` in
    /// `package` as `linked`. A name the linker kept is not recorded.
    pub fn insert(&mut self, linked: &str, package: &str, module: &str, name: &str) {
        if linked == name {
            return;
        }
        self.origins
            .entry(linked.to_string())
            .or_default()
            .insert(Origin {
                package: package.to_string(),
                module: module.to_string(),
                name: name.to_string(),
            });
    }

    /// Every record of `other`, added to this table.
    pub fn extend(&mut self, other: &LinkedSourceNames) {
        for (linked, origins) in &other.origins {
            self.origins
                .entry(linked.clone())
                .or_default()
                .extend(origins.iter().cloned());
        }
    }

    /// The spellings of this table for a program whose entry modules are
    /// `entry_modules`, each a `(package, module)` pair.
    pub fn for_entry_modules<I, P, M>(self, entry_modules: I) -> DiagnosticNames
    where
        I: IntoIterator<Item = (P, M)>,
        P: Into<String>,
        M: Into<String>,
    {
        DiagnosticNames {
            linked: self,
            entry_modules: entry_modules
                .into_iter()
                .map(|(package, module)| (package.into(), module.into()))
                .collect(),
            spellings: OnceLock::new(),
        }
    }
}

/// How far a declaration's spelling is qualified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Qualification {
    Bare,
    Module,
    Package,
}

impl Qualification {
    fn spell(self, origin: &Origin) -> String {
        match self {
            Self::Bare => origin.name.clone(),
            Self::Module => format!("{}.{}", origin.module, origin.name),
            Self::Package => format!("{}/{}.{}", origin.package, origin.module, origin.name),
        }
    }

    fn next(self) -> Option<Self> {
        match self {
            Self::Bare => Some(Self::Module),
            Self::Module => Some(Self::Package),
            Self::Package => None,
        }
    }
}

/// The authored spelling of each linked name of one program, as the
/// module documentation describes. Spellings are computed on first use, so
/// a program that reports no diagnostic never computes them.
#[derive(Debug, Clone, Default)]
pub struct DiagnosticNames {
    linked: LinkedSourceNames,
    entry_modules: BTreeSet<(String, String)>,
    spellings: OnceLock<BTreeMap<String, String>>,
}

impl DiagnosticNames {
    /// The spelling a diagnostic uses for the linked name `linked`, when it
    /// has one.
    pub fn source_name(&self, linked: &str) -> Option<&str> {
        self.spellings().get(linked).map(String::as_str)
    }

    /// `text` with every identifier that is exactly a spelled linked name
    /// replaced by its spelling.
    ///
    /// An identifier is a maximal run of ASCII letters, digits, `_` and
    /// non-ASCII characters, so a linked name is replaced only as a whole
    /// token: a longer identifier that merely begins with one is left alone.
    /// Text with no spelled name is returned borrowed.
    pub fn render<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if self.linked.origins.is_empty() {
            return Cow::Borrowed(text);
        }
        let spellings = self.spellings();
        let bytes = text.as_bytes();
        let mut rendered: Option<String> = None;
        let mut copied = 0;
        let mut index = 0;
        while index < bytes.len() {
            if !is_identifier_byte(bytes[index]) {
                index += 1;
                continue;
            }
            let start = index;
            while index < bytes.len() && is_identifier_byte(bytes[index]) {
                index += 1;
            }
            // Every token boundary is an ASCII byte, so a char boundary.
            if let Some(spelling) = spellings.get(&text[start..index]) {
                let out = rendered.get_or_insert_with(|| String::with_capacity(text.len()));
                out.push_str(&text[copied..start]);
                out.push_str(spelling);
                copied = index;
            }
        }
        match rendered {
            Some(mut out) => {
                out.push_str(&text[copied..]);
                Cow::Owned(out)
            }
            None => Cow::Borrowed(text),
        }
    }

    /// [`Self::render`] in place.
    pub fn render_in_place(&self, text: &mut String) {
        if let Cow::Owned(rendered) = self.render(text) {
            *text = rendered;
        }
    }

    fn spellings(&self) -> &BTreeMap<String, String> {
        self.spellings.get_or_init(|| self.compute_spellings())
    }

    fn compute_spellings(&self) -> BTreeMap<String, String> {
        // A linked name assigned to more than one declaration has no
        // authored spelling, and stays as linked.
        let mut chosen = self
            .linked
            .origins
            .iter()
            .filter_map(|(linked, origins)| {
                let mut origins = origins.iter();
                let (Some(origin), None) = (origins.next(), origins.next()) else {
                    return None;
                };
                let entry = self
                    .entry_modules
                    .contains(&(origin.package.clone(), origin.module.clone()));
                let qualification = if entry {
                    Qualification::Bare
                } else {
                    Qualification::Module
                };
                Some((linked.as_str(), (origin, qualification)))
            })
            .collect::<BTreeMap<_, _>>();
        // Qualify every member of a group of linked names spelled alike one
        // step further, until no two are. A bare spelling has no `.` and a
        // module-qualified one no `/`, so groups only form within one
        // qualification and two steps always suffice.
        loop {
            let mut grouped = BTreeMap::<String, Vec<&str>>::new();
            for (linked, (origin, qualification)) in &chosen {
                grouped
                    .entry(qualification.spell(origin))
                    .or_default()
                    .push(linked);
            }
            let mut qualified = false;
            for linked in grouped
                .into_values()
                .filter(|group| group.len() > 1)
                .flatten()
            {
                let (_, qualification) = chosen.get_mut(linked).expect("grouped name is chosen");
                if let Some(next) = qualification.next() {
                    *qualification = next;
                    qualified = true;
                }
            }
            if !qualified {
                break;
            }
        }
        let mut spellings = BTreeMap::<String, String>::new();
        let mut spelled_by = BTreeMap::<String, usize>::new();
        let spelled = chosen
            .into_iter()
            .map(|(linked, (origin, qualification))| (linked, qualification.spell(origin)))
            .collect::<Vec<_>>();
        for (_, spelling) in &spelled {
            *spelled_by.entry(spelling.clone()).or_default() += 1;
        }
        for (linked, spelling) in spelled {
            // Distinct origins spell distinctly once package-qualified; keep
            // the guarantee even so.
            if spelled_by[&spelling] == 1 {
                spellings.insert(linked.to_string(), spelling);
            }
        }
        spellings
    }
}

/// A byte of an identifier token. Bytes of a non-ASCII character count, so
/// a token never ends inside one and every token boundary is an ASCII byte.
fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || !byte.is_ascii()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(records: &[(&str, &str, &str, &str)]) -> LinkedSourceNames {
        let mut names = LinkedSourceNames::default();
        for (linked, package, module, name) in records {
            names.insert(linked, package, module, name);
        }
        names
    }

    #[test]
    fn entry_declarations_are_bare_and_others_module_qualified() {
        let names = table(&[
            (
                "pkg__demo__app__Demo__Main__out",
                "demo-app",
                "Demo.Main",
                "out",
            ),
            (
                "Pkg__demo__app__Demo__Main__Shape",
                "demo-app",
                "Demo.Main",
                "Shape",
            ),
            (
                "pkg__demo__app__Demo__Util__helper",
                "demo-app",
                "Demo.Util",
                "helper",
            ),
        ])
        .for_entry_modules([("demo-app", "Demo.Main")]);
        assert_eq!(
            names.render(
                "def 'pkg__demo__app__Demo__Main__out': got `Pkg__demo__app__Demo__Main__Shape`, \
                 see pkg__demo__app__Demo__Util__helper"
            ),
            "def 'out': got `Shape`, see Demo.Util.helper"
        );
        // A longer identifier that begins with a recorded name is a
        // different name and stays as written.
        assert_eq!(
            names.render(
                "`pkg__demo__app__Demo__Main__out_total` and pkg__demo__app__Demo__Main__out"
            ),
            "`pkg__demo__app__Demo__Main__out_total` and out"
        );
    }

    #[test]
    fn distinct_linked_names_are_never_spelled_alike() {
        // `T` in two modules of one package, and `Demo.Util.T` in two
        // packages that share a module path.
        let names = table(&[
            ("Pkg__app__Demo__A__T", "app", "Demo.A", "T"),
            ("Pkg__app__Demo__B__T", "app", "Demo.B", "T"),
            ("Pkg__app__Demo__Util__T", "app", "Demo.Util", "T"),
            ("Pkg__lib__Demo__Util__T", "lib", "Demo.Util", "T"),
            ("pkg__app__Demo__Main__go", "app", "Demo.Main", "go"),
            ("pkg__app__Demo__Other__go", "app", "Demo.Other", "go"),
        ])
        .for_entry_modules([("app", "Demo.Main"), ("app", "Demo.Other")]);
        assert_eq!(names.source_name("Pkg__app__Demo__A__T"), Some("Demo.A.T"));
        assert_eq!(names.source_name("Pkg__app__Demo__B__T"), Some("Demo.B.T"));
        assert_eq!(
            names.source_name("Pkg__app__Demo__Util__T"),
            Some("app/Demo.Util.T")
        );
        assert_eq!(
            names.source_name("Pkg__lib__Demo__Util__T"),
            Some("lib/Demo.Util.T")
        );
        // Two entry modules declaring one name are told apart by module.
        assert_eq!(
            names.source_name("pkg__app__Demo__Main__go"),
            Some("Demo.Main.go")
        );
        assert_eq!(
            names.source_name("pkg__app__Demo__Other__go"),
            Some("Demo.Other.go")
        );
    }

    #[test]
    fn a_linked_name_assigned_twice_stays_as_linked() {
        let names = table(&[
            (
                "Pkg__app__Demo__Main__My__Shape",
                "app",
                "Demo.Main",
                "My__Shape",
            ),
            (
                "Pkg__app__Demo__Main__My__Shape",
                "app",
                "Demo.Main.My",
                "Shape",
            ),
        ])
        .for_entry_modules([("app", "Demo.Main")]);
        assert_eq!(names.source_name("Pkg__app__Demo__Main__My__Shape"), None);
        assert!(matches!(
            names.render("`Pkg__app__Demo__Main__My__Shape` was already declared"),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn keeps_an_authored_double_underscore_whole() {
        // Inverting the encoding cuts `My__Shape` to `Shape`; the table
        // records the authored spelling the linker started from.
        let names = table(&[
            (
                "pkg__demo__Demo__Main__my__helper",
                "demo",
                "Demo.Main",
                "my__helper",
            ),
            (
                "Pkg__demo__Demo__Shapes__My__Shape",
                "demo",
                "Demo.Shapes",
                "My__Shape",
            ),
        ])
        .for_entry_modules([("demo", "Demo.Main")]);
        assert_eq!(
            names
                .render("`Pkg__demo__Demo__Shapes__My__Shape` / pkg__demo__Demo__Main__my__helper"),
            "`Demo.Shapes.My__Shape` / my__helper"
        );
    }

    #[test]
    fn leaves_unrecorded_and_lexical_text_borrowed() {
        let names = table(&[("pkg__demo__Demo__Main__out", "demo", "Demo.Main", "out")])
            .for_entry_modules([("demo", "Demo.Main")]);
        for text in [
            "def 'out' body doesn't match declared signature",
            "unbound variable: never_declared",
            // A linker-shaped name the table does not record is not guessed at.
            "pkg__demo__Demo__Other__out",
            "",
            // A non-ASCII character continues the identifier it touches.
            "naïve: pkg__demo__Demo__Main__outé",
        ] {
            assert!(matches!(names.render(text), Cow::Borrowed(_)), "{text}");
        }
        assert!(matches!(
            DiagnosticNames::default().render("pkg__demo__Demo__Main__out"),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn records_only_names_the_linker_changed() {
        let mut names = LinkedSourceNames::default();
        names.insert("out", "p", "M", "out");
        assert_eq!(names, LinkedSourceNames::default());
    }
}
