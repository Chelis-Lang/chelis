//! Authored spellings of the names the package linker qualifies, for
//! diagnostics (chelis#3269).
//!
//! The linker rewrites every top-level declaration of a package module to
//! its private `pkg__<package>__<Module>__<name>` (or `Pkg__…`) form before
//! the front end runs, so a checker or macro-expansion message that
//! interpolates a declaration name carries the linked spelling. A
//! diagnostic names what the author wrote: [`LinkedSourceNames`] maps each
//! linked name back to it, recorded from the same tables that assign the
//! linked names rather than recovered by inverting the encoding, so an
//! authored name that itself contains `__` is shown whole (chelis#2919).

use std::borrow::Cow;
use std::collections::BTreeMap;

/// Exact linked-name to authored-name table for one linked program.
///
/// Only names the linker changed are recorded: a lexical program, or a
/// single-file entry whose names the linker keeps, contributes nothing, so
/// rendering its diagnostics is the identity.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LinkedSourceNames {
    names: BTreeMap<String, String>,
}

impl LinkedSourceNames {
    /// Record that the linker spells the authored name `authored` as
    /// `linked`. A name the linker kept is not recorded. The first record of
    /// a linked name wins; the linker assigns each linked name from one
    /// `(package, module, name)` triple, so a second record repeats it.
    pub fn insert(&mut self, linked: String, authored: String) {
        if linked != authored {
            self.names.entry(linked).or_insert(authored);
        }
    }

    /// The authored spelling of the linked name `linked`, when the linker
    /// produced it.
    pub fn source_name(&self, linked: &str) -> Option<&str> {
        self.names.get(linked).map(String::as_str)
    }

    /// Every record of `other`, added to this table.
    pub fn extend(&mut self, other: &LinkedSourceNames) {
        for (linked, authored) in &other.names {
            self.insert(linked.clone(), authored.clone());
        }
    }

    /// `text` with every identifier that is exactly a recorded linked name
    /// replaced by its authored spelling.
    ///
    /// An identifier is a maximal run of ASCII letters, digits, `_` and
    /// non-ASCII characters, so a linked name is replaced only as a whole
    /// token: a longer identifier that merely begins with one is left alone.
    /// Text with no recorded name is returned borrowed.
    pub fn render<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if self.names.is_empty() {
            return Cow::Borrowed(text);
        }
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
            if let Some(authored) = self.names.get(&text[start..index]) {
                let out = rendered.get_or_insert_with(|| String::with_capacity(text.len()));
                out.push_str(&text[copied..start]);
                out.push_str(authored);
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
}

impl FromIterator<(String, String)> for LinkedSourceNames {
    fn from_iter<I: IntoIterator<Item = (String, String)>>(records: I) -> Self {
        let mut names = Self::default();
        for (linked, authored) in records {
            names.insert(linked, authored);
        }
        names
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

    fn table(records: &[(&str, &str)]) -> LinkedSourceNames {
        records
            .iter()
            .map(|(linked, authored)| (linked.to_string(), authored.to_string()))
            .collect()
    }

    #[test]
    fn renders_recorded_linked_names_as_whole_tokens() {
        let names = table(&[
            ("pkg__demo__app__Demo__Main__out", "out"),
            ("Pkg__demo__app__Demo__Main__Shape", "Shape"),
        ]);
        assert_eq!(
            names.render(
                "def 'pkg__demo__app__Demo__Main__out' body doesn't match declared signature: \
                 expected `i32`, got `Pkg__demo__app__Demo__Main__Shape`"
            ),
            "def 'out' body doesn't match declared signature: expected `i32`, got `Shape`"
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
    fn keeps_an_authored_double_underscore_whole() {
        // Inverting the encoding cuts `My__Shape` to `Shape`; the table
        // records the authored spelling the linker started from.
        let names = table(&[
            ("pkg__demo__Demo__Main__my__helper", "my__helper"),
            ("Pkg__demo__Demo__Shapes__My__Shape", "My__Shape"),
        ]);
        assert_eq!(
            names
                .render("`Pkg__demo__Demo__Shapes__My__Shape` / pkg__demo__Demo__Main__my__helper"),
            "`My__Shape` / my__helper"
        );
    }

    #[test]
    fn leaves_unrecorded_and_lexical_text_borrowed() {
        let names = table(&[("pkg__demo__Demo__Main__out", "out")]);
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
            LinkedSourceNames::default().render("pkg__demo__Demo__Main__out"),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn records_only_names_the_linker_changed_and_keeps_the_first() {
        let mut names = LinkedSourceNames::default();
        names.insert("out".to_string(), "out".to_string());
        assert_eq!(names, LinkedSourceNames::default());
        names.insert("pkg__p__M__out".to_string(), "out".to_string());
        names.insert("pkg__p__M__out".to_string(), "other".to_string());
        assert_eq!(names.source_name("pkg__p__M__out"), Some("out"));
        assert_eq!(names.source_name("out"), None);
    }
}
