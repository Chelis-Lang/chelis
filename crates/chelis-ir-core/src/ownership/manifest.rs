//! Borrowed manifest inputs for ordinary ownership calculations.

use chelis_type_core::RootPathStep;

#[derive(Clone, Copy)]
pub struct RootEntryView<'a> {
    pub name: &'a str,
    pub def_name: &'a str,
    pub path: &'a [RootPathStep],
    pub host: bool,
}

pub trait ManifestSource {
    fn len(&self) -> usize;
    fn entry(&self, index: usize) -> Option<RootEntryView<'_>>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Copy)]
pub struct ManifestEntries<'a>(pub &'a dyn ManifestSource);

impl<'a> ManifestEntries<'a> {
    pub fn len(self) -> usize {
        self.0.len()
    }
    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }
    pub fn get(self, index: usize) -> Option<RootEntryView<'a>> {
        self.0.entry(index)
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = RootEntryView<'a>> {
        (0..self.len()).map(move |index| self.get(index).expect("manifest index is in range"))
    }
}

impl dyn ManifestSource + '_ {
    pub fn entries(&self) -> ManifestEntries<'_> {
        ManifestEntries(self)
    }
}
