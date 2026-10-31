//! Native linker provenance guard.
use std::cell::Cell;
thread_local! { static LINKED_PROGRAM: Cell<bool> = const { Cell::new(false) }; }
pub fn install_linked_program_guard() -> LinkedProgramGuard {
    let previous = LINKED_PROGRAM.with(|cell| cell.replace(true));
    LinkedProgramGuard { previous }
}
pub struct LinkedProgramGuard {
    previous: bool,
}
impl Drop for LinkedProgramGuard {
    fn drop(&mut self) {
        LINKED_PROGRAM.with(|cell| cell.set(self.previous));
    }
}
pub(crate) fn linked_program() -> bool {
    LINKED_PROGRAM.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn linked_program_flag_defaults_false_and_guard_scopes() {
        assert!(!linked_program(), "default provenance is not-linked");
        {
            let _guard = install_linked_program_guard();
            assert!(linked_program(), "guard sets the linked flag");
            {
                let _nested = install_linked_program_guard();
                assert!(linked_program());
            }
            assert!(linked_program(), "nested guard restores TRUE on drop");
        }
        assert!(!linked_program(), "guard restores FALSE on drop");
    }
}
