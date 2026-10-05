//! A scoped copy of completed evaluator output for hosts with a forced-exit path.
use std::{
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
    sync::{Arc, Mutex},
};

/// Retains completed transcript lines until one terminal path claims them.
///
/// This capture supplements the ordinary result/error transcript; it does not
/// emit output. A host must choose either its normal result or this retained
/// copy, never both. Clones refer to the same capture and terminal claim.
#[derive(Clone)]
pub struct TranscriptCapture {
    lines: Arc<Mutex<Option<Vec<String>>>>,
}

impl Default for TranscriptCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl TranscriptCapture {
    pub fn new() -> Self {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        Self {
            lines: Arc::new(Mutex::new(Some(Vec::new()))),
        }
    }

    /// Claim terminal emission and drain completed lines. Exactly one caller
    /// receives `Some`, including when no lines have been produced. Subsequent
    /// appends are ignored and subsequent claims return `None`.
    ///
    /// Prepare normal result formatting before claiming, so a host's forced
    /// exit remains able to report completed effects if formatting stalls.
    pub fn finish(&self) -> Option<Vec<String>> {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        self.lines.lock().expect("transcript capture lock").take()
    }

    pub(crate) fn append(&self, line: String) {
        if let Some(lines) = self.lines.lock().expect("transcript capture lock").as_mut() {
            lines.push(line);
        }
    }
}

thread_local! {
    static CURRENT: RefCell<Option<TranscriptCapture>> = const { RefCell::new(None) };
}

/// Restores this thread's previous capture on drop. This guard cannot move
/// between threads; the capture handle itself can be shared with a watchdog.
pub struct TranscriptCaptureGuard {
    previous: Option<TranscriptCapture>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl Drop for TranscriptCaptureGuard {
    fn drop(&mut self) {
        CURRENT.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

/// Install a capture for evaluations started on the current thread. Nested
/// installations restore the outer capture rather than clearing it.
pub fn install_transcript_capture(capture: TranscriptCapture) -> TranscriptCaptureGuard {
    let _fp_env = chelis_runtime::FpEnvGuard::enter();
    TranscriptCaptureGuard {
        previous: CURRENT.with(|slot| slot.replace(Some(capture))),
        _thread_bound: PhantomData,
    }
}

pub(crate) fn current_transcript_capture() -> Option<TranscriptCapture> {
    CURRENT.with(|slot| slot.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_lines_are_drained_once_in_order() {
        let capture = TranscriptCapture::new();
        capture.append("first".into());
        capture.append("second\ncontinued".into());
        assert_eq!(
            capture.finish(),
            Some(vec!["first".into(), "second\ncontinued".into()])
        );
        capture.append("too late".into());
        assert_eq!(capture.finish(), None);
    }

    #[test]
    fn empty_capture_is_not_an_already_claimed_capture() {
        let capture = TranscriptCapture::new();
        assert_eq!(capture.finish(), Some(vec![]));
        assert_eq!(capture.finish(), None);
    }

    #[test]
    fn competing_terminal_paths_have_one_owner() {
        let capture = TranscriptCapture::new();
        capture.append("before timeout".into());
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let other_capture = capture.clone();
        let other_barrier = barrier.clone();
        let other = std::thread::spawn(move || {
            other_barrier.wait();
            other_capture.finish()
        });
        barrier.wait();
        let normal = capture.finish();
        let forced = other.join().unwrap();
        assert_ne!(normal.is_some(), forced.is_some());
        assert_eq!(normal.or(forced), Some(vec!["before timeout".into()]));
    }

    #[test]
    fn scoped_capture_restores_outer_and_stays_on_its_thread() {
        assert!(current_transcript_capture().is_none());
        let outer = TranscriptCapture::new();
        let inner = TranscriptCapture::new();
        {
            let _outer = install_transcript_capture(outer.clone());
            current_transcript_capture().unwrap().append("outer".into());
            {
                let _inner = install_transcript_capture(inner.clone());
                current_transcript_capture().unwrap().append("inner".into());
                std::thread::spawn(|| assert!(current_transcript_capture().is_none()))
                    .join()
                    .unwrap();
            }
            current_transcript_capture()
                .unwrap()
                .append("restored".into());
        }
        assert!(current_transcript_capture().is_none());
        assert_eq!(
            outer.finish(),
            Some(vec!["outer".into(), "restored".into()])
        );
        assert_eq!(inner.finish(), Some(vec!["inner".into()]));
    }
}
