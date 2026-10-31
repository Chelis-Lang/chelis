//! Portable cancellation classification.
use crate::errors::{CheckError, CheckErrorKind};
use crate::prelude::*;
pub const EVAL_CANCELLED_MSG: &str = "chelis::eval::cancelled";
pub fn cancellation_check_error() -> CheckError {
    CheckError::new(
        CheckErrorKind::Other,
        EVAL_CANCELLED_MSG.to_string(),
        vec![],
    )
}
pub fn is_cancellation(message: &str) -> bool {
    message.contains(EVAL_CANCELLED_MSG)
}

#[cfg(test)]
pub(crate) use native::{CancelToken, current_cancel_token, install_cancel_token};
#[cfg(test)]
mod native {

    use std::cell::RefCell;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug, Clone, Default)]
    pub struct CancelToken {
        flag: Arc<AtomicBool>,
    }

    impl CancelToken {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn cancel(&self) {
            self.flag.store(true, Ordering::Relaxed);
        }

        pub fn is_cancelled(&self) -> bool {
            self.flag.load(Ordering::Relaxed)
        }
    }

    thread_local! {
                    static CANCEL_TOKEN: RefCell<Option<CancelToken>> = const { RefCell::new(None) };
    }

    #[must_use = "cancellation is uninstalled when the guard drops"]
    pub fn install_cancel_token(token: CancelToken) -> CancelTokenGuard {
        let previous = CANCEL_TOKEN.with(|cell| cell.replace(Some(token)));
        CancelTokenGuard { previous }
    }

    pub fn current_cancel_token() -> Option<CancelToken> {
        CANCEL_TOKEN.with(|cell| cell.borrow().clone())
    }

    pub struct CancelTokenGuard {
        previous: Option<CancelToken>,
    }

    impl Drop for CancelTokenGuard {
        fn drop(&mut self) {
            let previous = self.previous.take();
            CANCEL_TOKEN.with(|cell| *cell.borrow_mut() = previous);
        }
    }

    impl crate::resources::Cancellation for CancelToken {
        fn is_cancelled(&self) -> bool {
            self.is_cancelled()
        }
    }
}
