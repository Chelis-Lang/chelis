use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};
struct Arena { storage: UnsafeCell<[u8; 1048576]>, next: AtomicUsize }
unsafe impl Sync for Arena {}
unsafe impl GlobalAlloc for Arena {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let base = self.storage.get().cast::<u8>();
        let mut used = self.next.load(Ordering::Relaxed);
        loop {
            let Some(address) = (base as usize).checked_add(used)
                .and_then(|address| address.checked_add(layout.align() - 1)) else {
                return core::ptr::null_mut();
            };
            let offset = (address & !(layout.align() - 1)) - base as usize;
            let Some(end) = offset.checked_add(layout.size()) else { return core::ptr::null_mut(); };
            if end > 1048576 { return core::ptr::null_mut(); }
            match self.next.compare_exchange_weak(used, end, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return unsafe { base.add(offset) },
                Err(current) => used = current,
            }
        }
    }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}
#[global_allocator]
static ALLOCATOR: Arena = Arena { storage: UnsafeCell::new([0; 1048576]), next: AtomicUsize::new(0) };
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { loop { core::hint::spin_loop(); } }
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    exercise();
    loop { core::hint::spin_loop(); }
}
