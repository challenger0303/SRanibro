//! Test-only per-thread allocation counter.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

pub(crate) struct CountingAllocator;

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record_allocation() {
    let enabled = ENABLED.try_with(Cell::get).unwrap_or(false);
    if enabled {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation();
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record_allocation();
        }
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() {
            record_allocation();
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
}

struct DisableOnDrop;

impl Drop for DisableOnDrop {
    fn drop(&mut self) {
        let _ = ENABLED.try_with(|enabled| enabled.set(false));
    }
}

pub(crate) fn count_current_thread_allocations<T>(f: impl FnOnce() -> T) -> (T, usize) {
    // Initialize both TLS cells before enabling the counter, so their initialization can
    // never be mistaken for an allocation performed by the measured closure.
    ENABLED.with(|enabled| {
        assert!(
            !enabled.get(),
            "nested allocation measurement is unsupported"
        );
    });
    ALLOCATIONS.with(|count| count.set(0));
    ENABLED.with(|enabled| enabled.set(true));
    let guard = DisableOnDrop;
    let result = f();
    let allocations = ALLOCATIONS.with(Cell::get);
    drop(guard);
    (result, allocations)
}
