use core::cell::UnsafeCell;

use halo2_proofs::{arithmetic::Field, circuit::Region};

#[cfg(feature = "thread-safe-region")]
use std::sync::Mutex;

/// Wrapper that enables sharing a [`Region`] across threads when the
/// `thread-safe-region` feature is enabled.
pub struct ThreadRegion<'a, F: Field> {
    inner: UnsafeCell<Region<'a, F>>,
    #[cfg(feature = "thread-safe-region")]
    lock: Mutex<()>,
}

impl<'a, F: Field> ThreadRegion<'a, F> {
    /// Creates a new [`ThreadRegion`] from a Halo2 [`Region`].
    pub fn new(region: Region<'a, F>) -> Self {
        Self {
            inner: UnsafeCell::new(region),
            #[cfg(feature = "thread-safe-region")]
            lock: Mutex::new(()),
        }
    }

    /// Executes a closure with a mutable borrow of the underlying [`Region`].
    pub fn with_mut<R>(&self, f: impl FnOnce(&mut Region<'a, F>) -> R) -> R {
        #[cfg(feature = "thread-safe-region")]
        let _guard = self
            .lock
            .lock()
            .expect("thread-safe region mutex should not be poisoned");

        // Safety: callers must uphold the contract that there is only a single
        // mutable access to the region at a time. The optional mutex above
        // enforces this when the `thread-safe-region` feature is enabled.
        let result = unsafe { f(&mut *self.inner.get()) };

        result
    }
}

/// Execution context for region-based assignments in the model synthesis
/// pipeline.
pub struct RegionCtx<'a, F: Field> {
    region: Option<ThreadRegion<'a, F>>,
    offset: usize,
    #[allow(dead_code)]
    constants_len: usize,
}

impl<'a, F: Field> RegionCtx<'a, F> {
    /// Creates a new region context that owns the provided region.
    pub fn new(region: Region<'a, F>) -> Self {
        Self {
            region: Some(ThreadRegion::new(region)),
            offset: 0,
            constants_len: 0,
        }
    }

    /// Creates a new region context with preloaded constants.
    pub fn new_with_constants(region: Region<'a, F>, constants_len: usize) -> Self {
        Self {
            region: Some(ThreadRegion::new(region)),
            offset: 0,
            constants_len,
        }
    }

    /// Creates a dummy region context without any underlying region.
    pub fn new_dummy() -> Self {
        Self {
            region: None,
            offset: 0,
            constants_len: 0,
        }
    }

    /// Applies a closure to the underlying region if one is present.
    pub fn with_region<R>(&self, f: impl FnOnce(&mut Region<'a, F>) -> R) -> Option<R> {
        self.region.as_ref().map(|region| region.with_mut(f))
    }

    /// Advances the current row offset within the region.
    pub fn advance(&mut self, delta: usize) {
        self.offset += delta;
    }

    /// Returns the current offset inside the region.
    pub fn offset(&self) -> usize {
        self.offset
    }
}

#[cfg(feature = "thread-safe-region")]
unsafe impl<'a, F: Field> Send for ThreadRegion<'a, F> {}
#[cfg(feature = "thread-safe-region")]
unsafe impl<'a, F: Field> Sync for ThreadRegion<'a, F> {}

#[cfg(feature = "thread-safe-region")]
unsafe impl<'a, F: Field> Send for RegionCtx<'a, F> {}
#[cfg(feature = "thread-safe-region")]
unsafe impl<'a, F: Field> Sync for RegionCtx<'a, F> {}
