//! Single pre-allocated bump arena.
//!
//! The embedded constraint that defines this project: after init, *every* working
//! buffer (activations and KV cache) is carved from one caller-provided block of
//! memory, and nothing in the forward pass allocates or grows.
//!
//! This arena works in units of `f32` — the element type of every buffer in the
//! model — which lets it hand out `&mut [f32]` views using
//! [`slice::split_at_mut`], with **no `unsafe` and no aliasing**: each allocation
//! is a disjoint sub-slice of the original block. The host supplies a properly
//! aligned `&mut [f32]` (e.g. a `vec![0.0f32; n]`), sized via
//! [`crate::memory::arena_floats`].
//!
//! The intended lifecycle is *allocate-once*: a `RunState` carves all of its
//! buffers from the arena during initialization and then reuses them in place for
//! every token, so the steady state is genuinely allocation-free. There is
//! deliberately no `reset` — buffers persist for the arena's lifetime.

use crate::error::EngineError;

/// A bump allocator over a borrowed block of `f32`.
///
/// Each [`alloc`](Arena::alloc) returns a fresh, disjoint `&mut [f32]` that lives
/// as long as the original block (`'buf`), so multiple buffers can be held at
/// once. Tracks a high-water mark for peak-usage reporting.
#[derive(Debug)]
pub struct Arena<'buf> {
    /// The not-yet-handed-out tail of the block. `None` only transiently while an
    /// allocation is in flight.
    free: Option<&'buf mut [f32]>,
    capacity: usize,
    used: usize,
}

impl<'buf> Arena<'buf> {
    /// Wrap a pre-allocated block. The arena never grows beyond `buf`.
    pub fn new(buf: &'buf mut [f32]) -> Self {
        let capacity = buf.len();
        Arena {
            free: Some(buf),
            capacity,
            used: 0,
        }
    }

    /// Carve `n` contiguous `f32` from the arena, zeroed-as-stored (the host
    /// hands us a zeroed block; values are whatever was last written).
    ///
    /// Returns [`EngineError::ArenaOverflow`] if fewer than `n` elements remain;
    /// the arena is left unchanged on failure.
    pub fn alloc(&mut self, n: usize) -> Result<&'buf mut [f32], EngineError> {
        // Take ownership of the free tail so we can split it without aliasing.
        let free = self.free.take().expect("arena free tail is always present");
        if n > free.len() {
            let available = core::mem::size_of_val(free);
            self.free = Some(free); // restore; allocation failed
            return Err(EngineError::ArenaOverflow {
                requested: n * core::mem::size_of::<f32>(),
                available,
            });
        }
        let (head, tail) = free.split_at_mut(n);
        self.free = Some(tail);
        self.used += n;
        head.fill(0.0);
        Ok(head)
    }

    /// Total capacity in `f32` elements.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Elements handed out so far (the high-water mark, since there is no reset).
    #[inline]
    pub fn used(&self) -> usize {
        self.used
    }

    /// Elements still available.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity - self.used
    }

    /// Peak usage in bytes — for the CLI's memory report.
    #[inline]
    pub fn peak_bytes(&self) -> usize {
        self.used * core::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_are_disjoint_and_track_usage() {
        let mut buf = [0.0f32; 16];
        let mut arena = Arena::new(&mut buf);
        assert_eq!(arena.capacity(), 16);

        let a = arena.alloc(4).unwrap();
        let b = arena.alloc(4).unwrap();
        // Both buffers are usable simultaneously and don't overlap.
        a[0] = 1.0;
        b[0] = 2.0;
        assert_eq!(a[0], 1.0);
        assert_eq!(b[0], 2.0);
        assert_eq!(arena.used(), 8);
        assert_eq!(arena.remaining(), 8);
        assert_eq!(arena.peak_bytes(), 8 * 4);
    }

    #[test]
    fn alloc_zeroes_the_buffer() {
        let mut buf = [7.0f32; 8];
        let mut arena = Arena::new(&mut buf);
        let s = arena.alloc(8).unwrap();
        assert!(s.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn overflow_returns_error_and_preserves_state() {
        let mut buf = [0.0f32; 8];
        let mut arena = Arena::new(&mut buf);
        let _ = arena.alloc(6).unwrap();
        let err = arena.alloc(4).unwrap_err();
        assert_eq!(
            err,
            EngineError::ArenaOverflow {
                requested: 4 * 4,
                available: 2 * 4,
            }
        );
        // State intact: the failed alloc didn't consume anything.
        assert_eq!(arena.used(), 6);
        assert_eq!(arena.remaining(), 2);
        // And we can still allocate within the remaining space.
        assert!(arena.alloc(2).is_ok());
    }

    #[test]
    fn exact_fit_succeeds_then_empty() {
        let mut buf = [0.0f32; 4];
        let mut arena = Arena::new(&mut buf);
        assert!(arena.alloc(4).is_ok());
        assert_eq!(arena.remaining(), 0);
        assert!(arena.alloc(1).is_err());
        // Zero-sized allocations are always fine.
        assert!(arena.alloc(0).is_ok());
    }
}
