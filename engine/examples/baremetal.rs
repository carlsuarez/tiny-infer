//! Freestanding `no_std` smoke example — proves the engine runs on bare metal.
//!
//! The engine crate is `#![no_std]` and allocation-free, but a *library* never has
//! to stand on its own: it borrows the host's `std`, allocator, and panic handler.
//! This example is the missing other half of that story — an actual `#![no_std]`,
//! `#![no_main]` firmware binary that supplies its own `#[panic_handler]`, owns every
//! buffer as a fixed-size stack array (no heap, no allocator), and drives a full
//! [`forward`] pass for one token. If it links, the engine genuinely composes into an
//! embedded target with nothing underneath it.
//!
//! Build it for a bare-metal target (this is a link check; there is no runtime to run
//! it on here):
//!
//! ```sh
//! cargo build -p engine --example baremetal --target thumbv7em-none-eabi
//! ```
//!
//! On a hosted OS the same file builds as an ordinary example whose `main` just
//! explains the above — the `#![no_std]`/`#![no_main]`/`#[panic_handler]` machinery is
//! conditioned on `target_os = "none"`, since a hosted build already has all three.

#![cfg_attr(target_os = "none", no_std)]
#![cfg_attr(target_os = "none", no_main)]

/// On a hosted target there is nothing to demonstrate — `std` already provides the
/// allocator, entry point, and panic handler this example would otherwise supply.
#[cfg(not(target_os = "none"))]
fn main() {
    println!("`baremetal` is a no_std/no_main firmware demo; nothing runs on a hosted OS.");
    println!("Build it for a bare-metal target to exercise the real path:");
    println!("  cargo build -p engine --example baremetal --target thumbv7em-none-eabi");
}

// A spinning `loop {}` is the correct idiom for a bare-metal entry point and panic
// handler: there is no OS thread to yield to, and clippy's `panic!()` suggestion is
// nonsensical inside the panic handler itself.
#[cfg(target_os = "none")]
#[allow(clippy::empty_loop)]
mod firmware {
    use core::panic::PanicInfo;

    use engine::llama::memory::{arena_floats, MemoryBudget};
    use engine::llama::weights::weight_floats;
    use engine::{forward, math, Arena, Config, Kernel, ModelWeights, RunState, Weights};

    /// A tiny synthetic model. Small enough that every buffer fits comfortably on the
    /// stack, so the whole forward pass runs with no allocator at all.
    const CFG: Config = Config {
        dim: 8,
        hidden_dim: 16,
        n_layers: 2,
        n_heads: 2,
        n_kv_heads: 1,
        vocab_size: 32,
        seq_len: 8,
        shared_weights: true,
    };

    // The arena and weight footprints are pure functions of the config, and both are
    // `const fn`, so the buffers are sized — and the budget is checked — at compile
    // time. A real firmware image can `const`-assert its arena fits a fixed RAM budget
    // exactly like this, before a single byte is allocated:
    const ARENA_FLOATS: usize = arena_floats(&CFG);
    const WEIGHT_FLOATS: usize = weight_floats(&CFG);
    const _: () = assert!(MemoryBudget::for_config(&CFG).total_bytes() <= 4 * 1024);

    /// Where the result lands. A real device would drive a peripheral; here a single
    /// `volatile` write is enough to keep the optimizer from eliding the forward pass.
    static mut SINK: u32 = 0;

    /// Firmware entry point. Builds the model on the stack, runs one decoder step, and
    /// publishes the greedily-chosen next token. There is no allocator and no `std`;
    /// every fallible engine call is handled without panicking (a real device would
    /// fault-handle the `Err` arms — they are unreachable here, since the buffers are
    /// sized from the very config they validate against).
    #[no_mangle]
    pub extern "C" fn _start() -> ! {
        // Deterministic small weights (an xorshift in [-0.1, 0.1]) so the pass produces
        // finite, non-trivial logits rather than the all-zero degenerate case.
        let mut weights_buf = [0.0f32; WEIGHT_FLOATS];
        let mut seed: u32 = 0x2545_f491;
        for w in weights_buf.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *w = (seed as f32 / u32::MAX as f32 - 0.5) * 0.2;
        }

        let Ok(view) = Weights::new(&weights_buf, &CFG) else {
            loop {}
        };
        let weights = ModelWeights::F32(view);

        let mut arena_buf = [0.0f32; ARENA_FLOATS];
        let mut arena = Arena::new(&mut arena_buf);
        let Ok(mut state) = RunState::new(&mut arena, &CFG) else {
            loop {}
        };

        // One decoder step for token 1 at position 0, scalar fp32 kernels, fp32 weights
        // (so no int8 activation scratch is needed — pass `None`).
        let logits = forward(&CFG, &weights, &mut state, 1, 0, Kernel::Scalar, &mut None);
        let next = math::argmax(logits);

        // SAFETY: a plain volatile store to our own static; no aliasing, no concurrency.
        unsafe { core::ptr::write_volatile(&raw mut SINK, next as u32) };

        loop {}
    }

    /// The binary's panic handler — the one piece the engine library deliberately does
    /// not provide, leaving the firmware to decide the policy (here: spin forever).
    #[panic_handler]
    fn panic(_: &PanicInfo) -> ! {
        loop {}
    }
}
