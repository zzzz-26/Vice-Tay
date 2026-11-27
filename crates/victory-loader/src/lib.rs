//! Runs a title's original x86-64 code natively, inside our process.
//!
//! Xbox Series X|S executables are x86-64 PE32+ images built for a Zen 2
//! CPU, so the machine code needs no translation. What differs is the
//! platform underneath, so the loader's job is to:
//!
//! 1. [`prepare()`] a lightly patched copy of the image (pure, portable): mark
//!    it as a DLL, remove its entry point and import directory, and detach
//!    its TLS callbacks, so the Windows loader maps it, relocates it,
//!    allocates its static TLS and registers its unwind tables, but runs
//!    none of its code and resolves none of its imports;
//! 2. load that copy as a real module (debuggers, profilers, crash dumps
//!    and stack walks all see it); this step is Windows-only;
//! 3. map the DLLs the title ships beside it the same way
//!    (`load_title_dlls`);
//! 4. bind every import ourselves through an [`ImportResolver`], our
//!    runtime (`victory-hle`), the host or the title's DLLs, and route
//!    anything unresolved to a generated stub that reports exactly what
//!    was called;
//! 5. initialise the title's DLLs, then run TLS callbacks and the entry
//!    point on a thread with the stack size the image asks for; from then
//!    on, pass every thread start and end on to the attached images.

#[cfg(windows)]
mod console_insn;
mod crash;
#[cfg(windows)]
pub mod descriptor_sites;
pub mod frame_rate;
pub mod prepare;
pub(crate) mod publish;
pub mod resolution;
pub mod resolve;
#[cfg(all(windows, feature = "diagnostics"))]
mod profile;
#[cfg(all(windows, feature = "diagnostics"))]
mod stacks;
#[cfg(windows)]
mod threads;
#[cfg(windows)]
mod thunks;
#[cfg(windows)]
mod tls;
#[cfg(windows)]
pub mod write_sites;
#[cfg(windows)]
mod windows;
pub mod working_set;

#[cfg(windows)]
pub use crash::{
    locate, log_crash, log_crashes_to, note_title_output, report_crashes, report_unhandled,
};
#[cfg(all(windows, feature = "diagnostics"))]
pub use crash::save_on_crash;
pub use prepare::{PrepareError, PreparedImage, prepare};
pub use resolve::{ImportResolver, Provenance, Resolution};
#[cfg(all(windows, feature = "diagnostics"))]
pub use stacks::{dump_stacks, dump_stacks_on_schedule};
#[cfg(all(windows, feature = "diagnostics"))]
pub use profile::profile_on_schedule;
#[cfg(windows)]
pub use threads::{disable_thread_calls, set_before_thread_shutdown};
#[cfg(windows)]
pub use thunks::{UnimplementedPolicy, unimplemented_call_count};
#[cfg(windows)]
pub use windows::{
    BindReport, LoadedImage, MissingImport, WithTitleDlls, load, load_title_dll, load_title_dlls,
};

/// `DLL_PROCESS_ATTACH`, for TLS callbacks.
pub const DLL_PROCESS_ATTACH: u32 = 1;
/// `DLL_PROCESS_DETACH`.
pub const DLL_PROCESS_DETACH: u32 = 0;
