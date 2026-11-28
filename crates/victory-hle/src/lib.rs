//! Our implementation of the Xbox system API surface.
//!
//! A title imports from three kinds of DLLs, and each gets its own policy:
//!
//! | kind | examples | handled by |
//! |------|----------|------------|
//! | Windows API sets the console shares with PC | `api-ms-win-core-synch-*`, `ws2_32.dll` | host pass-through, overridden here where console behaviour differs (paths, memory) |
//! | console-only system APIs | `xgameruntime.dll`, `xmem.dll`, `d3d12_xs.dll` | implemented here, in [`modules`] |
//! | anything else | | an unimplemented-import stub from the loader |
//!
//! Resolution order for every import: an implementation in [`modules`],
//! then the host if [`HostPolicy`] allows that DLL, then `Missing`.

mod host;
mod image_watch;
#[cfg(all(windows, feature = "diagnostics"))]
pub use victory_gpu::frame_phase;
pub use victory_gpu::frames_begun;

pub mod modules;
mod registry;
pub mod runtime;

pub use host::HostPolicy;
pub use registry::{HleExport, HleModule, registry};
pub use victory_gpu::console;
#[cfg(all(windows, feature = "diagnostics"))]
pub use victory_gpu::replay;

#[cfg(windows)]
pub use victory_gpu::pipeline::known as known_pipelines;
#[cfg(windows)]
pub use victory_gpu::finish_captures;
/// Renderer port #2's switch and marks, for the launcher.
#[cfg(windows)]
pub use victory_gpu::port_descriptors;

use victory_loader::{ImportResolver, Provenance, Resolution};
use victory_pe::ImportSymbol;

/// Resolves a title's imports against our runtime and the host.
#[derive(Debug, Default)]
pub struct HleResolver {
    pub host: HostPolicy,
}

impl HleResolver {
    pub fn new(host: HostPolicy) -> Self {
        HleResolver { host }
    }

    /// Where an import would come from, without binding anything.
    pub fn classify(&self, dll: &str, symbol: &ImportSymbol) -> Option<Provenance> {
        match self.resolve(dll, symbol) {
            Resolution::Found { provenance, .. } => Some(provenance),
            Resolution::Missing => None,
        }
    }
}

impl ImportResolver for HleResolver {
    fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Resolution {
        // Ordinal imports look up `#n`.
        if let Some(address) = registry::lookup(dll, &symbol.to_string()) {
            return Resolution::Found {
                address,
                provenance: Provenance::Hle,
            };
        }
        match self.host.resolve(dll, symbol) {
            Some(address) => Resolution::Found {
                address,
                provenance: Provenance::Host,
            },
            None => Resolution::Missing,
        }
    }
}

// The registry is code-only, so the "Missing" outcome can be checked on any
// host; host pass-through itself needs Windows.
#[cfg(test)]
mod tests {
    use super::*;

    fn name(n: &str) -> ImportSymbol {
        ImportSymbol::Name {
            hint: 0,
            name: n.into(),
        }
    }

    #[test]
    fn console_api_resolves_to_hle() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("XGameRuntime.dll", &name("XGameRuntimeInitialize")),
            Some(Provenance::Hle)
        );
    }

    #[test]
    fn strict_policy_reports_missing() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("api-ms-win-core-synch-l1-1-0.dll", &name("Sleep")),
            None
        );
    }

    #[cfg(windows)]
    #[test]
    fn shared_api_sets_pass_through_to_host() {
        let r = HleResolver::default();
        assert_eq!(
            r.classify("api-ms-win-core-synch-l1-2-0.dll", &name("Sleep")),
            Some(Provenance::Host)
        );
        // Console-only DLLs never pass through.
        assert_eq!(
            r.classify("xmem.dll", &name("XMemSetAllocationHooks")),
            None
        );
    }

    #[cfg(windows)]
    #[test]
    fn overrides_win_over_host() {
        let r = HleResolver::default();
        assert_eq!(
            r.classify(
                "api-ms-win-core-libraryloader-l1-2-0.dll",
                &name("GetModuleHandleW")
            ),
            Some(Provenance::Hle)
        );
    }

    // #510: the RPC runtime's GUID calls are ours, so the DLL itself stays
    // outside the host policy (its own `ws2_32` imports would be a network
    // path `modules::net` never sees).
    #[test]
    fn the_rpc_runtime_is_implemented_here() {
        let r = HleResolver::new(HostPolicy::none());
        for call in [
            "UuidCreate",
            "UuidCreateSequential",
            "UuidFromStringA",
            "UuidFromStringW",
            "UuidToStringA",
            "UuidToStringW",
            "RpcStringFreeA",
            "RpcStringFreeW",
        ] {
            assert_eq!(
                r.classify("rpcrt4.dll", &name(call)),
                Some(Provenance::Hle),
                "{call}"
            );
        }
        // A call we do not implement is not passed to the host either.
        assert_eq!(r.classify("rpcrt4.dll", &name("RpcServerRegisterIf")), None);
    }
}
