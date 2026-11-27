//! Tight alignment for small, internal buffers whose consumers respect Width.
//! Guest mirrors keep their existing allocation behavior.
use super::*;

// Agility 1.618+ definitions absent from windows 0.62.2. ABI checked against
// microsoft/DirectX-Headers include/directx/d3d12.h and the tight alignment spec.
const FEATURE: D3D12_FEATURE = D3D12_FEATURE(54);
const FLAG: D3D12_RESOURCE_FLAGS = D3D12_RESOURCE_FLAGS(0x400);
#[repr(C)]
#[derive(Default)]
struct Support {
    tier: i32,
}

pub(super) fn supported(device: &ID3D12Device) -> bool {
    if std::env::var_os("VICTORY_TIGHT_HASH_BUFFERS").is_some_and(|v| v == "0") {
        return false;
    }
    let mut support = Support::default();
    // SAFETY: this feature's output is a single 32-bit tier enum. Older
    // runtimes reject the query, retaining the original resource description.
    let supported = unsafe {
        device
            .CheckFeatureSupport(
                FEATURE,
                (&raw mut support).cast(),
                size_of::<Support>() as u32,
            )
            .is_ok()
    } && support.tier >= 1;
    tracing::info!(supported, "tight alignment for internal hash buffers");
    supported
}

pub(super) fn description(
    device: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
) -> Option<D3D12_RESOURCE_DESC> {
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
        || desc.Width >= 65536
        || desc.Alignment != 0
    {
        return None;
    }
    let mut tight = *desc;
    tight.Flags |= FLAG;
    // SAFETY: ordinary allocation queries; they allocate no backing.
    let (old, new) = unsafe {
        (
            device.GetResourceAllocationInfo(0, &[*desc]),
            device.GetResourceAllocationInfo(0, &[tight]),
        )
    };
    (new.SizeInBytes != u64::MAX
        && new.SizeInBytes > 0
        && new.Alignment > 0
        && new.SizeInBytes < old.SizeInBytes)
        .then_some(tight)
}
