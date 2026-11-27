//! Host work that a host library records into a title's command list in
//! place of a console one (the host's FidelityFX upscaler, say): the host
//! device, a title texture's host copy in the state the library is told,
//! and the host list itself.

use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
};

use crate::object::This;

/// The host D3D12 device, created on first use; `None` without a host GPU.
pub fn host_device() -> Option<ID3D12Device> {
    crate::host::host().map(|h| h.device.clone())
}

/// The active host list for one synchronous external recording call.
pub struct Recording<'a> {
    host: &'a crate::host::HostList,
    title_list: This,
}

impl Recording<'_> {
    /// The backing and its last state on this list, without recording a barrier.
    pub fn texture_binding(&self, resource: This) -> Option<(usize, Option<u32>)> {
        let copy = crate::resource::host_copy(resource)?;
        Some(self.host.texture_binding(&copy))
    }

    /// A failed external recording must not allocate a missing backing on rollback.
    pub fn existing_texture_binding(&self, resource: This) -> Option<(usize, Option<u32>)> {
        let copy = crate::resource::made_copy(resource)?;
        Some(self.host.texture_binding(&copy))
    }

    /// Move a title texture to the state the provider will see. The list
    /// holds the backing resource through submission.
    pub fn texture(&self, resource: This, state: u32) -> Option<(ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?.clone();
        crate::list::ensure_state(self.host, resource, state);
        let state = crate::host::legal_state(&texture, state);
        Some((texture, state))
    }

    /// Prepare aliases, mip uploads and depth twins like a compute SRV.
    pub fn prepare_input(&self, resource: This) -> Option<()> {
        let texture = crate::resource::host_resource(resource)?;
        let aliased = crate::resource::alias_of(resource)
            .is_some_and(|source| crate::texture::alias(self.host, resource, source));
        if !aliased && crate::texture::title_written(texture) {
            crate::texture::upload_from(self.host, resource, 0);
        }
        self.host.twin_use(resource, false, false);
        Some(())
    }

    /// Holds a reference to `object` (what the provider recorded uses, such
    /// as a DLSS feature) until the host list is reset, after its last
    /// submission has run.
    pub fn keep(&self, object: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        self.host.keep_shared(object);
    }

    /// Access flushes any pending host barriers before the provider runs.
    pub fn commands(&self) -> &ID3D12GraphicsCommandList {
        &self.host.list
    }

    /// Record AMD's preparation after the upscaler, on the same host list.
    pub fn prepare_framegen(&self, input: crate::framegen::Inputs<'_>) {
        if let Some(preparation) = crate::framegen::prepare(self.commands(), self.title_list, input) {
            self.host.keep_frame_preparation(preparation);
        }
    }
}

/// The producer recorded its output on the host's command list.
pub fn output_written(resource: This) {
    crate::texture::host_written(resource);
}

/// Settle submitted title work before releasing a provider's private GPU resources.
pub fn finish_submitted() {
    crate::queue::before_sync();
    crate::host::settle_for_fence();
}

/// Runs `f` with the host list behind the title list `list` (open for
/// recording); `None` without one. What `f` binds (descriptor heaps, root
/// signatures, pipeline) is replaced before the list's next draw or
/// dispatch.
pub fn record<R>(list: This, f: impl FnOnce(&Recording<'_>) -> R) -> Option<R> {
    if crate::framegen::requested() {
        // Finish the prior presentation before opening an upscaler list.
        // AMD's frame generator reuses its preparation buffers across presents.
        crate::queue::wait_before_framegen();
    }
    let recording = crate::list::thin::for_external(list)?;
    let mut out = None;
    crate::list::with_state(recording, |host, state| {
        let recording = Recording {
            host,
            title_list: list,
        };
        out = Some(f(&recording));
        state.external_compute();
    });
    out
}
