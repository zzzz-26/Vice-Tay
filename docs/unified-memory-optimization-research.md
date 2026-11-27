# Unified Memory Emulation and Optimization Research

Reviewed against local source `a827163` on 2026-11-27. This revision replaces
unsupported projections in the original research with code findings, scoped
changes, and explicit validation limits. Historical runs are evidence about
those builds and scenes, not predictions for the current build.

## Architecture and what must be preserved

Xbox Series X has 16 GB of shared GDDR6 with two bandwidth regions (10 GB at
560 GB/s and 6 GB at 336 GB/s). CPU/GPU sharing avoids many transfers required
by a discrete PC GPU. It does **not** remove synchronization requirements or
mean that CPU and GPU writes are instantly safe to consume in any order.
Title GPU addresses being represented as CPU addresses in this compatibility
layer is an address-space convention, not proof that all physical addresses,
cache behavior, or layouts can be reproduced with ordinary PC allocations.

On a discrete GPU, system RAM and local VRAM have different access costs.
Importing system memory into D3D12 does not turn it into local VRAM or remove
PCIe limits. `UMA` and `CacheCoherentUMA` are distinct D3D12 capabilities.
The correct strategy is to preserve sharing, aliasing and command ordering
while choosing backing and lifetimes for each actual usage pattern.

Sources:
- [Xbox Series X specifications](https://www.microsoft.com/en-gb/d/xbox-series-x/8wj714n3rbtl)
- [D3D12 architecture capabilities](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_architecture)
- [Memory management strategies](https://learn.microsoft.com/en-us/windows/win32/direct3d12/memory-management-strategies)
- [Resource states and synchronization](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12)

## Findings and implemented changes

### 1. Idle command allocators retain their high-water allocation

`crates/victory-gpu/src/list.rs` retains up to 512 idle host lists. Its 256 MiB
byte cap measures retained scratch storage, **not driver command-allocator
storage**. `list/thin.rs` previously had an unbounded returned-list pool.
`host/prelude_pool.rs` has an optional pool of 128 preludes.

Microsoft documents that command allocators grow but do not shrink, and that
`Reset` makes storage reusable without reducing its underlying size. This
can explain retention after a burst; it does not establish how many GiB
these objects retain in the user's current scene.

Implemented a shared idle-pool policy in `idle_pool.rs`:
- Preserve the existing host-list count/scratch caps and prelude count cap.
- Bound the thin returned-list pool to 512 entries when trimming is enabled.
- Expire idle entries after 30 seconds normally, or 2 seconds under existing
  host-memory pressure detection. Newly returned entries are preserved.
- Keep 32 warm host lists, 32 thin lists, and 16 pooled preludes.
- Release at most 32 expired entries per pool per once-per-second sweep.
- Preserve reuse order; release driver objects outside the pool lock.
- Only objects returned through the existing lifetime/completion paths are
  eligible. GPU fence handling and the existing retirement delay are unchanged.

`VICTORY_IDLE_TRIM=0` disables expiry and the new thin-list count cap for
comparison. Other pre-existing count/byte caps remain. Prelude pooling itself
is still opt-in; this change does not enable it globally.

[Microsoft command allocator guidance](https://learn.microsoft.com/en-us/windows/win32/direct3d12/recording-command-lists-and-bundles)

### 2. Optional growth must account for physical RAM and concurrent allocations

`residency.rs` previously admitted optional growth from available commit alone.
The staging-growth check charged zero bytes even when `pool.rs` was about to
allocate a 64 MiB block. Available commit can remain high while physical RAM
is exhausted and the application is paging.

Implemented reservations for speculative staging allocations and optional
CPU-backed second buffers:
- Charge the actual allocation size, rounded to a 64 KiB heap granularity.
- Preserve the existing commit reserve and leave at least 1 GiB of available
  physical RAM after optional growth. This is a heuristic, not a measured optimum.
- Serialize admission against a shared pending-allocation ledger.
- Keep successful allocations charged until the next successful OS sample;
  refund failed allocations automatically with an ownership guard.
- Refresh the sample at most every 32 ms. A sample taken while a driver call
  is pending can conservatively double-count part of that pending allocation.
- Do not hold the ledger lock across allocation or driver calls.

Existing staging slices remain reusable. Required table-bound texture uploads
and material bakes keep their existing paths; this gate must not cause them
to sample uninitialized or reused texture storage. The ahead-upload check is
only a hint; the allocation itself must obtain a size-specific reservation.
These reservations cover the named optional paths, not all game/OS allocation.

### 3. Completed submission upload buffers can retain 768 MiB

`host/send.rs` caches completed send buffers, with a 768 MiB system-memory cap
or a 256 MiB cap when the send-BAR option is selected. Previously entries had
no expiry. The cache already reused mapped buffers by best fit; a proposed
new persistent ring must therefore justify its benefit against that reuse.

Implemented the same age-based expiry, retaining two warm buffers and keeping
the existing best-fit lookup and largest-buffer-first byte-cap policy. Equal
size matches reuse the most recently returned buffer, so unused siblings can
age out instead of all being kept warm by round-robin reuse.
Only returned buffers are candidates. In-flight submission storage is never
trimmed. Actual savings depend on cache occupancy and recent demand; 768 MiB
is a cap, not a promised saving.

### 4. Use existing on-demand texture backing on the explicit 4070 preset

`tiles.rs` can back eligible streamed textures mip by mip instead of committing
the entire chain. Current upstream code also retires backing for discarded
standard mip levels, with hysteresis and GPU-completion lifetime handling;
the merged budget fixes constrain new tile allocation.

Enabled `VICTORY_LEVELS_ON_DEMAND=1` inside the non-Safe **explicit 4070**
preset in `scripts/play.ps1`. The implementation checks tiled-resource tier 2
support. Other presets and the launcher's existing automatic preset selection
are unchanged. `-More @{ VICTORY_LEVELS_ON_DEMAND = '0' }` can override it.

Historical source comments describe 2.2-3.0 GB of backing for absent levels in
particular captures. That is useful motivation, not a current saving estimate.
The intended behavior preserves resident texture quality; visual equivalence
and a performance improvement have not yet been established for this change.
Freeing VRAM can help RAM pressure only to the extent that migration or shared
backing is involved. A separate heap's virtual size is not its resident size.

- [D3D12 tiled-resource tiers](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_tiled_resources_tier)
- [DXGI memory budget and usage](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ns-dxgi1_4-dxgi_query_video_memory_info)

## Further opportunities, not implemented in this change

### Physical-page backing that can actually be reclaimed

`crates/victory-hle/src/modules/xmem.rs` allocates one pagefile-backed Windows
section per Xbox physical-page allocation. The source records a request of
about 6.1 GiB. Individual `Slice` objects share ownership of the section.
`XMemFreePhysicalPages` removes page records, but surviving slices and mappings
retain the section. Its persistent DirectStorage view is also retained until
the section is released. A second view aliases the same backing; it is not a
second physical allocation and must not be counted twice.

Windows does not allow committed `SEC_RESERVE` section pages to be decommitted
with ordinary `VirtualFree`. Thus partial page freeing need not return the
corresponding commit until the section and its views are gone. This is a
concrete retention mechanism, **not proof the title frees enough pages during
normal gameplay for a redesign to help**.

A future implementation could use smaller independently reclaimable sections.
It must handle ranges crossing sections, guest page identities, live aliases,
DirectStorage writes, and outstanding GPU references. Small sections also add
mapping/locking overhead; measure free-but-retained section bytes first.
Changing `SEC_RESERVE` to `SEC_COMMIT` is not this fix: it charges the whole
allocation up front and can worsen allocation failures.

- [Windows section commitment semantics](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-createfilemappingw)
- [OpenExistingHeapFromFileMapping](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device3-openexistingheapfromfilemapping)

### Window padding, sharing and GPU-upload backing

`shadow.rs` normally rounds loose buffer windows to a 4 MiB grid. Smaller
sized windows and root-view tails already exist behind `VICTORY_WINDOW_KB`
and `VICTORY_WINDOW_TAILS`. They can save padding but can also create multiple
overlapping copies. Unsized root bindings must retain their accessible range;
bytes not previously copied are not automatically safe to make inaccessible.
Compare allocated bytes, unique window coverage and copied bytes before
changing defaults. This change leaves window geometry unchanged.

`VICTORY_WINDOWS_BAR_MB` already supports eligible CPU-written, shader-read
windows directly in GPU-upload memory without a CPU-side mirror. A larger
budget can help only when VRAM headroom and support exist. It can instead
crowd out textures on a 12 GB card. GPU-upload memory is write-combined;
CPU reads are inefficient. GPU-write/readback resources need different paths.
`VICTORY_SEND_BAR` also already exists, and repository captures recorded CPU
write costs offsetting its GPU-copy savings. Neither option is enabled here.

- [GPU-upload heap specification](https://microsoft.github.io/DirectX-Specs/d3d/D3D12GPUUploadHeaps.html)
- [D3D12 heap usage guidance](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_heap_type)

### In-place mappings and asynchronous work

`INPLACE=2` and `PORT_CONSTANTS` are not universal zero-copy switches. The
implementation documents pinned-page lifetime restrictions and assumptions
about CPU operations versus pending GPU writes. System-memory import still
uses PCIe on a discrete GPU. A general solution needs explicit alias and
in-flight-write tracking; the proposed 5-10 GB default saving was unverified.

Batching residency and pipelining preparation could reduce serialized work,
but must preserve mapping completion, resource state and queue dependencies.
An asynchronous copy queue needs ownership transitions and waits before the
first consuming draw. Removing a CPU wait without reproducing its dependency
can cause rendering corruption. No such changes are made here.

## Corrections to the original research

- The launcher already sets `CARD_STEPS=2`, NVIDIA `PLACED_ORDER=1` outside
  Safe mode, and explicit 4070 card-buffer/window budgets. Their historical
  gains cannot be added again. Core card-buffer defaults now scale with the
  reported VRAM budget; the original description of the 4090 path is stale.
- Shared memory is a substantial architectural challenge, but the document
  did not prove it is the single dominant cause of every current FPS deficit.
  Overall CPU/GPU utilization percentages do not identify the critical path.
- RDNA supports both wave32 and wave64. Preserve each shader's actual guest
  semantics; globally forcing wave32 or broadening the barrier fallback is
  not justified. Register pressure and occupancy are shader-specific.
- `victory-formats/src/bcpack.rs` decodes losslessly packed BC1-BC7 block
  streams. The proposed universal BCPack-to-BC7 reinterpretation and assumed
  decompress-to-pixels/recompress cycle do not describe this implementation.
  Tiled block data still needs the correct host layout.
- A hard working-set ceiling removes residency, not allocation ownership or
  commit. It may force paging and worsen frame times. It is not a general
  memory-reclamation fix and is not enabled for the 4070 here.
- The projected +30-100% additional FPS, additive CPU millisecond savings,
  and target of 30 rendered FPS at <=16 GB private memory were not established
  by comparable current-source measurements. They are withdrawn.

Sources:
- [AMD: occupancy and wave sizes](https://gpuopen.com/learn/occupancy-explained/)
- [Xbox Velocity Architecture and BCPack](https://news.xbox.com/en-us/2020/07/14/a-closer-look-at-xbox-velocity-architecture/)
- [Windows working-set behavior](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsize)

## Validation and observation

Diagnostic builds now emit `memory, idle pools:` with arrays containing
`[count, bytes, hits, misses, released]` for host lists, thin lists, send
buffers and preludes. List bytes measure scratch only; driver allocator
storage is not reported by that counter. These counts distinguish actual
reclamation from a lower process working set caused by paging.

Focused checks cover expiry, retained warm entries, checked-out ownership,
reuse order, byte caps, physical/commit admission, and allocation reservations
across failure, completion and resampling. No game run or visual verification
is performed as part of this change. No new FPS or memory-saving measurement
is claimed.

For a later comparison, use the same scene, camera, resolution, upscaler and
frame-generation settings, and inspect both private commit and working set,
available physical RAM, page-in activity, DXGI local/nonlocal usage, rendered
frame times and these pool counters. Record and review the complete visual
verification per AGENTS.md before reporting rendering equivalence.

Validation on 2026-11-27:

- 26 distinct focused tests passed: residency (9), idle pool (4), other pool
  tests (10), and tiles (3). The full on-card texture test remains ignored.
- Clippy: all victory-gpu targets and features passed with -D warnings.
- Diagnostic executable rebuilt successfully with
  `cargo build --profile diag -p victory --features diagnostics --locked`;
  output: `target/diag/victory.exe`. It was not launched.
- PowerShell launcher syntax check and git diff --check passed.
- Cargo emitted filesystem hard-link fallback warnings, not code diagnostics.
- After git fetch origin, local main and origin/main both name a827163.
  These changes remain local on codex/unified-memory-retention, pending review
  and a PR; neither default branch was changed.
