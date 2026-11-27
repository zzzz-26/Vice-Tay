# Unified Memory Emulation and Optimization Research

Initial implementation reviewed against local source `a827163` on 2026-11-27;
the deeper research below reviews `0f44014`. This revision replaces
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

## Deeper research: progress saved 2026-11-27

This follow-up reviews source at `0f44014` (the preceding implementation is
now committed). The additions below are research proposals, not further runtime
changes. No game, profiler, benchmark, or installer was launched for this pass.
Sources were checked against Microsoft, AMD GPUOpen, NVIDIA, and upstream
open-source repositories. A mechanism being present in code does not establish
its current frame-time or memory cost.

### A. Separate useful allocations from fragmentation and delayed reuse

**Verified in code:** `pool.rs` allocates 64 MiB blocks, searches eligible
blocks for a fitting free range, and frees only completely empty blocks.
Consequently, a small long-lived allocation can retain a whole block. For
example, 1 MiB live in a 64 MiB block prevents releasing that block's other
63 MiB. This is an arithmetic example, not a measurement of this game.
The new idle-list trimming does not solve partially occupied texture heaps.

**Next concrete implementation candidate:** report, per pool, block bytes,
live piece bytes, bytes waiting for reuse, reusable free bytes, largest free
range, and occupancy distribution. Separate short-lived and persistent
allocations *within* a pool where measured lifetimes justify it; staging and
texture pools are already separate. Try packing new allocations into suitable
occupied blocks before retaining additional mostly empty ones. Do not move
live GPU resources just to implement a different placement policy.

[AMD D3D12MA statistics](https://gpuopen-librariesandsdks.github.io/D3D12MemoryAllocator/html/statistics.html)
provide a useful model for reporting allocations versus backing blocks.
[Its defragmentation documentation](https://gpuopen-librariesandsdks.github.io/D3D12MemoryAllocator/html/defragmentation.html)
explains that moving resources requires application-managed recreation, copies,
synchronization and descriptor updates. Importing an allocator does not make
those correctness obligations disappear.

**A second verified retention mechanism:** `pool.rs` has `REST = 2500` ms.
Returned pieces wait 2.5 seconds before their ranges become reusable. At a
hypothetical steady return rate of 200 MiB/s, that delay retains about 500 MiB
of pieces awaiting reuse, before fragmentation. Actual return rate is unknown.

A future fence-based retirement path could release proven-safe pieces sooner.
First count those bytes and audit all producers, GPU queue completion, aliases,
and the existing allocation/mapping workaround. Merely lowering `REST` is not
a safe optimization: elapsed time is not proof that every GPU reference ended,
and the historical workaround may protect more than ordinary resource use.
D3D12MA's [linear/ring allocator](https://gpuopen-librariesandsdks.github.io/D3D12MemoryAllocator/html/linear_algorithm.html)
is useful for truly submission-scoped storage, not arbitrary persistent caches.

### B. Reduce serialized residency work without removing its guarantee

**Verified in code:** `host::resident` calls `MakeResident` for one object.
`host::mapped` serializes a small copy submission and confirmed fence wait
under an `ASIDE` mutex. Comments record device-loss regressions without this
confirmation. These costs apply when new backing is created, not necessarily
on every steady-state frame.

**Candidate:** move backing creation into bounded batches ahead of demand;
track ready-to-use backing separately from allocations still being made
resident. Preserve the existing completion guarantee before publishing memory.
Microsoft's [EnqueueMakeResident](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device3-enqueuemakeresident)
accepts multiple pageable objects and signals a fence asynchronously. The
[open-source D3DX12Residency implementation](https://github.com/microsoft/DirectX-Graphics-Samples/tree/master/Libraries/D3DX12Residency)
is a concrete reference for residency sets and fence-aware management.

This is not a recommendation to set `VICTORY_WAIT_RESIDENT=0`, or proof that
an asynchronous API alone replaces the mapping workaround. First measure the
existing `RESIDENT` / `MAPPED` call counts, total times and maxima, and locate
them on the submission critical path. Batching that increases preallocated
memory can worsen the original RAM problem; budgets must bound it.

### C. Sparse buffer mirrors: an open-source emulator reference

[Xenia's D3D12 shared-memory implementation](https://github.com/xenia-project/xenia/blob/master/src/xenia/gpu/d3d12/d3d12_shared_memory.cc)
uses a reserved buffer and sparse backing for shared-memory emulation, with a
committed-buffer fallback. Its [common shared-memory code](https://github.com/xenia-project/xenia/blob/master/src/xenia/gpu/shared_memory.cc)
tracks page validity and memory invalidation.

**Design inference for Vice-Tay:** a stable host virtual address range backed
only for required pages could reduce overlapping loose-window mirrors and
4 MiB padding. This is more ambitious than changing `VICTORY_WINDOW_KB`.
It needs guest-address/physical-page identity, alias-aware dirty ownership,
GPU-write tracking, mapping generations, and fence-delayed reuse. Unknown
shader access ranges must be covered; returning zero for an accidentally
unmapped page is rendering corruption, not a memory saving.

Xenia targets Xbox 360, with different address-space and memory requirements.
Its code is a design reference, not a drop-in Series X implementation, and
sparse GPU backing does not make guest CPU memory disappear. Start with
read-only bounded windows if this is prototyped. Preserve the existing path
for unknown ranges and CPU/GPU-writable aliases.

### D. Tight resource alignment is a newer option worth checking

Microsoft's [tight placed-resource alignment specification](https://microsoft.github.io/DirectX-Specs/d3d/D3D12TightPlacedResourceAlignment.html)
adds a feature query and an opt-in resource flag. Supported placed buffers can
have much smaller alignment than the traditional 64 KiB requirement; committed
buffers also have a tighter allocation option. Support must be queried, and
the driver's allocation information must determine placement.

No tight-alignment integration was found in the reviewed pool code. Existing
small-texture placement already requests 4 KiB where appropriate, so that is
not a new gain. First inventory small committed scratch/argument/helper
buffers and their allocation overhead. Tight alignment can help that category;
it will not shrink a deliberately sized 4 MiB shadow window or 64 MiB pool
block by itself. Keep fallback behavior and the existing AMD-specific
placement restrictions until separately validated.

[AMD D3D12 Memory Allocator](https://gpuopen.com/d3d12-memory-allocator/)
implements feature-aware allocation, custom pools, statistics and a virtual
allocator. Its [MIT-licensed source](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator)
can inform the existing Rust allocator or be evaluated behind a narrow C ABI.
It works with D3D12 rather than requiring an AMD GPU. A wholesale replacement
would still need to preserve this repository's retirement, aliasing, and
mapping rules; it would not automatically reclaim guest sections or caches.

### E. Existing experimental descriptor and constant-buffer paths

The earlier document omitted two relevant paths already present in the repo:

- `port_descriptors.rs` / loader `descriptor_sites`: an opt-in descriptor
  write journal marks individual slots instead of scanning every slot on
  changed heap pages. Its comments describe about 600,000 examined slots and
  15-16 ms of work in a historical run. This is not a new measurement. It
  instruments specific guest stores and has a checking fallback; writer
  coverage, executable-version validation and publication ordering must be
  established before enabling it broadly. Sampled checking can find a missed
  write late and is not proof that every draw saw the correct descriptor.
- `port_cbv.rs`: an opt-in raw-view constant-buffer path works around the
  console's 16-byte versus D3D12 CBV's 256-byte alignment difference. It can
  avoid some per-draw copies when the memory is already available in place.
  Comments describe about 23,000 copies / 9 MB in a historical city frame.
  It depends on the in-place path and the title's fenced constant ring; host
  bindings and shader translation must change together. GPU reads of imported
  system memory still cross PCIe on the 4070.

These deserve targeted comparison before inventing another descriptor scanner
or constant ring. Neither is enabled by this research pass. Keep normal
copies for unsupported cases. Existing `VICTORY_PAGE_PARTS` already copies
known 4 KiB dirty parts, so recommending that switch again is not a new fix.

### F. Shader work: optimize measured translated programs, not wave size globally

[NVIDIA's shader guidance](https://developer.nvidia.com/blog/advanced-api-performance-shaders/)
identifies register pressure, divergent constant-buffer access, and buffer
load representation as useful targets. [AMD's RDNA guide](https://gpuopen.com/learn/rdna-performance-guide/)
also discusses register pressure, coalesced accesses and LDS bank conflicts.
These are hypotheses to test against emitted shaders, not permission to change
guest precision, control flow or thread-group semantics.

Concrete next steps for `victory-shader/src/emit`:

1. Rank expensive pipelines using the existing replay's DXIL hashes, then
   correlate those hashes with the guest shader and actual host shader metrics.
2. Inspect register spills, peak live values, shared-memory use and stalled
   memory accesses. Wave64 lowering may amplify those costs, but aggregate
   GPU utilization does not prove that it is the main bottleneck.
3. Look for redundant address/descriptor loads, unnecessarily long live ranges
   and missed adjacent-load grouping. Prove alignment, bounds, alias behavior
   and uniformity before changing the generated operation.
4. Use exact transformations first. Do not broadly force wave32, fast math,
   early depth, `readFirstLane`, or all-resources-bound assumptions. A uniform
   value must actually be uniform, and a resource must actually be bound.

The raw/structured distinction already has handling in `emit/lower.rs`;
NVIDIA's preference for structured access is not evidence that every raw view
can safely become structured. Similarly, [Root Signature 1.1 promises](https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signature-version-1-1)
are valid only for descriptors/data that really remain unchanged for the
promised interval. Guest-visible mutable heaps need their current guarantees.
[DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler)
is an open-source reference for DXIL tooling and compiler passes, but IR
validation alone does not establish faithful translation of guest shaders.

### G. Tool shortlist and what each would answer

No tools were installed or executed during this research. Check current tool,
driver, API-feature and capture support before using a particular combination.
Use one capture/profiling system at a time when diagnosing overhead.

| Tool / implementation | Availability and hardware scope | Specific use here |
|---|---|---|
| [Perfetto](https://perfetto.dev/docs/) | Open source; can view exported traces without changing GPU hardware | Open the repo's existing Chrome-trace timeline; identify the thread/queue dependency that delays each present. |
| [PresentMon](https://github.com/GameTechDev/PresentMon) | Open source; Windows ETW capture, multiple GPU vendors | Record frame-time distributions and correlate stalls. Distinguish presented/generated frames from actual rendered work. |
| [RenderDoc](https://github.com/baldurk/renderdoc) | Open-source graphics debugger; D3D12 support | Inspect host draw resources, descriptors and pipeline state for regressions. Capture compatibility with imported/sparse resources must be checked. |
| [Tracy](https://github.com/wolfpld/tracy) | Open source; CPU, D3D12 GPU, allocation and lock instrumentation | Optional deeper allocation/lock attribution if the existing timeline cannot answer it. Avoid adding a second instrumentation system without a gap to fill. |
| [D3D12MA](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator) | Open source, MIT; D3D12 across GPU vendors | Allocation maps, fragmentation statistics, lifetime-specific pools and tight-alignment implementation reference. |
| [D3DX12Residency](https://github.com/microsoft/DirectX-Graphics-Samples/tree/master/Libraries/D3DX12Residency) | Open-source Microsoft sample/library | Learn from batched residency sets and completion tracking; adapt rather than blindly replace the existing workaround. |
| [Xenia shared memory](https://github.com/xenia-project/xenia/blob/master/src/xenia/gpu/d3d12/d3d12_shared_memory.cc) | Open-source Xbox 360 emulator code | Reference for sparse backing, stable GPU addresses and invalidation; different console, not compatible code by default. |
| [PIX Timing Captures](https://devblogs.microsoft.com/pix/timing-captures-new/) | Microsoft tool; not presented here as open source | Correlate CPU samples, context switches, allocations, file activity and GPU queue timing. |
| [GPUView](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/using-gpuview) | Microsoft WPT/ADK tool; Windows ETW | Distinguish CPU preparation gaps, GPU queue waits, scheduling and page-fault events. |
| [VMMap](https://learn.microsoft.com/en-us/sysinternals/downloads/vmmap) / [RAMMap](https://learn.microsoft.com/en-us/sysinternals/downloads/rammap) | Microsoft Sysinternals tools | Separate committed allocation categories, resident pages and system file cache. Viewing alias mappings does not mean each mapping consumes unique RAM. |
| [Nsight Graphics](https://developer.nvidia.com/nsight-graphics) | NVIDIA tool; appropriate for RTX 4070 GPU analysis | GPU Trace, memory/cache throughput and shader bottlenecks; establish whether the GPU is starved or busy on expensive work. |
| [Radeon Memory Visualizer](https://github.com/GPUOpen-Tools/radeon_memory_visualizer) | Source available; [capture requires supported AMD hardware/driver](https://gpuopen.com/rmv/) | Have AMD community testers inspect resource placement, fragmentation and paging. It is not a 4070 capture tool. |
| [Radeon GPU Analyzer](https://github.com/GPUOpen-Tools/radeon_gpu_analyzer) | Open-source tool; [mode-specific hardware/driver requirements](https://gpuopen.com/rga/) | AMD ISA/register/LDS analysis. Some offline modes need no AMD GPU; D3D12 mode needs the appropriate AMD driver. It does not predict NVIDIA machine code. |

### H. Use the instrumentation already in this repository

`crates/victory-gpu/src/timeline.rs` already exports CPU spans, synchronization
waits/wakeups, slow address-space operations and GPU submission timestamps.
`VICTORY_TIMELINE` produces Chrome-trace JSON for Perfetto plus a text summary.
This is the first tool to try, before adding new instrumentation libraries.
`crates/victory-loader/src/profile.rs` also has scheduled stack sampling.

[The existing replay guide](replay.md) describes frame capture, detailed
pipeline timing, buffer-placement comparisons and shader replacement by hash.
Replay removes normal title-side preparation and alters pacing, so its frame
rate is not the game's frame rate. Its detailed event timings are not isolated
shader costs. Capture can consume gigabytes of extra RAM and pause the title;
do not use a heavy capture as the baseline for the original memory-pressure
problem. A smaller low-overhead timing trace should come first.

At 18 rendered FPS, a frame is about 55.6 ms; 30 FPS requires about 33.3 ms.
The required reduction is about 22.2 ms of critical-path time. Savings on
parallel CPU threads or overlapped GPU work cannot simply be added together.
Section M ranks proposals by the evidence needed to show that they actually
remove work from that critical path.

### I. Follow the allocation, not the old comment

Further source inspection found an important distinction in `shadow.rs`:
`Card::before` is still described in several comments as a before-image or
twin, but `Host::card_before` actually calls
`uav_buffer(card_hash_bytes(size))`. `card_hash_bytes` allocates 16 bytes per
64 KiB piece, with two extra entries in its formula. A 256 MiB mirror asks
for 65,568 bytes of hash storage before D3D12 allocation rounding, not another
256 MiB copy. **Compact GPU before-images are already implemented.** Do not
count these as full-size duplicates or propose hashing as a new optimization.
The initial hash pass can still read the whole mirror; reduced storage does
not imply reduced memory traffic. Measure initialization frequency separately
from steady-state comparisons.

CPU snapshots are different. `Shadow::writing` calls `pooled(self.len)` and
stores range contents at their original offsets. `pooled` reserves a `Vec`
capacity for the full mirror; it does not initialize every byte. Therefore
requested capacity, allocator commitment and physically touched pages must
be distinguished. Existing pool limits and pressure trimming constrain
returned snapshots, not every snapshot held by live shadows.

A concrete next candidate is **range-backed CPU preimages**, if telemetry
shows that declared writes cover a small fraction of large mirrors. Record
active snapshot capacity, the union of initialized byte ranges, and peak
concurrent snapshots first. A design could store bounded chunks with an
offset index, retaining the first preimage for overlapping declarations and
releasing chunks after every reader/submission completes. It must update
`take_snapshots`, `differing`, landing/merge paths and kept-snapshot handling
together: those paths currently depend on direct `snapshot + offset` access.
Merely shortening the vector would cause out-of-bounds accesses. Preserve
guest writes that overlap delayed GPU writeback and do not overwrite an
earlier preimage with a later declaration. This is a substantive change,
not a safe one-line cap adjustment. Sparse coverage could reduce reservation
and allocator pressure; FPS benefit remains conditional on the actual stall.

### J. Existing wide-load experiment: useful control, not a default win

`emit/bitcode/opt.rs` already defines pass bit 64 (`WIDE`): eligible console
loads use one DXIL call per four dwords instead of a call per dword.
`VICTORY_SHADER_OPT=127` enables it alongside the normal passes; default 63
leaves it disabled. The source records RX 7900 XTX replay results of
18.47 to 18.45 ms and 18.23 to 18.16 ms and attributes the negligible change
to the driver already joining loads. These are historical source comments,
not measurements made in this research or evidence of an RTX improvement.

The source includes a software-adapter test for loads crossing view ends,
and the optimization setting participates in the shader cache key. Those
are useful foundations, but do not establish speed or driver correctness
on the user's GPU. A future matched NVIDIA replay can compare selected
expensive shader hashes, generated instruction/register metrics, and output
with 63 versus 127. Separate cold compilation from steady-state timing.
Only promote it if representative shaders improve without increasing spills
or changing bounds behavior. No default was changed or experiment run here.

The checked-in Agility loader requests SDK 620 / package 1.620.2. This makes
the older-SDK explanation insufficient for missing tight-alignment usage;
the [Microsoft feature specification](https://microsoft.github.io/DirectX-Specs/d3d/D3D12TightPlacedResourceAlignment.html)
still requires runtime capability detection and driver-reported allocation
requirements. SDK presence alone does not demonstrate device support.

### K. More precise GPU-upload and budget handling

`Host::mapped_card_buffer` already queries `GPUUploadHeapSupported` and
creates GPU-upload resources. It maps them with `Map(0, None, ...)`, although
the function's contract says the CPU should only write. Microsoft's
[Map documentation](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12resource-map)
defines a null read range as potentially reading the whole resource; an empty
range declares no CPU reads. A narrowly scoped improvement is to audit all
callers and use an empty range for genuinely write-only buffers. This makes
intent accurate and can help tools; it is **not evidence of a steady-state
FPS improvement**, does not restrict which addresses can be accessed, and
does not replace GPU synchronization. Keep readback/shared buffers' actual
read requirements intact. Audit generated stores and CPU compare paths too:
write-combined mappings are a poor destination for read-modify-write or
CPU hashing, even when the source expression looks like a simple assignment.

`window_mapped_for` checks local usage against 80% of the current budget and
has its own atomic byte cap. This is already a guard, not unlimited BAR use.
The [GPU-upload specification](https://microsoft.github.io/DirectX-Specs/d3d/D3D12GPUUploadHeaps.html)
also warns that oversubscribing local memory can force backing into system
memory. Its write-tracking discussion matters for captures: ordinary
write-watch cannot be assumed to detect every GPU-upload write. If adding
manual capture write tracking, feature-query it and follow the tool's
contract; it is not a replacement for the layer's guest-memory dirty tracking.

`Host::video_memory` currently queries both DXGI segments at each call. An
optional improvement is a device-scoped cached budget with
[budget-change notification](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-registervideomemorybudgetchangenotificationevent),
plus periodic usage refresh and a shared ledger for pending local allocations.
Budget notifications do not report every allocation or usage change. Do not
replace live usage accounting with an event-only cached number. Before
implementing, count query calls/time and audit existing per-class reservations
so the same backing is not charged twice. Keep host physical-RAM/commit
admission separate from local VRAM admission; the existing `residency.rs`
growth reservation addresses the former. This proposal is about avoiding
stale admission decisions and repeated queries, not increasing the OS budget.

### L. Descriptor preparation backlog and pressure feedback

`host/aside.rs` executes descriptor preparation on one worker using Rust's
[unbounded channel](https://doc.rust-lang.org/std/sync/mpsc/fn.channel.html).
Jobs are boxed closures, and completed jobs return vectors holding shadow
references. However, `HostList::refreshes_coming` first consumes its previous
job, and list closure also waits. Thus the queue is not proof of unbounded
growth during normal playback: the number of concurrently preparing lists
and their completion paths constrain practical demand.

Record queue depth, oldest-job age, jobs' retained resource bytes where
attributable, worker service time and the existing `WAITED` counter. If
backlog grows while memory is pressured, consider admission/backpressure at
a point with no locks needed by the worker. Do not just replace `channel`
with blocking `sync_channel`: a producer holding a resource lock while
waiting for space can deadlock the worker. Do not add workers without auditing
descriptor generations, guest-memory snapshots and ordering. Jobs for the
same heap cannot be coalesced merely because their heap addresses match.

This is also why raising frames/submissions in flight is not automatically
an improvement. `host/send.rs` intentionally keeps each submission's bytes
separate so a later CPU preparation cannot change an earlier GPU submission's
input. More overlap retains more versions. Preserve that correctness fix;
measure retained in-flight bytes and queue idle intervals before changing
pacing. NVIDIA's [synchronization guidance](https://developer.nvidia.com/blog/advanced-api-performance-synchronization/)
supports reducing unnecessary dependencies and checking actual overlap, not
removing waits indiscriminately or assuming every extra queue is useful.

### M. Implementation order and evidence required

This order favors attribution and bounded changes before redesigning memory
ownership. It is not a ranking of promised FPS gains. Move a conditional item
up only when a trace shows its cost on the frame's critical path.

| Priority | Concrete work | Evidence that justifies it | Acceptance condition / main constraint |
|---|---|---|---|
| First | Add active CPU snapshot capacity/coverage counters and per-pool live, resting, reusable and stranded bytes | RAM growth is not explained by idle caches alone | Account backing once; distinguish active versus cached ownership, commitment versus residency, and aliases versus duplicates. |
| Next, if sparse writes dominate | Replace whole-mirror CPU preimages with bounded range/chunk storage (I) | Large active snapshot capacity with low written-range coverage | Lower peak commitment or allocation stalls; preserve overlapping preimages, delayed landing and all concurrent readers. |
| Next, if allocator fragmentation dominates | Separate allocation lifetime cohorts and pack compatible occupied blocks (A) | Many mostly empty 64 MiB blocks remain pinned by small live allocations | Fewer committed blocks at comparable live bytes; no unsafe moves of guest-addressed or in-flight resources. |
| Next, if delayed reuse dominates | Audit retirement, then use completion-aware reuse where proven safe (A) | High resting bytes after every relevant GPU use has completed | Preserve mapping-workaround requirements and all-queue completion; do not replace the 2.5-second delay with an arbitrary shorter delay. |
| Next, if mapping/residency waits dominate | Batch compatible residency requests behind explicit readiness (B) | Repeated serialized `RESIDENT`/`MAPPED` intervals delay submission | Reduced wait time without removing the historical device-loss workaround before its cause is understood. |
| Next, if descriptor work dominates | Validate existing descriptor journaling/raw-CBV paths (E), then address backlog (L) | Heap preparation, copying or worker waits occupy the critical path | Exact build/site matching, all writers covered, correct descriptor generations and submission-owned data. |
| Conditional GPU work | Profile expensive shader hashes, then try existing wide loads or targeted exact simplifications (F, J) | Hardware metrics show instruction, register or memory-access cost in those shaders | Lower representative GPU time with unchanged output; no additional spills or invalid bounds assumptions. |
| Conditional allocation improvement | Add tight alignment for eligible small allocations (D, J) | Actual allocation-size histograms show material alignment waste | Feature fallback and driver allocation requirements respected; no expectation of shrinking large logical mirrors. |
| Small targeted cleanup | Audit write-only mapping ranges and budget query frequency (K) | Capture overhead or measured repeated-query cost | Accurate API usage; no claim that a mapping hint fixes RAM exhaustion. |
| Larger redesign | Reclaimable guest sections or sparse canonical mirrors (earlier sections, C) | Significant freed-but-retained sections or overlapping mirror backing remains after simpler changes | Guest identity, aliases, CPU/GPU ownership and DirectStorage lifetimes all preserved. |

For CPU snapshots, telemetry should report both the sum of initialized ranges
and their union so repeated declarations do not inflate useful coverage.
For pools, distinguish alignment waste inside an allocation from free gaps
between allocations and whole idle blocks. For work queues, retained resource
sizes should be deduplicated: multiple `Arc`s can prolong one allocation's
lifetime without creating multiple allocations.

### N. Minimal future measurement plan and research status

No measurement in this section has been run. Start with existing memory
counters and a short timeline in the same scene, camera and launch settings.
Record rendered frame times, local/non-local DXGI usage and budgets, process
private commitment, working set and available physical RAM. These quantities
overlap and must not be added into a purported total memory footprint.
Distinguish generated/presented frames from rendered frames. Capture warm-up
and steady state separately; shader compilation and streaming can dominate
the former. Compare repeated samples, including frame-time tails, rather
than a single FPS reading.

Use GPUView/PIX when the GPU queue is starved, memory faults are suspected or
CPU preparation blocks submission; use Nsight hardware metrics when an
identified GPU pass remains expensive. For AMD contributors,
[Radeon GPU Profiler](https://gpuopen.com/rgp/) adds hardware wave timing and
pipeline analysis alongside RMV and RGA. Its
[public repository](https://github.com/GPUOpen-Tools/radeon_gpu_profiler)
provides documentation/releases; a public GitHub repository alone does not
establish that the profiler implementation is open source. Capturing Radeon
hardware behavior does not predict NVIDIA wave64-emulation cost.

Only after identifying the bottleneck should a heavy replay or frame capture
be taken with sufficient memory headroom. Validate one change at a time. Any
visual-equivalence claim needs the full recorded and reviewed comparison
required by `AGENTS.md`, including temporal artifacts. A valid improvement
can be lower memory usage with unchanged FPS, but label it that way; a lower
working set caused by paging is not a memory-management win.

Do not use memory-discard calls as a general leak fix. Microsoft's
[DiscardVirtualMemory contract](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-discardvirtualmemory)
leaves memory committed and makes contents undefined. It cannot reclaim
live guest data or repair ownership. It would require a separate design for
dead, fully owned, page-aligned scratch storage that is rewritten before use;
ordinary pooled Rust vectors may share allocator pages. Existing idle-buffer
release is the safer starting point.

Research checkpoint, 2026-11-27: only this Markdown document was changed in
the deeper research pass. No code, defaults, tools or dependencies were
changed, and no game, tests or profiling session was launched. Earlier build
and test results above apply to the preceding implementation, not these
unimplemented proposals. Several findings are already implemented upstream
(GPU hash preimages, asynchronous descriptor preparation, dirty-run scatter
uploads); they are explicitly not presented as new work.

Repository checkpoint after fetching origin: local `main` and `origin/main`
both remain at `a827163`. The active branch and its remote counterpart are
both at `0f44014`, one commit ahead of the default branch. This document's
research additions remain uncommitted locally. A future commit/PR is needed
to publish these notes and review/merge the branch into the default branch;
no merge or push was performed during research.

## Implementation follow-up: 2026-11-27 (in progress)

The request to implement the findings supersedes the research-only scope
above. Changes are being checked against actual ownership paths; benchmark
results are still unavailable and the game has not been launched.

- Added stable-offset sparse CPU snapshot storage. For snapshots of at least
  8 MiB, the default selects it when host-memory pressure is active;
  `VICTORY_SPARSE_SNAPSHOTS=1` selects it for all such snapshots and `=0`
  preserves dense pooled storage. Only declared pages are committed before
  deferred writers receive addresses. Existing overlap/landing rules remain.
  Small snapshots retain the established vector pool. Sparse buffers return
  only after the existing completion path and have a separate 128 MiB / eight
  buffer cache, released under pressure or after 30 idle seconds.
- Added active snapshot reservation/capacity, backing estimate and declared
  range-union counters, separate from the sparse idle cache. Dense capacity
  is an allocator-request estimate, not an OS commitment measurement.
- Texture/staging allocation now tries occupied ripe blocks before empty
  ripe blocks, preserving whole empty blocks for reclamation and larger
  requests. Ripeness, delayed reuse, alignment and live addresses are intact.
- Added pool backing/live/resting/free/empty/partial-gap accounting. This
  distinguishes retention from fragmentation without summing overlapping
  categories or counting aliases as new allocations.

The sparse snapshot approach uses private address reservations, not guest
sections or imported D3D12 memory. Microsoft's [VirtualAlloc](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc)
and [VirtualFree](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualfree)
contracts permit committing subranges and releasing the reservation after
its final user. No live snapshot pages are discarded or decommitted.

Further implementation checkpoint:

- `VICTORY_BATCH_RESIDENCY=1` coalesces concurrent `MakeResident` requests
  (up to 32 per batch, same device), and concurrent mapping-confirmation
  requests. It inserts no batching sleep. Requests arriving after a batch
  starts wait for a later batch; every requester receives completion or
  failure. The existing mapping-confirmation operation remains, once per
  batch. This experimental path is off by default until a driver comparison.
- Small internal GPU hash buffers query tight-alignment support (Agility
  feature 54) and allocation information. Supported devices can use tighter
  backing; failed creation retries the unchanged description. This is scoped
  to internal hash storage, not guest mirrors or textures whose extra padding
  may currently be relied upon. `VICTORY_TIGHT_HASH_BUFFERS=0` disables it.
  The ABI constants are checked against Microsoft's
  [DirectX-Headers](https://github.com/microsoft/DirectX-Headers/blob/main/include/directx/d3d12.h).
- Submission-owned GPU-upload buffers now declare an empty CPU read range.
  Shared buffers and general mapped windows retain their original range,
  because those paths may read/compare. No general zero-copy switch was enabled.
- Descriptor-worker diagnostics record queued/peak/active jobs, queue latency
  and service time. Failure fallback no longer runs a job while holding the
  sender mutex. A blocking queue cap is intentionally not introduced without
  establishing producer/worker lock dependencies.
- `VICTORY_SECTION_CHUNK_MB=<MiB>` optionally subdivides physical allocations
  of at least 1 GiB on Windows. All sections are prepared before page IDs are
  published, with rollback on preparation failure. Existing section-aware
  run splitting handles writes and mappings across boundaries. Smaller heaps
  and Wine remain unchanged. Default 0 retains the original single section;
  a request crossing a chunk boundary can lose DirectStorage's direct-decode
  fast path and use its existing copy fallback, so this needs comparison.

At this checkpoint, sparse-storage tests (2), residency batch tests (2), and
GPU Clippy checks passed. Further integration checks are ongoing; no game run
or measured FPS improvement is claimed.

### Final scope and controls

Implementation is saved locally. The user requested stopping tests; no further
tests will be run in this pass. Eight focused tests had completed beforehand
(two snapshot, two batch, three physical-section and one allocator test).
The earlier GPU Clippy pass preceded the final physical-section accounting
and launcher edits; it is not a claim that all final code was linted again.

| Control | Default | Purpose |
|---|---|---|
| `VICTORY_SPARSE_SNAPSHOTS` | adaptive | Large new CPU snapshots use sparse commitment under pressure; 1 forces large sparse snapshots, 0 restores pooled vectors. |
| `VICTORY_POOL_PACK` | 1 | Prefer occupied ripe allocator blocks; 0 restores the prior search order. |
| `VICTORY_TIGHT_HASH_BUFFERS` | 1, capability checked | Tighter small internal hash-buffer allocations, with creation fallback; 0 disables. |
| `VICTORY_BATCH_RESIDENCY` | 0 | Experimental concurrent residency/mapping-confirmation batching; 1 enables. |
| `VICTORY_SECTION_CHUNK_MB` | 0 | Experimental independent sections for physical allocations >=1 GiB on Windows; e.g. 64 requests 64 MiB chunks. |

Pass overrides through the launcher's existing `-More` hashtable. `-Safe`
disables all five changes, except explicit `-More` overrides applied later.
Physical-section debug logs now distinguish surviving section capacity from
owned page-ID bytes; neither is labeled committed RAM.

Not every research hypothesis became a runtime change. Fence-only retirement
cannot replace the unexplained mapping/ripeness workaround on present
evidence. Canonical sparse GPU mirrors need guest-alias and GPU-write ownership
tracking beyond this patch. Budget-event caching remains deferred because
usage changes are not budget events and allocation admission must not become
stale. A blocking descriptor queue remains deferred pending the lock audit;
its new telemetry identifies whether it is a bottleneck. Existing shader
wide-load, raw-CBV and descriptor-journal experiments remain independently
selectable; this patch does not enable them globally or claim new shader gains.

The final diagnostic build succeeded (`cargo build --profile diag -p victory
--features diagnostics --locked`), producing `target/diag/victory.exe`.
`git diff --check` passed. After fetching origin, local main and origin/main
still agree; the active branch remains one committed change ahead plus these
uncommitted edits. The game was not launched. No visual
verification or measured FPS/RAM improvement is claimed. Changes are not
committed, pushed or merged by this implementation pass.
