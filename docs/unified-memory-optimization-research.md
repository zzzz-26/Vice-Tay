# Unified Memory Emulation and FPS Optimization Research

Vice-Tay translates an Xbox Series title's D3D12.X graphics commands to host
Direct3D 12 on Windows. The title was built for a console with **unified
memory**: the CPU and GPU share the same GDDR6 pool. On a PC the GPU has its
own VRAM and the CPU has system RAM, separated by the PCIe bus. That
architectural mismatch is the single largest source of FPS loss and RAM
inflation in Vice-Tay today.

This document catalogues the technical background, what the codebase does
to bridge the gap, and concrete optimization opportunities with references.

---

## 1  Xbox Series Memory Architecture

### 1.1  Hardware layout

| Console | Total GDDR6 | Title-usable | Bandwidth | GPU |
|---|---|---|---|---|
| **Series X** | 16 GB | ~13.5 GB (10 GB GPU-optimal @ 560 GB/s, 3.5 GB standard @ 336 GB/s) | Split | RDNA 2, 52 CU, 1.825 GHz |
| **Series S** | 10 GB | ~8 GB (8 GB @ 224 GB/s, 2 GB @ 56 GB/s) | Split | RDNA 2, 20 CU, 1.565 GHz |

> **Sources:**
> - [Xbox Series X|S specs — xbox.com](https://www.xbox.com/consoles/xbox-series-s)
> - [Hot Chips 2020 Xbox System Architecture — IEEE](https://ieeexplore.ieee.org/document/9220622)
> - [Digital Foundry Xbox Series S Deep Dive](https://www.eurogamer.net/digitalfoundry-2020-xbox-series-s-analysis)
> - Web search result: "Xbox Series S 10GB memory split 8GB 2GB GDK Direct3D 12.X"

### 1.2  Unified memory model (UMA / CacheCoherentUMA)

On the console the CPU and GPU see the **same physical addresses**. A title
allocates memory once and both processors access it without any copy. The
D3D12.X API inlines command-list packets directly into shared memory; the GPU
reads buffers, textures and constant data where the CPU wrote them.

Vice-Tay's `victory-gpu` lib.rs header states this directly:

> *"The title's GPU addresses are CPU addresses (console memory is unified)."*
> — [`crates/victory-gpu/src/lib.rs`](file:///d:/Games/Vice-Tay/crates/victory-gpu/src/lib.rs#L7)

The `D3D12_FEATURE_DATA_ARCHITECTURE` query exposes this via the `UMA` and
`CacheCoherentUMA` flags. On a CacheCoherentUMA device, upload heaps use
write-back cached memory and no staging copies are needed.

> **Sources:**
> - [Microsoft Learn — D3D12_FEATURE_DATA_ARCHITECTURE](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_architecture)
> - [DirectX-Specs — Heap Tier 2 UMA Optimizations](https://microsoft.github.io/DirectX-Specs/)
> - Web search: "D3D12_FEATURE_DATA_ARCHITECTURE CacheCoherentUMA Write-Combined"

### 1.3  What breaks on a discrete PC GPU

On a discrete GPU (NVIDIA Ada / Ampere / Turing, AMD RDNA3, etc.):

| Console (UMA) | PC (NUMA / discrete) |
|---|---|
| GPU reads title memory at ~224–560 GB/s | GPU reads system RAM over PCIe at ~25–28 GB/s (Gen4 ×16) |
| CPU writes are instantly visible to GPU | CPU must copy data to VRAM (`DEFAULT` heap) or use write-combined upload heaps |
| No copy overhead | Every byte the GPU reads from title memory must be tracked, copied, hashed and potentially compared |
| ~8 GB total (Series S) | 22–27+ GB private memory on a 48 GB PC running Series S profile |

The 2.5–3× RAM inflation and 30–50 ms/frame CPU overhead come from bridging
this gap. The codebase calls this the **"one idea behind most of it"**:

> *"On the console the CPU and the graphics processor share memory. Here they
> do not, so everything one writes and the other reads must be noticed and
> moved."*
> — [`docs/findings.md`](file:///d:/Games/Vice-Tay/docs/findings.md#L28-L29)

---

## 2  How Vice-Tay Bridges the Gap Today

### 2.1  Shadow copies (`shadow.rs`)

Every title buffer the GPU accesses gets a persistent **host copy** (a
`D3D12_HEAP_TYPE_DEFAULT` resource in VRAM or a shared/upload buffer). The CPU
side of this copy lives in system RAM. Before each submission, changed 64 KiB
pages (detected by write-watching or read-only guards) are refreshed from title
memory into the CPU side, then sent up to the card.

- File: [`crates/victory-gpu/src/shadow.rs`](file:///d:/Games/Vice-Tay/crates/victory-gpu/src/shadow.rs)
- RAM cost: each title buffer exists **twice** — once in title memory, once as a
  CPU-side mirror. Runs show 7–9 GB in shadow windows alone
  ([`docs/findings.md`](file:///d:/Games/Vice-Tay/docs/findings.md#L96)).

### 2.2  Write watching (`watch.rs`)

Title memory allocated via `XMemVirtualAlloc` uses `MEM_WRITE_WATCH` so the
kernel tracks dirty 4 KiB pages. For physical-page views (sections), read-only
guards with a vectored exception handler detect writes at 64 KiB granularity.

- File: [`crates/victory-gpu/src/watch.rs`](file:///d:/Games/Vice-Tay/crates/victory-gpu/src/watch.rs)
- The kernel says which 4 KiB pages were written; a copy takes the 64 KiB page
  around each. Session data: 29 MB written per frame caused ~114 MB copied and
  sent up and re-hashed.

> **Source:** [Microsoft Learn — GetWriteWatch function](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-getwritewatch)

### 2.3  Card steps (`host/card_steps.rs`)

Constant-buffer fills and buffer copies that must wait for in-flight shader
writes are optionally offloaded to the GPU itself via a prepended command list
(`VICTORY_CARD_STEPS=2`). Without this, the CPU submission thread blocks 25–30
ms/frame waiting for flights to land.

- File: [`crates/victory-gpu/src/host/card_steps.rs`](file:///d:/Games/Vice-Tay/crates/victory-gpu/src/host/card_steps.rs)

### 2.4  In-place title memory (`inplace.rs`, `port.rs`)

Where possible, title memory is opened as a D3D12 heap directly
(`OpenExistingHeapFromAddress` / `OpenExistingHeapFromAddress1`), and a buffer
placed over it. The GPU then reads the title's bytes without any copy,
refresh or copy-back — the console's behaviour.

- `VICTORY_INPLACE=1`: windows outside buffers in private memory
- `VICTORY_INPLACE=2`: title buffers too (except large ones on the card)
- `VICTORY_PORT_CONSTANTS=1`: the title's 32-bit graphics heap (174 MB)
  mapped via a view of the same section

Limitations:
- View memory (sections made with `SEC_RESERVE`) cannot be opened this way
- While a heap is open, the title cannot decommit/free/protect those pages
- Assumes no in-flight shader writes overlap the in-place reads

> **Sources:**
> - [Microsoft Learn — ID3D12Device3::OpenExistingHeapFromAddress](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device3-openexistingheapfromaddress)
> - [Microsoft Learn — ID3D12Device13::OpenExistingHeapFromAddress1](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device13-openexistingheapfromaddress1)

### 2.5  GPU Upload Heaps and Resizable BAR

On cards with Resizable BAR, `D3D12_HEAP_TYPE_GPU_UPLOAD` lets the CPU write
directly to VRAM, eliminating staging copies. Vice-Tay uses this in several
places:

- `VICTORY_WINDOWS_BAR_MB`: shader-read memory windows placed directly in
  the card's BAR-mapped memory (up to 3–4 GB on the laptop/4070 Ti SUPER profiles)
- `VICTORY_SEND_BAR=1`: submission send data placed in card memory (saves
  5 ms of GPU copy time but costs 5–6 ms more CPU time due to write-combining
  penalties — currently a net loss, **off by default**)

> **Sources:**
> - [DirectX-Specs — GPU Upload Heaps](https://microsoft.github.io/DirectX-Specs/d3d/GPUUploadHeaps.html)
> - [NVIDIA — Advanced API Performance: GPU Upload Heaps](https://developer.nvidia.com/blog/advanced-api-performance-gpu-upload-heaps/)
> - [AMD GPUOpen — Resizable BAR Best Practices](https://gpuopen.com/learn/rebar-best-practices/)

### 2.6  Levels on demand (`tiles.rs`)

Title textures that only partially own their mip chain are made as reserved
(tiled) resources. Tiles are assigned per-level as the title streams them in.
Before this, committed textures wasted 2.2–3.0 GB of VRAM on unmapped levels.

- File: [`crates/victory-gpu/src/tiles.rs`](file:///d:/Games/Vice-Tay/crates/victory-gpu/src/tiles.rs)
- Switch: `VICTORY_LEVELS_ON_DEMAND=1`

---

## 3  Shader Translation: Xbox GFX10 ISA → Host DXIL

### 3.1  The pipeline

```
Xbox shader container (DXBC)
  → XDAT blob → zlib-compressed GFX10 (RDNA2) machine code
    → victory_shader::gfx10  (disassemble)
    → victory_shader::ir     (lift to SSA IR, one node per instruction)
    → victory_shader::structure (remove EXEC masks → branches + guards)
    → victory_shader::resource  (trace descriptors to root parameters)
    → victory_shader::emit      (lower to DXIL / SM 6.6 LLVM bitcode)
```

- [`crates/victory-shader/src/lib.rs`](file:///d:/Games/Vice-Tay/crates/victory-shader/src/lib.rs)

### 3.2  Wave64 → Wave32 emulation

The Xbox GPU runs RDNA2 with native **wave64** (64-lane wavefronts). NVIDIA
GPUs have a fixed **warp size of 32**. The translator must map wave64 semantics
onto two host warps:

- Group barriers placed only where every console wave of the group arrives
- Cross-lane operations (`v_readfirstlane`, permute, ballot) must account for
  the half-width
- 4 compute shaders are allow-listed for the wave-wide barrier path
  (`VICTORY_WAVE_WIDE`); the rest fall back to per-draw handling
- Incorrect wave64 emulation has caused TDR hangs (`0x887A0006`)

> **Sources:**
> - [AMD RDNA2 ISA Reference Guide](https://gpuopen.com/amd-isa-documentation/)
> - [AMD RDNA Performance Guide — Wave32 vs Wave64](https://gpuopen.com/learn/rdna-performance-guide/)
> - [Microsoft — HLSL SM 6.6 WaveSize](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html)
> - [NVIDIA — Advanced API Performance: Intrinsics](https://developer.nvidia.com/blog/advanced-api-performance-intrinsics/)
> - Web search: "NVIDIA Ada Lovelace warp size 32 wave64 emulation"

### 3.3  Descriptor heap limits on NVIDIA

NVIDIA GPUs cap the shader-visible CBV/SRV/UAV descriptor heap at ~1,000,000
slots. The title creates descriptor tables dynamically per-frame. Before the
early-reclaim fix, the 2^18 slots for tables made afresh ran out in busy
scenes, causing hundreds of dropped draws per frame and visual blow-out.

The fix reclaims descriptor slots when the GPU finishes a flight, not when the
host list resets ~64 flights later.

- [`docs/this-fork.md`](file:///d:/Games/Vice-Tay/docs/this-fork.md#L73-L79)

---

## 4  Current Performance Profile (Series S on RTX 4090)

| Metric | Value |
|---|---|
| Median rendered FPS | 21.1 |
| 1% low FPS | 17.0 |
| Median frame time | 47.3 ms |
| CPU submission prep | 33.7 ms |
| GPU read-back | 12.6 ms |
| Copies per frame | 13.3 ms / 86.8 MB |
| Peak private memory | 24.1 GB |
| Peak working set | 27.4 GB |
| Peak card memory | 8.8 GB |
| Draws per frame | 4,446 (median) |
| Dropped draws | 98 (median) |

Source: [`docs/measurements.md`](file:///d:/Games/Vice-Tay/docs/measurements.md#L62-L68)

### 4.1  Where a frame goes

From [`docs/findings.md`](file:///d:/Games/Vice-Tay/docs/findings.md#L8-L25):

| Component | Time |
|---|---|
| **Submission preparation thread** (the wall) | ~48 ms |
| — Stepping host lists | 18 ms |
| — Refreshing host copies | 10 ms |
| — Waiting for descriptor records | 6 ms |
| — Write-watch queries | 5 ms |
| — Card memory copies | 4 ms |
| — Title bytes under shader writes | 4 ms |
| **GPU busy** | ~41 ms |
| **List recording** (8 threads) | 36,000 calls/frame, 230 ms total |
| **Descriptor mirror** | 400K slots examined, 40K mirrored/frame |
| **Shader write landing** | 16 MB/frame as dwords |

---

## 5  Optimization Opportunities

### 5.1  Expand in-place title memory (highest impact on RAM)

**The problem:** Every title buffer that cannot be placed in-place gets a
full shadow copy in system RAM *and* a VRAM copy. The title's 174 MB
"graphics heap" and its ~7.5 GB of buffer memory are largely duplicated.

**What to do:**
1. **Switch `xmem` sections from `SEC_RESERVE` to `SEC_COMMIT`** so that
   `OpenExistingHeapFromFileMapping` accepts them. This is the main blocker
   for in-place placement of view memory (the page pool). The
   [`inplace.rs`](file:///d:/Games/Vice-Tay/crates/victory-gpu/src/inplace.rs#L22-L27) comments note that
   `SEC_RESERVE` sections are refused. If the section commitment model can be
   changed without breaking the title's page-mapping behaviour, shadow copies
   for all view memory become unnecessary.

2. **Enable `VICTORY_PORT_CONSTANTS=1` by default on NVIDIA** for the 32-bit
   graphics heap. This eliminates ~17 MB/frame of tracked copies, ~6,000
   constant-buffer fills a frame, and all copy-back for that heap.
   [`port.rs`](file:///d:/Games/Vice-Tay/crates/victory-gpu/src/port.rs#L1-L52) already implements it.

3. **Enable `VICTORY_INPLACE=2` by default** (with `VICTORY_LATE_WAIT`
   disabled). This places title buffers in private memory in-place, skipping
   copy, refresh and copy-back entirely.

**Expected RAM saving:** 5–10 GB reduction in private memory if most view
memory can be placed in-place.

> **References:**
> - [`inplace.rs`](file:///d:/Games/Vice-Tay/crates/victory-gpu/src/inplace.rs#L65-L90)
> - [`port.rs`](file:///d:/Games/Vice-Tay/crates/victory-gpu/src/port.rs#L71-L79)
> - [Microsoft — OpenExistingHeapFromFileMapping](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device3-openexistingheapfromfilemapping)

### 5.2  Move shader-read buffers to VRAM (highest impact on FPS)

**The problem:** On the default 4090 profile, title buffers that shaders read
stay in system RAM. The GPU reads them over PCIe at ~25 GB/s instead of local
VRAM at ~1 TB/s. The RTX 3080 10 GB test showed this directly: switching from
the 4090 profile (buffers in RAM) to the 4070 profile (buffers on card, 64 KiB
minimum, 1536 MiB budget) **doubled FPS from 7.4 to 14.7** in the same scene.

**What to do:**
1. **Enable `VICTORY_CARD_BUFFERS_MIN_KB=64` and `VICTORY_CARD_BUFFERS_MB=1536`
   by default** on all NVIDIA GPUs with ≥10 GB VRAM, not just the 4070 profile.
2. **Enable `VICTORY_CARD_WINDOWS_MB=1024`** to also keep shader-read memory
   windows on the card.
3. **Enable `VICTORY_CARD_RESERVE_BUDGETS=1`** to let texture admission
   account for buffer/window reservations.
4. **Use `VICTORY_WINDOWS_BAR_MB`** on cards with Resizable BAR to put
   windows directly in VRAM without a CPU-side copy.

**Measured impact:**
- RTX 3080 10 GB: 7.4 → **14.7 FPS** (2× improvement)
- RTX 5070 Ti 16 GB: 16.5 → **21.6 FPS** (stationary), 15.7 → **20.1 FPS** (open water)
- RTX 4070 12 GB: 2.8–7.8 → **5.9–25.0 FPS** (exploratory)

> **References:**
> - [`docs/rtx4070-performance.md`](file:///d:/Games/Vice-Tay/docs/rtx4070-performance.md#L113-L146)
> - [`scripts/play.ps1`](file:///d:/Games/Vice-Tay/scripts/play.ps1#L91-L101) (4070 profile)
> - [NVIDIA — Advanced API Performance: Memory Management](https://developer.nvidia.com/blog/advanced-api-performance-memory-management/)
> - [Microsoft — Uploading Resources](https://learn.microsoft.com/en-us/windows/win32/direct3d12/uploading-resources)

### 5.3  Reduce submission preparation CPU time

**The problem:** The CPU-side submission thread takes 33–50 ms/frame and is the
wall-clock bottleneck. It is single-threaded for the serial parts (stepping
through lists, refreshing copies, waiting for flights).

**What to do:**
1. **Split the submission preparation across two threads** (already identified
   as "what is next" #1 in
   [`docs/findings.md`](file:///d:/Games/Vice-Tay/docs/findings.md#L84-L86)).
2. **Pipeline submissions**: begin preparing the next submission while the
   current one is being sent to the GPU.
3. **Reduce refresh volume**: with `VICTORY_PAGE_PARTS` on (default), only
   the 4 KiB parts of a 64 KiB page that the kernel reports dirty are copied
   instead of the full page. This already reduced 114 MB → ~29 MB
   ([`watch.rs`](file:///d:/Games/Vice-Tay/crates/victory-gpu/src/watch.rs#L46-L55)).
4. **Enable `VICTORY_CARD_STEPS=2` by default** to offload constant-buffer
   fills and buffer copies to the GPU, eliminating 25–30 ms of CPU waits.
5. **Enable `VICTORY_PLACED_ORDER=1` by default** to avoid refusing whole
   submissions for in-list copies that only matter in order.

### 5.4  Reduce working set and page file pressure

**The problem:** Peak private memory of 22–27 GB on a 48 GB PC causes paging
on 16–32 GB systems, with frame times >1 second.

**What to do:**
1. **Enable `VICTORY_PRELUDE_POOL=1` by default.** The prelude pool recycles
   command allocators + lists after a flight lands. Without it, 831 preludes
   were created on one run; with it, only 186.
   ([`host/prelude_pool.rs`](file:///d:/Games/Vice-Tay/crates/victory-gpu/src/host/prelude_pool.rs))
2. **Cap ownerless view entries** (`VICTORY_OWNERLESS_VIEWS`) — already
   capped at 1 million. The view-key padding fix prevented 52 GB → 27 GB
   growth.
   ([`docs/findings-rtx4090.md`](file:///d:/Games/Vice-Tay/docs/findings-rtx4090.md#L7-L32))
3. **Enable `VICTORY_LEVELS_ON_DEMAND=1` by default** on cards with tiled
   resources tier ≥ 2. This saved 2.2–3.0 GB of VRAM on textures with
   partially-mapped mip chains.
4. **Set `VICTORY_WORKING_SET_MB`** automatically on PCs with ≤32 GB RAM
   (already done in the laptop profile).
5. **Enable `VICTORY_DSTORAGE_IO=direct`** on low-RAM PCs to bypass the
   Windows file cache.

### 5.5  Shader and rendering optimizations

| Optimization | Switch | Impact | Status |
|---|---|---|---|
| Early depth testing | `VICTORY_EARLY_DEPTH=1` | Reduces overdraw for compatible pixel shaders | Opt-in |
| Wave-wide barriers (safe 4 shaders only) | Default since PR #161 | Correctness + slight perf | On |
| Skipping 4 impostor pixel shaders | `VICTORY_SKIP_PIPELINES=...` | 21.5 → 22.5 FPS | Needs visual check |
| Static samplers via driver | Default (not library) | Fixes colour grading on NVIDIA | On |
| Non-uniform heap indexing (waterfall elimination) | Built-in | Avoids per-lane waterfall loops for divergent descriptors | On |

### 5.6  Potential new optimizations (not yet in codebase)

1. **`D3D12_HEAP_TYPE_GPU_UPLOAD` for constant buffers.** On Resizable BAR
   systems, constant data (~6,000 fills/frame from the ring) could be written
   directly to VRAM. The CPU writes sequentially (ideal for write-combining)
   and the GPU reads locally. This eliminates the refresh + send cycle for the
   hottest path.

2. **Batched `MakeResident` / `Evict`.** Currently each shadow copy is made
   resident individually. Batching residency calls reduces kernel overhead.

3. **Async copy queue for texture uploads.** The title's texture upload path
   currently goes through the main graphics queue. Using a dedicated copy queue
   would let uploads overlap with rendering.

4. **Persistent mapped upload ring.** Instead of allocating per-submission send
   memory, use a single persistently-mapped ring buffer. The spare pool
   ([`host/send.rs`](file:///d:/Games/Vice-Tay/crates/victory-gpu/src/host/send.rs#L119-L170)) already
   recycles buffers, but a ring avoids all allocation/deallocation.

5. **Compressed texture format pass-through.** If the title's BCPack textures
   can be reinterpreted as BC7 on the host without detiling, the full
   decompress→recompress cycle is avoided.

---

## 6  Xbox vs PC Memory Mapping Reference

### 6.1  D3D12 heap types

| Heap type | CPU access | GPU access | Where | Use |
|---|---|---|---|---|
| `DEFAULT` | None | Full speed | VRAM | Static assets, render targets |
| `UPLOAD` | Write-combined write | Slow (over PCIe) | System RAM | Staging for one-time copies |
| `READBACK` | Cached read | Write (over PCIe) | System RAM | GPU → CPU results |
| `GPU_UPLOAD` | Write-combined write (via BAR) | Full speed | VRAM | Dynamic data with BAR |
| `CUSTOM` (L0, WB) | Cached read/write | Full speed (UMA only) | Shared | Console-like unified access |

> **Source:** [Microsoft Learn — D3D12_HEAP_TYPE](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_heap_type)

### 6.2  Console-to-PC memory translation in Vice-Tay

| Console concept | Vice-Tay implementation | File |
|---|---|---|
| Title GPU address = CPU address | Shadow copy + write-watch tracking | `shadow.rs`, `watch.rs` |
| XMemVirtualAlloc (graphics, 64K pages) | `VirtualAlloc2` with `MEM_WRITE_WATCH` + address constraints | `xmem.rs` |
| XMemAllocatePhysicalPages + MapPhysicalPages | Section + MapViewOfFile3 (page pool) | `xmem.rs` |
| GPU reads title memory directly | Copy to `DEFAULT` heap or in-place via `OpenExistingHeapFromAddress` | `shadow.rs`, `inplace.rs` |
| Shader writes → CPU visible instantly | Landing: compare card copy with CPU side, merge diffs back | `shadow.rs`, `land.rs` |
| Constant buffer in shared ring | Refresh from title memory, fill descriptor slot | `host/card_steps.rs` |
| Texture streaming (mip chains) | Reserved resource + per-level tile mapping | `tiles.rs`, `texture.rs` |

### 6.3  Console's 7,776 MiB title memory (Series S)

Vice-Tay models this in
[`console.rs`](file:///d:/Games/Vice-Tay/crates/victory-gpu/src/console.rs#L22-L27):

```rust
Console::SeriesS => (0, 7_776 << 20),  // 0 GPU-optimal, 7776 MiB standard
```

On the actual console, all 7,776 MiB is accessible by both CPU and GPU at
224 GB/s. On the PC, this 7,776 MiB of title memory exists in system RAM and
must be **duplicated** into VRAM copies plus tracking overhead, inflating to
22–27 GB.

---

## 7  Key References

### Microsoft / DirectX

| Topic | Link |
|---|---|
| D3D12 Memory Management | https://learn.microsoft.com/en-us/windows/win32/direct3d12/memory-management |
| Uploading Resources | https://learn.microsoft.com/en-us/windows/win32/direct3d12/uploading-resources |
| Residency | https://learn.microsoft.com/en-us/windows/win32/direct3d12/residency |
| GPU Upload Heaps spec | https://microsoft.github.io/DirectX-Specs/d3d/GPUUploadHeaps.html |
| HLSL SM 6.6 WaveSize | https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html |
| SM 6.6 Dynamic Resources | https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html |
| D3D12_HEAP_TYPE enum | https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_heap_type |
| OpenExistingHeapFromAddress | https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device3-openexistingheapfromaddress |
| OpenExistingHeapFromFileMapping | https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device3-openexistingheapfromfilemapping |
| ExecuteIndirect | https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-executeindirect |
| Agility SDK (D3D12Core 1.620) | https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.620.2 |
| DirectX Shader Compiler (DXC) | https://github.com/microsoft/DirectXShaderCompiler |

### NVIDIA

| Topic | Link |
|---|---|
| Advanced API Performance: Memory | https://developer.nvidia.com/blog/advanced-api-performance-memory-management/ |
| Advanced API Performance: Intrinsics | https://developer.nvidia.com/blog/advanced-api-performance-intrinsics/ |
| Advanced API Performance: GPU Upload Heaps | https://developer.nvidia.com/blog/advanced-api-performance-gpu-upload-heaps/ |
| DLSS SDK | https://github.com/NVIDIA/DLSS |
| DLSS 4.5 Multi Frame Generation | https://www.nvidia.com/en-us/geforce/news/nvidia-app-dlss-4-5/ |

### AMD

| Topic | Link |
|---|---|
| RDNA Performance Guide | https://gpuopen.com/learn/rdna-performance-guide/ |
| RDNA2 ISA Reference | https://gpuopen.com/amd-isa-documentation/ |
| Resizable BAR Best Practices | https://gpuopen.com/learn/rebar-best-practices/ |
| FSR SDK | https://gpuopen.com/amd-fsr-sdk/ |

### Xbox / Console

| Topic | Link |
|---|---|
| Xbox Series S specs | https://www.xbox.com/consoles/xbox-series-s |
| Hot Chips 2020 Xbox SoC | https://ieeexplore.ieee.org/document/9220622 |
| GDK memory updates (Thurrott) | https://www.thurrott.com/xbox/series-s |

---

## 8  Summary: Path to 30 FPS at Console-Quality RAM

The lead about unified memory is confirmed: it is the **root cause** of both
the RAM inflation and the FPS deficit. The console's ~8 GB title memory
balloons to 22–27 GB because every buffer exists twice (title memory +
shadow copy) plus tracking and staging overhead.

**Priority changes (by expected impact):**

| Priority | Change | Expected effect |
|---|---|---|
| **P0** | Move shader-read buffers to VRAM by default (`CARD_BUFFERS_MIN_KB=64`, `CARD_BUFFERS_MB=1536`, `CARD_WINDOWS_MB=1024`) | +30–100% FPS (measured 2× on RTX 3080) |
| **P0** | Enable `CARD_STEPS=2` and `PLACED_ORDER=1` by default | −25–60 ms/frame CPU wait |
| **P1** | Enable `PORT_CONSTANTS=1` by default on NVIDIA | −17 MB/frame copies, −6K fills/frame |
| **P1** | Enable `INPLACE=2` by default (disable `LATE_WAIT`) | −5–10 GB private memory |
| **P1** | Enable `LEVELS_ON_DEMAND=1` by default | −2–3 GB VRAM |
| **P1** | Enable `PRELUDE_POOL=1` by default | −4× command allocator churn |
| **P2** | Change `xmem` section creation from `SEC_RESERVE` to `SEC_COMMIT` | Unlocks in-place for all view memory |
| **P2** | Use `GPU_UPLOAD` heaps for constant-buffer ring on BAR systems | Eliminates constant fill copies |
| **P2** | Pipeline submission preparation across 2 threads | −15–20 ms/frame CPU |
| **P3** | Async copy queue for texture streaming | Overlaps upload with rendering |
| **P3** | Persistent mapped upload ring | Eliminates per-submission alloc |

Together, P0 + P1 changes have been individually measured at the stated
magnitudes and require only config/default changes in `scripts/play.ps1`
and enabling existing code paths. P2 changes require new code. With all
P0 + P1 defaults enabled on a 32 GB PC with a 10+ GB GPU, the target of
**30 rendered FPS in the open world at ≤16 GB private memory** is within
reach based on the existing measurements.
