# Memory retention follow-up, 2026-11-27

This pass starts from the user's reverted fork, `c3fc5da`, and merges upstream
`99a7f9b` (mapping-fence completion must be confirmed before publishing memory).
It does not reapply the old memory-pressure commit wholesale. In particular,
upstream already accounts for pinned windows in its optional window budget and
preserves normal idle expiry. The new copy-page tracking, texture readback
lifetimes and shader fixes are retained.

## Observed pressure

The user's Series X run on the RTX 4070 / 7800X3D / 32 GB PC reported:

- 27,458 to 29,053 MiB private memory in the supplied log excerpt.
- Roughly 3.8–4.1 GiB of window copies, almost all referenced by descriptors or
  lists. They cannot safely be freed just because RAM is low.
- Zero retained snapshot buffers; texture staging around 50–80 MiB; Rust-owned
  allocations around 1.6–1.7 GiB. These are not a reason to cut every cache.
- The external 32-second sample had 556–942 MiB available RAM and roughly
  7.1–7.4 GiB of GPU-shared memory. System-wide disk page-ins occurred during
  five of seven samples; these counters do not attribute every fault to the game.

The executable for that run had SHA-256
`60714481e8a921a51df1502ba43387051b554c3cbf010f39edddf513c2997979`
and a November 26 write timestamp. Its source revision is not established.
Those observations identify pressure, not a matched performance baseline for
today's source. Virtual-address region totals include aliases and GPU mappings;
adding them together would overstate physical memory consumption.

## Changes

Completed host command lists retain their existing 512-entry / 256 MiB scratch
reuse limits. Cold entries can now expire after 30 seconds, shortened to two
seconds under the existing RAM-pressure signal. The pool keeps 32 warm entries
and releases at most 32 per sweep. LIFO order is preserved so eviction does not
replace the most recently used allocator with a cold one.

The translation-front pool receives the same aging and a 512-entry cap. It
contains only fronts with no remaining recording owner. Submitted native lists
have already been detached and retain independent ownership. Existing fence
completion, retirement lag and resource references are unchanged. Rejected and
expired objects are destroyed outside the pool locks.

`VICTORY_LIST_TRIM=0` disables aging and the new front cap for an A/B comparison.
The `memory, idle lists:` diagnostic reports retained entries, scratch bytes,
reuse hits, misses and releases. Driver allocator storage is excluded from the
byte count because D3D12 does not expose it directly. The hit/miss counters are
necessary to detect allocation churn instead of assuming lower retention is
always faster.

Speculative staging and optional CPU-backed buffers must leave at least 1 GiB
of physical RAM after the requested allocation, in addition to the existing
commit reserve. A new staging block counts as 64 MiB. Required uploads,
reusing committed staging slices and decoration bakes keep their existing
rules. This is sampled admission control, not a hard concurrent allocator cap.

## Architecture and research

Xbox Series X has 16 GB GDDR6 shared by CPU and GPU, with different bandwidth
regions. The PC's GPU-shared system memory does not have that bandwidth. See
[Xbox Velocity Architecture](https://news.xbox.com/en-us/2020/07/14/a-closer-look-at-xbox-velocity-architecture/)
and [Microsoft's UMA/discrete architecture flags](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_architecture).

Microsoft's D3D12 guidance is to classify, budget and stream allocations;
residency budgets cover both local and nonlocal memory. Exceeding budgets can
cause paging and stalls. The compatibility layer must preserve memory visibility
and GPU completion when reclaiming resources, not simply release every mirror.
See [memory-management strategies](https://learn.microsoft.com/en-us/windows/win32/direct3d12/memory-management-strategies)
and [residency](https://learn.microsoft.com/en-us/windows/win32/direct3d12/residency).

Command allocators grow but do not shrink on reset. Reusing a warm working set
is efficient; retaining a loading peak forever is not required. This motivates
aging completed pools, rather than disabling pooling or shortening fences.
See [recording command lists](https://learn.microsoft.com/en-us/windows/win32/direct3d12/recording-command-lists-and-bundles).

GPU upload heaps help CPU-write/GPU-read traffic, but CPU reads from their
write-combined memory are expensive. The layer also compares and merges shader
writes, so converting every mirror to GPU upload memory is not appropriate.
See [NVIDIA's GPU-upload guidance](https://developer.nvidia.com/blog/optimizing-dx12-resource-uploads-to-the-gpu-using-gpu-upload-heaps/).

Wave32 versus wave64 affects occupancy and cross-lane semantics, not just a
compiler preference. This tree already restricts unsafe cross-wave fallback
forms after recorded device losses. This patch preserves that logic; overall
CPU/GPU utilization does not justify removing wave barriers or narrowing lanes.
See [AMD's occupancy explanation](https://gpuopen.com/learn/occupancy-explained/).

## Validation

Four focused pool tests cover bounded eviction, ordering, ownership, byte/count
limits and a synthetic 48 MiB touched-allocation burst. Eviction releases 15 MiB
of that burst while retaining 32 MiB in the warm pool and one checked-out MiB.
Six residency tests cover physical/commit admission, required-upload behavior
and nested upload deferrals. All ten passed.

Gameplay performance and visual quality require matched runs. Keep console,
scene, camera, output, upscaling, frame generation and cache state fixed. Compare
rendered frame times, private bytes, GPU-shared usage, disk page-ins and pool
misses. Record and review the full video before making a visual-quality claim.
