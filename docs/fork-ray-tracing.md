# Ray tracing in this fork: what the title does and what has been built

Status, 2026-11-25, midday: **with `VICTORY_BVH_BUILDER=1` rooms get their
indirect light** (runs light-10, light-11, n19), for some 40 ms more of the
card's time a frame (the dorm: 75 of 90 ms busy against 34 of 64). Off by
default. **`VICTORY_AMBIENT=1`** puts an estimate of that light in its place
for next to nothing (below).

## What the title does

The title's indirect light (probes, bounce, reflections) and the
shadows of its lamps and sun are ray traced. Without them, anything the sun
does not hit directly is black: interiors, the shadowed side of a face, rooms
lit only by lamps.

The title does not use a ray-tracing API:

- Traversal is ordinary compute-shader code: a per-thread stack in shared
  memory, and loads through raw 40-bit addresses of title memory.
- The static world's acceleration structures ship pre-built in the title's
  data, in the console GPU's native node format (64-byte nodes: box nodes with
  four children and 16-bit float bounds, 128-byte box nodes with 32-bit float
  bounds, triangle nodes, instance nodes). They are unpacked on the GPU.
- A set of builder programs (from the title's `XboxBvhBuilder.dll`, 52 compute
  programs) builds and refits the dynamic structures on the GPU every frame.
- Only the per-node test is hardware: the instructions
  `image_bvh64_intersect_ray` and `image_bvh_intersect_ray`. One call tests a
  ray against a box node's four children (returning the hit children, nearest
  first) or against a triangle node (returning the intersection's numerator,
  denominator and barycentric numerators).

So mapping this onto the host's own ray tracing (DXR) is not practical: the
layer would have to know the geometry behind structures the title never
describes to it, and the traversal loops live in 64 of the title's own
programs. This fork runs the title's builder and traversal as translated
shaders and emulates the one instruction.

## What this fork adds

1. **The builder programs translate** (all 52, with host pipelines on the RX
   7900 XTX; the title creates 53 such pipelines). They pass buffer addresses in a spilled parameter table
   rather than in descriptors. Added: the spilled user-data table, root
   unordered-access views (translator and host binding), byte and 16-bit buffer
   accesses, and several control-flow shapes.
2. **Title memory by raw address from shaders.** A directory with one entry per
   64 KiB of title address space leads a shader to the layer's host copy of
   that memory; loads, stores, atomic add/sub and float min/max go through it.
   Before a dispatch the host makes ready the memory its root views and root
   constants name.
3. **The ray test, emulated in DXIL.** Four slab tests and a sorting network
   for box nodes; a watertight ray/triangle test for triangle nodes; both forms
   of the instruction. A plain Rust reference lives in
   `crates/victory-shader/src/bvh.rs`; tests run the emitted DXIL on the GPU
   over 311 ray/node cases and compare with the reference bit for bit.
4. **The trace and shading programs translate** (59 of 64 trace programs, and
   the irradiance-cache shading program). This needed lane masks treated as
   plain numbers on some paths, and exact forms for programs written for
   64-lane waves running on a GPU with 32-lane waves.
5. **A safety rule on the host side.** The builder's stages run only while
   every builder pipeline the title has created has a host pipeline; otherwise
   none run (`VICTORY_BVH_BUILDER=1` asks for it; `VICTORY_BVH_QUARANTINE=1`
   also keeps the builder's stores out of title memory, for a first test). A
   partly working builder wrote garbage bounds into title memory and crashed
   the title.

## The structures (from dumps of the arenas, `VICTORY_MEMORY_DUMP`)

- Header, 64 bytes: dword 0 the version 6 in the low half and marks in the high
  (0x8000 bottom level, 0x4 compacted, 0x2 rehydrated, 0x100 still packed),
  dword 1 the size in 64-byte nodes, 3 the number of elements, 4-9 the bounds,
  10 the offset of the box nodes, 12 the root's node pointer, 13 the offset of
  the elements, 14 the offset of the parent table (from the header; the static
  ones' index counts the leaves too, so it is negative).
- An instance is 128 bytes at the top level's offset 13: a 3x4 matrix in dwords
  0-11, id and mask, flags, the root pointer (bit 30 set for static ones), the
  bottom level's address in 256-byte units.
- The static world's structures are shipped **packed ("dehydrated")** and
  unpacked in place, once, by DecompressHeader, Box32/Box16Rehydration and
  LeafRehydration when they have been streamed in (3,000 in the opening).
- Dynamic ones are built, their compacted size read back by the title from a
  small buffer, and copied to a place of that size.

## How title memory by address is held (`memory.rs`)

Pages of 64 KiB in arenas on the card, whole title buffers in their own copies,
pages with small title buffers as windows of title memory. The module's
comment has the rules. In short: what the builder stores stays on the card
(only small title buffers it was given to write come back); a packed structure
goes up whole when the builder comes to it; title buffers take its stores only
after it was given them as an unordered access view.

| variable | does |
|---|---|
| `VICTORY_BVH_BUILDER=1` | the builder's stages run |
| `VICTORY_BVH_FENCES=0` | no fences around the builder's buffer atomics (beloooo's 72e748ea has them too) |
| `VICTORY_BVH_SIZES_WAIT=1` | submissions that ask for structures' sizes are waited for (off: no help, slow) |
| `VICTORY_BVH_BOUNDS=0` | the builder's accesses are not bounded |
| `VICTORY_BVH_TRACE=1` | logs what sizes are asked, when they came, and the copies (diagnostics) |
| `VICTORY_BVH_QUARANTINE=1` | nothing it stores comes back to title memory |
| `VICTORY_BVH_FAULTS_ALL=<n>` | pages taken from faults in all (2048). Do not raise it: run rtf-2 (no bound) gave every committed page a copy, still had 110-730 faulted pages a frame of memory that is not the title's, and filled the 3 GiB of pages in 4 minutes (reports/RT-FLICKER.md) |
| `VICTORY_MEMORY_CARD_MB` | the most of it on the card (4096) |
| `VICTORY_MEMORY_DUMP=<folder>` | a file `go` there writes the arenas out (diagnostics) |
| `VICTORY_UPLOAD_LOW=0`, `VICTORY_LENT_HEAP=0` | lists' upload memory above 2^40, no view heap for lists without one, as before |

Every ten seconds the log has a line "title memory by address: ..." with pages,
dispatches, faults and what came.

## Known gaps

- Dynamic structures overlap each other (8 of 65 good in run light-10, 48 of
  275 in light-15). Part of the cause, run light-11's trace (`VICTORY_BVH_TRACE=1`): the title asks for a new
  structure's compacted size, reads it one or two frames later without looking
  whether it is there, takes memory of that size and has the structure copied
  into it. Our submissions land up to two frames late; the title read what the
  slot held before, another structure's size, and the copy wrote over the
  neighbours. Waiting for a submission that asks (`VICTORY_BVH_SIZES_WAIT=1`)
  did not mend it and costs some 20 ms a frame (light-15): the title asks in
  every frame, and what it read was as old as before. Open.
- What a structure that was written over says of its size sent the builder's
  copy of it through gigabytes of other structures; bounded now
  (`VICTORY_BVH_BOUNDS`). Pages touched that are not there: 2,500 a frame
  before, 90 after (the dorm).
- The card's time in the dorm with the builder (timing run light-13, title's
  commands 84 ms): `CopyForBVHUpdate` 13.3 ms in 63 dispatches (before the
  bound), traces 24 ms (`RaytraceVBuffer` 7.9, screen probes 6.4, irradiance
  cache 3.5, primary probes 2.3, shadows 1.6).
- Captures made on 2026-11-25 (builder on or off) hang the card in a replay
  (cause unknown; do not try).
- Five shadow-mask trace programs do not translate (a structure's address out
  of a descriptor picked from the heap by a computed index: the host would have
  to give a table of addresses by slot). The title did not create them in any
  run so far.
- An access to a page that is not there is lost (a read gives zero, a store is
  dropped).
- Not publicly documented, and assumed: the instruction's rounding, the order
  of two boxes entered at the same distance, and the result for a ray exactly
  through an edge.

## Without rays: the stand-in (`crates/victory-gpu/src/ambient.rs`)

Without structures every ray misses. The title's program that resolves the
diffuse light of its screen probes (`CS_ScreenProbes_TemporalResolve_*`, list
184 of a frame) then leaves the sky's light outdoors, nothing indoors, and in
blocks of tiles what the probes' memory held. `VICTORY_AMBIENT=<share>` raises
it to an estimate right after that program: the title keeps, for its own screen
traces, the light visible surfaces send out (`CS_CacheScreenSpaceRadiance`,
1280x720, the frame before); its mean in 64x36 cells is gathered at 160x90
points, each cell weighted by the share of the surroundings it covers, as a
directional value; each pixel takes it between the points by depth, with its
normal and a shade in creases. So level and tint follow the hour, the room and
its lamps. What the program left counts up to four times the estimate.

Where it can be put, and the crease shade, are beloooo's (VIctory-Belobug-AMD
06031237, 4f45ab88, cd9f78ce); `VICTORY_AMBIENT=constant:<level>` is that
fork's constant.

| variable | does |
|---|---|
| `VICTORY_AMBIENT=<share>` | the estimate, 1 all of it (unset: the title's own) |
| `VICTORY_AMBIENT=constant:<level>` | the same light everywhere |
| `VICTORY_AMBIENT_SHADE=<strength>,<metres>` | the creases (1,0.6) |
| `VICTORY_AMBIENT_AIR=<multiple>` | what the air scatters is bounded by a multiple of the light around (off: the brown veil in the day room stays with it, light-14) |

It is off while the builder runs.

### Limit reflection brightness with the builder off

`VICTORY_AMBIENT_REFLECT` caps reflection brightness at a multiple of the
ambient estimate and preserves hue. For a comparison with ray tracing off,
keep the usual launcher settings and add `-More $reflectionHold`.

```powershell
$reflectionHold = @{
    VICTORY_AMBIENT = '0.8'
    VICTORY_AMBIENT_REFLECT = '1'
    VICTORY_AMBIENT_REFLECT_IN = 'CS_TemporalResolve2_Opaque,CS_TemporalResolve1_Alpha'
}
```

The two names are case-sensitive shader name prefixes for the opaque and
alpha resolves. The `held=1` startup message confirms the option was parsed.
Look for `the reflection hold is enabled` and a nonzero `direct` or
`indirect_gated` count in the resolve summary to confirm a hold was recorded.
The counters also report direct dispatches with zero groups, counted indirect
dispatches and indirect dispatches with multiple records.
[Original reflection hold](https://github.com/avatarmanz/VIctory-7900XTX/pull/314)

## Where to look

- `crates/victory-shader/src/bvh.rs`: the reference for the node test.
- `crates/victory-shader/src/emit/lower.rs`: the lowering of the two
  instructions and of memory by address (`Lower::memory`).
- `crates/victory-shader/src/resource.rs`: address tracking for descriptors
  and raw addresses.
- `crates/victory-gpu/src/pipeline.rs`, `draw.rs`: the builder rule and the
  host side of root views and title memory.
