# VIctory-NVIDIA changes in Vice-Tay

The comparison used [VIctory-NVIDIA at `9d6300f`](https://github.com/trustno1-coder/VIctory-NVIDIA/tree/9d6300fa2dd29cd85ef22b3d6198572a9b5ffe25),
including its earlier renderer history. Both projects use GPL-3.0-or-later.

| Change | Source |
| --- | --- |
| Execute the buffer comparison after the title lists so the relevant D3D12 buffers can decay to `COMMON`. The same submission fence covers both calls and the timing ranges include the comparison. | Jeremy Goldberg, [#1](https://github.com/trustno1-coder/VIctory-NVIDIA/pull/1), [#3](https://github.com/trustno1-coder/VIctory-NVIDIA/pull/3) |
| Give the ambient-light resolve its own root signature and require a matching 2D depth SRV, including placed textures and depth planes. | Jeremy Goldberg, [#2](https://github.com/trustno1-coder/VIctory-NVIDIA/pull/2), readmet3xt-lab, [`bda6e59`](https://github.com/trustno1-coder/VIctory-NVIDIA/commit/bda6e597dabd903dd2589855e6c09563b3a3d231) |
| Create NVIDIA pipelines with static samplers through the driver. The fork reproduced incorrect colour grading after loading those pipelines from its D3D12 pipeline library on an RTX 4070 SUPER. | readmet3xt-lab, [`7fa0e95`](https://github.com/trustno1-coder/VIctory-NVIDIA/commit/7fa0e958e8947f5351e28b437214af675fb9cd03) |
| Keep typed buffer writes intact when translation cannot determine the view kind. | Tissma95, [#15](https://github.com/trustno1-coder/VIctory-NVIDIA/pull/15) |
| Allow early depth testing for compatible pixel shaders. | readmet3xt-lab, [`41b4a05`](https://github.com/trustno1-coder/VIctory-NVIDIA/commit/41b4a056d6e176656a062355d7198058d0065ca6) |
| Count occupied and free descriptor slots by kind, using the existing table lock and updating the counts on allocation and reuse. | LucaruluDev, [#5](https://github.com/trustno1-coder/VIctory-NVIDIA/pull/5) |
| Measure texture and buffer readbacks separately in diagnostics. The title's pending reads still wait for their writes in landing order. | Indy2l, [#7](https://github.com/trustno1-coder/VIctory-NVIDIA/pull/7) |
| Wake only the helpers that can share a batch of memory copy or comparison jobs. | readmet3xt-lab, [`8c1fa5d`](https://github.com/trustno1-coder/VIctory-NVIDIA/commit/8c1fa5d578b859f8399da3963a2de64b980f8f98) |
| Expose an optional GameInput mouse using relative motion and wheel input from the focused window. | superboynick, [#9](https://github.com/trustno1-coder/VIctory-NVIDIA/pull/9), [#17](https://github.com/trustno1-coder/VIctory-NVIDIA/pull/17) |
| Build with mimalloc as an alternative Rust allocator, with the same allocation counters in diagnostics. | Dastano, [#11](https://github.com/trustno1-coder/VIctory-NVIDIA/pull/11) |
| Read fault captures and show guard-lock waits, hot blocks and thread costs. | readmet3xt-lab, [`d900cb8`](https://github.com/trustno1-coder/VIctory-NVIDIA/commit/d900cb8bf5245960827bc7031d3d53db550e1cd9) |

Pass `-More @{VICTORY_EARLY_DEPTH='1'}` to `scripts\play.ps1` to try early
depth testing. This creates a separate shader cache key. The translator
version also changes for the typed buffer fix, so allow time for shader
translation after updating.

Pass `-More @{VICTORY_GAMEPAD_MOUSE='1'}` for the mouse device. It uses
Windows Raw Input in the foreground and leaves the pointer, fullscreen and
controller settings to the existing flows. An existing process-wide mouse
registration takes priority.

For a matched colour-grading comparison on NVIDIA, use the same build, scene,
render settings and warmed driver cache in two runs. Set
`-More @{VICTORY_LIBRARY_STATIC_SAMPLERS='1'}` in the control run to use the
pipeline library for those shaders. Compare the same frame after restarting
each process. The default may take longer to start while pipelines rebuild.

Build the optional allocator with
`cargo build --locked --profile diag -p victory --features diagnostics,mimalloc`.
For an allocator comparison, keep the build profile, scene, cache and options
matched and record rendered frame times and peak private bytes for each run.

Capture memory faults with
`-More @{VICTORY_FAULT_BOOK='C:\traces\faults'; VICTORY_FAULT_BOOK_AT='150'}`
in a diagnostics build. Once the log reports that the capture was written,
run `scripts\faultbook.ps1 C:\traces\faults.0.bin`. The frequency defaults to
10 MHz and can be supplied in ticks per second with `-Frequency`.

Vice-Tay already has a 16 GB NVIDIA laptop preset, an opt-in Agility SDK,
`VICTORY_D3D12_ADAPTER` for hardware selection, the frame-interval override,
and diagnostics-only placed-texture filler tracking. The fork's F1 menu and
live DLSS multipliers belong to its foxhound and Streamline stack. NVIDIA's
6× Multi Frame Generation targets GeForce RTX 50 Series hardware.
[NVIDIA DLSS 4.5](https://www.nvidia.com/en-us/geforce/news/gfecnt/20264/nvidia-app-dlss-4-5-dynamic-multi-frame-generation-available-now/)
