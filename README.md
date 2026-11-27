# Vice-Tay

**Xbox Series compatibility on Windows, tuned on a GeForce RTX 4090.**

Vice-Tay builds on [VIctory](https://github.com/Marvisak/VIctory) to run an Xbox Series title from your own files. The title's x86-64 code runs on the host CPU. The compatibility layer provides the console APIs it calls and translates its graphics work to Direct3D 12.

[Quick start](#getting-started) · [Gameplay](#gameplay) · [Measurements](#measurements) · [Benchmark dashboard](https://taynangt.github.io/Vice-Tay/)

## Measurements

The goal is **30 rendered FPS in the open world with correct visuals**. Both sessions used the Series S profile.

| Session | Result |
| --- | --- |
| **30-minute playtest** · `bc090416` | Median of **21.1 rendered FPS**. The run covered the prologue and boat section, then continued on foot on the island. |
| **6-minute frame generation test** · `88e80129` | Medians of **22.4 rendered FPS** and **44.9 presented FPS** with 2× frame generation on an experimental build. |

The playtest logged a 17.0 FPS 1% low, a measure of its slowest frames, and peaked at 24.1 GB of private game memory. The frame generation sample used DLSS Quality to scale 720p to 1080p and AMD FSR 3.1.6 to generate frames between rendered frames.

[Playtest measurements](docs/measurements.md#series-s-baseline) · [Frame generation settings and sample](docs/measurements.md#series-s-frame-generation)

**Test PC** · RTX 4090 (24 GB), Ryzen 9 9950X3D, 48 GiB DDR5, Windows 11 Pro x64

<details>
<summary>Full test configuration</summary>

| Component | Configuration |
| --- | --- |
| CPU and motherboard | AMD Ryzen 9 9950X3D (16 cores, 32 threads), MSI MAG X870 TOMAHAWK WIFI (MS-7E51), BIOS 1.A93 |
| Graphics | NVIDIA GeForce RTX 4090 (24 GB GDDR6X, driver 621.40), integrated AMD Radeon Graphics |
| Memory | 2 × 24 GiB DDR5 at 8,000 MT/s, 43.4 GiB visible to Windows, 128 GiB page file |
| Storage | Samsung 990 PRO 4 TB and 2 TB, Samsung 970 PRO 1 TB |
| Display | Samsung Odyssey OLED G8 (G81SF), 3840 × 2160 at 240 Hz |
| OS | Windows 11 Pro 26H2 x64, build 26300.10075 |

</details>

## Gameplay

Checkpoint retries worked after the boat ran aground. The HUD, minimap, mission prompts and pause menu were usable along the route.

This clip shows the boat section with the diagnostics overlay. It displays frame times, draw counts, CPU and GPU load, and memory use. Press **F3** to toggle it in a diagnostics build.

https://github.com/user-attachments/assets/bd07891c-c53f-4f46-9b7e-80463c379b84

[Browse ten screenshots from the Series S playtest on build `672cf831`](https://github.com/TaynanGT/Vice-Tay-Screenshots#readme)

## Getting started

**Requirements.** Windows 10 or 11 x64, an AVX2-capable CPU, a Direct3D 12 GPU with Shader Model 6.6 and mesh shader support, Rust, and MSVC build tools with a Windows SDK. Rust uses the version pinned in [`rust-toolchain.toml`](rust-toolchain.toml).

> [!IMPORTANT]
> Bring your own copy of the title. Keep the executable, game data and derived cache outside this checkout. See the [asset policy](docs/asset-policy.md).

1. Clone the repository and copy the configuration template.

   ```powershell
   git clone https://github.com/TaynanGT/Vice-Tay.git
   Set-Location Vice-Tay
   Copy-Item .\victory.example.toml .\victory.toml
   ```

2. Set `game_dir` in `victory.toml` to the directory containing your title files. Build the diagnostics executable and launch the Series S profile with the path to your executable.

   ```powershell
   cargo build --locked --profile diag -p victory --features diagnostics
   .\scripts\play.ps1 -Exe 'D:\victory-data\game\game.exe' -Console series-s
   ```

Use [`Play-VIctory.cmd`](Play-VIctory.cmd) to pass the same options from Command Prompt. Double-click it to enter the executable's path at a prompt. The launcher defaults to full screen, the Series X profile and an automatic GPU preset. **F11** or **Alt+Enter** toggles full screen during play.

| Option | Use it to |
| --- | --- |
| `-Console series-s` | Select the lighter console profile used in the sessions above. |
| `-Machine 4090` / `-Machine 4070` / `-Machine 7900xtx` | Select a GPU preset explicitly. See the [RTX 4070 test notes](docs/rtx4070-performance.md). |
| `-Machine laptop` | Try the [16 GB NVIDIA laptop preset](docs/laptop-16gb.md), measured on one RTX 3050 laptop. |
| `-Safe` | Launch with the conservative settings when investigating a graphics or stability issue. |
| `-Windowed` / `-NoPanel` | Start in a window or hide the diagnostics overlay at launch. |
| `-RayTracing` | Enable the structures needed for the title's ray-traced indirect lighting. |
| `-Fsr4K quality` / `-Dlss4K quality` | Upscale to 3840 × 2160 with PC FSR or DLSS on Series X. |
| `-Build release` | Use the executable built with `cargo build --locked --release -p victory`. The overlay requires a diagnostics build. |

<details>
<summary>PC upscaling and Series S frame generation</summary>

The Series X 4K profiles support `quality`, `balanced` and `performance` on
build 1.0.2.1. They render at 2560 × 1440, 2258 × 1270 and 1920 × 1080
respectively. For Series S, use `-Dlss` or `-Fsr` with the same three
presets to keep the console profile's output size.

For FSR, point `VICTORY_FIDELITYFX_DIR` or `[paths].fidelityfx_dir` at the
folder containing `amd_fidelityfx_upscaler_dx12.dll`. The 4K mode prefers
FSR 4.1.2 and uses 3.1.6 when that is what the adapter supports. Series S
uses FSR 3.1.6. Both profiles can use FSR when DLSS fails.
For DLSS on an RTX card, get `nvngx_dlss.dll` from the NVIDIA DLSS SDK.
Set `VICTORY_DLSS_DIR` or `[paths].dlss_dir` to its folder, or put the DLL
beside `victory.exe`. Some drivers ship without this library. Vice-Tay
checks for it before initializing NGX and uses the configured FSR
fallback when it is missing. `VICTORY_DLSS_NO_UPDATE=1` uses the installed
selection without an NGX update. [AMD FSR SDK](https://gpuopen.com/amd-fsr-sdk/) ·
[NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS)

For 2× Series S frame generation, pass `-FrameGen`. Point
`VICTORY_FFX_FG_DIR` or `[paths].ffx_fg_dir` at the SDK 2.4.0
`Kits/FidelityFX/signedbin` folder with `amd_fidelityfx_loader_dx12.dll`
and `amd_fidelityfx_framegeneration_dx12.dll`. If neither is set, the FSR
upscaler folder is used. This uses AMD's analytical
3.1.6 frame generation provider. The launch below uses DLSS on an RTX card.

```powershell
$env:VICTORY_FFX_FG_DIR = '<SDK signedbin>'
.\scripts\play.ps1 -Exe 'D:\victory-data\game\game.exe' -Console series-s -Dlss quality -FrameGen
```

With neither PC upscaler selected, frame generation uses depth and motion
vectors from the title's native FidelityFX loader. `RENDER FPS` in the
diagnostics panel counts rendered frames, and `FG OUTPUT FPS` counts
swap-chain presentations. The PC FSR bridge follows
[DexNom's implementation](https://github.com/DexNom/VIctory/commit/ab34a8b21477c94da2a6cf07249138e8bc84602e)
through beloooo's port. The frame generation bridge follows
[PastiiJay's DirectX 12 integration](https://github.com/PastiiJay/klbf-lab/commit/793c56c36fda760b027122418b922fc359abf580).

</details>

<details>
<summary>Shader cache and pipeline warm-up</summary>

The launcher pretranslates shaders when the cache has fewer than 1,000 entries. It uses `game_dir` when configured and falls back to the executable's directory. The driver may still build pipelines during the first run.

Run these steps separately from the repository root to warm the shader cache and compile pipelines seen in previous runs. Both steps can resume from cached work.

```powershell
.\target\diag\victory.exe shaders
.\target\diag\victory.exe warm --used
```

Pass an archive path to `shaders` to scan just that archive. `-SkipShaders` skips pretranslation at launch.

</details>

<details>
<summary>Direct3D 12 Agility SDK on Windows 10</summary>

For a Windows runtime that rejects root signature flags `c00` or `c01`, use the application-local [Microsoft.Direct3D.D3D12 1.620.2 package](https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.620.2).

1. Open the `.nupkg` as a ZIP and copy `build/native/bin/x64/D3D12Core.dll` to `target\diag\D3D12\`. The optional debug layer uses `D3D12SDKLayers.dll` from the same package.
2. Build with the Agility SDK exports enabled.

   ```powershell
   $env:VICTORY_AGILITY_SDK = '1'
   try { cargo build --locked --profile diag -p victory --features diagnostics }
   finally { Remove-Item Env:VICTORY_AGILITY_SDK }
   ```

3. Launch with `scripts\play.ps1`. Keep the `D3D12` folder beside the executable. For a release build, use `target\release\D3D12\` and build with `--release`.

</details>

<details>
<summary>Linux and Proton: a faster wineserver</summary>

Under Proton the title keeps 90,000 to 136,000 mapped views alive and maps or unmaps about a thousand a second. Stock `wineserver` keeps a process's views in a list and searches it from the front on every request, so everything that waits on the server slows down. [`linux/wineserver`](linux/wineserver) builds GE-Proton11-14's `wineserver` with a red-black tree of views and puts it in a copy of the GE-Proton folder, `GE-Proton11-14-viewtree`.

```sh
bash linux/wineserver/build.sh
```

Restart Steam and pick `GE-Proton11-14-viewtree`. GE-Proton's `master` already carries an equivalent change ([`em-fixups/0078-server-store-process-views-in-an-rb-tree.patch`](https://github.com/GloriousEggroll/proton-ge-custom/blob/master/patches/wine-hotfixes/em-fixups/0078-server-store-process-views-in-an-rb-tree.patch)); once a GE-Proton release includes it, this build is no longer needed. On one RTX 5080 (Series S profile, the same build and settings), the opening area ran at 2.7 to 4 fps with 190 to 294 ms of copies a frame on stock GE-Proton11-14, and at about 20 fps with 22 to 25 ms on the patched server.

</details>

## Known problems

- **RX 6700 XT on Windows 11 LTSC.** Drivers 32.0.21045.26001 and 32.0.32029.2010 can reset the device while loading the world. The AMD launcher profile defaults to relay mode 0 on those drivers. See [#138](https://github.com/TaynanGT/Vice-Tay/issues/138) for the reported runs. The launcher still accepts an explicit relay mode through its settings override.
- **Frame time.** The 30-minute run recorded a median frame time of 47.3 ms, with 33.7 ms for CPU-side submission preparation and 12.6 ms for GPU read-back. [Frame breakdown](docs/measurements.md#series-s-baseline)
- **Memory pressure.** The five-minute overlay session peaked at 27.1 GB of private game memory. Low free RAM can lead to paging and long stalls. [Memory findings](docs/findings-rtx4090.md)
- **Rendering.** Colour banding, flat-colour surfaces, blocky clouds and reflections, and missing or incorrect shading remain visible in some scenes. Ray-traced indirect lighting requires `-RayTracing`.
- **GPU timing.** Series S game logs report GPU time near zero. The diagnostics overlay reads GPU load from Windows when those counters are available. [Overlay details](docs/this-fork.md#playing)

For banding that varies between launches, add `-More @{VICTORY_RTV_FORMAT_LOG='1'}` to the diagnostics launch. Compare the `RTV format check` totals and `RTV format mismatch` examples at the same checkpoint in both runs.

[Report a performance or rendering issue](https://github.com/TaynanGT/Vice-Tay/issues/new?template=performance-or-graphics.yml)

## Development and testing

- **Loader and runtime.** `victory-loader` maps the title executable. `victory-hle` supplies the console services it expects. [Architecture](docs/architecture.md)
- **Graphics.** `victory-gpu` maps console command lists and resource views to host D3D12. `victory-shader` recompiles shaders, including Xbox wave64 programs on NVIDIA hardware. Vice-Tay reclaims temporary resource views when the GPU finishes using them, reducing dropped draws when the descriptor heap fills. [Changes in Vice-Tay](docs/this-fork.md)
- **Verification.** CI runs Clippy, the workspace tests in default and diagnostics configurations, and the asset policy check. Performance changes are compared on the same route, build base and session length. [Test procedure and run logs](docs/tests.md) · [CI workflow](.github/workflows/ci.yml)

See the [upstream usage notes](docs/README-fxceless.md) for additional commands and [frame replay](docs/replay.md) for capturing and replaying a frame locally.

## Credits

Vice-Tay grew out of codec-d's `cool_fox`, which brought NVIDIA work to avatarmanz's VIctory tree. Imported work retains its original authorship and commit references. The [full credits](docs/credits.md) list the contributors in the project's history.

- [Marvisak/VIctory](https://github.com/Marvisak/VIctory) and its contributors built the loader, console runtime, Direct3D 12 backend and shader recompiler.
- [fxceless/victory-fork](https://github.com/fxceless/victory-fork) and its contributors provided the fork history Vice-Tay builds on.
- [avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX) contributed parallel command recording and translation, accelerated submissions, texture uploads on a copy queue and frame replay.
- **codec-d/cool_fox** contributed two-plane composition, fallback heaps, descriptor table plans, residency and fence handling, NVIDIA wave32 lowering, movie uploads and XMA audio fixes.
- [DexNom/VIctory](https://github.com/DexNom/VIctory) contributed ray tracing fixes, cube gradients and sparse textures.
- [Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD) contributed the overlay, ambient lighting work and pipeline workers, with work by romainhedouin.
- [FQY55kai/ZJOU-VICTORY](https://github.com/FQY55kai/ZJOU-VICTORY) and [myokex/victory-amd](https://github.com/myokex/victory-amd) are part of the project's lineage. The latter includes page-view and title-memory work by Spexivus, joroyt, tightfall and Joshua Odom.
- **Geckomacabre** and **cxy0072019-sketch** contributed texture read-back. **meshahid973**, **nullexxx**, **cxy0072019-sketch** and **welisonzika14-jpg** contributed fixes and reviewed changes through Vice-Tay pull requests.
- [ivall/XCL](https://github.com/ivall/xcl) (GPL-2.0-only) served as a reference for title behaviour.

## License

[GPL-3.0-or-later](LICENSE), inherited from VIctory. The [asset policy](docs/asset-policy.md#the-one-exception) records the separate BCPack format tables. Vice-Tay is independent of the title's developer and publisher.
