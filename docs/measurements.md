# Measurements

To compare two runs at the same camera position, select the time spent in
that scene from each diagnostics log. Supply times with an explicit offset.
This PowerShell command selects one minute with more than 1,000 draws per frame.

```powershell
.\scripts\measure-frames.ps1 .\run.log -From '2026-11-27T12:00:00Z' -Until '2026-11-27T12:01:00Z' -MinDraws 1001
```

The start is included and the end is excluded. All three filters are optional.
The script counts periodic rendered-frame reports and gives each one equal
weight. P10 is the nearest-rank 10th percentile of their rendered FPS. A draw
threshold can exclude menu samples. Keep the same scene and check the sample
count in both runs. The script reads console reports copied with line breaks
between fields, including in the header.

Pass `-TextureProgress` to report the indexed-texture counters from each
matching frame report. Compare the UTC time, draws per frame, uploaded
textures and MiB, waiting textures and slots passed over as unmapped.
Use the same time and draw filters to compare a selected load interval.

In the diagnostics memory report, the private-byte delta near the host-list
counts sums process-wide changes sampled during list creation. Concurrent
calls can overlap, and other threads can allocate or free memory in the same
intervals. Measure allocator cost in a separate run.

## Series S 4K boat test

On November 27, 2026, I repeated a 40-second boat sequence from the same
checkpoint on the RTX 4090 (driver 621.40) and Ryzen 9 9950X3D. Each run
used the Series S profile with DLSS Quality (2560 × 1440 → 3840 × 2160),
AMD FSR 3.1.6 frame generation at 2×, the frame rate unlocked and the
performance overlay visible. Both runs started with the same shader and
pipeline cache seed and loaded the same local BCPack format tables. After
the objective appeared, the sequence turned the camera in both directions
and accelerated for four seconds.

| Local build | Intervals | Sample time | Rendered frames | Generated frames | Presented frames | Mean time per rendered frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K reference · `e9a30f4f` | 38 | 39.26 s | 633 | 633 | 1,266 | 62.02 ms |
| 4K experimental · `424fb3f6` | 38 | 38.73 s | 1,035 | 1,035 | 2,070 | 37.43 ms |

| Local build | Rendered FPS | Generated FPS | Presented FPS | Rendered P10 | Rendered median | Rendered P90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K reference | 16.12 | 16.12 | 32.25 | 13.98 | 16.68 | 17.17 |
| 4K experimental | 26.72 | 26.72 | 53.44 | 23.48 | 27.80 | 28.50 |

Average FPS divides the frame counts by the total time of the intervals
that fall entirely within the measured sequence. The percentiles and
median use the individual intervals, each about one second long.

The reference adds local 4K support to `01482583`. The experimental build
adds texture and descriptor changes, starts placing buffers on the card
at 64 KiB, uses GPU-placed constants, and sets a 4 GiB Resizable BAR
window pool. The result measures the combined configuration on this route.
The proposed launcher and the texture changes are tracked separately in
[PR #329](https://github.com/TaynanGT/Vice-Tay/pull/329) and
[PR #370](https://github.com/TaynanGT/Vice-Tay/pull/370).

## Series S 1080p GPU read-through

On November 27, 2026, I switched GPU read-through for compatible fills
on and off every five seconds in one diagnostic build, with the boat
stationary and the camera fixed on the bridge. The RTX 4090 used DLSS
Quality (1280 × 720 → 1920 × 1080) and AMD FSR 3.1.6 frame generation
at 2×. The 115-second test yielded 11 complete phases and 33 telemetry
intervals per setting after dropping the first second of each phase.

| GPU read-through | Rendered FPS | Generated FPS | Presented FPS | Mean time per rendered frame |
| --- | ---: | ---: | ---: | ---: |
| Off | 22.60 | 22.60 | 45.13 | 44.25 ms |
| On | 26.38 | 26.38 | 52.78 | 37.91 ms |

All ten consecutive off-to-on pairs improved. The median reduction in
frame time across those pairs was 6.45 ms. The change from Maxime Cambou
is enabled in compatible NVIDIA profiles as of `01482583`. [PR #374](https://github.com/TaynanGT/Vice-Tay/pull/374)

## Series S frame generation

On November 27, 2026, I measured a boat traversal on the RTX 4090 and Ryzen 9
9950X3D. The Series S diagnostics build was `88e80129`, based on `d945e99c`
with the current test patches. DLSS Quality scaled 1280×720 to 1920×1080, and
AMD FSR 3.1.6 frame generation was set to 2×. The frame rate was unlocked,
with the performance overlay on and capture running at up to 60 FPS.

The sample contains 356 intervals over 364 seconds of piloting. It excludes
the checkpoint transition.

| FPS measure | Median | 10th percentile |
|---|---:|---:|
| Rendered | 22.4 | 20.0 |
| Presented, including generated frames | 44.9 | 40.3 |

Frame generation added a median of 22.4 FPS. The swap chain presented 2.00×
as many frames as the renderer produced across the sample. The output
reached 30 FPS in 355 of the 356 intervals.

## Series S baseline

Build bc090416, 2026-11-26 (`cargo build --profile diag -p victory --features diagnostics`).
On the test PC the README describes, with `scripts\play.ps1 -Console series-s`
and the RTX 4090 settings, in full screen, played by hand: the prologue from the
prison into the boat section, the boat driven to the package and along the
marked route to an island, three failed attempts picked up again from a
checkpoint (one of them skipped), and a capture every minute.

Frame columns are medians over the world lines (once a second, over 1,000
draws a frame). Memory and card columns are samples every 20 s over the whole
session, loading included. The GPU ms column reads near zero on this profile
and is left out of conclusions (README, Known problems).

| Profile | world lines | fps median | p10 | p90 | 1% low | ms/frame | preparing | copies ms / MB | textures ms / MB | GPU ms | copying back ms / MB | draws | dropped draws | captures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Series S | 349 | 21.1 | 19.1 | 23.2 | 17.0 | 47.3 | 33.7 | 13.3 / 86.8 | 0.6 / 31.0 | 0.3 | 12.6 / 29.9 | 4446.0 | 98.0 | 183 |

| Profile | samples | game memory median / max (GB) | working set max (GB) | card memory max (GB) | card load median (%) | power median (W) | temperature max (C) | clock median (MHz) | CPU seconds per minute |
|---|---|---|---|---|---|---|---|---|---|
| Series S | 94 | 23.5 / 24.1 | 27.4 | 8.8 | 86.0 | 134.5 | 49.6 | 2760.0 | 624 |

CPU seconds per minute: the game's CPU time over all its threads; 60 is one core busy all the time.

## RTX 5070 session observation

t90 recorded a session lasting 1 hour, 47 minutes and 54 seconds from launch
through manual close on an RTX 5070 (12 GB). The diagnostics build combined
`2dd3479`, `755dc92` and `e11801e` locally and used a 1,152 MiB card-buffer cap.

The log contains 6,322 renderer submission reports, with a maximum gap of
3.457 seconds between reports. The elapsed time includes startup and loading
and may include menu or idle time. It describes this session's continuity.
[Contributor report](https://github.com/TaynanGT/Vice-Tay/pull/240)
