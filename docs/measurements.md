# Measurements

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
