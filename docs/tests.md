# Tests

How changes to this tree are tested on the test PC the README describes, and
what each recent session played and measured. Newest first.

## How a session is run

- **Build.** A diagnostics build (`cargo build --profile diag -p victory
  --features diagnostics`), its shaders translated ahead with `victory shaders`.
- **Launch.** `scripts\play.ps1 -Console series-s` with the RTX 4090 settings,
  in full screen, the performance panel on. Nothing else compiles meanwhile; one
  game runs at a time.
- **Play.** Either by hand, through scripted pad input sent while watching a
  capture after every move, or by a scripted route (walk, run, turn the camera,
  step aside). Take a capture every minute and record the full verification
  from the game window on the card's encoder.
- **Measure.** fps and the frame's parts from the game's own once-a-second frame
  lines in the world (lines with over 1,000 draws a frame); the game's memory,
  the card's memory, load, power and temperature sampled every 20 s.
- **Compare.** A change that could cost frames is run against a control built
  from the same base, played the same way for the same length. Match the scene,
  camera, resolution, upscaler, frame generation and recording settings.
- **Review.** Watch the full video before reporting results. Check colour,
  lighting, textures, reflections, HUD legibility, flicker and ghosting.
  Record the timestamps of visual changes and artifacts beside the frame-rate
  results. Keep recordings local under the [asset policy](asset-policy.md).

## Sessions

### Performance panel, 2026-11-26

Build `5963f415`, the branch of
[#57](https://github.com/TaynanGT/Vice-Tay/pull/57) (main at `22c6eac8` plus the
panel). Scripted route, 5 minutes in the world.

| Start screen / world | fps median (p10-p90; 1% low) | ms per frame: preparing, copying back | Game memory median / max | Card memory max, load | Draws / dropped |
|---|---|---|---|---|---|
| 88 s / 126 s | 20.8 (16.8-23.9; 1.4) | 48.2: 32.7, 12.6 | 26.2 / 27.1 GB | 11.9 GB, 79% | 3,532 / 82 |

The panel showed 22 fps at 1920×1080 with 2,889 draws, a 45.9 ms frame, the
GPU at 87% as Windows reads it, 9.5 of 21.3 GB of VRAM, the CPU at 44% with its
per-core bars, and the game at 26.3 GB, all matching the log. Frame rate within
the boat session's range: the panel costs nothing measurable. The 1% low comes
from a few slow frames while the world streamed in.

### The boat section, 2026-11-26

Build `bc090416`. Played by hand for 30 minutes in the world.

| Minute | What happened |
|---|---|
| 0 | The world, in the prison (start screen after 72 s, world after 120 s) |
| 0-4 | Cutscenes from the prison to the boat |
| 4 | The boat under control; open water, a road bridge ahead |
| 7 | "Use Focus to find the package", then "Follow the plane" and the package |
| 9 | Ran aground on a mangrove island: mission failed, picked up again at the checkpoint |
| 12 | Ran aground again: "the boat is no longer usable"; the checkpoint was skipped and the map drew a route |
| 13 | Failed once more; the checkpoint repeated |
| 14-24 | The route: under an iron bridge, along a mangrove channel, past a broken pier, across a bay; a few groundings backed out of without failing; the pause menu once |
| 25 | The island with the crashed plane; off the boat, the inventory tutorial |
| 25-30 | On foot on the beach |

| fps median (p10-p90; 1% low) | ms per frame: preparing, copying back | Game memory median / max | Card memory max, load, power, temperature | Draws / dropped | Captures |
|---|---|---|---|---|---|
| 21.1 (19.1-23.2; 17.0) | 47.3: 33.7, 12.6 | 23.5 / 24.1 GB | 8.8 GB, 86%, 134.5 W, 49.6 C | 4,446 / 98 | 183 |

By thirds of the session the median fps was 21.7, 21.4 and 20.6, and the game's
memory 22.9, 23.6 and 23.9 GB. Open water reached 30 fps for a moment; one
sharp turn that streamed a new area dipped to 11 fps. Clouds showed the square
blocks listed under the README's known problems.

## Comparisons under way

On the base of the panel session (`5963f415`), 10 minutes on the scripted route
each, against a control of the base itself:

- Belobug's texture uploads at the start of each submission
  ([`5d6bd841`](https://github.com/Belobug/VIctory-Belobug-AMD/commit/5d6bd841)).
- The C01 translator batch (translator version 3).

PC upscalers on the Series X profile, compared at the moment the boat is first
under control: the title's own upscale, AMD FSR 4.1.2 (`-Fsr4K quality`) and
NVIDIA DLSS 4.5 (`-Dlss4K quality`).

## CI and test reliability

`ci-local.ps1` runs what GitHub runs: clippy with and without the diagnostics
feature (warnings are errors), every test in both configurations, and
`cargo xtask` for the repository policy.

Four `victory-gpu` tests failed CI now and then:
`unchanged_ranges_are_known`, `no_write_is_lost_with_port_7`,
`records_copied_as_bytes_are_made_at_the_next_sync` and the descriptor test of
views off their elements. The first three shared global state with tests
running beside them (the one asking of `watch::ask_once`, the heap-sync thread);
the fourth assumed where a heap would land. Running the whole `victory-gpu`
suite 15 times per commit:

| Commit | Failures |
|---|---|
| before the fixes | 3 of 15 |
| tests that ask hold a test-only lock | 0 of 15 |
| and the heap-sync test syncs again | 0 of 15 |
