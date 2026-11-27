# What this repository adds

Vice-Tay carries the work of the trees it comes from (see the README's
credits) and the changes below, made and measured on one PC: RTX 4090, Ryzen 9
9950X3D, 48 GB of RAM, a 4K screen. Each change names the switch that controls it
where there is one; the default is the behaviour described.

## Playing

- **Full screen.** The title's window covers the whole monitor with no frame
  (`VICTORY_FULLSCREEN=1`, set by `scripts\play.ps1`; `-Windowed` leaves it out).
  **F11** or **Alt+Enter** switches between full screen and a window fitted to the
  screen while the title runs; the picture is stretched to either. DXGI's own
  Alt+Enter, which would make the swap chain exclusive full screen, is turned off.
- **Settings for an RTX 4090 / Ryzen 9 9950X3D / 48 GB** (`scripts\play.ps1
  -Machine 4090`, picked for any NVIDIA card; `-Machine laptop` for 16 GB of RAM,
  [laptop-16gb.md](laptop-16gb.md)): the thread counts and switches
  `scripts\play.ps1` sets. `-Safe` keeps the wave-wide list there: without it four
  wave64 programs do not translate on a 32-lane card and their draws are dropped.
- **A performance panel** in the top-left corner (diagnostics builds; on by
  default in `scripts\play.ps1`, **F3** hides it, `-NoPanel` leaves it out):
  FPS against the target, frame times with a graph, 1% low and worst frame,
  draws, resolution, GPU name and load, VRAM, CPU name with per-core bars, the
  game's RAM and the PC's. The GPU load is the busiest engine of this process on
  the game's adapter, from Windows' "GPU Engine" counters (labelled `APP`), or an
  estimate from the queue's idle time when Windows has no sample (`EST`). The
  HUD is Belo's.

- **A chosen monitor** (`VICTORY_MONITOR`): the device name (`DISPLAY3`) or the
  monitor's 1-based number in Windows' enumeration order (the order
  `EnumDisplayMonitors` lists them, not the numbers in Windows' display settings).
  The title's own top-level window is centred on that monitor's work area (a size
  left to Windows becomes three quarters of it, then `VICTORY_WINDOW_MAX` applies),
  and with `VICTORY_FULLSCREEN=1` full screen covers that monitor. Tool, child and
  owned windows stay where Windows puts them; the separate DXGI fallback window
  keeps its own placement. Example: `scripts\play.ps1 -Exe <exe> -More @{VICTORY_MONITOR='DISPLAY3'}`.

## Memory and frame rate

How each of these was found, with the numbers:
[findings-rtx4090.md](findings-rtx4090.md).

- **A placed view keeps one shape.** The words the title leaves unwritten in
  view and resource descriptions (padding, unused union tails) are zeroed before
  a view is keyed, so the same view no longer makes a new shape each time it
  arrives. Before, the game grew to 52 GB in 70 minutes, past physical RAM.
  Entries no resource releases are capped at a million
  (`VICTORY_OWNERLESS_VIEWS`).
- **Small pooled textures sit at 4 KiB placements**, except on AMD cards, which
  keep the 64 KiB that avoided device losses on an RX 7900 XTX
  (`VICTORY_POOL_SMALL=1` or `=0` picks either on any card).
- **Structured views that begin off an element get no copy of their bytes** by
  default. Making one holds a global lock; on a card whose view heap has no raw
  twins (NVIDIA) every such view took it, and the list threads queued on it down
  to 6 fps (`VICTORY_STRUCTURED_RAW=1` makes the copies).

## Picture

- **Dynamic descriptor slots are given back when the card has run a flight**, not
  when the host list is reset some 64 flights later. On a card whose
  shader-visible view heap is capped at 1,000,000 (NVIDIA), the 2^18 slots for
  tables made afresh ran out in busy scenes: hundreds of draws a frame were
  dropped, and the picture blew out to white and filled with black blobs. The
  frame line counts the slots tables take, the refusals whose free slots were
  in pieces, and the tables that waited for flights to land for slots or were
  refused again and dropped (diagnostics builds).
- **Rectangle lists** expanded on the CPU find their right angles from the element
  the input layout names `POSITION`, not from the first two-component element,
  which may be a texture coordinate.
- **Wave64 programs on a 32-lane card** (two host waves per console wave):
  - a group barrier is placed only where every console wave of the group arrives
    (a scalar branch can take one console wave of a group and not the other; the
    first then waited on the host for threads that never came);
  - an operand ending at `v255` counts `v255`;
  - a lane write (`v_writelane`) inside a region a branch skips keeps the skip.
- The translator's version is 2: the shader cache translates again what version 1
  made.

## Loading and the runtime

- **DirectStorage**: a queue submits itself when it holds half its capacity, unless
  the title created it with automatic submission off (`DSTORAGE_QUEUE_OPTIONS`).
- **Shader cache**: a shader that translated is used even when its entry cannot
  be written. `VICTORY_WAVE_WIDE=0` turns the wave-wide form off for the listed
  shaders too, and `victory warm` makes a pipeline again when the wave-wide list
  that decides its shaders changes.
- **XAsync**: a call is over by the time its result shows, so a title that begins
  the next call on the block at once is not refused. The call stays registered
  through its provider's cleanup: a wait for its status sleeps until that call's
  result is published, even if the block is reused, and a status query from
  inside the cleanup returns at once instead of waiting for itself.
- **Late DLLs**: a title DLL whose imports fail to bind leaves nothing loaded
  behind for the next `LoadLibrary` of its name.
- **Closing**: only a `WM_CLOSE` to the title's own window ends the title; other
  windows it shows close by themselves.
- **Saves**: a version whose manifest names a blob that is gone is still the
  save when no older version is intact (that blob reads as not found).
- **Scripted keys** (`VICTORY_KEYS_FILE`): the file is taken by renaming it before
  it is read, so a batch written meanwhile is never deleted unread.

## Measuring

- Every host list that ran is kept for reuse.
- Frame dumps (`VICTORY_CAPTURE`) are encoded on a thread of their own, and the
  ones still queued are written before the process exits.
- Diagnostics builds: `VICTORY_WALLS_TRACE=1` logs the source of each read-back;
  a placed texture view that gets no host view says why once per kind.
- [VIctory-NVIDIA changes](victory-nvidia.md) covers the source revisions,
  graphics fixes, optional settings and fault capture reader.
