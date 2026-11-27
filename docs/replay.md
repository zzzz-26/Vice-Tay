# Recording a frame and running it again

A diagnostics build can write down one frame's D3D12 work (what the host
layer sends to the card, and the memory it reads) and run it again without
the title, to see where the card's time goes and to try changes in seconds.
Code: `crates/victory-gpu/src/replay`.

## Inspect a dropped draw

For one frame's operation listing, create the capture folder first. Use a
diagnostics build and add the folder and frame number to your usual launch.

```powershell
-More @{VICTORY_CAPTURE='C:\captures\vice-tay'; VICTORY_CAPTURE_FRAMES='1000'}
```

Read `frame-1000-ops.txt` in the capture folder. A failed preparation for a
direct draw, indexed draw or mesh dispatch includes the reason, pipeline
recipe key and guest shader hashes when a pipeline was set. Draws that reach
the GPU can still produce incorrect geometry, depth or materials.

## Recording

Off unless `VICTORY_RECORD` is set. Build with
`cargo build --locked --profile diag -p victory --features diagnostics`.

| variable | meaning |
|---|---|
| `VICTORY_RECORD=<folder>` | on; the capture goes to `<folder>\capture-<n>` (outside the repository) |
| `VICTORY_RECORD_AT=<seconds>` or `f<frame>` | when: seconds after the device was made, or a number of presents. Unset: only the trigger file |
| file `<folder>\go` | when it appears, the capture starts (always watched) |
| `VICTORY_RECORD_FRAMES=<n>` | frames to capture (1) |
| `VICTORY_RECORD_ALWAYS=1` | record lists from the start: needed only if notes say lists were not recorded (a title that submits old lists again) |

At the time given, lists are recorded for a few frames, until a frame's lists
were all recorded; the next frame is captured: everything sent to the drawing
queue between two presents (the first may be the swap chain's own work for
the frame before). One capture per process. At the frame's first submission
the card is drained and its memory copied out; before each later one the card
is drained again and what the CPU changed is written. The title stands still
meanwhile. `frame recorder: the frame is captured` in the log and `frame.bin`
in the folder mean it is done; `failed.txt` that it failed.

**Read `notes.txt`**: everything that could not be recorded is listed there
(and in every replay's report). Only the line with the totals, waits of the
drawing queue and lists of another queue (the relay's copies) are expected.

Measured on a made-up frame of a title's size (12,000 resources, 1.5 million
views, 30 submissions, 40,000 calls): a frame takes 3-10 ms more with the
recorder on, the capture stands still for 5.4 s and is 9 GB (8 GB of the card's
memory at 2.3 GB/s, 1 GB the CPU wrote). The recorder holds a copy of the
memory both sides see that the frame reads, and 1 GB or more of read-back
buffers, until it is done.

## Running it

    victory replay <folder> [--loops 5] [--timing detailed] [experiments]

`<folder>` is the one given as `VICTORY_RECORD` (its last capture) or a
capture. Each loop puts back what the one before changed and sends the frame
without waiting between submissions. Reported, from timestamps on the card:
the frame (first list's start to last list's end), each submission, copies
that stand for what the CPU wrote between submissions (not the title's cost),
and gaps. With `--timing detailed`, each command: sums by kind and the
costliest pipelines with the names the runtime logs and their DXIL hashes
(blake3). A command's time is from the end of the one before to its own end:
draws overlap, so it is a share of the list, not a cost alone; the extra
timestamps cost a little too. The picture goes to `replay.png` and is compared
with the presented one.

| switch | does |
|---|---|
| `--compare` | runs the frame unchanged and with the experiments turn by turn, and prints the difference |
| `--skip-clamps` | no argument clamps and no barriers of their scratch buffers; draws read arguments kept from a first run |
| `--batch-clamps` / `--batch-clamps 2` | argument clamps in batches, as `VICTORY_INDIRECT_BATCH=2` / `=1` does |
| `--skip-hash-passes` | no compare/rehash passes over card buffers |
| `--no-barriers` | no unordered-access barriers |
| `--merge-barriers` | joins barrier calls with no work between them |
| `--buffers vram` / `sysmem` | all buffers in the card's / the PC's memory |
| `--only-buffers <what>` | with `--buffers`: only the buffers all of these match: `name=<text>`, `min=<bytes>`, `max=<bytes>`, `seen=No`, `Cached` or `Combined`, `id=<a>..<b>`, or `!` and one of them (repeatable) |
| `--listing 3..7` | only writes the frame's buffers and those lists' commands |
| `--one-submit` | one `ExecuteCommandLists`, with a full wait between the former ones |
| `--only 3..7` | runs these lists; of the others only barriers |
| `--no-merge` | copies of what the CPU wrote also overwrite the card's own results (below) |
| `--skip-shader <text>` | leaves out what runs pipelines with the text in their name or DXIL hash (repeatable) |
| `--shaders <folder>` | takes shaders from the folder where it has them: `<hash>.dxil`, by the hash the runtime names the title's shader by (a pass of ours: its own DXIL hash); with `--compare --shaders-b <folder>` a second set, turn by turn with the first |
| `--retranslate <folder>` | only writes that folder: the capture's shaders translated by this build with the `VICTORY_*` switches of this run (`--view-limit`, `--raw-views`: the view heap's null slot and where raw views start, as the capture's shaders have them) |
| `--no-restore` | leaves what a loop changed, and makes no copies to restore from (of a capture that does not fit in the card's memory with them) |
| `--pace <ms>` | a wait after each submission sent: the card stands idle between them, as in a game that prepares the next one |
| `--awake` | keeps the card's clocks up (`VICTORY_CARD_AWAKE=1`) |
| `--breadcrumbs` | the system's breadcrumbs (`VICTORY_D3D12_DRED=light`) |
| `--debug-layer` | D3D12's debug layer; its errors in the report |

Use the median of several loops; the first is slower. An experiment that
leaves out waits or work can draw another picture and, with wild data, lose
the device: save work in other programs first. Do not run the title at the
same time.

## What draws a pixel

    victory replay <folder> --pixel 1234,567 [--of 1280x720] [--lists 40..80]

runs the frame once with copies of the texel under the point after every
command that can change it, and names each command that did with its pipeline
(id, name, DXIL hashes) and the values before and after: draws with their
targets (where the viewport has the picture's shape), dispatches and draws with
the textures of the picture's shape shaders may write at that time, copies,
resolves and clears. `--of` is the size of the picture the point was read from
(a screenshot's). The commands are numbered `<list>:<command>` as in
`--listing`, `--slots` and `--looks`.

| switch | does |
|---|---|
| `--pixel <id>[.<sub>]:<x>,<y>` | a texel of one texture instead (repeatable, both kinds) |
| `--depth` | depth targets too: the whole target is copied after each draw |
| `--targets-only` | no textures of unordered access: fewer copies |
| `--split <list>:<command>` | the draws of that indirect call one by one, each named |
| `--lists 40..80` | only these lists are looked into |

A texture that was changed by what is not looked into is reported as "changed
unseen". Copies of 3D textures' targets are of the slice drawn to.

## Runs that differ

Shaders that give out places with atomics (the title's culling) give other
ones in every run. The host layer brings such results to title memory and
sends 64 KiB pieces of it up again, so the recorded copies hold the recorded
run's results. Put over a replay's own, lists and what points into them are of
two runs, and things are drawn from other things' lists. So copies of what the
CPU wrote leave alone words the card has written in this run (`merge.rs`; the
report counts them). `--no-merge` runs the copies as recorded.

| switch | does |
|---|---|
| `--wait-all` / `--wait-lists` | a full wait after every command / each list sent alone and waited for: if pictures still differ, waits are not what is missing |
| `--diverge [runs]` | runs each list this often from the very same memory, finds the first commands that make different memory of it, and says what differs (same values in another order: places given out); contents go to `diverge` in the capture |
| `--diverge-picture [runs]` | from each list on, runs the rest this often from the same memory: where the pictures start to agree, what lies before decides. `--lists 76 --commands 0..900 --step 8` goes into one list |
| `--diverge 10 --lists 76 --draws <text>` | each draw of the pipelines named, alone, this often: how many outcomes, and what it reads |
| `--watch <id>:<offset>:<bytes>` | which lists (with `--lists`: commands) change those words |
| `--slots <list>:<command>` | the heap slots its constants name |
| `--check-written` | that everything the frame changes is put back |
| `--upcopies` | runs the frame twice: which words the copies of the CPU's sides of card buffers bring that the card holds another value of (those the card wrote alike in both runs are the ones to look at), and which pieces shaders wrote that were not declared as written. The live check is `VICTORY_UPCOPY_CHECK=1` (`host/upcheck.rs`) |
| `--flow <lists>` | what the dispatches (`--flow-draws`: draws too) have bound: pipeline, root arguments, heap slots their constants lead to; no device is made |
| `--images <folder>` | runs the frame once and saves the textures its shaders write (or `--texture <id>[.<sub>]`, repeatable) as pictures, with least, mean and greatest values; `--raw`: the values too; of a buffer named, its bytes |
| `--until <list>:<command>` | nothing but barriers after that command |
| `--dump <text>` | the DXIL of the pipelines named, in `shaders` |

The `--diverge` family runs the recorded copies without the merge.

## Limits

- Not recorded, but reported when used: reserved (tiled) resources, bundles,
  ray tracing, render passes, enhanced barriers, predication, stream output,
  `ClearUnorderedAccessView`, shading rate, newer creation methods
  (`CreateCommittedResource1` and on).
- Not kept: contents of multisampled textures, of textures in memory the CPU
  sees, and of texture levels over 256 MB; resources of other queues' work.
- Texture states come from the barriers seen. A resource that overlaps the
  one last used (aliasing) is taken as without contents.
- Addresses in memory are not rewritten: indirect arguments with addresses
  (vertex buffers, root views) are reported, and would be wrong. Descriptors
  and root arguments are.
- Views replaced during the frame get a heap of their own per submission in
  the replay (64 MB for two million views).
- Upload heaps become custom write-combined ones in the replay, as the card
  writes what the CPU wrote.
- What the CPU changes in memory both see is taken before each submission, by
  pages of 4 KiB, of what that submission reads of it (all of a buffer that a
  descriptor names). Pages the card wrote there are part of it: they count as
  copies, and set the card's own results to the recorded ones.
- The card's clocks are the system's: other programs on the card change the
  times.

Tests: `cargo test -p victory-gpu --features diagnostics` (`tests/replay.rs`
runs a small program and compares every byte; `VICTORY_TEST_D3D_DEBUG=1` adds
the debug layer) and `tests/replay_big.rs` (`--ignored`).
