# wineserver: an index for mapped views

`wineserver` keeps each process's mapped views in a list and searches it from
the front on every map and unmap request (`is_valid_view_addr`,
`find_mapped_view`, `find_mapped_addr` in `server/mapping.c`). A title mapping
physical pages holds 90,000 to 136,000 live views and maps or unmaps about a
thousand a second, so those searches took 37–43% of a core in `perf`, and
everything waiting on `wineserver` waited behind them.

`view-tree.patch` adds a red-black tree of all views, keyed by process and base
address. Views of a process never overlap (each is checked before it is added),
so a lookup by address is a search for the nearest base at or below it. The list
stays, for its order (main exe first, then load order).

`build.sh` builds GE-Proton11-14's `wineserver` with the patch:

- from GE's Wine and wine-staging, with the Wine part of GE's own patch script;
- the result goes in a copy of the GE-Proton folder, `GE-Proton11-14-viewtree`;
- the original is kept beside it as `files/bin/wineserver.ge`.

The patch was checked against the original searches with 2 million random
operations (up to 66,000 live views across three processes).
