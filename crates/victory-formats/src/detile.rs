//! High-throughput CPU texture detiling and tiling for RDNA2 (GFX10.3) layouts.
//!
//! Every thin tiled swizzle mode (`SW_256B_*`, `SW_4KB_*`, `SW_64KB_*`) places
//! aligned 256-byte micro-tiles contiguously in tiled memory (including mip
//! tails, whose origins are always micro-tile aligned). Processing full
//! micro-tile strips lets us:
//!
//! 1. Read each 256-byte micro-tile once sequentially (4 cache lines) instead
//!    of striding across macro-blocks on every scanline.
//! 2. Write 4, 8, or 16 sequential cache-line-aligned output streams into
//!    linear memory (friendly to both L1/L2 cache and write-combining upload
//!    heaps).
//! 3. Replace per-element address arithmetic and heap-allocated run tables with
//!    a 64-byte stack-resident micro-tile offset LUT per macro-block.
//! 4. Use runtime-detected AVX2/SSE2 micro-tile transposes on `x86_64`, with a
//!    portable fixed-width fallback everywhere else.
//! 5. Fan out large subresources (`>= PARALLEL_MIN_BYTES`) across a persistent
//!    worker pool on macro-block-row boundaries so streaming bursts avoid
//!    repeated OS thread creation while small mips run inline with zero
//!    synchronization overhead.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use crate::texture::{BLOCK_256_2D, Dimension, Error, Layout, Mip, Swizzle};

/// Subresource active byte threshold above which strip work is fanned out
/// across the persistent worker pool. Subresources under 2 MiB (e.g., 1 MiB
/// 1K BC7 mips) run inline so thread-pool wake/join latency does not outweigh
/// parallel gain on high-core / virtualized CPUs.
const PARALLEL_MIN_BYTES: usize = 2 * 1024 * 1024;

/// Which 256-byte micro-tile byte permutation applies to a thin tiled layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MicroKind {
    /// 4 rows x 64 bytes: 4x4 transpose of 16-byte chunks (Thin S, 16B & 8B).
    R4x64S,
    /// 4 rows x 64 bytes: interleaved 32-byte halves (Thin D, 16B).
    R4x64D16,
    /// 4 rows x 64 bytes: interleaved 16-byte row pairs (Thin D, 8B).
    R4x64D8,
    /// 8 rows x 32 bytes: left 16B half in 0..128, right 16B half in 128..256
    /// (Thin S 4B, Thin S & D 2B).
    R8x32S,
    /// 8 rows x 32 bytes: 4-row quadrants of left/right 16B halves (Thin D, 4B).
    R8x32D4,
    /// 16 rows x 16 bytes: already row-major inside the micro-tile (Thin S, 1B).
    R16x16S1,
    /// 16 rows x 16 bytes: 8-byte row-pair interleave (Thin D, 1B).
    R16x16D1,
}

impl MicroKind {
    pub(crate) fn from_layout(display: bool, element_log2: u32) -> Self {
        match (display, element_log2) {
            (false, 4 | 3) => Self::R4x64S,
            (true, 4) => Self::R4x64D16,
            (true, 3) => Self::R4x64D8,
            (false, 2 | 1) | (true, 1) => Self::R8x32S,
            (true, 2) => Self::R8x32D4,
            (false, 0) => Self::R16x16S1,
            (true, 0) => Self::R16x16D1,
            _ => unreachable!("validated element_log2 is 0..=4"),
        }
    }

    #[inline(always)]
    const fn row_bytes(self) -> usize {
        match self {
            Self::R4x64S | Self::R4x64D16 | Self::R4x64D8 => 64,
            Self::R8x32S | Self::R8x32D4 => 32,
            Self::R16x16S1 | Self::R16x16D1 => 16,
        }
    }
}

/// Direction of a texture copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Detile,
    Tile,
}

/// Validated subresource copy parameters shared by both `detile` and `tile`.
struct CopyPlan<'a> {
    layout: &'a Layout,
    mip_info: Mip,
    first_slice: u32,
    depth: u32,
    element_bytes: usize,
    row_bytes: usize,
    row_pitch: usize,
    depth_pitch: usize,
}

#[allow(clippy::too_many_arguments)]
fn validate_plan<'a>(
    layout: &'a Layout,
    tiled_len: usize,
    mip: u32,
    slice: u32,
    linear_len: usize,
    row_pitch: usize,
    depth_pitch: usize,
    dir: Direction,
) -> Result<CopyPlan<'a>, Error> {
    let m = *layout.mips.get(mip as usize).ok_or(Error::Invalid)?;
    let depth = m.depth;
    if m.width == 0 || m.height == 0 || depth == 0 {
        return Err(Error::Invalid);
    }
    if !layout.desc.element_bytes.is_power_of_two() || layout.desc.element_bytes > 16 {
        return Err(Error::Invalid);
    }
    let first_slice = if layout.desc.dimension == Dimension::Tex3d {
        if slice != 0 {
            return Err(Error::Invalid);
        }
        0
    } else {
        slice
    };
    if layout.thick {
        if layout.block[2] == 0 {
            return Err(Error::Invalid);
        }
        let last_z = first_slice.checked_add(depth - 1).ok_or(Error::Invalid)?;
        if last_z / layout.block[2] >= layout.slices {
            return Err(Error::Invalid);
        }
    } else {
        let slice_end = first_slice.checked_add(depth).ok_or(Error::Invalid)?;
        if slice_end > layout.slices {
            return Err(Error::Invalid);
        }
    }
    if !layout.addressable() {
        return Err(Error::Unsupported);
    }
    let element_bytes = layout.desc.element_bytes as usize;
    let row_bytes = (m.width as usize)
        .checked_mul(element_bytes)
        .ok_or(Error::TooSmall)?;
    let slice_bytes = (m.height as usize - 1)
        .checked_mul(row_pitch)
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or(Error::TooSmall)?;

    // Multiple depth slices must not overwrite each other.
    if depth > 1 && depth_pitch < slice_bytes {
        return Err(Error::TooSmall);
    }

    let need = (depth as usize - 1)
        .checked_mul(depth_pitch)
        .and_then(|offset| offset.checked_add(slice_bytes))
        .ok_or(Error::TooSmall)?;
    let min_tiled = match dir {
        Direction::Detile => layout.bytes_from(mip),
        Direction::Tile => layout.size,
    };
    let min_tiled = usize::try_from(min_tiled).map_err(|_| Error::TooSmall)?;
    if tiled_len < min_tiled || linear_len < need || row_pitch < row_bytes {
        return Err(Error::TooSmall);
    }
    Ok(CopyPlan {
        layout,
        mip_info: m,
        first_slice,
        depth,
        element_bytes,
        row_bytes,
        row_pitch,
        depth_pitch,
    })
}

/// Detiles one subresource from `tiled` into `linear`.
pub(crate) fn detile(
    layout: &Layout,
    tiled: &[u8],
    mip: u32,
    slice: u32,
    linear: &mut [u8],
    row_pitch: usize,
    depth_pitch: usize,
) -> Result<(), Error> {
    let plan = validate_plan(
        layout,
        tiled.len(),
        mip,
        slice,
        linear.len(),
        row_pitch,
        depth_pitch,
        Direction::Detile,
    )?;
    if layout.desc.swizzle == Swizzle::LINEAR {
        copy_linear_detile(&plan, tiled, linear);
        return Ok(());
    }
    if let Some((params, full_mx, full_my, full_w, full_h, strips_per_block)) =
        prepare_thin_geometry(&plan, tiled.len(), linear.len(), Direction::Detile, use_avx2())
    {
        copy_thin_tiled(
            &plan,
            &params,
            full_mx,
            full_my,
            full_w,
            full_h,
            strips_per_block,
            tiled.as_ptr(),
            tiled.len(),
            linear.as_mut_ptr(),
            linear.len(),
        );
    } else {
        copy_thick_detile(&plan, tiled, linear);
    }
    Ok(())
}

/// Tiles one subresource from `linear` into `tiled`.
pub(crate) fn tile(
    layout: &Layout,
    tiled: &mut [u8],
    mip: u32,
    slice: u32,
    linear: &[u8],
    row_pitch: usize,
    depth_pitch: usize,
) -> Result<(), Error> {
    let plan = validate_plan(
        layout,
        tiled.len(),
        mip,
        slice,
        linear.len(),
        row_pitch,
        depth_pitch,
        Direction::Tile,
    )?;
    if layout.desc.swizzle == Swizzle::LINEAR {
        copy_linear_tile(&plan, tiled, linear);
        return Ok(());
    }
    if let Some((params, full_mx, full_my, full_w, full_h, strips_per_block)) =
        prepare_thin_geometry(&plan, tiled.len(), linear.len(), Direction::Tile, use_avx2())
    {
        copy_thin_tiled(
            &plan,
            &params,
            full_mx,
            full_my,
            full_w,
            full_h,
            strips_per_block,
            tiled.as_mut_ptr(),
            tiled.len(),
            linear.as_ptr().cast_mut(),
            linear.len(),
        );
    } else {
        copy_thick_tile(&plan, tiled, linear);
    }
    Ok(())
}

/// Forces the portable (non-AVX2) thin micro-tile path for testing parity on
/// AVX2-capable CPUs.
#[cfg(test)]
pub(crate) fn detile_portable(
    layout: &Layout,
    tiled: &[u8],
    mip: u32,
    slice: u32,
    linear: &mut [u8],
    row_pitch: usize,
    depth_pitch: usize,
) -> Result<(), Error> {
    let plan = validate_plan(
        layout,
        tiled.len(),
        mip,
        slice,
        linear.len(),
        row_pitch,
        depth_pitch,
        Direction::Detile,
    )?;
    if layout.desc.swizzle == Swizzle::LINEAR {
        copy_linear_detile(&plan, tiled, linear);
    } else if let Some((params, full_mx, full_my, full_w, full_h, strips_per_block)) =
        prepare_thin_geometry(&plan, tiled.len(), linear.len(), Direction::Detile, false)
    {
        copy_thin_tiled(
            &plan,
            &params,
            full_mx,
            full_my,
            full_w,
            full_h,
            strips_per_block,
            tiled.as_ptr(),
            tiled.len(),
            linear.as_mut_ptr(),
            linear.len(),
        );
    } else {
        copy_thick_detile(&plan, tiled, linear);
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn tile_portable(
    layout: &Layout,
    tiled: &mut [u8],
    mip: u32,
    slice: u32,
    linear: &[u8],
    row_pitch: usize,
    depth_pitch: usize,
) -> Result<(), Error> {
    let plan = validate_plan(
        layout,
        tiled.len(),
        mip,
        slice,
        linear.len(),
        row_pitch,
        depth_pitch,
        Direction::Tile,
    )?;
    if layout.desc.swizzle == Swizzle::LINEAR {
        copy_linear_tile(&plan, tiled, linear);
    } else if let Some((params, full_mx, full_my, full_w, full_h, strips_per_block)) =
        prepare_thin_geometry(&plan, tiled.len(), linear.len(), Direction::Tile, false)
    {
        copy_thin_tiled(
            &plan,
            &params,
            full_mx,
            full_my,
            full_w,
            full_h,
            strips_per_block,
            tiled.as_mut_ptr(),
            tiled.len(),
            linear.as_ptr().cast_mut(),
            linear.len(),
        );
    } else {
        copy_thick_tile(&plan, tiled, linear);
    }
    Ok(())
}

#[inline]
fn use_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static HAS_AVX2: OnceLock<bool> = OnceLock::new();
        *HAS_AVX2.get_or_init(|| std::is_x86_feature_detected!("avx2"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Linear layout detile: single bulk copy when tightly packed, else per-row.
fn copy_linear_detile(plan: &CopyPlan<'_>, tiled: &[u8], linear: &mut [u8]) {
    let m = &plan.mip_info;
    let tiled_pitch = m.pitch as usize * plan.element_bytes;
    let height = m.height as usize;
    let packed = tiled_pitch == plan.row_bytes && plan.row_pitch == plan.row_bytes;
    for z in 0..plan.depth {
        let (s, _) = plan.layout.split(plan.first_slice + z);
        let tiled_base = (u64::from(s) * plan.layout.slice_size + m.offset) as usize;
        let linear_base = z as usize * plan.depth_pitch;
        if packed {
            let total = height * plan.row_bytes;
            linear[linear_base..linear_base + total]
                .copy_from_slice(&tiled[tiled_base..tiled_base + total]);
        } else {
            for y in 0..height {
                let t = tiled_base + y * tiled_pitch;
                let l = linear_base + y * plan.row_pitch;
                linear[l..l + plan.row_bytes].copy_from_slice(&tiled[t..t + plan.row_bytes]);
            }
        }
    }
}

/// Linear layout tile: single bulk copy when tightly packed, else per-row.
fn copy_linear_tile(plan: &CopyPlan<'_>, tiled: &mut [u8], linear: &[u8]) {
    let m = &plan.mip_info;
    let tiled_pitch = m.pitch as usize * plan.element_bytes;
    let height = m.height as usize;
    let packed = tiled_pitch == plan.row_bytes && plan.row_pitch == plan.row_bytes;
    for z in 0..plan.depth {
        let (s, _) = plan.layout.split(plan.first_slice + z);
        let tiled_base = (u64::from(s) * plan.layout.slice_size + m.offset) as usize;
        let linear_base = z as usize * plan.depth_pitch;
        if packed {
            let total = height * plan.row_bytes;
            tiled[tiled_base..tiled_base + total]
                .copy_from_slice(&linear[linear_base..linear_base + total]);
        } else {
            for y in 0..height {
                let t = tiled_base + y * tiled_pitch;
                let l = linear_base + y * plan.row_pitch;
                tiled[t..t + plan.row_bytes].copy_from_slice(&linear[l..l + plan.row_bytes]);
            }
        }
    }
}

/// Builds contiguous column runs for `[start_x..end_x)` elements.
fn build_column_runs(
    layout: &Layout,
    m: &Mip,
    start_x: u32,
    end_x: u32,
) -> Vec<(usize, usize, usize)> {
    if start_x >= end_x {
        return Vec::new();
    }
    let n = layout.desc.element_bytes as usize;
    let mut runs = Vec::new();
    let mut first_linear = start_x as usize * n;
    let mut first_tiled = layout.column(m, start_x) as usize;
    for x in (start_x + 1)..end_x {
        let offset = layout.column(m, x) as usize;
        let at = x as usize * n;
        if offset != first_tiled + (at - first_linear) {
            runs.push((first_tiled, first_linear, at - first_linear));
            first_tiled = offset;
            first_linear = at;
        }
    }
    runs.push((first_tiled, first_linear, end_x as usize * n - first_linear));
    runs
}

#[inline]
fn copy_runs_detile<const N: usize>(
    runs: &[(usize, usize, usize)],
    tiled: &[u8],
    linear: &mut [u8],
    tiled_row: usize,
    linear_row: usize,
) {
    for &(t, l, len) in runs {
        let len = if N == 0 { len } else { N };
        linear[linear_row + l..][..len].copy_from_slice(&tiled[tiled_row + t..][..len]);
    }
}

#[inline]
fn copy_runs_tile<const N: usize>(
    runs: &[(usize, usize, usize)],
    tiled: &mut [u8],
    linear: &[u8],
    tiled_row: usize,
    linear_row: usize,
) {
    for &(t, l, len) in runs {
        let len = if N == 0 { len } else { N };
        tiled[tiled_row + t..][..len].copy_from_slice(&linear[linear_row + l..][..len]);
    }
}

fn uniform_run_len(body: &[(usize, usize, usize)]) -> Option<usize> {
    body.first()
        .map(|run| run.2)
        .filter(|&len| body.iter().all(|run| run.2 == len))
}

fn copy_thick_detile(plan: &CopyPlan<'_>, tiled: &[u8], linear: &mut [u8]) {
    let m = &plan.mip_info;
    let runs = build_column_runs(plan.layout, m, 0, m.width);
    let (body, last) = runs.split_at(runs.len().saturating_sub(1));
    let run_len = uniform_run_len(body);
    for z in 0..plan.depth {
        let (s, inner) = plan.layout.split(plan.first_slice + z);
        let base = u64::from(s) * plan.layout.slice_size + m.offset;
        for y in 0..m.height {
            let tiled_row = (base + plan.layout.row(m, y, inner)) as usize;
            let linear_row = z as usize * plan.depth_pitch + y as usize * plan.row_pitch;
            match run_len {
                Some(4) => copy_runs_detile::<4>(body, tiled, linear, tiled_row, linear_row),
                Some(8) => copy_runs_detile::<8>(body, tiled, linear, tiled_row, linear_row),
                Some(16) => copy_runs_detile::<16>(body, tiled, linear, tiled_row, linear_row),
                _ => copy_runs_detile::<0>(body, tiled, linear, tiled_row, linear_row),
            }
            copy_runs_detile::<0>(last, tiled, linear, tiled_row, linear_row);
        }
    }
}

fn copy_thick_tile(plan: &CopyPlan<'_>, tiled: &mut [u8], linear: &[u8]) {
    let m = &plan.mip_info;
    let runs = build_column_runs(plan.layout, m, 0, m.width);
    let (body, last) = runs.split_at(runs.len().saturating_sub(1));
    let run_len = uniform_run_len(body);
    for z in 0..plan.depth {
        let (s, inner) = plan.layout.split(plan.first_slice + z);
        let base = u64::from(s) * plan.layout.slice_size + m.offset;
        for y in 0..m.height {
            let tiled_row = (base + plan.layout.row(m, y, inner)) as usize;
            let linear_row = z as usize * plan.depth_pitch + y as usize * plan.row_pitch;
            match run_len {
                Some(4) => copy_runs_tile::<4>(body, tiled, linear, tiled_row, linear_row),
                Some(8) => copy_runs_tile::<8>(body, tiled, linear, tiled_row, linear_row),
                Some(16) => copy_runs_tile::<16>(body, tiled, linear, tiled_row, linear_row),
                _ => copy_runs_tile::<0>(body, tiled, linear, tiled_row, linear_row),
            }
            copy_runs_tile::<0>(last, tiled, linear, tiled_row, linear_row);
        }
    }
}

/// Wrapper allowing raw buffer pointers to cross worker threads when disjoint
/// ranges are proven by the dispatcher.
#[derive(Clone, Copy)]
struct SendPtr(*mut u8);

impl SendPtr {
    #[inline(always)]
    fn as_ptr(self) -> *mut u8 {
        self.0
    }
}

// SAFETY: `copy_thin_tiled` guarantees that worker tasks access strictly
// disjoint memory ranges (disjoint linear rows and disjoint 256-byte tiled
// micro-tiles) and waits for all workers before returning.
unsafe impl Send for SendPtr {}
// SAFETY: as above.
unsafe impl Sync for SendPtr {}

/// Precomputed per-strip parameters for thin tiled micro-tile processing.
struct ThinStripParams {
    kind: MicroKind,
    micro_h: u32,
    micro_row_bytes: usize,
    block_bytes: usize,
    row_block_stride: usize,
    mx_per_block: usize,
    my_per_block_shift: u32,
    my_per_block_mask: usize,
    start_mx: usize,
    start_my: usize,
    full_blocks_x: usize,
    rem_full_mx: usize,
    x_in_block: [usize; 16],
    y_in_block: [usize; 16],
    row_pitch: usize,
    #[cfg(target_arch = "x86_64")]
    avx2: bool,
    dir: Direction,
}

impl ThinStripParams {
    #[inline(always)]
    fn strip_tiled_base(&self, tiled_plane_base: usize, my: usize) -> usize {
        let my_shifted = my + self.start_my;
        let by = my_shifted >> self.my_per_block_shift;
        let in_by = my_shifted & self.my_per_block_mask;
        tiled_plane_base + by * self.row_block_stride + self.y_in_block[in_by]
    }

    /// Processes `my_start..my_end` full micro-tile strips on one depth/array
    /// plane whose base offsets are `tiled_plane_base` and `linear_plane_base`.
    ///
    /// # Safety
    /// - Every full micro-tile offset computed for `my in my_start..my_end`
    ///   must lie within `tiled_ptr[..tiled_len]` (with at least 256 bytes).
    /// - Every linear row for `y in my_start*micro_h .. my_end*micro_h` must
    ///   lie within `linear_ptr[..linear_len]`.
    /// - When called in parallel across disjoint `my_start..my_end` ranges,
    ///   neither tiled micro-tiles nor linear rows may overlap across threads.
    unsafe fn run_strips(
        &self,
        tiled_ptr: *mut u8,
        linear_ptr: *mut u8,
        tiled_plane_base: usize,
        linear_plane_base: usize,
        my_start: usize,
        my_end: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: caller verified `self.avx2` via `is_x86_feature_detected!("avx2")`
            // and all pointer bounds/disjointness invariants.
            unsafe {
                self.run_strips_avx2(
                    tiled_ptr,
                    linear_ptr,
                    tiled_plane_base,
                    linear_plane_base,
                    my_start,
                    my_end,
                );
            }
            return;
        }
        // SAFETY: caller verified all pointer bounds/disjointness invariants.
        unsafe {
            self.run_strips_portable(
                tiled_ptr,
                linear_ptr,
                tiled_plane_base,
                linear_plane_base,
                my_start,
                my_end,
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn run_strips_avx2(
        &self,
        tiled_ptr: *mut u8,
        linear_ptr: *mut u8,
        tiled_plane_base: usize,
        linear_plane_base: usize,
        my_start: usize,
        my_end: usize,
    ) {
        macro_rules! dispatch_kind {
            ($kernel:ident) => {
                for my in my_start..my_end {
                    let y = (my as u32) * self.micro_h;
                    let row_tiled_base = self.strip_tiled_base(tiled_plane_base, my);
                    let row_linear_base = linear_plane_base + (y as usize) * self.row_pitch;
                    let mut block_tiled = row_tiled_base;
                    let mut lin_col = row_linear_base;
                    if self.start_mx == 0 {
                        for _ in 0..self.full_blocks_x {
                            for &x_off in &self.x_in_block[..self.mx_per_block] {
                                // SAFETY: bounds and disjointness verified by caller.
                                unsafe {
                                    $kernel(
                                        tiled_ptr.add(block_tiled + x_off),
                                        linear_ptr.add(lin_col),
                                        self.row_pitch,
                                    );
                                }
                                lin_col += self.micro_row_bytes;
                            }
                            block_tiled += self.block_bytes;
                        }
                        for &x_off in &self.x_in_block[..self.rem_full_mx] {
                            // SAFETY: bounds and disjointness verified by caller.
                            unsafe {
                                $kernel(
                                    tiled_ptr.add(block_tiled + x_off),
                                    linear_ptr.add(lin_col),
                                    self.row_pitch,
                                );
                            }
                            lin_col += self.micro_row_bytes;
                        }
                    } else {
                        let end_mx = self.start_mx + self.rem_full_mx;
                        for &x_off in &self.x_in_block[self.start_mx..end_mx] {
                            // SAFETY: bounds and disjointness verified by caller.
                            unsafe {
                                $kernel(
                                    tiled_ptr.add(block_tiled + x_off),
                                    linear_ptr.add(lin_col),
                                    self.row_pitch,
                                );
                            }
                            lin_col += self.micro_row_bytes;
                        }
                    }
                }
            };
        }

        match (self.dir, self.kind) {
            (Direction::Detile, MicroKind::R4x64S) => dispatch_kind!(detile_4x64_s_avx2),
            (Direction::Detile, MicroKind::R4x64D16) => dispatch_kind!(detile_4x64_d16_avx2),
            (Direction::Detile, MicroKind::R4x64D8) => dispatch_kind!(detile_4x64_d8_avx2),
            (Direction::Detile, MicroKind::R8x32S) => dispatch_kind!(detile_8x32_s_avx2),
            (Direction::Detile, MicroKind::R8x32D4) => dispatch_kind!(detile_8x32_d4_avx2),
            (Direction::Detile, MicroKind::R16x16S1) => dispatch_kind!(detile_16x16_s1_sse2),
            (Direction::Detile, MicroKind::R16x16D1) => dispatch_kind!(detile_16x16_d1_sse2),
            (Direction::Tile, MicroKind::R4x64S) => dispatch_kind!(tile_4x64_s_avx2),
            (Direction::Tile, MicroKind::R4x64D16) => dispatch_kind!(tile_4x64_d16_avx2),
            (Direction::Tile, MicroKind::R4x64D8) => dispatch_kind!(tile_4x64_d8_avx2),
            (Direction::Tile, MicroKind::R8x32S) => dispatch_kind!(tile_8x32_s_avx2),
            (Direction::Tile, MicroKind::R8x32D4) => dispatch_kind!(tile_8x32_d4_avx2),
            (Direction::Tile, MicroKind::R16x16S1) => dispatch_kind!(tile_16x16_s1_sse2),
            (Direction::Tile, MicroKind::R16x16D1) => dispatch_kind!(tile_16x16_d1_sse2),
        }
    }

    unsafe fn run_strips_portable(
        &self,
        tiled_ptr: *mut u8,
        linear_ptr: *mut u8,
        tiled_plane_base: usize,
        linear_plane_base: usize,
        my_start: usize,
        my_end: usize,
    ) {
        macro_rules! dispatch_kind {
            ($kernel:ident) => {
                for my in my_start..my_end {
                    let y = (my as u32) * self.micro_h;
                    let row_tiled_base = self.strip_tiled_base(tiled_plane_base, my);
                    let row_linear_base = linear_plane_base + (y as usize) * self.row_pitch;
                    let mut block_tiled = row_tiled_base;
                    let mut lin_col = row_linear_base;
                    if self.start_mx == 0 {
                        for _ in 0..self.full_blocks_x {
                            for &x_off in &self.x_in_block[..self.mx_per_block] {
                                // SAFETY: bounds and disjointness verified by caller.
                                unsafe {
                                    $kernel(
                                        tiled_ptr.add(block_tiled + x_off),
                                        linear_ptr.add(lin_col),
                                        self.row_pitch,
                                    );
                                }
                                lin_col += self.micro_row_bytes;
                            }
                            block_tiled += self.block_bytes;
                        }
                        for &x_off in &self.x_in_block[..self.rem_full_mx] {
                            // SAFETY: bounds and disjointness verified by caller.
                            unsafe {
                                $kernel(
                                    tiled_ptr.add(block_tiled + x_off),
                                    linear_ptr.add(lin_col),
                                    self.row_pitch,
                                );
                            }
                            lin_col += self.micro_row_bytes;
                        }
                    } else {
                        let end_mx = self.start_mx + self.rem_full_mx;
                        for &x_off in &self.x_in_block[self.start_mx..end_mx] {
                            // SAFETY: bounds and disjointness verified by caller.
                            unsafe {
                                $kernel(
                                    tiled_ptr.add(block_tiled + x_off),
                                    linear_ptr.add(lin_col),
                                    self.row_pitch,
                                );
                            }
                            lin_col += self.micro_row_bytes;
                        }
                    }
                }
            };
        }

        match (self.dir, self.kind) {
            (Direction::Detile, MicroKind::R4x64S) => dispatch_kind!(detile_4x64_s_portable),
            (Direction::Detile, MicroKind::R4x64D16) => dispatch_kind!(detile_4x64_d16_portable),
            (Direction::Detile, MicroKind::R4x64D8) => dispatch_kind!(detile_4x64_d8_portable),
            (Direction::Detile, MicroKind::R8x32S) => dispatch_kind!(detile_8x32_s_portable),
            (Direction::Detile, MicroKind::R8x32D4) => dispatch_kind!(detile_8x32_d4_portable),
            (Direction::Detile, MicroKind::R16x16S1) => dispatch_kind!(detile_16x16_s1_portable),
            (Direction::Detile, MicroKind::R16x16D1) => dispatch_kind!(detile_16x16_d1_portable),
            (Direction::Tile, MicroKind::R4x64S) => dispatch_kind!(tile_4x64_s_portable),
            (Direction::Tile, MicroKind::R4x64D16) => dispatch_kind!(tile_4x64_d16_portable),
            (Direction::Tile, MicroKind::R4x64D8) => dispatch_kind!(tile_4x64_d8_portable),
            (Direction::Tile, MicroKind::R8x32S) => dispatch_kind!(tile_8x32_s_portable),
            (Direction::Tile, MicroKind::R8x32D4) => dispatch_kind!(tile_8x32_d4_portable),
            (Direction::Tile, MicroKind::R16x16S1) => dispatch_kind!(tile_16x16_s1_portable),
            (Direction::Tile, MicroKind::R16x16D1) => dispatch_kind!(tile_16x16_d1_portable),
        }
    }
}

/// Validates all geometry, bounds, and micro-tile disjointness invariants
/// required by the unsafe thin strip kernels. Returns `None` if any `pub` field
/// of `Layout` was mutated into an inconsistent state so callers safely fall
/// back to `copy_thick_*` without ever entering `unsafe`.
fn prepare_thin_geometry(
    plan: &CopyPlan<'_>,
    tiled_len: usize,
    linear_len: usize,
    dir: Direction,
    avx2: bool,
) -> Option<(ThinStripParams, usize, usize, u32, u32, usize)> {
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx2;
    let layout = plan.layout;
    if layout.thick {
        return None;
    }
    let m = &plan.mip_info;
    let element_log2 = layout.desc.element_bytes.trailing_zeros();
    let [micro_w, micro_h] = BLOCK_256_2D[element_log2 as usize];
    let kind = MicroKind::from_layout(layout.desc.swizzle.display(), element_log2);
    let micro_row_bytes = kind.row_bytes();

    let block_bytes = usize::try_from(layout.alignment).ok()?;
    let bw = layout.block[0];
    let bh = layout.block[1];
    if !bw.is_power_of_two() || !bh.is_power_of_two() || bw < micro_w || bh < micro_h {
        return None;
    }
    let mx_per_block = (bw / micro_w) as usize;
    let my_per_block = (bh / micro_h) as usize;
    if !(1..=16).contains(&mx_per_block)
        || !(1..=16).contains(&my_per_block)
        || !my_per_block.is_power_of_two()
    {
        return None;
    }
    if mx_per_block
        .checked_mul(my_per_block)?
        .checked_mul(256)?
        != block_bytes
    {
        return None;
    }

    let mut x_in_block = [0usize; 16];
    for (i, slot) in x_in_block.iter_mut().take(mx_per_block).enumerate() {
        let idx = i.checked_mul(micro_w as usize)?;
        let off = *layout.pattern[0].get(idx)? as usize;
        if (off & 0xff) != 0 || off >= block_bytes {
            return None;
        }
        *slot = off;
    }
    let mut y_in_block = [0usize; 16];
    for (i, slot) in y_in_block.iter_mut().take(my_per_block).enumerate() {
        let idx = i.checked_mul(micro_h as usize)?;
        let off = *layout.pattern[1].get(idx)? as usize;
        if (off & 0xff) != 0 || off >= block_bytes {
            return None;
        }
        *slot = off;
    }

    // Prove every `(ix, iy)` inside a macro-block maps to a strictly distinct
    // 256-byte micro-tile completely inside `0..block_bytes` so parallel
    // macro-row / strip workers never race or step out of `block_base..block_base+block_bytes`.
    // Because `mx_per_block <= 16` and `my_per_block <= 16`, `block_bytes <= 65536`
    // and every 256-byte tile index `sum >> 8` fits in a 256-bit mask.
    let mut seen_tiles = [0u64; 4];
    for &yo in &y_in_block[..my_per_block] {
        for &xo in &x_in_block[..mx_per_block] {
            let sum = yo.checked_add(xo)?;
            if (sum & 0xff) != 0 || sum.checked_add(256)? > block_bytes {
                return None;
            }
            let slot = sum >> 8;
            let word = &mut seen_tiles[slot >> 6];
            let bit = 1u64 << (slot & 63);
            if (*word & bit) != 0 {
                return None;
            }
            *word |= bit;
        }
    }

    let full_mx = (m.width / micro_w) as usize;
    let full_my = (m.height / micro_h) as usize;
    let full_w = u32::try_from(full_mx).ok()?.checked_mul(micro_w)?;
    let full_h = u32::try_from(full_my).ok()?.checked_mul(micro_h)?;

    let tail_x = m.tail.map_or(0, |t| t.x);
    let tail_y = m.tail.map_or(0, |t| t.y);
    if !tail_x.is_multiple_of(micro_w) || !tail_y.is_multiple_of(micro_h) {
        return None;
    }
    let start_mx = (tail_x / micro_w) as usize;
    let start_my = (tail_y / micro_h) as usize;
    if m.tail.is_some()
        && (start_mx.checked_add(full_mx)? > mx_per_block
            || start_my.checked_add(full_my)? > my_per_block)
    {
        return None;
    }
    let (full_blocks_x, rem_full_mx) = if start_mx == 0 {
        (full_mx / mx_per_block, full_mx % mx_per_block)
    } else {
        if start_mx.checked_add(full_mx)? > mx_per_block {
            return None;
        }
        (0, full_mx)
    };
    let pitch_blocks = (m.pitch / bw) as usize;
    let row_block_stride = pitch_blocks.checked_mul(block_bytes)?;

    let params = ThinStripParams {
        kind,
        micro_h,
        micro_row_bytes,
        block_bytes,
        row_block_stride,
        mx_per_block,
        my_per_block_shift: my_per_block.trailing_zeros(),
        my_per_block_mask: my_per_block - 1,
        start_mx,
        start_my,
        full_blocks_x,
        rem_full_mx,
        x_in_block,
        y_in_block,
        row_pitch: plan.row_pitch,
        #[cfg(target_arch = "x86_64")]
        avx2,
        dir,
    };

    if full_mx > 0 && full_my > 0 {
        let last_linear_end = (plan.depth as usize - 1)
            .checked_mul(plan.depth_pitch)?
            .checked_add((full_h as usize - 1).checked_mul(plan.row_pitch)?)?
            .checked_add(full_mx.checked_mul(micro_row_bytes)?)?;
        if last_linear_end > linear_len {
            return None;
        }

        let last_slice = plan.first_slice.checked_add(plan.depth - 1)?;
        let (last_s, _) = layout.split(last_slice);
        let last_plane_base_u64 = u64::from(last_s)
            .checked_mul(layout.slice_size)?
            .checked_add(m.offset)?;
        let last_plane_base = usize::try_from(last_plane_base_u64).ok()?;

        let last_my_shifted = (full_my - 1).checked_add(start_my)?;
        let last_by = last_my_shifted >> params.my_per_block_shift;
        if m.tail.is_some() && last_by != 0 {
            return None;
        }
        let blocks_x = if start_mx > 0 {
            1
        } else {
            full_blocks_x.checked_add(usize::from(rem_full_mx > 0))?
        };
        if last_by > 0 && pitch_blocks < blocks_x {
            return None;
        }
        let last_bx = blocks_x - 1;
        let _ = u32::try_from(last_by).ok()?.checked_mul(bh)?;
        let _ = u32::try_from(last_bx).ok()?.checked_mul(bw)?;

        let last_block_base = last_by
            .checked_mul(row_block_stride)?
            .checked_add(last_plane_base)?
            .checked_add(last_bx.checked_mul(block_bytes)?)?;
        if last_block_base.checked_add(block_bytes)? > tiled_len {
            return None;
        }
    }

    Some((params, full_mx, full_my, full_w, full_h, my_per_block))
}

/// Fast path for all addressable thin tiled layouts (`SW_256B_*`, `SW_4KB_*`,
/// `SW_64KB_*`), handling full 256-byte micro-tiles with SIMD/LUT kernels and
/// partial right/bottom edges with minimal run copies.
#[allow(clippy::too_many_arguments)]
fn copy_thin_tiled(
    plan: &CopyPlan<'_>,
    params: &ThinStripParams,
    full_mx: usize,
    full_my: usize,
    full_w: u32,
    full_h: u32,
    strips_per_block: usize,
    tiled_ptr: *const u8,
    tiled_len: usize,
    linear_ptr: *mut u8,
    linear_len: usize,
) {
    let layout = plan.layout;
    let m = &plan.mip_info;

    // Full micro-tile body.
    if full_mx > 0 && full_my > 0 {
        let body_bytes = full_mx.saturating_mul(full_my).saturating_mul(256);
        let tiled_send = SendPtr(tiled_ptr.cast_mut());
        let linear_send = SendPtr(linear_ptr);

        for z in 0..plan.depth {
            let (s, _) = layout.split(plan.first_slice + z);
            let tiled_plane_base = (u64::from(s) * layout.slice_size + m.offset) as usize;
            let linear_plane_base = z as usize * plan.depth_pitch;

            if body_bytes >= PARALLEL_MIN_BYTES && full_my >= 4 {
                // Partition on macro-block-row boundaries when multiple macro-block rows exist
                // so every thread owns disjoint 64 KiB / 4 KiB blocks (zero cache-line sharing).
                let unit = if full_my >= 2 * strips_per_block {
                    strips_per_block
                } else {
                    1
                };
                let total_units = full_my.div_ceil(unit);
                DetilePool::global().run(total_units, |u_start, u_end| {
                    let my_start = (u_start * unit).min(full_my);
                    let my_end = (u_end * unit).min(full_my);
                    if my_start < my_end {
                        // SAFETY: disjoint `my_start..my_end` strips own disjoint linear rows
                        // and disjoint 256-byte micro-tiles; bounds and disjointness were
                        // verified in `prepare_thin_geometry`.
                        unsafe {
                            params.run_strips(
                                tiled_send.as_ptr(),
                                linear_send.as_ptr(),
                                tiled_plane_base,
                                linear_plane_base,
                                my_start,
                                my_end,
                            );
                        }
                    }
                });
            } else {
                // SAFETY: bounds and disjointness were verified in `prepare_thin_geometry`.
                unsafe {
                    params.run_strips(
                        tiled_send.0,
                        linear_send.0,
                        tiled_plane_base,
                        linear_plane_base,
                        0,
                        full_my,
                    );
                }
            }
        }
    }

    // Partial right edge (`full_w..m.width` for `y in 0..full_h`) and
    // partial bottom edge (`0..m.width` for `y in full_h..m.height`).
    let has_right = full_w < m.width && full_h > 0;
    let has_bottom = full_h < m.height;
    if !has_right && !has_bottom {
        return;
    }

    let right_runs = if has_right {
        build_column_runs(layout, m, full_w, m.width)
    } else {
        Vec::new()
    };
    let bottom_runs = if has_bottom {
        build_column_runs(layout, m, 0, m.width)
    } else {
        Vec::new()
    };

    match params.dir {
        Direction::Detile => {
            // SAFETY: `tiled_ptr[..tiled_len]` comes from the caller's `&[u8]` slice
            // and all parallel workers above have joined.
            let tiled = unsafe { std::slice::from_raw_parts(tiled_ptr, tiled_len) };
            // SAFETY: `linear_ptr[..linear_len]` comes from the caller's `&mut [u8]`
            // slice, does not overlap `tiled`, and all parallel workers above have joined.
            let linear = unsafe { std::slice::from_raw_parts_mut(linear_ptr, linear_len) };
            for z in 0..plan.depth {
                let (s, _) = layout.split(plan.first_slice + z);
                let base = u64::from(s) * layout.slice_size + m.offset;
                if has_right {
                    for y in 0..full_h {
                        let tiled_row = (base + layout.row(m, y, 0)) as usize;
                        let linear_row = z as usize * plan.depth_pitch + y as usize * plan.row_pitch;
                        copy_runs_detile::<0>(&right_runs, tiled, linear, tiled_row, linear_row);
                    }
                }
                if has_bottom {
                    for y in full_h..m.height {
                        let tiled_row = (base + layout.row(m, y, 0)) as usize;
                        let linear_row = z as usize * plan.depth_pitch + y as usize * plan.row_pitch;
                        copy_runs_detile::<0>(&bottom_runs, tiled, linear, tiled_row, linear_row);
                    }
                }
            }
        }
        Direction::Tile => {
            // SAFETY: `tiled_ptr[..tiled_len]` comes from the caller's `&mut [u8]`
            // slice, does not overlap `linear`, and all parallel workers above have joined.
            let tiled = unsafe { std::slice::from_raw_parts_mut(tiled_ptr.cast_mut(), tiled_len) };
            // SAFETY: `linear_ptr[..linear_len]` comes from the caller's `&[u8]` slice
            // and all parallel workers above have joined.
            let linear = unsafe { std::slice::from_raw_parts(linear_ptr.cast_const(), linear_len) };
            for z in 0..plan.depth {
                let (s, _) = layout.split(plan.first_slice + z);
                let base = u64::from(s) * layout.slice_size + m.offset;
                if has_right {
                    for y in 0..full_h {
                        let tiled_row = (base + layout.row(m, y, 0)) as usize;
                        let linear_row = z as usize * plan.depth_pitch + y as usize * plan.row_pitch;
                        copy_runs_tile::<0>(&right_runs, tiled, linear, tiled_row, linear_row);
                    }
                }
                if has_bottom {
                    for y in full_h..m.height {
                        let tiled_row = (base + layout.row(m, y, 0)) as usize;
                        let linear_row = z as usize * plan.depth_pitch + y as usize * plan.row_pitch;
                        copy_runs_tile::<0>(&bottom_runs, tiled, linear, tiled_row, linear_row);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Persistent worker pool for large subresources
// ============================================================================

struct Job {
    func: fn(*const (), usize, usize),
    ctx: *const (),
    total_units: usize,
    active_threads: usize,
}

// SAFETY: `Job` carries a pointer to a `Sync` closure on the caller's stack;
// `DetilePool::run` waits for every worker to finish (even on unwind) before
// returning.
unsafe impl Send for Job {}

struct PoolState {
    generation: u64,
    job: Option<Job>,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

struct DetilePool {
    busy: Mutex<()>,
    state: Mutex<PoolState>,
    work_cv: Condvar,
    remaining: AtomicUsize,
    done_lock: Mutex<()>,
    done_cv: Condvar,
    workers: AtomicUsize,
}

struct GlobalPool(&'static DetilePool);

impl DetilePool {
    fn global() -> &'static Self {
        static POOL: OnceLock<GlobalPool> = OnceLock::new();
        POOL.get_or_init(|| {
            // Use up to half the logical CPUs (capped at 6 total threads including
            // the caller) to saturate memory bandwidth without starving title threads.
            let total = std::thread::available_parallelism()
                .map_or(1, |n| (n.get() / 2).clamp(1, 6));
            let desired_workers = total.saturating_sub(1);
            let pool: &'static DetilePool = Box::leak(Box::new(DetilePool {
                busy: Mutex::new(()),
                state: Mutex::new(PoolState {
                    generation: 0,
                    job: None,
                    panic: None,
                }),
                work_cv: Condvar::new(),
                remaining: AtomicUsize::new(0),
                done_lock: Mutex::new(()),
                done_cv: Condvar::new(),
                workers: AtomicUsize::new(0),
            }));
            let mut spawned = 0usize;
            for idx in 0..desired_workers {
                if std::thread::Builder::new()
                    .name(format!("victory-detile-{idx}"))
                    .spawn(move || pool.worker_loop(idx + 1))
                    .is_ok()
                {
                    spawned += 1;
                } else {
                    break;
                }
            }
            pool.workers.store(spawned, Ordering::Release);
            GlobalPool(pool)
        })
        .0
    }

    fn worker_loop(&'static self, worker_id: usize) {
        let mut seen_gen = 0u64;
        loop {
            let (func, ctx, start, end, participates) = {
                let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
                while guard.generation == seen_gen {
                    guard = self.work_cv.wait(guard).unwrap_or_else(|e| e.into_inner());
                }
                seen_gen = guard.generation;
                if let Some(ref job) = guard.job {
                    if worker_id < job.active_threads {
                        let (s, e) = split_range(job.total_units, job.active_threads, worker_id);
                        (job.func, job.ctx, s, e, true)
                    } else {
                        (job.func, job.ctx, 0, 0, false)
                    }
                } else {
                    continue;
                }
            };
            if participates {
                struct DoneGuard<'a>(&'a DetilePool);
                impl Drop for DoneGuard<'_> {
                    fn drop(&mut self) {
                        if self.0.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                            let _done = self.0.done_lock.lock().unwrap_or_else(|e| e.into_inner());
                            self.0.done_cv.notify_one();
                        }
                    }
                }
                let _done_guard = DoneGuard(self);
                if start < end
                    && let Err(payload) =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            func(ctx, start, end);
                        }))
                {
                    let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
                    if guard.panic.is_none() {
                        guard.panic = Some(payload);
                    }
                }
            }
        }
    }

    fn run<F: Fn(usize, usize) + Sync>(&'static self, total_units: usize, f: F) {
        let workers = self.workers.load(Ordering::Acquire);
        if workers == 0 || total_units <= 1 {
            f(0, total_units);
            return;
        }
        // If another thread is already using the pool, run inline to avoid
        // blocking or deadlocking under concurrent callers. Recover cleanly
        // if a previous caller panicked while holding `busy`.
        let _busy_guard = match self.busy.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                f(0, total_units);
                return;
            }
        };

        let active_threads = total_units.min(workers + 1);
        let active_workers = active_threads - 1;

        fn trampoline<F: Fn(usize, usize) + Sync>(ctx: *const (), start: usize, end: usize) {
            // SAFETY: `ctx` points to `f` on the caller's stack, which remains
            // alive until `WaitGuard::drop` confirms `remaining == 0`.
            let f = unsafe { &*ctx.cast::<F>() };
            f(start, end);
        }

        struct WaitGuard<'a> {
            pool: &'a DetilePool,
        }

        impl Drop for WaitGuard<'_> {
            fn drop(&mut self) {
                let mut done = self.pool.done_lock.lock().unwrap_or_else(|e| e.into_inner());
                while self.pool.remaining.load(Ordering::Acquire) != 0 {
                    done = self
                        .pool
                        .done_cv
                        .wait(done)
                        .unwrap_or_else(|e| e.into_inner());
                }
                drop(done);
                let mut state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
                state.job = None;
            }
        }

        self.remaining.store(active_workers, Ordering::Release);
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.panic = None;
            state.job = Some(Job {
                func: trampoline::<F>,
                ctx: (&raw const f).cast::<()>(),
                total_units,
                active_threads,
            });
            state.generation = state.generation.wrapping_add(1);
            self.work_cv.notify_all();
        }
        let wait = WaitGuard { pool: self };

        // Caller processes slice 0 while workers process slices 1..active_threads.
        let (s0, e0) = split_range(total_units, active_threads, 0);
        if s0 < e0 {
            f(s0, e0);
        }

        drop(wait);
        let worker_panic = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.panic.take()
        };
        if let Some(payload) = worker_panic {
            std::panic::resume_unwind(payload);
        }
    }
}

#[inline]
fn split_range(total: usize, parts: usize, idx: usize) -> (usize, usize) {
    let base = total / parts;
    let rem = total % parts;
    let start = idx * base + idx.min(rem);
    let len = base + usize::from(idx < rem);
    (start, start + len)
}

// ============================================================================
// Portable 256-byte micro-tile kernels (fixed-width copies, zero bounds checks)
// ============================================================================

#[inline(always)]
unsafe fn copy_bytes<const N: usize>(src: *const u8, dst: *mut u8) {
    // SAFETY: caller guarantees `src[..N]` and `dst[..N]` are valid and non-overlapping.
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, N);
    }
}

/// Thin S, 16B & 8B (`--------y0y1x0x1` / `------x0y0y1x1x2`): 4 rows x 64B.
#[inline(always)]
unsafe fn detile_4x64_s_portable(tiled: *const u8, linear: *mut u8, pitch: usize) {
    // SAFETY: `tiled[..256]` and 4 rows of `linear[r * pitch .. r * pitch + 64]` are valid.
    unsafe {
        for r in 0..4 {
            let dst = linear.add(r * pitch);
            copy_bytes::<16>(tiled.add(r * 16), dst);
            copy_bytes::<16>(tiled.add(64 + r * 16), dst.add(16));
            copy_bytes::<16>(tiled.add(128 + r * 16), dst.add(32));
            copy_bytes::<16>(tiled.add(192 + r * 16), dst.add(48));
        }
    }
}

#[inline(always)]
unsafe fn tile_4x64_s_portable(tiled: *mut u8, linear: *const u8, pitch: usize) {
    // SAFETY: `tiled[..256]` and 4 rows of `linear[r * pitch .. r * pitch + 64]` are valid.
    unsafe {
        for r in 0..4 {
            let src = linear.add(r * pitch);
            copy_bytes::<16>(src, tiled.add(r * 16));
            copy_bytes::<16>(src.add(16), tiled.add(64 + r * 16));
            copy_bytes::<16>(src.add(32), tiled.add(128 + r * 16));
            copy_bytes::<16>(src.add(48), tiled.add(192 + r * 16));
        }
    }
}

/// Thin D, 16B (`--------x0y0x1y1`): 4 rows x 64B.
#[inline(always)]
unsafe fn detile_4x64_d16_portable(tiled: *const u8, linear: *mut u8, pitch: usize) {
    // SAFETY: `tiled[..256]` and 4 rows of `linear[r * pitch .. r * pitch + 64]` are valid.
    unsafe {
        let r0 = linear;
        let r1 = linear.add(pitch);
        let r2 = linear.add(2 * pitch);
        let r3 = linear.add(3 * pitch);
        copy_bytes::<32>(tiled, r0);
        copy_bytes::<32>(tiled.add(64), r0.add(32));
        copy_bytes::<32>(tiled.add(32), r1);
        copy_bytes::<32>(tiled.add(96), r1.add(32));
        copy_bytes::<32>(tiled.add(128), r2);
        copy_bytes::<32>(tiled.add(192), r2.add(32));
        copy_bytes::<32>(tiled.add(160), r3);
        copy_bytes::<32>(tiled.add(224), r3.add(32));
    }
}

#[inline(always)]
unsafe fn tile_4x64_d16_portable(tiled: *mut u8, linear: *const u8, pitch: usize) {
    // SAFETY: `tiled[..256]` and 4 rows of `linear[r * pitch .. r * pitch + 64]` are valid.
    unsafe {
        let r0 = linear;
        let r1 = linear.add(pitch);
        let r2 = linear.add(2 * pitch);
        let r3 = linear.add(3 * pitch);
        copy_bytes::<32>(r0, tiled);
        copy_bytes::<32>(r1, tiled.add(32));
        copy_bytes::<32>(r0.add(32), tiled.add(64));
        copy_bytes::<32>(r1.add(32), tiled.add(96));
        copy_bytes::<32>(r2, tiled.add(128));
        copy_bytes::<32>(r3, tiled.add(160));
        copy_bytes::<32>(r2.add(32), tiled.add(192));
        copy_bytes::<32>(r3.add(32), tiled.add(224));
    }
}

/// Thin D, 8B (`------x0y0x1x2y1`): 4 rows x 64B.
#[inline(always)]
unsafe fn detile_4x64_d8_portable(tiled: *const u8, linear: *mut u8, pitch: usize) {
    // SAFETY: `tiled[..256]` and 4 rows of `linear[r * pitch .. r * pitch + 64]` are valid.
    unsafe {
        let r0 = linear;
        let r1 = linear.add(pitch);
        let r2 = linear.add(2 * pitch);
        let r3 = linear.add(3 * pitch);
        for c in 0..4 {
            copy_bytes::<16>(tiled.add(c * 32), r0.add(c * 16));
            copy_bytes::<16>(tiled.add(c * 32 + 16), r1.add(c * 16));
            copy_bytes::<16>(tiled.add(128 + c * 32), r2.add(c * 16));
            copy_bytes::<16>(tiled.add(128 + c * 32 + 16), r3.add(c * 16));
        }
    }
}

#[inline(always)]
unsafe fn tile_4x64_d8_portable(tiled: *mut u8, linear: *const u8, pitch: usize) {
    // SAFETY: `tiled[..256]` and 4 rows of `linear[r * pitch .. r * pitch + 64]` are valid.
    unsafe {
        let r0 = linear;
        let r1 = linear.add(pitch);
        let r2 = linear.add(2 * pitch);
        let r3 = linear.add(3 * pitch);
        for c in 0..4 {
            copy_bytes::<16>(r0.add(c * 16), tiled.add(c * 32));
            copy_bytes::<16>(r1.add(c * 16), tiled.add(c * 32 + 16));
            copy_bytes::<16>(r2.add(c * 16), tiled.add(128 + c * 32));
            copy_bytes::<16>(r3.add(c * 16), tiled.add(128 + c * 32 + 16));
        }
    }
}

/// Thin S 4B & Thin S/D 2B: 8 rows x 32B.
#[inline(always)]
unsafe fn detile_8x32_s_portable(tiled: *const u8, linear: *mut u8, pitch: usize) {
    // SAFETY: `tiled[..256]` and 8 rows of `linear[r * pitch .. r * pitch + 32]` are valid.
    unsafe {
        for r in 0..8 {
            let dst = linear.add(r * pitch);
            copy_bytes::<16>(tiled.add(r * 16), dst);
            copy_bytes::<16>(tiled.add(128 + r * 16), dst.add(16));
        }
    }
}

#[inline(always)]
unsafe fn tile_8x32_s_portable(tiled: *mut u8, linear: *const u8, pitch: usize) {
    // SAFETY: `tiled[..256]` and 8 rows of `linear[r * pitch .. r * pitch + 32]` are valid.
    unsafe {
        for r in 0..8 {
            let src = linear.add(r * pitch);
            copy_bytes::<16>(src, tiled.add(r * 16));
            copy_bytes::<16>(src.add(16), tiled.add(128 + r * 16));
        }
    }
}

/// Thin D 4B (`----x0x1y0y1x2y2`): 8 rows x 32B.
#[inline(always)]
unsafe fn detile_8x32_d4_portable(tiled: *const u8, linear: *mut u8, pitch: usize) {
    // SAFETY: `tiled[..256]` and 8 rows of `linear[r * pitch .. r * pitch + 32]` are valid.
    unsafe {
        for r in 0..4 {
            let dst0 = linear.add(r * pitch);
            copy_bytes::<16>(tiled.add(r * 16), dst0);
            copy_bytes::<16>(tiled.add(64 + r * 16), dst0.add(16));
            let dst1 = linear.add((r + 4) * pitch);
            copy_bytes::<16>(tiled.add(128 + r * 16), dst1);
            copy_bytes::<16>(tiled.add(192 + r * 16), dst1.add(16));
        }
    }
}

#[inline(always)]
unsafe fn tile_8x32_d4_portable(tiled: *mut u8, linear: *const u8, pitch: usize) {
    // SAFETY: `tiled[..256]` and 8 rows of `linear[r * pitch .. r * pitch + 32]` are valid.
    unsafe {
        for r in 0..4 {
            let src0 = linear.add(r * pitch);
            copy_bytes::<16>(src0, tiled.add(r * 16));
            copy_bytes::<16>(src0.add(16), tiled.add(64 + r * 16));
            let src1 = linear.add((r + 4) * pitch);
            copy_bytes::<16>(src1, tiled.add(128 + r * 16));
            copy_bytes::<16>(src1.add(16), tiled.add(192 + r * 16));
        }
    }
}

/// Thin S 1B (`x0x1x2x3y0y1y2y3`): 16 rows x 16B.
#[inline(always)]
unsafe fn detile_16x16_s1_portable(tiled: *const u8, linear: *mut u8, pitch: usize) {
    // SAFETY: `tiled[..256]` and 16 rows of `linear[r * pitch .. r * pitch + 16]` are valid.
    unsafe {
        for r in 0..16 {
            copy_bytes::<16>(tiled.add(r * 16), linear.add(r * pitch));
        }
    }
}

#[inline(always)]
unsafe fn tile_16x16_s1_portable(tiled: *mut u8, linear: *const u8, pitch: usize) {
    // SAFETY: `tiled[..256]` and 16 rows of `linear[r * pitch .. r * pitch + 16]` are valid.
    unsafe {
        for r in 0..16 {
            copy_bytes::<16>(linear.add(r * pitch), tiled.add(r * 16));
        }
    }
}

/// Thin D 1B (`x0x1x2y1y0y2x3y3`): 16 rows x 16B.
#[inline(always)]
unsafe fn detile_16x16_d1_portable(tiled: *const u8, linear: *mut u8, pitch: usize) {
    const QUAD_OFFS: [usize; 4] = [0, 32, 128, 160];
    // SAFETY: `tiled[..256]` and 16 rows of `linear[r * pitch .. r * pitch + 16]` are valid.
    unsafe {
        for (q, &base) in QUAD_OFFS.iter().enumerate() {
            let y0 = q * 4;
            let d0 = linear.add(y0 * pitch);
            let d1 = linear.add((y0 + 1) * pitch);
            let d2 = linear.add((y0 + 2) * pitch);
            let d3 = linear.add((y0 + 3) * pitch);
            copy_bytes::<8>(tiled.add(base), d0);
            copy_bytes::<8>(tiled.add(base + 64), d0.add(8));
            copy_bytes::<8>(tiled.add(base + 16), d1);
            copy_bytes::<8>(tiled.add(base + 80), d1.add(8));
            copy_bytes::<8>(tiled.add(base + 8), d2);
            copy_bytes::<8>(tiled.add(base + 72), d2.add(8));
            copy_bytes::<8>(tiled.add(base + 24), d3);
            copy_bytes::<8>(tiled.add(base + 88), d3.add(8));
        }
    }
}

#[inline(always)]
unsafe fn tile_16x16_d1_portable(tiled: *mut u8, linear: *const u8, pitch: usize) {
    const QUAD_OFFS: [usize; 4] = [0, 32, 128, 160];
    // SAFETY: `tiled[..256]` and 16 rows of `linear[r * pitch .. r * pitch + 16]` are valid.
    unsafe {
        for (q, &base) in QUAD_OFFS.iter().enumerate() {
            let y0 = q * 4;
            let s0 = linear.add(y0 * pitch);
            let s1 = linear.add((y0 + 1) * pitch);
            let s2 = linear.add((y0 + 2) * pitch);
            let s3 = linear.add((y0 + 3) * pitch);
            copy_bytes::<8>(s0, tiled.add(base));
            copy_bytes::<8>(s0.add(8), tiled.add(base + 64));
            copy_bytes::<8>(s1, tiled.add(base + 16));
            copy_bytes::<8>(s1.add(8), tiled.add(base + 80));
            copy_bytes::<8>(s2, tiled.add(base + 8));
            copy_bytes::<8>(s2.add(8), tiled.add(base + 72));
            copy_bytes::<8>(s3, tiled.add(base + 24));
            copy_bytes::<8>(s3.add(8), tiled.add(base + 88));
        }
    }
}

// ============================================================================
// x86_64 AVX2 / SSE2 256-byte micro-tile kernels (unaligned load/store)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn detile_4x64_s_avx2(tiled: *const u8, linear: *mut u8, pitch: usize) {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 enabled by target_feature; unaligned 32-byte accesses inside
    // caller-verified `tiled[..256]` and 4 rows of `linear[r * pitch .. +64]`.
    unsafe {
        let c0_01 = _mm256_loadu_si256(tiled.cast());
        let c0_23 = _mm256_loadu_si256(tiled.add(32).cast());
        let c1_01 = _mm256_loadu_si256(tiled.add(64).cast());
        let c1_23 = _mm256_loadu_si256(tiled.add(96).cast());
        let c2_01 = _mm256_loadu_si256(tiled.add(128).cast());
        let c2_23 = _mm256_loadu_si256(tiled.add(160).cast());
        let c3_01 = _mm256_loadu_si256(tiled.add(192).cast());
        let c3_23 = _mm256_loadu_si256(tiled.add(224).cast());

        let r0_01 = _mm256_permute2x128_si256(c0_01, c1_01, 0x20);
        let r1_01 = _mm256_permute2x128_si256(c0_01, c1_01, 0x31);
        let r0_23 = _mm256_permute2x128_si256(c2_01, c3_01, 0x20);
        let r1_23 = _mm256_permute2x128_si256(c2_01, c3_01, 0x31);

        let r2_01 = _mm256_permute2x128_si256(c0_23, c1_23, 0x20);
        let r3_01 = _mm256_permute2x128_si256(c0_23, c1_23, 0x31);
        let r2_23 = _mm256_permute2x128_si256(c2_23, c3_23, 0x20);
        let r3_23 = _mm256_permute2x128_si256(c2_23, c3_23, 0x31);

        let r0 = linear;
        let r1 = linear.add(pitch);
        let r2 = linear.add(2 * pitch);
        let r3 = linear.add(3 * pitch);

        _mm256_storeu_si256(r0.cast(), r0_01);
        _mm256_storeu_si256(r0.add(32).cast(), r0_23);
        _mm256_storeu_si256(r1.cast(), r1_01);
        _mm256_storeu_si256(r1.add(32).cast(), r1_23);
        _mm256_storeu_si256(r2.cast(), r2_01);
        _mm256_storeu_si256(r2.add(32).cast(), r2_23);
        _mm256_storeu_si256(r3.cast(), r3_01);
        _mm256_storeu_si256(r3.add(32).cast(), r3_23);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tile_4x64_s_avx2(tiled: *mut u8, linear: *const u8, pitch: usize) {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 enabled by target_feature; unaligned 32-byte accesses inside
    // caller-verified `tiled[..256]` and 4 rows of `linear[r * pitch .. +64]`.
    unsafe {
        let r0 = linear;
        let r1 = linear.add(pitch);
        let r2 = linear.add(2 * pitch);
        let r3 = linear.add(3 * pitch);

        let r0_01 = _mm256_loadu_si256(r0.cast());
        let r0_23 = _mm256_loadu_si256(r0.add(32).cast());
        let r1_01 = _mm256_loadu_si256(r1.cast());
        let r1_23 = _mm256_loadu_si256(r1.add(32).cast());
        let r2_01 = _mm256_loadu_si256(r2.cast());
        let r2_23 = _mm256_loadu_si256(r2.add(32).cast());
        let r3_01 = _mm256_loadu_si256(r3.cast());
        let r3_23 = _mm256_loadu_si256(r3.add(32).cast());

        let c0_01 = _mm256_permute2x128_si256(r0_01, r1_01, 0x20);
        let c1_01 = _mm256_permute2x128_si256(r0_01, r1_01, 0x31);
        let c0_23 = _mm256_permute2x128_si256(r2_01, r3_01, 0x20);
        let c1_23 = _mm256_permute2x128_si256(r2_01, r3_01, 0x31);

        let c2_01 = _mm256_permute2x128_si256(r0_23, r1_23, 0x20);
        let c3_01 = _mm256_permute2x128_si256(r0_23, r1_23, 0x31);
        let c2_23 = _mm256_permute2x128_si256(r2_23, r3_23, 0x20);
        let c3_23 = _mm256_permute2x128_si256(r2_23, r3_23, 0x31);

        _mm256_storeu_si256(tiled.cast(), c0_01);
        _mm256_storeu_si256(tiled.add(32).cast(), c0_23);
        _mm256_storeu_si256(tiled.add(64).cast(), c1_01);
        _mm256_storeu_si256(tiled.add(96).cast(), c1_23);
        _mm256_storeu_si256(tiled.add(128).cast(), c2_01);
        _mm256_storeu_si256(tiled.add(160).cast(), c2_23);
        _mm256_storeu_si256(tiled.add(192).cast(), c3_01);
        _mm256_storeu_si256(tiled.add(224).cast(), c3_23);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn detile_4x64_d16_avx2(tiled: *const u8, linear: *mut u8, pitch: usize) {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 enabled by target_feature; unaligned 32-byte accesses inside
    // caller-verified `tiled[..256]` and 4 rows of `linear[r * pitch .. +64]`.
    unsafe {
        let t0 = _mm256_loadu_si256(tiled.cast());
        let t1 = _mm256_loadu_si256(tiled.add(32).cast());
        let t2 = _mm256_loadu_si256(tiled.add(64).cast());
        let t3 = _mm256_loadu_si256(tiled.add(96).cast());
        let t4 = _mm256_loadu_si256(tiled.add(128).cast());
        let t5 = _mm256_loadu_si256(tiled.add(160).cast());
        let t6 = _mm256_loadu_si256(tiled.add(192).cast());
        let t7 = _mm256_loadu_si256(tiled.add(224).cast());

        let r0 = linear;
        let r1 = linear.add(pitch);
        let r2 = linear.add(2 * pitch);
        let r3 = linear.add(3 * pitch);

        _mm256_storeu_si256(r0.cast(), t0);
        _mm256_storeu_si256(r0.add(32).cast(), t2);
        _mm256_storeu_si256(r1.cast(), t1);
        _mm256_storeu_si256(r1.add(32).cast(), t3);
        _mm256_storeu_si256(r2.cast(), t4);
        _mm256_storeu_si256(r2.add(32).cast(), t6);
        _mm256_storeu_si256(r3.cast(), t5);
        _mm256_storeu_si256(r3.add(32).cast(), t7);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tile_4x64_d16_avx2(tiled: *mut u8, linear: *const u8, pitch: usize) {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 enabled by target_feature; unaligned 32-byte accesses inside
    // caller-verified `tiled[..256]` and 4 rows of `linear[r * pitch .. +64]`.
    unsafe {
        let r0 = linear;
        let r1 = linear.add(pitch);
        let r2 = linear.add(2 * pitch);
        let r3 = linear.add(3 * pitch);

        let a0 = _mm256_loadu_si256(r0.cast());
        let a1 = _mm256_loadu_si256(r0.add(32).cast());
        let b0 = _mm256_loadu_si256(r1.cast());
        let b1 = _mm256_loadu_si256(r1.add(32).cast());
        let c0 = _mm256_loadu_si256(r2.cast());
        let c1 = _mm256_loadu_si256(r2.add(32).cast());
        let d0 = _mm256_loadu_si256(r3.cast());
        let d1 = _mm256_loadu_si256(r3.add(32).cast());

        _mm256_storeu_si256(tiled.cast(), a0);
        _mm256_storeu_si256(tiled.add(32).cast(), b0);
        _mm256_storeu_si256(tiled.add(64).cast(), a1);
        _mm256_storeu_si256(tiled.add(96).cast(), b1);
        _mm256_storeu_si256(tiled.add(128).cast(), c0);
        _mm256_storeu_si256(tiled.add(160).cast(), d0);
        _mm256_storeu_si256(tiled.add(192).cast(), c1);
        _mm256_storeu_si256(tiled.add(224).cast(), d1);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn detile_4x64_d8_avx2(tiled: *const u8, linear: *mut u8, pitch: usize) {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 enabled by target_feature; unaligned 32-byte accesses inside
    // caller-verified `tiled[..256]` and 4 rows of `linear[r * pitch .. +64]`.
    unsafe {
        let t0 = _mm256_loadu_si256(tiled.cast());
        let t1 = _mm256_loadu_si256(tiled.add(32).cast());
        let t2 = _mm256_loadu_si256(tiled.add(64).cast());
        let t3 = _mm256_loadu_si256(tiled.add(96).cast());
        let t4 = _mm256_loadu_si256(tiled.add(128).cast());
        let t5 = _mm256_loadu_si256(tiled.add(160).cast());
        let t6 = _mm256_loadu_si256(tiled.add(192).cast());
        let t7 = _mm256_loadu_si256(tiled.add(224).cast());

        let r0 = linear;
        let r1 = linear.add(pitch);
        let r2 = linear.add(2 * pitch);
        let r3 = linear.add(3 * pitch);

        _mm256_storeu_si256(r0.cast(), _mm256_permute2x128_si256(t0, t1, 0x20));
        _mm256_storeu_si256(r1.cast(), _mm256_permute2x128_si256(t0, t1, 0x31));
        _mm256_storeu_si256(r0.add(32).cast(), _mm256_permute2x128_si256(t2, t3, 0x20));
        _mm256_storeu_si256(r1.add(32).cast(), _mm256_permute2x128_si256(t2, t3, 0x31));

        _mm256_storeu_si256(r2.cast(), _mm256_permute2x128_si256(t4, t5, 0x20));
        _mm256_storeu_si256(r3.cast(), _mm256_permute2x128_si256(t4, t5, 0x31));
        _mm256_storeu_si256(r2.add(32).cast(), _mm256_permute2x128_si256(t6, t7, 0x20));
        _mm256_storeu_si256(r3.add(32).cast(), _mm256_permute2x128_si256(t6, t7, 0x31));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tile_4x64_d8_avx2(tiled: *mut u8, linear: *const u8, pitch: usize) {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 enabled by target_feature; unaligned 32-byte accesses inside
    // caller-verified `tiled[..256]` and 4 rows of `linear[r * pitch .. +64]`.
    unsafe {
        let r0 = linear;
        let r1 = linear.add(pitch);
        let r2 = linear.add(2 * pitch);
        let r3 = linear.add(3 * pitch);

        let a0 = _mm256_loadu_si256(r0.cast());
        let a1 = _mm256_loadu_si256(r0.add(32).cast());
        let b0 = _mm256_loadu_si256(r1.cast());
        let b1 = _mm256_loadu_si256(r1.add(32).cast());
        let c0 = _mm256_loadu_si256(r2.cast());
        let c1 = _mm256_loadu_si256(r2.add(32).cast());
        let d0 = _mm256_loadu_si256(r3.cast());
        let d1 = _mm256_loadu_si256(r3.add(32).cast());

        _mm256_storeu_si256(tiled.cast(), _mm256_permute2x128_si256(a0, b0, 0x20));
        _mm256_storeu_si256(tiled.add(32).cast(), _mm256_permute2x128_si256(a0, b0, 0x31));
        _mm256_storeu_si256(tiled.add(64).cast(), _mm256_permute2x128_si256(a1, b1, 0x20));
        _mm256_storeu_si256(tiled.add(96).cast(), _mm256_permute2x128_si256(a1, b1, 0x31));

        _mm256_storeu_si256(tiled.add(128).cast(), _mm256_permute2x128_si256(c0, d0, 0x20));
        _mm256_storeu_si256(tiled.add(160).cast(), _mm256_permute2x128_si256(c0, d0, 0x31));
        _mm256_storeu_si256(tiled.add(192).cast(), _mm256_permute2x128_si256(c1, d1, 0x20));
        _mm256_storeu_si256(tiled.add(224).cast(), _mm256_permute2x128_si256(c1, d1, 0x31));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn detile_8x32_s_avx2(tiled: *const u8, linear: *mut u8, pitch: usize) {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 enabled by target_feature; unaligned 32-byte accesses inside
    // caller-verified `tiled[..256]` and 8 rows of `linear[r * pitch .. +32]`.
    unsafe {
        for k in 0..4 {
            let a = _mm256_loadu_si256(tiled.add(32 * k).cast());
            let b = _mm256_loadu_si256(tiled.add(128 + 32 * k).cast());
            _mm256_storeu_si256(
                linear.add((2 * k) * pitch).cast(),
                _mm256_permute2x128_si256(a, b, 0x20),
            );
            _mm256_storeu_si256(
                linear.add((2 * k + 1) * pitch).cast(),
                _mm256_permute2x128_si256(a, b, 0x31),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tile_8x32_s_avx2(tiled: *mut u8, linear: *const u8, pitch: usize) {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 enabled by target_feature; unaligned 32-byte accesses inside
    // caller-verified `tiled[..256]` and 8 rows of `linear[r * pitch .. +32]`.
    unsafe {
        for k in 0..4 {
            let r0 = _mm256_loadu_si256(linear.add((2 * k) * pitch).cast());
            let r1 = _mm256_loadu_si256(linear.add((2 * k + 1) * pitch).cast());
            _mm256_storeu_si256(
                tiled.add(32 * k).cast(),
                _mm256_permute2x128_si256(r0, r1, 0x20),
            );
            _mm256_storeu_si256(
                tiled.add(128 + 32 * k).cast(),
                _mm256_permute2x128_si256(r0, r1, 0x31),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn detile_8x32_d4_avx2(tiled: *const u8, linear: *mut u8, pitch: usize) {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 enabled by target_feature; unaligned 32-byte accesses inside
    // caller-verified `tiled[..256]` and 8 rows of `linear[r * pitch .. +32]`.
    unsafe {
        let a0 = _mm256_loadu_si256(tiled.cast());
        let b0 = _mm256_loadu_si256(tiled.add(64).cast());
        let a1 = _mm256_loadu_si256(tiled.add(32).cast());
        let b1 = _mm256_loadu_si256(tiled.add(96).cast());
        let a2 = _mm256_loadu_si256(tiled.add(128).cast());
        let b2 = _mm256_loadu_si256(tiled.add(192).cast());
        let a3 = _mm256_loadu_si256(tiled.add(160).cast());
        let b3 = _mm256_loadu_si256(tiled.add(224).cast());

        _mm256_storeu_si256(linear.cast(), _mm256_permute2x128_si256(a0, b0, 0x20));
        _mm256_storeu_si256(linear.add(pitch).cast(), _mm256_permute2x128_si256(a0, b0, 0x31));
        _mm256_storeu_si256(linear.add(2 * pitch).cast(), _mm256_permute2x128_si256(a1, b1, 0x20));
        _mm256_storeu_si256(linear.add(3 * pitch).cast(), _mm256_permute2x128_si256(a1, b1, 0x31));
        _mm256_storeu_si256(linear.add(4 * pitch).cast(), _mm256_permute2x128_si256(a2, b2, 0x20));
        _mm256_storeu_si256(linear.add(5 * pitch).cast(), _mm256_permute2x128_si256(a2, b2, 0x31));
        _mm256_storeu_si256(linear.add(6 * pitch).cast(), _mm256_permute2x128_si256(a3, b3, 0x20));
        _mm256_storeu_si256(linear.add(7 * pitch).cast(), _mm256_permute2x128_si256(a3, b3, 0x31));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tile_8x32_d4_avx2(tiled: *mut u8, linear: *const u8, pitch: usize) {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 enabled by target_feature; unaligned 32-byte accesses inside
    // caller-verified `tiled[..256]` and 8 rows of `linear[r * pitch .. +32]`.
    unsafe {
        let r0 = _mm256_loadu_si256(linear.cast());
        let r1 = _mm256_loadu_si256(linear.add(pitch).cast());
        let r2 = _mm256_loadu_si256(linear.add(2 * pitch).cast());
        let r3 = _mm256_loadu_si256(linear.add(3 * pitch).cast());
        let r4 = _mm256_loadu_si256(linear.add(4 * pitch).cast());
        let r5 = _mm256_loadu_si256(linear.add(5 * pitch).cast());
        let r6 = _mm256_loadu_si256(linear.add(6 * pitch).cast());
        let r7 = _mm256_loadu_si256(linear.add(7 * pitch).cast());

        _mm256_storeu_si256(tiled.cast(), _mm256_permute2x128_si256(r0, r1, 0x20));
        _mm256_storeu_si256(tiled.add(64).cast(), _mm256_permute2x128_si256(r0, r1, 0x31));
        _mm256_storeu_si256(tiled.add(32).cast(), _mm256_permute2x128_si256(r2, r3, 0x20));
        _mm256_storeu_si256(tiled.add(96).cast(), _mm256_permute2x128_si256(r2, r3, 0x31));
        _mm256_storeu_si256(tiled.add(128).cast(), _mm256_permute2x128_si256(r4, r5, 0x20));
        _mm256_storeu_si256(tiled.add(192).cast(), _mm256_permute2x128_si256(r4, r5, 0x31));
        _mm256_storeu_si256(tiled.add(160).cast(), _mm256_permute2x128_si256(r6, r7, 0x20));
        _mm256_storeu_si256(tiled.add(224).cast(), _mm256_permute2x128_si256(r6, r7, 0x31));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn detile_16x16_s1_sse2(tiled: *const u8, linear: *mut u8, pitch: usize) {
    use std::arch::x86_64::*;
    // SAFETY: SSE2/AVX2 enabled; unaligned 16-byte accesses inside caller-verified bounds.
    unsafe {
        for r in 0..16 {
            let v = _mm_loadu_si128(tiled.add(r * 16).cast());
            _mm_storeu_si128(linear.add(r * pitch).cast(), v);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tile_16x16_s1_sse2(tiled: *mut u8, linear: *const u8, pitch: usize) {
    use std::arch::x86_64::*;
    // SAFETY: SSE2/AVX2 enabled; unaligned 16-byte accesses inside caller-verified bounds.
    unsafe {
        for r in 0..16 {
            let v = _mm_loadu_si128(linear.add(r * pitch).cast());
            _mm_storeu_si128(tiled.add(r * 16).cast(), v);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn detile_16x16_d1_sse2(tiled: *const u8, linear: *mut u8, pitch: usize) {
    use std::arch::x86_64::*;
    const QUAD_OFFS: [usize; 4] = [0, 32, 128, 160];
    // SAFETY: SSE2/AVX2 enabled; unaligned 16-byte accesses inside caller-verified bounds.
    unsafe {
        for (q, &base) in QUAD_OFFS.iter().enumerate() {
            let y0 = q * 4;
            let a0 = _mm_loadu_si128(tiled.add(base).cast());
            let b0 = _mm_loadu_si128(tiled.add(base + 64).cast());
            let a1 = _mm_loadu_si128(tiled.add(base + 16).cast());
            let b1 = _mm_loadu_si128(tiled.add(base + 80).cast());

            _mm_storeu_si128(linear.add(y0 * pitch).cast(), _mm_unpacklo_epi64(a0, b0));
            _mm_storeu_si128(linear.add((y0 + 2) * pitch).cast(), _mm_unpackhi_epi64(a0, b0));
            _mm_storeu_si128(linear.add((y0 + 1) * pitch).cast(), _mm_unpacklo_epi64(a1, b1));
            _mm_storeu_si128(linear.add((y0 + 3) * pitch).cast(), _mm_unpackhi_epi64(a1, b1));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tile_16x16_d1_sse2(tiled: *mut u8, linear: *const u8, pitch: usize) {
    use std::arch::x86_64::*;
    const QUAD_OFFS: [usize; 4] = [0, 32, 128, 160];
    // SAFETY: SSE2/AVX2 enabled; unaligned 16-byte accesses inside caller-verified bounds.
    unsafe {
        for (q, &base) in QUAD_OFFS.iter().enumerate() {
            let y0 = q * 4;
            let r0 = _mm_loadu_si128(linear.add(y0 * pitch).cast());
            let r1 = _mm_loadu_si128(linear.add((y0 + 1) * pitch).cast());
            let r2 = _mm_loadu_si128(linear.add((y0 + 2) * pitch).cast());
            let r3 = _mm_loadu_si128(linear.add((y0 + 3) * pitch).cast());

            _mm_storeu_si128(tiled.add(base).cast(), _mm_unpacklo_epi64(r0, r2));
            _mm_storeu_si128(tiled.add(base + 64).cast(), _mm_unpackhi_epi64(r0, r2));
            _mm_storeu_si128(tiled.add(base + 16).cast(), _mm_unpacklo_epi64(r1, r3));
            _mm_storeu_si128(tiled.add(base + 80).cast(), _mm_unpackhi_epi64(r1, r3));
        }
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;
    use std::collections::HashSet;

    /// Verifies that a panic inside a worker thread propagates to the caller,
    /// keeps the worker alive (no deadlock on subsequent calls), and that a
    /// panic on the caller thread does not poison `busy` into single-threaded
    /// fallback.
    #[test]
    fn detile_pool_recovers_from_worker_and_caller_panics() {
        let pool = DetilePool::global();
        let workers = pool.workers.load(Ordering::Acquire);
        if workers == 0 {
            return;
        }
        let total = workers + 1;

        // 1. Worker panic propagates cleanly and leaves all workers alive.
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(total, |s, _e| {
                if s > 0 {
                    panic!("simulated worker panic");
                }
            });
        }));
        assert!(res.is_err(), "worker panic must propagate to caller");

        // 2. Caller panic unwinds cleanly without poisoning `busy`.
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(total, |s, _e| {
                if s == 0 {
                    panic!("simulated caller panic");
                }
            });
        }));
        assert!(res.is_err(), "caller panic must propagate");

        // 3. Subsequent dispatch still succeeds across multiple threads (no
        //    hang and no permanent fallback to single-threaded execution).
        let seen = Mutex::new(HashSet::new());
        let sum = AtomicUsize::new(0);
        pool.run(total * 4, |s, e| {
            seen.lock().unwrap().insert(std::thread::current().id());
            for i in s..e {
                sum.fetch_add(i + 1, Ordering::Relaxed);
            }
        });
        let n = total * 4;
        assert_eq!(sum.load(Ordering::Relaxed), n * (n + 1) / 2);
        assert!(
            seen.lock().unwrap().len() >= 2,
            "pool should still execute across worker threads after recovery"
        );
    }

    #[test]
    fn mutated_public_layout_fields_never_enter_unsound_fast_path() {
        use crate::texture::{Desc, Tail};

        let base_desc = Desc {
            dimension: Dimension::Tex2d,
            width: 512,
            height: 512,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_4KB_D,
        };

        let mut pitch_zero = Layout::new(&base_desc).unwrap();
        pitch_zero.mips[0].pitch = 0;
        let row = 512 * 4;
        let lin_len = row * 512;
        let plan = validate_plan(
            &pitch_zero,
            pitch_zero.size as usize,
            0,
            0,
            lin_len,
            row,
            0,
            Direction::Tile,
        )
        .unwrap();
        assert!(
            prepare_thin_geometry(&plan, pitch_zero.size as usize, lin_len, Direction::Tile, true)
                .is_none(),
            "m.pitch = 0 with last_by > 0 must reject thin fast path"
        );

        let small_desc = Desc {
            width: 64,
            height: 64,
            ..base_desc
        };
        let row3 = 64 * 4;
        let plane3 = row3 * 64;
        let mut wrap_slice = Layout::new(&small_desc).unwrap();
        wrap_slice.slices = 3;
        wrap_slice.mips[0].depth = 3;
        wrap_slice.slice_size = 1u64 << 63;
        wrap_slice.size = 65536;
        let plan3 = validate_plan(
            &wrap_slice,
            65536,
            0,
            0,
            plane3 * 3,
            row3,
            plane3,
            Direction::Detile,
        )
        .unwrap();
        assert!(
            prepare_thin_geometry(&plan3, 65536, plane3 * 3, Direction::Detile, true).is_none(),
            "wrapping slice_size must reject thin fast path"
        );

        let mut wrap_off = Layout::new(&small_desc).unwrap();
        wrap_off.slices = 2;
        wrap_off.mips[0].depth = 2;
        wrap_off.slice_size = 1;
        wrap_off.mips[0].offset = u64::MAX;
        wrap_off.size = 65536;
        let plan4 = validate_plan(
            &wrap_off,
            65536,
            0,
            0,
            plane3 * 2,
            row3,
            plane3,
            Direction::Detile,
        )
        .unwrap();
        assert!(
            prepare_thin_geometry(&plan4, 65536, plane3 * 2, Direction::Detile, true).is_none(),
            "wrapping m.offset must reject thin fast path"
        );

        let mut bad_block = Layout::new(&base_desc).unwrap();
        bad_block.block = [8, 128, 1];
        let plan_b = validate_plan(
            &bad_block,
            bad_block.size as usize,
            0,
            0,
            lin_len,
            row,
            0,
            Direction::Tile,
        )
        .unwrap();
        assert!(
            prepare_thin_geometry(&plan_b, bad_block.size as usize, lin_len, Direction::Tile, true)
                .is_none(),
            "mutated layout.block with colliding micro-tiles must reject thin fast path"
        );

        let mut bad_tail = Layout::new(&base_desc).unwrap();
        bad_tail.mips[0].tail = Some(Tail {
            offset: 0,
            x: 0,
            y: 0,
        });
        let plan_t = validate_plan(
            &bad_tail,
            bad_tail.size as usize,
            0,
            0,
            lin_len,
            row,
            0,
            Direction::Detile,
        )
        .unwrap();
        assert!(
            prepare_thin_geometry(&plan_t, bad_tail.size as usize, lin_len, Direction::Detile, true)
                .is_none(),
            "oversized mutated tail mip must reject thin fast path"
        );

        let tiled = vec![0u8; 65536];
        let mut linear = vec![0u8; 65536];

        let mut zero_h = Layout::new(&base_desc).unwrap();
        zero_h.mips[0].height = 0;
        assert_eq!(zero_h.detile(&tiled, 0, 0, &mut linear, row, 0), Err(Error::Invalid));

        let mut zero_d = Layout::new(&base_desc).unwrap();
        zero_d.mips[0].depth = 0;
        assert_eq!(zero_d.detile(&tiled, 0, 0, &mut linear, row, 0), Err(Error::Invalid));

        let mut zero_w = Layout::new(&base_desc).unwrap();
        zero_w.mips[0].width = 0;
        assert_eq!(zero_w.detile(&tiled, 0, 0, &mut linear, row, 0), Err(Error::Invalid));

        let mut bad_bpe = Layout::new(&base_desc).unwrap();
        bad_bpe.desc.element_bytes = 3;
        assert_eq!(bad_bpe.detile(&tiled, 0, 0, &mut linear, row, 0), Err(Error::Invalid));

        let mut huge_bpe = Layout::new(&base_desc).unwrap();
        huge_bpe.desc.element_bytes = 32;
        assert_eq!(huge_bpe.detile(&tiled, 0, 0, &mut linear, row, 0), Err(Error::Invalid));

        let mut bad_depth = Layout::new(&small_desc).unwrap();
        bad_depth.mips[0].depth = 10;
        assert_eq!(
            bad_depth.detile(&tiled, 0, 0, &mut linear, row3, plane3),
            Err(Error::Invalid)
        );
    }
}
