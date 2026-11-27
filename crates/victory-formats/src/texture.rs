//! Texture layouts of the console GPU (RDNA2, GFX10.3): a texture's size and
//! alignment, where each mip and array slice starts, the address of each
//! element, and copies between tiled memory and linear rows.
//!
//! Units: an element is a texel, or a 4x4 block of a block-compressed
//! format. Mips of one array slice (or one depth slab of a thick 3D texture)
//! are stored together, smallest first: the mip tail (the small mips sharing
//! one block) at offset 0, mip 0 last.
//!
//! The algorithm follows AMD's address library (addrlib), as shipped in Mesa
//! under `src/amd/addrlib`: block dimensions, mip placement and mip tails
//! from `gfx10addrlib.cpp` and `addrlib2.cpp`
//! (`ComputeSurfaceInfoMacroTiled`, `ComputeThinBlockDimension`,
//! `GetMipTailDim`, `GetMaxNumMipsInTail`, `HwlComputeSurfaceInfoLinear`,
//! `ComputeSurfaceAddrFromCoordMacroTiled`), and the 256-byte micro tile
//! patterns in `MICRO` from `gfx10SwizzlePattern.h` (the RB+ tables, as on
//! RDNA2). addrlib carries this notice:
//!
//! Copyright (C) 2007-2026 Advanced Micro Devices, Inc. All rights reserved.
//! SPDX-License-Identifier: MIT
//!
//! Permission is hereby granted, free of charge, to any person obtaining a
//! copy of this software and associated documentation files (the
//! "Software"), to deal in the Software without restriction, including
//! without limitation the rights to use, copy, modify, merge, publish,
//! distribute, sublicense, and/or sell copies of the Software, and to permit
//! persons to whom the Software is furnished to do so, subject to the
//! following conditions: the above copyright notice and this permission
//! notice shall be included in all copies or substantial portions of the
//! Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//! EXPRESS OR IMPLIED.

use std::sync::OnceLock;

/// A swizzle mode, by its hardware number (addrlib's `ADDR_SW_*` order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Swizzle(pub u32);

impl Swizzle {
    pub const LINEAR: Swizzle = Swizzle(0);
    pub const SW_256B_S: Swizzle = Swizzle(1);
    pub const SW_256B_D: Swizzle = Swizzle(2);
    pub const SW_4KB_S: Swizzle = Swizzle(5);
    pub const SW_4KB_D: Swizzle = Swizzle(6);
    pub const SW_64KB_S: Swizzle = Swizzle(9);
    pub const SW_64KB_D: Swizzle = Swizzle(10);
    pub const SW_64KB_S_T: Swizzle = Swizzle(17);
    pub const SW_64KB_D_T: Swizzle = Swizzle(18);
    pub const SW_4KB_S_X: Swizzle = Swizzle(21);
    pub const SW_4KB_D_X: Swizzle = Swizzle(22);
    pub const SW_4KB_R_X: Swizzle = Swizzle(23);
    pub const SW_64KB_Z_X: Swizzle = Swizzle(24);
    pub const SW_64KB_S_X: Swizzle = Swizzle(25);
    pub const SW_64KB_D_X: Swizzle = Swizzle(26);
    pub const SW_64KB_R_X: Swizzle = Swizzle(27);

    /// log2 of the block size in bytes (0 for linear); `None` for modes
    /// GFX10 does not have.
    pub fn block_log2(self) -> Option<u32> {
        Some(match self.0 {
            0 => 0,
            1 | 2 => 8,
            5 | 6 | 21..=23 => 12,
            9 | 10 | 17 | 18 | 24..=27 => 16,
            _ => return None,
        })
    }

    fn standard(self) -> bool {
        matches!(self.0, 1 | 5 | 9 | 17 | 21 | 25)
    }

    pub(crate) fn display(self) -> bool {
        matches!(self.0, 2 | 6 | 10 | 18 | 22 | 26)
    }

    fn z_order(self) -> bool {
        self.0 == 24
    }

    /// Pipe and bank XOR modes (`_T`, `_X`): their addresses depend on the
    /// GPU's pipe configuration.
    pub fn xor(self) -> bool {
        self.0 >= 16
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    Tex1d,
    Tex2d,
    Tex3d,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The description makes no texture (zero size, bad element size, a
    /// mode this dimension cannot use, too many mips).
    Invalid,
    /// A valid layout whose addressing is not implemented here.
    Unsupported,
    /// A buffer is smaller than the layout or the rows need.
    TooSmall,
}

/// One plane of a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Desc {
    pub dimension: Dimension,
    /// In texels.
    pub width: u32,
    pub height: u32,
    /// Depth of a 3D texture, else the array size.
    pub depth: u32,
    pub mips: u32,
    /// 1, 2, 4, 8 or 16.
    pub element_bytes: u32,
    /// Texels per element side: 1, or 4 for block compression.
    pub block: u32,
    pub samples: u32,
    pub swizzle: Swizzle,
}

/// Elements along one side of mip `mip`, as data and copies see it: texels
/// halve rounding down (to at least 1), then round up to whole elements.
fn dim(texels: u32, mip: u32, block: u32) -> u32 {
    (texels >> mip).max(1).div_ceil(block)
}

/// Elements along one side of mip `mip` for placement (padding, tail,
/// offsets): addrlib halves mip 0's elements rounding up, which can be one
/// more than [`dim`] for odd sizes.
fn padded_dim(texels: u32, mip: u32, block: u32) -> u32 {
    texels.max(1).div_ceil(block).div_ceil(1 << mip)
}

/// Where one mip lives inside a slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mip {
    /// Size in elements (depth 1 unless the texture is 3D).
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Padded size in elements: whole blocks, or the tail's share of one.
    pub pitch: u32,
    pub rows: u32,
    pub depth_pitch: u32,
    /// Byte offset of its first block in the slice (0 in the tail).
    pub offset: u64,
    /// In the mip tail: byte offset inside the tail block and the element
    /// position the mip starts at.
    pub tail: Option<Tail>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tail {
    pub offset: u32,
    pub x: u32,
    pub y: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub desc: Desc,
    /// Block dimensions in elements (linear: the pitch alignment, 1, 1).
    pub block: [u32; 3],
    /// 3D blocks span `block[2]` depth slices (GFX10 3D `S` and `D` modes).
    pub thick: bool,
    /// Array slices, depth slabs of `block[2]` slices (thick), or depth
    /// slices (thin and linear 3D).
    pub slices: u32,
    /// Bytes of one slice: its whole mip chain.
    pub slice_size: u64,
    pub size: u64,
    pub alignment: u64,
    /// The first mip in the tail; `mips.len()` without one.
    pub first_tail_mip: u32,
    pub mips: Vec<Mip>,
    /// Byte offsets inside a block for each coordinate, per axis.
    /// Empty for linear and unsupported modes.
    pub(crate) pattern: &'static SwizzlePattern,
}

pub(crate) type SwizzlePattern = [Vec<u32>; 3];
static EMPTY_PATTERN: SwizzlePattern = [const { Vec::new() }; 3];

type CopyRun = (usize, usize, usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TileShape {
    pattern: usize,
    extent: [u32; 2],
    element_bytes: u32,
    block: [u32; 3],
    block_shift: u32,
    pitch: u32,
    tail: (u32, u32),
    inner_slices: u32,
}

impl TileShape {
    fn new(layout: &Layout, mip: &Mip) -> Self {
        Self {
            // The swizzle tables are static for the process.
            pattern: std::ptr::from_ref(layout.pattern) as usize,
            extent: [mip.width, mip.height],
            element_bytes: layout.desc.element_bytes,
            block: layout.block,
            block_shift: layout.alignment.trailing_zeros(),
            pitch: mip.pitch,
            tail: mip.tail.map_or((0, 0), |tail| (tail.x, tail.y)),
            inner_slices: if layout.thick { layout.block[2] } else { 1 },
        }
    }
}

struct TilePlan {
    runs: Vec<CopyRun>,
    chunk_size: Option<usize>,
    rows: Vec<u64>,
}

impl TilePlan {
    fn new(layout: &Layout, m: &Mip, remember_rows: bool) -> Self {
        let n = layout.desc.element_bytes as usize;
        let mut runs = Vec::new();
        let mut first_linear = 0;
        let mut first_tiled = layout.column(m, 0) as usize;
        for x in 1..m.width {
            let offset = layout.column(m, x) as usize;
            let at = x as usize * n;
            if offset != first_tiled + (at - first_linear) {
                runs.push((first_tiled, first_linear, at - first_linear));
                first_tiled = offset;
                first_linear = at;
            }
        }
        runs.push((first_tiled, first_linear, m.width as usize * n - first_linear));
        // A mip can end partway through the last run.
        let body = &runs[..runs.len() - 1];
        let chunk_size = body.first().map(|run| run.2).filter(|&size| {
            body.iter().all(|&(offset, _, len)| len == size && offset.is_multiple_of(size))
        });
        let rows = if remember_rows {
            let slices = if layout.thick { layout.block[2] } else { 1 };
            (0..slices)
                .flat_map(|z| (0..m.height).map(move |y| layout.row(m, y, z)))
                .collect()
        } else {
            Vec::new()
        };
        Self { runs, chunk_size, rows }
    }
}

/// Reuses row offsets and column runs across copies of the same geometry.
#[derive(Default)]
pub struct TilingCache {
    plans: std::collections::HashMap<TileShape, TilePlan>,
}

impl TilingCache {
    /// Tiles one `(mip, slice)` from linear rows with `(row, depth)` pitches.
    pub fn tile(
        &mut self,
        layout: &Layout,
        tiled: &mut [u8],
        subresource: (u32, u32),
        linear: &[u8],
        pitches: (usize, usize),
    ) -> Result<(), Error> {
        if !layout.thick {
            return crate::detile::tile(
                layout,
                tiled,
                subresource.0,
                subresource.1,
                linear,
                pitches.0,
                pitches.1,
            );
        }
        layout.walk::<false>(
            linear,
            tiled,
            subresource.0,
            subresource.1,
            pitches.0,
            pitches.1,
            Some(self),
        )
    }

    fn plan(&mut self, layout: &Layout, mip: &Mip) -> &TilePlan {
        self.plans.entry(TileShape::new(layout, mip)).or_insert_with(|| TilePlan::new(layout, mip, true))
    }
}

/// Micro tile (256 bytes) patterns by log2 of the element size: for each
/// address bit, the element coordinate bit it takes (`x0`, `y1`, `z0`...),
/// `--` for the bits inside an element. Thin `S`, thin `D`, thick `S`
/// (also the `D` 3D modes). From addrlib's `GFX10_SW_256_S_RBPLUS`,
/// `GFX10_SW_256_D_RBPLUS` and `GFX10_SW_4K_S3_RBPLUS` patterns; the larger
/// blocks add bits Y, X (thin) or Y, X, Z (thick) in turn.
const MICRO: [[&[u8; 16]; 5]; 3] = [
    [
        b"x0x1x2x3y0y1y2y3",
        b"--x0x1x2y0y1y2x3",
        b"----x0x1y0y1y2x2",
        b"------x0y0y1x1x2",
        b"--------y0y1x0x1",
    ],
    [
        b"x0x1x2y1y0y2x3y3",
        b"--x0x1x2y0y1y2x3",
        b"----x0x1y0y1x2y2",
        b"------x0y0x1x2y1",
        b"--------x0y0x1y1",
    ],
    [
        b"x0x1z0y0z1y1x2z2",
        b"--x0z0y0z1y1x1z2",
        b"----x0y0z0y1x1z1",
        b"------x0z0y0x1z1",
        b"--------z0y0x0z1",
    ],
];

/// For each axis, the address bit each coordinate bit lands on.
fn pattern(kind: usize, element_log2: u32, block_log2: u32) -> [Vec<u32>; 3] {
    let mut axes: [Vec<u32>; 3] = Default::default();
    let micro = MICRO[kind][element_log2 as usize];
    for bit in 0..8u32 {
        let (axis, index) = (micro[2 * bit as usize], micro[2 * bit as usize + 1]);
        let axis = match axis {
            b'x' => 0,
            b'y' => 1,
            b'z' => 2,
            _ => continue,
        };
        let index = usize::from(index - b'0');
        let to = &mut axes[axis];
        if to.len() <= index {
            to.resize(index + 1, 0);
        }
        to[index] = bit;
    }
    let order: &[usize] = if kind == 2 { &[1, 0, 2] } else { &[1, 0] };
    for (bit, axis) in (8..block_log2).zip(order.iter().cycle()) {
        axes[*axis].push(bit);
    }
    axes
}

/// Byte offsets inside a block for coordinates 0..2^bits of one axis.
fn spread(bits: &[u32]) -> Vec<u32> {
    (0..1u32 << bits.len())
        .map(|c| {
            bits.iter()
                .enumerate()
                .map(|(i, &to)| ((c >> i) & 1) << to)
                .sum()
        })
        .collect()
}

/// Tables depend only on the micro-tile kind, block size and element size.
/// Texture extents, mip placement and array slices only affect the base
/// address, so layouts with the same key can share these immutable tables.
fn shared_pattern(kind: usize, element_log2: u32, block_log2: u32) -> &'static SwizzlePattern {
    // Indexed by MICRO kind, block size (256 B, 4 KiB, 64 KiB), then element size.
    static PATTERNS: [[[OnceLock<SwizzlePattern>; 5]; 3]; 3] =
        [const { [const { [const { OnceLock::new() }; 5] }; 3] }; 3];
    let block_index = ((block_log2 - 8) / 4) as usize;
    PATTERNS[kind][block_index][element_log2 as usize]
        .get_or_init(|| pattern(kind, element_log2, block_log2).map(|bits| spread(&bits)))
}

const MAX_MIPS: u32 = 16;

/// The largest width, height and depth (or array size) accepted, in texels.
/// GFX10's image descriptors hold at most 16384 texels per side, so no
/// texture the GPU can use is refused. The bound also keeps every size
/// computed below within `u64` (and its `u32` steps), which unchecked
/// values from a title would otherwise overflow.
const MAX_EXTENT: u32 = 1 << 14;

/// 256-byte micro tile dimensions by log2 element size, 2D and 3D.
pub(crate) const BLOCK_256_2D: [[u32; 2]; 5] = [[16, 16], [16, 8], [8, 8], [8, 4], [4, 4]];
const BLOCK_256_3D: [[u32; 3]; 5] = [[8, 4, 8], [4, 4, 8], [4, 4, 4], [4, 2, 4], [2, 2, 4]];
const BLOCK_1K_3D: [[u32; 3]; 5] = [[16, 8, 8], [8, 8, 8], [8, 8, 4], [8, 4, 4], [4, 4, 4]];

impl Layout {
    pub fn new(desc: &Desc) -> Result<Layout, Error> {
        let d = *desc;
        let block_log2 = d.swizzle.block_log2().ok_or(Error::Invalid)?;
        let valid = (1..=MAX_EXTENT).contains(&d.width)
            && (1..=MAX_EXTENT).contains(&d.height)
            && (1..=MAX_EXTENT).contains(&d.depth)
            && (1..=MAX_MIPS).contains(&d.mips)
            && d.element_bytes.is_power_of_two()
            && d.element_bytes <= 16
            && matches!(d.block, 1 | 4)
            && d.samples.is_power_of_two()
            && d.samples <= 8
            && (d.samples == 1
                || (d.dimension == Dimension::Tex2d && d.mips == 1 && d.swizzle != Swizzle::LINEAR))
            && (d.dimension != Dimension::Tex1d || d.height == 1)
            && (d.dimension != Dimension::Tex3d
                || matches!(d.swizzle.0, 0 | 5 | 9 | 17 | 21 | 24..=27));
        if !valid {
            return Err(Error::Invalid);
        }
        if block_log2 == 0 {
            Ok(Layout::linear(d))
        } else {
            Ok(Layout::tiled(d, block_log2))
        }
    }

    /// `HwlComputeSurfaceInfoLinear`: rows aligned to 256 bytes, mips packed
    /// smallest first, a 3D texture's depth slices each hold the chain.
    fn linear(d: Desc) -> Layout {
        let bpe = u64::from(d.element_bytes);
        let align = 256 / d.element_bytes;
        let three_d = d.dimension == Dimension::Tex3d;
        let mut mips = vec![];
        let mut offset = 0;
        for i in (0..d.mips).rev() {
            let h = padded_dim(d.height, i, d.block);
            let pitch = padded_dim(d.width, i, d.block).next_multiple_of(align);
            mips.push(Mip {
                width: dim(d.width, i, d.block),
                height: dim(d.height, i, d.block),
                depth: if three_d { dim(d.depth, i, 1) } else { 1 },
                pitch,
                rows: h,
                depth_pitch: if three_d { d.depth } else { 1 },
                offset,
                tail: None,
            });
            offset += u64::from(pitch) * u64::from(h) * bpe;
        }
        mips.reverse();
        Layout {
            desc: d,
            block: [align, 1, 1],
            thick: false,
            slices: d.depth,
            slice_size: offset,
            size: offset * u64::from(d.depth),
            alignment: 256,
            first_tail_mip: d.mips,
            mips,
            pattern: &EMPTY_PATTERN,
        }
    }

    fn tiled(d: Desc, block_log2: u32) -> Layout {
        let element_log2 = d.element_bytes.trailing_zeros();
        let bpe = u64::from(d.element_bytes);
        let block_bytes = 1u64 << block_log2;
        let thick = d.dimension == Dimension::Tex3d && (d.swizzle.standard() || d.swizzle.display());
        let block = if thick {
            // ComputeThickBlockDimension
            let amp = (block_log2 - 10) / 3;
            let rest = (block_log2 - 10) % 3;
            let b = BLOCK_1K_3D[element_log2 as usize];
            [b[0] << amp, b[1] << (amp + rest / 2), b[2] << (amp + u32::from(rest != 0))]
        } else {
            // ComputeThinBlockDimension
            let samples_log2 = d.samples.trailing_zeros();
            let elements_log2 = block_log2 - element_log2 - samples_log2;
            let wide = samples_log2 & 1 == 0 || block_log2 & 1 != 0;
            let w = (elements_log2 + u32::from(wide)) / 2;
            [1 << w, 1 << (elements_log2 - w), 1]
        };
        let slices = d.depth.next_multiple_of(block[2]);
        let three_d = d.dimension == Dimension::Tex3d;
        // Mip depths placed by a thick layout (thin ones store depth as
        // slices).
        let mip_depth = |i: u32| if thick { padded_dim(d.depth, i, 1) } else { 1 };
        let depth_of = |i: u32| if three_d { dim(d.depth, i, 1) } else { 1 };
        let (width, height) = (|i| dim(d.width, i, d.block), |i| dim(d.height, i, d.block));
        let padded = |i| (padded_dim(d.width, i, d.block), padded_dim(d.height, i, d.block));
        let mut mips = Vec::with_capacity(d.mips as usize);
        let mut first_tail_mip = d.mips;
        let mut slice_size = 0u64;
        if d.mips == 1 {
            mips.push(Mip {
                width: width(0),
                height: height(0),
                depth: depth_of(0),
                pitch: width(0).next_multiple_of(block[0]),
                rows: height(0).next_multiple_of(block[1]),
                depth_pitch: if three_d { slices } else { 1 },
                offset: 0,
                tail: None,
            });
            slice_size = u64::from(mips[0].pitch)
                * u64::from(mips[0].rows)
                * bpe
                * u64::from(d.samples);
        } else {
            // GetMipTailDim: the largest mip the tail takes.
            let mut tail_max = block;
            if thick {
                match block_log2 % 3 {
                    0 => tail_max[1] >>= 1,
                    1 => tail_max[0] >>= 1,
                    _ => tail_max[2] >>= 1,
                }
            } else {
                tail_max[0] >>= 1;
            }
            // GetMaxNumMipsInTail; 256-byte blocks have no tail
            // (`ComputeSurfaceInfoMicroTiled`).
            let effective = if thick {
                block_log2 - (block_log2 - 8) / 3
            } else {
                block_log2
            };
            let max_in_tail = match effective {
                8 => 0,
                9..=11 => 1 + (1 << (effective - 9)),
                _ => effective - 4,
            };
            // dsMipmapHtileFix: 8- and 16-bit depth enter the tail as 32-bit.
            let mut fixed = tail_max;
            if d.swizzle.z_order() && element_log2 <= 1 {
                let b = BLOCK_256_2D[element_log2 as usize];
                fixed[0] /= b[0] / BLOCK_256_2D[2][0];
                fixed[1] /= b[1] / BLOCK_256_2D[2][1];
            }
            let mut sizes = vec![];
            for i in 0..d.mips {
                let (w, h) = padded(i);
                if w <= fixed[0] && h <= fixed[1] && d.mips - i <= max_in_tail {
                    first_tail_mip = i;
                    slice_size += block_bytes / u64::from(block[2]);
                    break;
                }
                let pitch = w.next_multiple_of(block[0]);
                let rows = h.next_multiple_of(block[1]);
                let depth = mip_depth(i).next_multiple_of(block[2]);
                let one = u64::from(pitch) * u64::from(rows) * bpe;
                slice_size += one;
                sizes.push(one * u64::from(block[2]));
                mips.push(Mip {
                    width: width(i),
                    height: height(i),
                    depth: depth_of(i),
                    pitch,
                    rows,
                    depth_pitch: if three_d { depth } else { 1 },
                    offset: 0,
                    tail: None,
                });
            }
            // Standard mips from the smallest up, after the tail block.
            let mut offset = if first_tail_mip < d.mips {
                block_bytes
            } else {
                0
            };
            for i in (0..first_tail_mip as usize).rev() {
                mips[i].offset = offset;
                offset += sizes[i];
            }
            let micro = if thick {
                let b = BLOCK_256_3D[element_log2 as usize];
                [b[0], b[1]]
            } else {
                BLOCK_256_2D[element_log2 as usize]
            };
            let (mut pitch, mut rows) = (tail_max[0], tail_max[1]);
            for i in first_tail_mip..d.mips {
                let m = max_in_tail - 1 - (i - first_tail_mip);
                let offset = if m > 6 { 16 << m } else { m << 8 };
                // Address bits 8.. of the offset, odd ones x, even ones y.
                let bits = |from: u32| (0..6).map(|b| ((offset >> (from + 2 * b)) & 1) << b).sum();
                let (mut x, mut y): (u32, u32) = (bits(9), bits(8));
                if block_log2 & 1 != 0 {
                    std::mem::swap(&mut x, &mut y);
                    if element_log2 & 1 != 0 {
                        y = (y << 1) | (x & 1);
                        x >>= 1;
                    }
                }
                mips.push(Mip {
                    width: width(i),
                    height: height(i),
                    depth: depth_of(i),
                    pitch,
                    rows,
                    depth_pitch: if thick {
                        mip_depth(i).next_multiple_of(BLOCK_256_3D[element_log2 as usize][2])
                    } else {
                        1
                    },
                    offset: 0,
                    tail: Some(Tail {
                        offset,
                        x: x * micro[0],
                        y: y * micro[1],
                    }),
                });
                pitch = (pitch >> 1).max(micro[0]);
                rows = (rows >> 1).max(micro[1]);
            }
        }
        // Thick layouts store each depth slab's chain together.
        let slab = if thick { u64::from(block[2]) } else { 1 };
        let slice_size = slice_size * slab;
        let slices = slices / block[2];
        let addressable = !d.swizzle.xor() && d.samples == 1;
        let pattern = if addressable {
            let kind = if thick {
                2
            } else {
                usize::from(d.swizzle.display())
            };
            shared_pattern(kind, element_log2, block_log2)
        } else {
            &EMPTY_PATTERN
        };
        Layout {
            desc: d,
            block,
            thick,
            slices,
            slice_size,
            size: slice_size * u64::from(slices),
            alignment: block_bytes,
            first_tail_mip,
            mips,
            pattern,
        }
    }

    /// Whether [`Layout::offset`] and the copies work for this layout:
    /// linear and the modes without pipe or bank XOR, single-sampled.
    ///
    /// shortcut: no addressing for the XOR modes (`_X`, `_T`) and MSAA;
    /// they need the GPU's pipe and bank configuration (Series X and S
    /// differ). Titles use them for render targets and depth, which the
    /// host renders; add them when a title copies into or out of one.
    pub fn addressable(&self) -> bool {
        self.desc.swizzle == Swizzle::LINEAR || !self.pattern[0].is_empty()
    }

    /// The slice and the depth inside it for `z` (an array slice or a 3D
    /// depth slice).
    pub(crate) fn split(&self, z: u32) -> (u32, u32) {
        if self.thick {
            (z / self.block[2], z % self.block[2])
        } else {
            (z, 0)
        }
    }

    /// Byte offset of element (`x`, `y`) of mip `mip` in array slice or
    /// depth slice `z`. Coordinates must lie inside the mip.
    pub fn offset(&self, mip: u32, x: u32, y: u32, z: u32) -> Result<u64, Error> {
        let m = self.mips.get(mip as usize).ok_or(Error::Invalid)?;
        let depth = if self.desc.dimension == Dimension::Tex3d {
            m.depth
        } else {
            self.slices
        };
        let inside = x < m.width && y < m.height && z < depth;
        if !inside || !self.addressable() {
            return Err(if inside {
                Error::Unsupported
            } else {
                Error::Invalid
            });
        }
        let (slice, z) = self.split(z);
        let base = u64::from(slice) * self.slice_size + m.offset;
        if self.desc.swizzle == Swizzle::LINEAR {
            let row = u64::from(y) * u64::from(m.pitch) + u64::from(x);
            return Ok(base + row * u64::from(self.desc.element_bytes));
        }
        Ok(base + self.row(m, y, z) + self.column(m, x))
    }

    /// The part of an address that depends on the row (and depth in a
    /// thick block).
    #[inline]
    pub(crate) fn row(&self, m: &Mip, y: u32, z: u32) -> u64 {
        let [bw, bh, _] = self.block;
        let log2 = self.alignment.trailing_zeros();
        // In the tail the pattern of the shifted position gives the mip's
        // offset inside the tail block.
        let y = y + m.tail.map_or(0, |t| t.y);
        let blocks = u64::from(y / bh) * u64::from(m.pitch / bw);
        (blocks << log2) + u64::from(self.pattern[1][(y % bh) as usize] | self.pattern[2][z as usize])
    }

    /// The part of an address that depends on the column.
    #[inline]
    pub(crate) fn column(&self, m: &Mip, x: u32) -> u64 {
        let bw = self.block[0];
        let x = x + m.tail.map_or(0, |t| t.x);
        (u64::from(x / bw) << self.alignment.trailing_zeros())
            + u64::from(self.pattern[0][(x % bw) as usize])
    }

    /// Depth slices a subresource of mip `mip` covers: the mip's depth for
    /// 3D textures, else 1 (one array slice).
    pub fn subresource_depth(&self, mip: u32) -> u32 {
        self.mips.get(mip as usize).map_or(0, |m| m.depth)
    }

    /// The bytes from the texture's start that mip `first` and the smaller
    /// ones take: the smaller a mip, the nearer the start it lies, so a
    /// title that streams textures has memory for the smaller mips only and
    /// another texture right behind them. All of the layout where slices
    /// follow each other (each holds its whole chain).
    pub fn bytes_from(&self, first: u32) -> u64 {
        let whole = first == 0
            || self.slices != 1
            || self.thick
            || self.desc.dimension == Dimension::Tex3d
            || self.desc.samples != 1
            || first >= self.desc.mips;
        if whole {
            return self.size;
        }
        self.mips[first as usize..]
            .iter()
            .map(|m| match m.tail {
                Some(_) => self.alignment,
                None => m.offset + u64::from(m.pitch) * u64::from(m.rows) * u64::from(self.desc.element_bytes),
            })
            .max()
            .unwrap_or(self.size)
            .min(self.size)
    }

    /// Copies one subresource out of tiled memory (the whole texture,
    /// [`Layout::size`] bytes, or the [`Layout::bytes_from`] of the mip)
    /// into linear rows: mip `mip` of array slice
    /// `slice` (3D: every depth slice of the mip, `slice` 0). Rows are
    /// `row_pitch` bytes apart, depth slices `depth_pitch`.
    pub fn detile(
        &self,
        tiled: &[u8],
        mip: u32,
        slice: u32,
        linear: &mut [u8],
        row_pitch: usize,
        depth_pitch: usize,
    ) -> Result<(), Error> {
        crate::detile::detile(self, tiled, mip, slice, linear, row_pitch, depth_pitch)
    }

    /// The reverse of [`Layout::detile`]: linear rows into tiled memory.
    pub fn tile(
        &self,
        tiled: &mut [u8],
        mip: u32,
        slice: u32,
        linear: &[u8],
        row_pitch: usize,
        depth_pitch: usize,
    ) -> Result<(), Error> {
        crate::detile::tile(self, tiled, mip, slice, linear, row_pitch, depth_pitch)
    }

    /// Copies contiguous runs in one direction between the two layouts.
    #[allow(clippy::too_many_arguments)]
    fn walk<const DETILE: bool>(
        &self,
        src: &[u8],
        dst: &mut [u8],
        mip: u32,
        slice: u32,
        row_pitch: usize,
        depth_pitch: usize,
        cache: Option<&mut TilingCache>,
    ) -> Result<(), Error> {
        let m = self.mips.get(mip as usize).ok_or(Error::Invalid)?;
        let depth = m.depth;
        if m.width == 0 || m.height == 0 || depth == 0 {
            return Err(Error::Invalid);
        }
        let first = if self.desc.dimension == Dimension::Tex3d {
            if slice != 0 {
                return Err(Error::Invalid);
            }
            0
        } else {
            slice
        };
        if first >= self.slices {
            return Err(Error::Invalid);
        }
        if !self.addressable() {
            return Err(Error::Unsupported);
        }
        let n = self.desc.element_bytes as usize;
        let bytes = m.width as usize * n;
        let slice_bytes = (m.height as usize - 1)
            .checked_mul(row_pitch)
            .and_then(|offset| offset.checked_add(bytes))
            .ok_or(Error::TooSmall)?;

        // Multiple depth slices must not overwrite each other.
        if depth > 1 && depth_pitch < slice_bytes {
            return Err(Error::TooSmall);
        }

        let need = (depth as usize - 1)
            .checked_mul(depth_pitch)
            .and_then(|offset| offset.checked_add(slice_bytes))
            .ok_or(Error::TooSmall)?;
        let (tiled_len, linear_len) = if DETILE {
            (src.len(), dst.len())
        } else {
            (dst.len(), src.len())
        };
        let tiled_need = if DETILE { self.bytes_from(mip) } else { self.size };
        if tiled_len < tiled_need as usize || linear_len < need || row_pitch < bytes {
            return Err(Error::TooSmall);
        }
        let linear = self.desc.swizzle == Swizzle::LINEAR;
        let local;
        // Every row uses the same column runs, including in the mip tail.
        let plan = if linear {
            None
        } else {
            Some(match cache {
                Some(cache) => cache.plan(self, m),
                None => {
                    local = TilePlan::new(self, m, false);
                    &local
                }
            })
        };
        let runs = plan.map_or(&[][..], |p| p.runs.as_slice());
        let (body, last) = runs.split_at(runs.len().saturating_sub(1));
        let chunk_size = plan.and_then(|p| p.chunk_size);
        let offsets = plan.map_or(&[][..], |p| p.rows.as_slice());
        for z in 0..depth {
            let (s, inner) = self.split(first + z);
            let base = u64::from(s) * self.slice_size + m.offset;
            let rows = inner as usize * m.height as usize;
            for y in 0..m.height {
                let linear_row = z as usize * depth_pitch + y as usize * row_pitch;
                if linear {
                    let tiled_row = base + u64::from(y) * u64::from(m.pitch) * n as u64;
                    copy_run::<DETILE>(src, dst, tiled_row as usize, linear_row, bytes);
                } else {
                    let row = offsets.get(rows + y as usize).copied().unwrap_or_else(|| self.row(m, y, inner));
                    let tiled_row = (base + row) as usize;
                    match chunk_size {
                        Some(8) if tiled_row.is_multiple_of(8) => {
                            copy_runs::<8, DETILE>(src, dst, body, tiled_row, linear_row)
                        }
                        Some(16) if tiled_row.is_multiple_of(16) => {
                            copy_runs::<16, DETILE>(src, dst, body, tiled_row, linear_row)
                        }
                        Some(32) if tiled_row.is_multiple_of(32) => {
                            copy_runs::<32, DETILE>(src, dst, body, tiled_row, linear_row)
                        }
                        _ => copy_runs::<0, DETILE>(src, dst, body, tiled_row, linear_row),
                    }
                    copy_runs::<0, DETILE>(src, dst, last, tiled_row, linear_row);
                }
            }
        }
        Ok(())
    }
}

fn copy_run<const DETILE: bool>(
    src: &[u8],
    dst: &mut [u8],
    tiled: usize,
    linear: usize,
    len: usize,
) {
    let (from, to) = if DETILE {
        (tiled, linear)
    } else {
        (linear, tiled)
    };
    dst[to..][..len].copy_from_slice(&src[from..][..len]);
}

/// A row of equal runs is contiguous in linear memory. Treat it as a
/// slice of arrays so only the tiled chunk needs a bounds check per copy.
/// Use the byte ranges for irregular runs and the last run of each row.
fn copy_runs<const N: usize, const DETILE: bool>(
    src: &[u8],
    dst: &mut [u8],
    runs: &[CopyRun],
    tiled_row: usize,
    linear_row: usize,
) {
    if N == 0 {
        for &(tiled, linear, len) in runs {
            copy_run::<DETILE>(src, dst, tiled_row + tiled, linear_row + linear, len);
        }
    } else if DETILE {
        let tiled = src.as_chunks::<N>().0;
        let linear = dst[linear_row..][..runs.len() * N].as_chunks_mut::<N>().0;
        for (out, &(offset, ..)) in linear.iter_mut().zip(runs) {
            *out = tiled[(tiled_row + offset) / N];
        }
    } else {
        let tiled = dst.as_chunks_mut::<N>().0;
        let linear = src[linear_row..][..runs.len() * N].as_chunks::<N>().0;
        for (input, &(offset, ..)) in linear.iter().zip(runs) {
            tiled[(tiled_row + offset) / N] = *input;
        }
    }
}

/// What a DXGI format stores per element: bytes, texels per element side
/// (4 for block compression), and whether its stencil is a second plane
/// (the GPU keeps depth and stencil apart).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub bytes: u32,
    pub block: u32,
    pub stencil: bool,
}

/// By DXGI format number. `None` for 96-bit, 1-bit, packed 4:2:2 and video
/// formats.
///
/// shortcut: those formats have no layout here; add them when a title uses
/// one.
pub fn format(dxgi: u32) -> Option<Format> {
    let (bytes, block, stencil) = match dxgi {
        1..=4 => (16, 1, false),
        9..=18 => (8, 1, false),
        19..=22 | 44..=47 => (4, 1, true),
        23..=43 | 67 | 87..=93 | 116 | 117 => (4, 1, false),
        48..=59 | 85 | 86 | 115 => (2, 1, false),
        118..=120 => (2, 1, true),
        60..=65 => (1, 1, false),
        70..=72 | 79..=81 => (8, 4, false),
        73..=78 | 82..=84 | 94..=99 => (16, 4, false),
        _ => return None,
    };
    Some(Format {
        bytes,
        block,
        stencil,
    })
}

/// A texture as the console allocates it: a D3D12-style description with
/// a swizzle mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDesc {
    pub dimension: Dimension,
    /// In texels.
    pub width: u32,
    pub height: u32,
    /// Depth of a 3D texture, else the array size.
    pub depth_or_array: u32,
    /// 0: the full chain.
    pub mips: u32,
    /// DXGI format number.
    pub format: u32,
    pub samples: u32,
    pub swizzle: Swizzle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    /// From the start of the texture.
    pub offset: u64,
    pub stencil: bool,
    pub layout: Layout,
}

/// Every data plane of a texture (depth, then stencil), placed one after
/// the other.
///
/// shortcut: no compression metadata planes (HTILE, CMASK, FMASK, DCC);
/// the console adds them to render targets and depth buffers unless the
/// description denies them, so those are smaller here than on the console.
/// The host GPU renders them, so nothing reads that memory; add the planes
/// if a title sizes a pool from console sizes and runs out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Texture {
    pub planes: Vec<Plane>,
    pub size: u64,
    pub alignment: u64,
}

impl Texture {
    pub fn new(desc: &TextureDesc) -> Result<Texture, Error> {
        let f = format(desc.format).ok_or(Error::Invalid)?;
        let depth3d = if desc.dimension == Dimension::Tex3d {
            desc.depth_or_array
        } else {
            1
        };
        let mips = match desc.mips {
            0 => 32 - desc.width.max(desc.height).max(depth3d).max(1).leading_zeros(),
            n => n,
        };
        let plane = |bytes| Desc {
            dimension: desc.dimension,
            width: desc.width,
            height: desc.height,
            depth: desc.depth_or_array,
            mips,
            element_bytes: bytes,
            block: f.block,
            samples: desc.samples.max(1),
            swizzle: desc.swizzle,
        };
        let mut planes = vec![Plane {
            offset: 0,
            stencil: false,
            layout: Layout::new(&plane(f.bytes))?,
        }];
        if f.stencil {
            let layout = Layout::new(&plane(1))?;
            let offset = planes[0].layout.size.next_multiple_of(layout.alignment);
            planes.push(Plane {
                offset,
                stencil: true,
                layout,
            });
        }
        let last = &planes[planes.len() - 1];
        let size = last.offset + last.layout.size;
        let alignment = planes.iter().map(|p| p.layout.alignment).max().unwrap_or(1);
        Ok(Texture {
            planes,
            size,
            alignment,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(w: u32, h: u32, mips: u32, bpe: u32, swizzle: Swizzle) -> Desc {
        Desc {
            dimension: Dimension::Tex2d,
            width: w,
            height: h,
            depth: 1,
            mips,
            element_bytes: bpe,
            block: 1,
            samples: 1,
            swizzle,
        }
    }

    #[test]
    fn linear_layouts_require_one_sample() {
        let linear = desc(256, 256, 1, 4, Swizzle::LINEAR);
        let single = Layout::new(&linear).unwrap();
        assert!(single.addressable());
        assert_eq!(single.size, 256 * 256 * 4);

        for samples in [2, 4, 8] {
            let multisampled = Desc { samples, ..linear };
            assert_eq!(Layout::new(&multisampled), Err(Error::Invalid));

            let tiled = Layout::new(&Desc {
                swizzle: Swizzle::SW_64KB_R_X,
                ..multisampled
            })
            .unwrap();
            assert_eq!(tiled.size, single.size * u64::from(samples));
        }
    }

    /// Sizes past the hardware's 16384 texels per side are refused instead
    /// of overflowing the size arithmetic (found by random-input testing).
    #[test]
    fn extents_past_the_hardware_limit_are_invalid() {
        for swizzle in [Swizzle::LINEAR, Swizzle::SW_64KB_D_X] {
            let largest = Layout::new(&desc(MAX_EXTENT, MAX_EXTENT, 15, 16, swizzle)).unwrap();
            assert!(largest.size >= u64::from(MAX_EXTENT) * u64::from(MAX_EXTENT) * 16);
            for (w, h) in [
                (MAX_EXTENT + 1, 1),
                (1, MAX_EXTENT + 1),
                (u32::MAX, u32::MAX),
            ] {
                let result = Layout::new(&desc(w, h, 1, 16, swizzle));
                assert_eq!(result, Err(Error::Invalid), "{w}x{h}");
            }
            let mut deep = desc(1, 1, 1, 16, swizzle);
            deep.depth = u32::MAX;
            assert_eq!(Layout::new(&deep), Err(Error::Invalid));
        }
    }

    #[test]
    fn blocks_hold_their_size_in_elements() {
        for log2 in [8u32, 12, 16] {
            let swizzle = match log2 {
                8 => Swizzle::SW_256B_S,
                12 => Swizzle::SW_4KB_S,
                _ => Swizzle::SW_64KB_D,
            };
            for e in 0..5 {
                let l = Layout::new(&desc(1, 1, 1, 1 << e, swizzle)).unwrap();
                let [w, h, d] = l.block;
                assert_eq!(u64::from(w * h * d) << e, 1 << log2, "{log2} {e}");
                // The pattern covers exactly the block's coordinates.
                assert_eq!(l.pattern[0].len() as u32, w);
                assert_eq!(l.pattern[1].len() as u32, h);
            }
        }
        // 64 KiB, 16-byte elements: 64x64 (a 256x256 BC7 texture).
        let l = Layout::new(&desc(64, 64, 1, 16, Swizzle::SW_64KB_D)).unwrap();
        assert_eq!(l.block, [64, 64, 1]);
        assert_eq!((l.size, l.alignment), (65536, 65536));
    }

    /// Every element of every mip and slice lands on its own address, inside
    /// the texture, and tiling then detiling gives the rows back.
    #[test]
    fn addresses_are_a_bijection_and_copies_round_trip() {
        let mut cache = TilingCache::default();
        let cases = [
            desc(64, 64, 7, 16, Swizzle::SW_64KB_D),
            desc(100, 60, 7, 4, Swizzle::SW_64KB_D),
            desc(250, 175, 8, 8, Swizzle::SW_4KB_S),
            desc(33, 17, 6, 1, Swizzle::SW_4KB_D),
            desc(40, 9, 3, 2, Swizzle::SW_64KB_S),
            desc(30, 20, 5, 4, Swizzle::LINEAR),
            desc(83, 31, 6, 1, Swizzle::SW_256B_S),
            Desc {
                block: 4,
                ..desc(109, 71, 6, 16, Swizzle::SW_256B_D)
            },
            Desc {
                depth: 3,
                ..desc(70, 70, 4, 16, Swizzle::SW_4KB_S)
            },
            Desc {
                dimension: Dimension::Tex3d,
                depth: 16,
                ..desc(64, 32, 7, 4, Swizzle::SW_4KB_S)
            },
        ];
        for d in cases {
            let l = Layout::new(&d).unwrap();
            let mut seen = vec![false; l.size as usize];
            let mut tiled = vec![0u8; l.size as usize];
            let n = d.element_bytes as usize;
            let slices = if d.dimension == Dimension::Tex3d { 1 } else { d.depth };
            for slice in 0..slices {
                for mip in 0..d.mips {
                    let m = l.mips[mip as usize];
                    let active = m.width as usize * n;
                    let row = active + 13;
                    let plane = row * m.height as usize + 11;
                    let linear: Vec<u8> = (0..plane * m.depth as usize)
                        .map(|i| (i * 7 + mip as usize * 13 + slice as usize) as u8)
                        .collect();
                    cache.tile(&l, &mut tiled, (mip, slice), &linear, (row, plane)).unwrap();
                    let mut back = vec![0xcc; linear.len()];
                    l.detile(&tiled, mip, slice, &mut back, row, plane).unwrap();
                    for z in 0..m.depth {
                        let base = z as usize * plane;
                        for y in 0..m.height {
                            let start = base + y as usize * row;
                            assert_eq!(back[start..start + active], linear[start..start + active]);
                            assert!(back[start + active..start + row].iter().all(|&b| b == 0xcc));
                            for x in 0..m.width {
                                let at = l.offset(mip, x, y, slice + z).unwrap() as usize;
                                let source = start + x as usize * n;
                                assert_eq!(tiled[at..at + n], linear[source..source + n]);
                                assert!(at + n <= seen.len(), "{d:?} mip {mip} outside");
                                assert!(at.is_multiple_of(n));
                                assert!(!seen[at], "{d:?} mip {mip} ({x},{y},{z}) twice");
                                seen[at] = true;
                            }
                        }
                        assert!(back[base + plane - 11..base + plane].iter().all(|&b| b == 0xcc));
                    }
                }
            }
        }
    }

    #[test]
    fn tiled_chunks_match_element_offsets_with_odd_row_pitches() {
        let mut cache = TilingCache::default();
        for swizzle in [
            Swizzle::SW_256B_S,
            Swizzle::SW_256B_D,
            Swizzle::SW_4KB_S,
            Swizzle::SW_4KB_D,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ] {
            for n in [1, 2, 4, 8, 16] {
                let l = Layout::new(&Desc {
                    depth: 2,
                    ..desc(73, 39, 7, n, swizzle)
                })
                .unwrap();
                let n = n as usize;
                let tiled: Vec<_> = (0..l.size)
                    .map(|i| (i.wrapping_mul(0x9e37_79b9) >> 24) as u8)
                    .collect();
                let mut retiled = vec![0xbb; tiled.len()];
                let mut cached = retiled.clone();
                let mut expected_tiled = retiled.clone();
                for slice in 0..l.slices {
                    for (mip, m) in l.mips.iter().enumerate() {
                        let mip = mip as u32;
                        let bytes = m.width as usize * n;
                        let pitch = bytes + 3;
                        // Keep padding between rows, with exactly the active
                        // bytes in the last row.
                        let len = (m.height as usize - 1) * pitch + bytes;
                        let mut linear = vec![0xcc; len];
                        let mut expected = linear.clone();
                        for y in 0..m.height {
                            for x in 0..m.width {
                                let src = l.offset(mip, x, y, slice).unwrap() as usize;
                                let dst = y as usize * pitch + x as usize * n;
                                expected[dst..][..n].copy_from_slice(&tiled[src..][..n]);
                                expected_tiled[src..][..n].copy_from_slice(&tiled[src..][..n]);
                            }
                        }
                        l.detile(&tiled, mip, slice, &mut linear, pitch, len)
                            .unwrap();
                        assert_eq!(linear, expected, "{swizzle:?} {n} mip {mip} slice {slice}");
                        l.tile(&mut retiled, mip, slice, &expected, pitch, len)
                            .unwrap();
                        cache.tile(&l, &mut cached, (mip, slice), &expected, (pitch, len))
                            .unwrap();
                    }
                }
                assert_eq!(retiled, expected_tiled, "{swizzle:?} {n}");
                assert_eq!(cached, expected_tiled, "{swizzle:?} {n} with cached rows and columns");
            }
        }
    }

    /// Hand-checked: in a 64 KiB `D` block of 4-byte elements, x bits go to
    /// address bits 2, 3, 6, 9, 11, 13, 15 and y bits to 4, 5, 7, 8, 10, 12,
    /// 14; a 4 KiB `S` block of 16-byte elements is 16x16 with y first.
    #[test]
    fn hand_checked_addresses() {
        let l = Layout::new(&desc(128, 128, 1, 4, Swizzle::SW_64KB_D)).unwrap();
        assert_eq!(l.offset(0, 1, 0, 0), Ok(1 << 2));
        assert_eq!(l.offset(0, 4, 0, 0), Ok(1 << 6));
        assert_eq!(l.offset(0, 0, 4, 0), Ok(1 << 7));
        assert_eq!(l.offset(0, 0, 8, 0), Ok(1 << 8));
        assert_eq!(l.offset(0, 64, 64, 0), Ok((1 << 15) | (1 << 14)));
        let l = Layout::new(&desc(32, 32, 1, 16, Swizzle::SW_4KB_S)).unwrap();
        assert_eq!(l.block, [16, 16, 1]);
        assert_eq!(l.offset(0, 0, 1, 0), Ok(1 << 4));
        assert_eq!(l.offset(0, 1, 0, 0), Ok(1 << 6));
        // Blocks row by row: (16, 0) is the second block, (0, 16) the third.
        assert_eq!(l.offset(0, 16, 0, 0), Ok(4096));
        assert_eq!(l.offset(0, 0, 16, 0), Ok(8192));
    }

    #[test]
    fn layouts_share_identical_swizzle_tables() {
        for (swizzle, kind, dimension, block_log2) in [
            (Swizzle::SW_256B_S, 0, Dimension::Tex2d, 8),
            (Swizzle::SW_256B_D, 1, Dimension::Tex2d, 8),
            (Swizzle::SW_4KB_S, 0, Dimension::Tex2d, 12),
            (Swizzle::SW_4KB_D, 1, Dimension::Tex2d, 12),
            (Swizzle::SW_64KB_S, 0, Dimension::Tex2d, 16),
            (Swizzle::SW_64KB_D, 1, Dimension::Tex2d, 16),
            (Swizzle::SW_4KB_S, 2, Dimension::Tex3d, 12),
            (Swizzle::SW_64KB_S, 2, Dimension::Tex3d, 16),
        ] {
            for element_log2 in 0..5 {
                let d = Desc {
                    dimension,
                    depth: 8,
                    ..desc(128, 96, 7, 1 << element_log2, swizzle)
                };
                let first = Layout::new(&d).unwrap();
                let second = Layout::new(&Desc {
                    width: 32,
                    height: 24,
                    mips: 3,
                    depth: 1,
                    ..d
                })
                .unwrap();
                let expected = pattern(kind, element_log2, block_log2).map(|bits| spread(&bits));
                assert_eq!(first.pattern, &expected);
                assert!(std::ptr::eq(first.pattern, second.pattern));
                assert!(std::ptr::eq(first.pattern, first.clone().pattern));
            }
        }
    }

    /// A 256x256 BC7 chain in 64 KiB `D`: one tail block holding mips 1-8,
    /// then mip 0; mip 1 sits at x = 32 elements, 32 KiB into the tail.
    #[test]
    fn mip_tail_comes_first() {
        let l = Layout::new(&desc(64, 64, 9, 16, Swizzle::SW_64KB_D)).unwrap();
        assert_eq!(l.first_tail_mip, 1);
        assert_eq!(l.size, 2 * 65536);
        assert_eq!(l.mips[0].offset, 65536);
        assert_eq!(
            l.mips[1].tail,
            Some(Tail {
                offset: 32768,
                x: 32,
                y: 0
            })
        );
        assert_eq!(l.offset(1, 0, 0, 0), Ok(32768));
        // 4 KiB blocks hold fewer tail mips: 16x16 BC7 mips stay outside.
        let l = Layout::new(&desc(64, 64, 9, 16, Swizzle::SW_4KB_S)).unwrap();
        assert_eq!(l.first_tail_mip, 3);
        assert_eq!(l.mips[2].offset, 4096);
        assert_eq!(l.mips[1].offset, 8192);
        assert_eq!(l.mips[0].offset, 8192 + 16384);
    }

    /// A volume deeper than its block keeps its whole chain in the tail when
    /// every mip's width and height fit it (addrlib's `mipChainInTail`): each
    /// depth slab has its own tail block, holding that slab's slices of
    /// every mip. 16x16x64 in 64 KiB `S` blocks of 32x32x16 is four slabs
    /// of one block; mip 1's 32 slices span the first two. Addresses from
    /// addrlib, as in [`layouts_match_addrlib`].
    #[test]
    fn deep_volumes_keep_their_chain_in_each_slabs_tail() {
        let l = Layout::new(&Desc {
            dimension: Dimension::Tex3d,
            depth: 64,
            ..desc(16, 16, 4, 4, Swizzle::SW_64KB_S)
        })
        .unwrap();
        assert_eq!(l.block, [32, 32, 16]);
        assert_eq!((l.first_tail_mip, l.size), (0, 4 * 65536));
        for (mip, x, y, z, address) in [
            (0, 0, 0, 0, 32768),
            (0, 0, 0, 15, 42128),
            (0, 0, 0, 16, 65536 + 32768),
            (0, 15, 15, 63, 3 * 65536 + 49148),
            (1, 0, 0, 0, 16384),
            (1, 0, 0, 16, 65536 + 16384),
            (1, 0, 0, 31, 65536 + 25744),
            (2, 0, 0, 0, 4096),
            (3, 1, 1, 7, 3740),
        ] {
            let at = l.offset(mip, x, y, z);
            assert_eq!(at, Ok(address), "mip {mip} ({x},{y},{z})");
        }
    }

    /// Layouts as AMD's addrlib computes them (Mesa `src/amd/addrlib` at ed09d38,
    /// `Addr2ComputeSurfaceInfo` and `Addr2ComputeSurfaceAddrFromCoord`,
    /// Navi 21 settings, no pipe-bank XOR): its surface size, its first mip
    /// in the tail (`mipChainInTail` is 0; linear has no tail) and an
    /// FNV-1a hash of the address of every element (mips in order, then
    /// slices, rows and columns; 8 little-endian bytes each). Element sizes
    /// 1-16 bytes, block compression, arrays, mip tails and deep volumes, in
    /// every mode [`Layout::offset`] addresses.
    #[test]
    fn layouts_match_addrlib() {
        use Dimension::{Tex1d, Tex2d, Tex3d};
        #[rustfmt::skip]
        let cases = [
            // mode, dimension, element bytes, texels per element side,
            // [width, height, depth or array], mips,
            //     (size, first tail mip, hash)
            (Swizzle::SW_256B_S, Tex2d, 1, 1, [83, 31, 1], 6,
                (5120, 6, 0x435e_9cad_96ff_dd81)),
            (Swizzle::SW_256B_D, Tex2d, 16, 4, [109, 71, 1], 6,
                (13824, 6, 0xbc2b_b84f_c528_7cb3)),
            (Swizzle::SW_4KB_S, Tex2d, 8, 1, [250, 175, 1], 8,
                (495616, 4, 0x1e87_7240_4dcd_c56e)),
            (Swizzle::SW_4KB_S, Tex2d, 16, 1, [70, 70, 3], 4,
                (479232, 4, 0xf948_6d1d_511c_d75c)),
            (Swizzle::SW_4KB_D, Tex2d, 1, 1, [33, 17, 1], 6,
                (8192, 1, 0xaae1_0e3f_15a1_892e)),
            (Swizzle::SW_64KB_S, Tex2d, 2, 1, [40, 9, 1], 3,
                (65536, 0, 0xf7a3_7fe6_41c2_d565)),
            (Swizzle::SW_64KB_S, Tex2d, 4, 1, [100, 60, 6], 7,
                (786432, 1, 0xe6b7_50a0_b65e_09a3)),
            (Swizzle::SW_64KB_S, Tex2d, 8, 1, [65, 33, 1], 7,
                (131072, 1, 0xdf0a_7ddb_9b93_41ee)),
            (Swizzle::SW_64KB_D, Tex2d, 16, 4, [256, 256, 1], 9,
                (131072, 1, 0xe569_7630_03d8_3e28)),
            (Swizzle::SW_64KB_D, Tex2d, 4, 1, [100, 60, 1], 7,
                (131072, 1, 0xb1db_1641_adaf_77a6)),
            (Swizzle::SW_64KB_D, Tex2d, 1, 1, [255, 129, 2], 8,
                (262144, 1, 0xd706_42f5_9a01_5115)),
            (Swizzle::LINEAR, Tex2d, 4, 1, [30, 20, 1], 5,
                (10240, 5, 0x819a_9c12_81e6_40f2)),
            (Swizzle::LINEAR, Tex2d, 2, 1, [17, 9, 3], 5,
                (15360, 5, 0xbf1a_74f1_7803_df62)),
            (Swizzle::LINEAR, Tex1d, 4, 1, [1000, 1, 1], 10,
                (9216, 10, 0xd37e_f9ac_a5de_cab4)),
            (Swizzle::SW_4KB_S, Tex3d, 4, 1, [64, 32, 16], 7,
                (188416, 3, 0xf06b_d42a_7183_90e6)),
            (Swizzle::SW_4KB_S, Tex3d, 4, 1, [4, 4, 16], 5,
                (8192, 0, 0xbf87_001b_ad72_d375)),
            (Swizzle::SW_4KB_S, Tex3d, 1, 1, [1, 1, 17], 2,
                (8192, 0, 0xcdf4_4ee3_33e5_e03d)),
            (Swizzle::SW_64KB_S, Tex3d, 4, 1, [16, 16, 64], 4,
                (262144, 0, 0x5d1a_a7af_c371_e8e5)),
            (Swizzle::SW_64KB_S, Tex3d, 1, 1, [8, 8, 128], 4,
                (262144, 0, 0x8fec_f010_6542_6225)),
            (Swizzle::SW_64KB_S, Tex3d, 16, 1, [33, 17, 40], 6,
                (1966080, 3, 0xec9c_fd44_b89c_2fd5)),
            (Swizzle::LINEAR, Tex3d, 4, 1, [9, 7, 5], 3,
                (16640, 3, 0x6d5b_351d_4a3d_7b5a)),
        ];
        for (swizzle, dimension, element_bytes, block, extent, mips, expected) in cases {
            let [width, height, depth] = extent;
            let d = Desc {
                dimension,
                width,
                height,
                depth,
                mips,
                element_bytes,
                block,
                samples: 1,
                swizzle,
            };
            let l = Layout::new(&d).unwrap();
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for (mip, m) in (0..).zip(&l.mips) {
                let slices = if dimension == Tex3d { m.depth } else { depth };
                for z in 0..slices {
                    for y in 0..m.height {
                        for x in 0..m.width {
                            for byte in l.offset(mip, x, y, z).unwrap().to_le_bytes() {
                                h = (h ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
                            }
                        }
                    }
                }
            }
            assert_eq!((l.size, l.first_tail_mip, h), expected, "{d:?}");
        }
    }

    /// The smaller mips lie first: they are read from the memory that holds
    /// them alone, as from the whole texture's.
    #[test]
    fn smaller_mips_need_only_the_start_of_the_texture() {
        for swizzle in [Swizzle::SW_4KB_S, Swizzle(9)] {
            let l = Layout::new(&Desc { block: 4, ..desc(1024, 1024, 9, 8, swizzle) }).unwrap();
            assert_eq!(l.bytes_from(0), l.size);
            let needs: Vec<u64> = (0..9).map(|mip| l.bytes_from(mip)).collect();
            assert!(needs.windows(2).all(|w| w[0] >= w[1]), "{needs:?}");
            assert!(needs[3] * 4 < l.size, "{needs:?}");
            let tiled: Vec<u8> = (0..l.size).map(|i| ((i * 7) >> 3) as u8).collect();
            for mip in 1..9 {
                let m = l.mips[mip as usize];
                let pitch = m.width as usize * 8;
                let read = |tiled: &[u8]| {
                    let mut linear = vec![0; pitch * m.height as usize];
                    l.detile(tiled, mip, 0, &mut linear, pitch, pitch * m.height as usize).map(|()| linear)
                };
                let len = l.bytes_from(mip) as usize;
                assert_eq!(read(&tiled[..len]), read(&tiled));
                assert_eq!(read(&tiled[..len - 1]), Err(Error::TooSmall));
            }
        }
        // Slices each hold their chain: the whole texture.
        let l = Layout::new(&Desc { depth: 2, block: 4, ..desc(1024, 1024, 9, 8, Swizzle::SW_4KB_S) }).unwrap();
        assert_eq!(l.bytes_from(4), l.size);
    }

    #[test]
    fn depth_stencil_is_two_planes() {
        let t = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            width: 1920,
            height: 1080,
            depth_or_array: 1,
            mips: 1,
            format: 20,
            samples: 1,
            swizzle: Swizzle::SW_64KB_Z_X,
        })
        .unwrap();
        assert_eq!(t.planes.len(), 2);
        // 4-byte depth in 128x128 blocks, 1-byte stencil in 256x256 blocks.
        assert_eq!(t.planes[0].layout.size, 1920 * 1152 * 4);
        assert_eq!(t.planes[1].offset, t.planes[0].layout.size);
        assert_eq!(t.planes[1].layout.size, 2048 * 1280);
        assert!(!t.planes[0].layout.addressable());
    }

    #[test]
    fn texture_copies_reject_overflowing_pitches() {
        let layout = Layout::new(&desc(4, 2, 1, 4, Swizzle::LINEAR)).unwrap();
        let mut tiled = vec![0; layout.size as usize];
        let mut linear = [0; 32];
        assert_eq!(
            layout.detile(&tiled, 0, 0, &mut linear, usize::MAX, 0),
            Err(Error::TooSmall)
        );
        assert_eq!(
            layout.tile(&mut tiled, 0, 0, &linear, usize::MAX, 0),
            Err(Error::TooSmall)
        );
    }

    fn volume_pitch_layout(swizzle: Swizzle) -> Layout {
        Layout::new(&Desc {
            dimension: Dimension::Tex3d,
            depth: 2,
            ..desc(4, 2, 1, 4, swizzle)
        })
        .unwrap()
    }

    #[test]
    fn depth_pitch_rejects_overlapping_detile() {
        for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S] {
            let l = volume_pitch_layout(swizzle);
            let tiled = vec![0x42; l.size as usize];
            let mut linear = [0xcc; 72];

            for pitch in [0, 16, 35] {
                assert_eq!(
                    l.detile(&tiled, 0, 0, &mut linear, 20, pitch),
                    Err(Error::TooSmall),
                    "{swizzle:?}, pitch {pitch}"
                );

                assert_eq!(linear, [0xcc; 72]);
            }
        }
    }

    #[test]
    fn depth_pitch_rejects_overlapping_tile() {
        for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S] {
            let l = volume_pitch_layout(swizzle);
            let mut tiled = vec![0xcc; l.size as usize];
            let linear = [0x42; 72];

            for pitch in [0, 16, 35] {
                assert_eq!(
                    l.tile(&mut tiled, 0, 0, &linear, 20, pitch),
                    Err(Error::TooSmall),
                    "{swizzle:?}, pitch {pitch}"
                );

                assert!(tiled.iter().all(|&b| b == 0xcc));
            }
        }
    }

    #[test]
    fn depth_pitch_accepts_exact_and_padded_slices() {
        for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S] {
            let l = volume_pitch_layout(swizzle);

            for pitch in [36usize, 40, 64] {
                let len = pitch + 36;
                let mut tiled = vec![0; l.size as usize];

                let linear: Vec<u8> = (0..len).map(|i| i as u8).collect();

                let mut back = vec![0xcc; len];

                l.tile(&mut tiled, 0, 0, &linear, 20, pitch).unwrap();

                l.detile(&tiled, 0, 0, &mut back, 20, pitch).unwrap();

                for z in 0..2 {
                    for y in 0..2 {
                        let start = z * pitch + y * 20;

                        assert_eq!(back[start..start + 16], linear[start..start + 16]);
                    }
                }
            }
        }
    }

    #[test]
    fn volume_copy_rejects_overflowing_depth_pitch() {
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex3d,
            depth: 3,
            ..desc(4, 2, 1, 4, Swizzle::LINEAR)
        })
        .unwrap();
        let tiled = vec![0; layout.size as usize];
        let mut linear = [0; 96];
        assert_eq!(
            layout.detile(&tiled, 0, 0, &mut linear, 16, usize::MAX),
            Err(Error::TooSmall)
        );
    }

    #[test]
    fn bad_descriptions_and_buffers_are_refused() {
        assert_eq!(
            Layout::new(&desc(0, 4, 1, 4, Swizzle::SW_4KB_S)),
            Err(Error::Invalid)
        );
        assert_eq!(
            Layout::new(&desc(4, 4, 1, 12, Swizzle::SW_4KB_S)),
            Err(Error::Invalid)
        );
        assert_eq!(
            Layout::new(&desc(4, 4, 1, 4, Swizzle(3))),
            Err(Error::Invalid)
        );
        let l = Layout::new(&desc(64, 64, 1, 4, Swizzle::SW_4KB_S)).unwrap();
        let mut out = vec![0; 64 * 64 * 4];
        assert_eq!(
            l.detile(&[0; 100], 0, 0, &mut out, 256, 0),
            Err(Error::TooSmall)
        );
        assert_eq!(l.offset(0, 64, 0, 0), Err(Error::Invalid));
    }

    /// Pure element-by-element scalar reference using `Layout::offset`.
    fn detile_scalar_ref(
        layout: &Layout,
        tiled: &[u8],
        mip: u32,
        slice: u32,
        linear: &mut [u8],
        row_pitch: usize,
        depth_pitch: usize,
    ) -> Result<(), Error> {
        let m = layout.mips.get(mip as usize).ok_or(Error::Invalid)?;
        let n = layout.desc.element_bytes as usize;
        let first = if layout.desc.dimension == Dimension::Tex3d {
            if slice != 0 {
                return Err(Error::Invalid);
            }
            0
        } else {
            slice
        };
        if first >= layout.slices {
            return Err(Error::Invalid);
        }
        if !layout.addressable() {
            return Err(Error::Unsupported);
        }
        for z in 0..m.depth {
            for y in 0..m.height {
                let row_base = z as usize * depth_pitch + y as usize * row_pitch;
                for x in 0..m.width {
                    let t = layout.offset(mip, x, y, first + z)? as usize;
                    let l = row_base + x as usize * n;
                    linear[l..l + n].copy_from_slice(&tiled[t..t + n]);
                }
            }
        }
        Ok(())
    }

    /// Pure element-by-element scalar reference for tiling.
    fn tile_scalar_ref(
        layout: &Layout,
        tiled: &mut [u8],
        mip: u32,
        slice: u32,
        linear: &[u8],
        row_pitch: usize,
        depth_pitch: usize,
    ) -> Result<(), Error> {
        let m = layout.mips.get(mip as usize).ok_or(Error::Invalid)?;
        let n = layout.desc.element_bytes as usize;
        let first = if layout.desc.dimension == Dimension::Tex3d {
            if slice != 0 {
                return Err(Error::Invalid);
            }
            0
        } else {
            slice
        };
        if first >= layout.slices {
            return Err(Error::Invalid);
        }
        if !layout.addressable() {
            return Err(Error::Unsupported);
        }
        for z in 0..m.depth {
            for y in 0..m.height {
                let row_base = z as usize * depth_pitch + y as usize * row_pitch;
                for x in 0..m.width {
                    let t = layout.offset(mip, x, y, first + z)? as usize;
                    let l = row_base + x as usize * n;
                    tiled[t..t + n].copy_from_slice(&linear[l..l + n]);
                }
            }
        }
        Ok(())
    }

    /// Pre-optimization `walk` detile helper for BEFORE vs AFTER benchmarking.
    fn detile_baseline_walk(
        layout: &Layout,
        tiled: &[u8],
        mip: u32,
        slice: u32,
        linear: &mut [u8],
        row_pitch: usize,
        depth_pitch: usize,
    ) -> Result<(), Error> {
        layout.walk::<true>(tiled, linear, mip, slice, row_pitch, depth_pitch, None)
    }

    /// Deterministic non-repeating pseudo-random byte fill so any swapped byte
    /// or misaddressed micro-tile is caught immediately.
    fn fill_pattern(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
        let mut out = vec![0u8; len];
        for b in &mut out {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *b = (s >> 24) as u8;
        }
        out
    }

    /// Exhaustively compares optimized `detile`/`tile` (and forced portable
    /// fallback) bit-for-bit against the exact scalar `Layout::offset`
    /// reference across every addressable swizzle mode, every element size
    /// (1, 2, 4, 8, 16 bytes -> R8, RG8/R16, RGBA8/R32, BC1..BC7), block
    /// sizes 1 and 4, NPOT sizes, odd pitches, array slices, 3D textures, and
    /// partial mip chains (`bytes_from`).
    #[test]
    fn optimized_detile_and_tile_match_scalar_reference_exhaustively() {
        let swizzles = [
            Swizzle::LINEAR,
            Swizzle::SW_256B_S,
            Swizzle::SW_256B_D,
            Swizzle::SW_4KB_S,
            Swizzle::SW_4KB_D,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        // Diverse power-of-two and NPOT sizes exercising full micro-tiles,
        // right-edge remainders, bottom-edge remainders, and mip tails.
        let dims = [
            (1u32, 1u32, 1u32),
            (3, 5, 2),
            (17, 31, 5),
            (63, 65, 6),
            (109, 71, 6),
            (128, 128, 8),
            (257, 131, 6),
        ];
        let mut cache = TilingCache::default();
        for &swizzle in &swizzles {
            for &bpe in &[1u32, 2, 4, 8, 16] {
                let blocks: &[u32] = if bpe >= 8 { &[1, 4] } else { &[1] };
                for &block in blocks {
                    for &(w, h, mips) in &dims {
                        let d = Desc {
                            dimension: Dimension::Tex2d,
                            width: w * block,
                            height: h * block,
                            depth: 1,
                            mips,
                            element_bytes: bpe,
                            block,
                            samples: 1,
                            swizzle,
                        };
                        let layout = Layout::new(&d).unwrap();
                        let tiled_src = fill_pattern(
                            layout.size as usize,
                            u64::from(swizzle.0) * 97 + u64::from(bpe) * 31 + u64::from(w),
                        );
                        for mip in 0..d.mips {
                            let m = layout.mips[mip as usize];
                            let active = m.width as usize * bpe as usize;
                            // Test both tightly packed and odd/unaligned pitches.
                            for extra_pitch in [0usize, 1, 7, 13] {
                                let row_pitch = active + extra_pitch;
                                let plane = row_pitch * m.height as usize + 5;
                                let partial_len = layout.bytes_from(mip) as usize;
                                let partial = &tiled_src[..partial_len];

                                let mut ref_lin = vec![0xA5u8; plane];
                                let mut opt_lin = vec![0xA5u8; plane];
                                let mut port_lin = vec![0xA5u8; plane];

                                detile_scalar_ref(
                                    &layout,
                                    partial,
                                    mip,
                                    0,
                                    &mut ref_lin,
                                    row_pitch,
                                    plane,
                                )
                                .unwrap();
                                layout
                                    .detile(partial, mip, 0, &mut opt_lin, row_pitch, plane)
                                    .unwrap();
                                crate::detile::detile_portable(
                                    &layout,
                                    partial,
                                    mip,
                                    0,
                                    &mut port_lin,
                                    row_pitch,
                                    plane,
                                )
                                .unwrap();

                                assert_eq!(
                                    opt_lin, ref_lin,
                                    "detile mismatch: {d:?} mip={mip} extra_pitch={extra_pitch}"
                                );
                                assert_eq!(
                                    port_lin, ref_lin,
                                    "portable detile mismatch: {d:?} mip={mip} extra_pitch={extra_pitch}"
                                );

                                // Verify tiling matches scalar reference bit-for-bit too.
                                let full_len = layout.size as usize;
                                let mut ref_tiled = vec![0x5Au8; full_len];
                                let mut opt_tiled = vec![0x5Au8; full_len];
                                let mut port_tiled = vec![0x5Au8; full_len];
                                let mut cache_tiled = vec![0x5Au8; full_len];
                                tile_scalar_ref(
                                    &layout,
                                    &mut ref_tiled,
                                    mip,
                                    0,
                                    &ref_lin,
                                    row_pitch,
                                    plane,
                                )
                                .unwrap();
                                layout
                                    .tile(&mut opt_tiled, mip, 0, &ref_lin, row_pitch, plane)
                                    .unwrap();
                                crate::detile::tile_portable(
                                    &layout,
                                    &mut port_tiled,
                                    mip,
                                    0,
                                    &ref_lin,
                                    row_pitch,
                                    plane,
                                )
                                .unwrap();
                                cache
                                    .tile(
                                        &layout,
                                        &mut cache_tiled,
                                        (mip, 0),
                                        &ref_lin,
                                        (row_pitch, plane),
                                    )
                                    .unwrap();

                                assert_eq!(
                                    opt_tiled, ref_tiled,
                                    "tile mismatch: {d:?} mip={mip} extra_pitch={extra_pitch}"
                                );
                                assert_eq!(
                                    port_tiled, ref_tiled,
                                    "portable tile mismatch: {d:?} mip={mip} extra_pitch={extra_pitch}"
                                );
                                assert_eq!(
                                    cache_tiled, ref_tiled,
                                    "cached tile mismatch: {d:?} mip={mip} extra_pitch={extra_pitch}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Verifies multi-slice arrays, 1D textures, and 3D thick/linear textures
    /// match scalar reference bit-for-bit.
    #[test]
    fn arrays_1d_and_3d_match_scalar_reference() {
        let cases = [
            Desc {
                dimension: Dimension::Tex1d,
                width: 137,
                height: 1,
                depth: 3,
                mips: 5,
                element_bytes: 4,
                block: 1,
                samples: 1,
                swizzle: Swizzle::SW_4KB_D,
            },
            Desc {
                dimension: Dimension::Tex2d,
                width: 193,
                height: 117,
                depth: 3,
                mips: 5,
                element_bytes: 16,
                block: 4,
                samples: 1,
                swizzle: Swizzle::SW_64KB_D,
            },
            Desc {
                dimension: Dimension::Tex3d,
                width: 45,
                height: 29,
                depth: 11,
                mips: 4,
                element_bytes: 8,
                block: 1,
                samples: 1,
                swizzle: Swizzle::SW_4KB_S,
            },
            Desc {
                dimension: Dimension::Tex3d,
                width: 37,
                height: 21,
                depth: 7,
                mips: 3,
                element_bytes: 4,
                block: 1,
                samples: 1,
                swizzle: Swizzle::LINEAR,
            },
        ];
        let mut cache = TilingCache::default();
        for d in cases {
            let layout = Layout::new(&d).unwrap();
            let tiled_src = fill_pattern(
                layout.size as usize,
                u64::from(d.width) * 131 + u64::from(d.swizzle.0),
            );
            let slices = if d.dimension == Dimension::Tex3d { 1 } else { d.depth };
            for slice in 0..slices {
                for mip in 0..d.mips {
                    let m = layout.mips[mip as usize];
                    let active = m.width as usize * d.element_bytes as usize;
                    let row_pitch = active + 19;
                    let depth_pitch = row_pitch * m.height as usize + 23;
                    let total = depth_pitch * m.depth as usize;

                    let mut ref_lin = vec![0x3Cu8; total];
                    let mut opt_lin = vec![0x3Cu8; total];
                    detile_scalar_ref(
                        &layout,
                        &tiled_src,
                        mip,
                        slice,
                        &mut ref_lin,
                        row_pitch,
                        depth_pitch,
                    )
                    .unwrap();
                    layout
                        .detile(&tiled_src, mip, slice, &mut opt_lin, row_pitch, depth_pitch)
                        .unwrap();
                    assert_eq!(
                        opt_lin, ref_lin,
                        "array/3D detile mismatch: {d:?} mip={mip} slice={slice}"
                    );

                    let mut ref_tiled = vec![0xC3u8; layout.size as usize];
                    let mut opt_tiled = vec![0xC3u8; layout.size as usize];
                    let mut cache_tiled = vec![0xC3u8; layout.size as usize];
                    tile_scalar_ref(
                        &layout,
                        &mut ref_tiled,
                        mip,
                        slice,
                        &ref_lin,
                        row_pitch,
                        depth_pitch,
                    )
                    .unwrap();
                    layout
                        .tile(&mut opt_tiled, mip, slice, &ref_lin, row_pitch, depth_pitch)
                        .unwrap();
                    cache
                        .tile(
                            &layout,
                            &mut cache_tiled,
                            (mip, slice),
                            &ref_lin,
                            (row_pitch, depth_pitch),
                        )
                        .unwrap();
                    assert_eq!(
                        opt_tiled, ref_tiled,
                        "array/3D tile mismatch: {d:?} mip={mip} slice={slice}"
                    );
                    assert_eq!(
                        cache_tiled, ref_tiled,
                        "array/3D cached tile mismatch: {d:?} mip={mip} slice={slice}"
                    );
                }
            }
        }
    }

    /// Exercises both sub-threshold inline (`1 MiB`), exact-boundary (`2 MiB`),
    /// and strictly above-threshold parallel worker pool (`> 2 MiB`, e.g. 3 MiB
    /// NPOT and 4 MiB POT) execution across `SW_64KB_D`, `SW_64KB_S`, and
    /// `SW_4KB_D`, plus concurrent multi-caller `try_lock` fallback, verifying
    /// bit-exact parity for both `detile` and `tile`.
    #[test]
    fn parallel_threshold_and_concurrent_callers_match_scalar_reference() {
        let cases = [
            // 1024x1024 BC7 -> 256x256x16 = 1 MiB inline (SW_64KB_D)
            Desc {
                dimension: Dimension::Tex2d,
                width: 1024,
                height: 1024,
                depth: 1,
                mips: 1,
                element_bytes: 16,
                block: 4,
                samples: 1,
                swizzle: Swizzle::SW_64KB_D,
            },
            // 1024x1024 BC7 -> 256x256x16 = 1 MiB inline (SW_64KB_S)
            Desc {
                dimension: Dimension::Tex2d,
                width: 1024,
                height: 1024,
                depth: 1,
                mips: 1,
                element_bytes: 16,
                block: 4,
                samples: 1,
                swizzle: Swizzle::SW_64KB_S,
            },
            // NPOT 1027x515 RGBA8 -> exact 2 MiB full micro-tile body + right/bottom remainders
            Desc {
                dimension: Dimension::Tex2d,
                width: 1027,
                height: 515,
                depth: 1,
                mips: 1,
                element_bytes: 4,
                block: 1,
                samples: 1,
                swizzle: Swizzle::SW_64KB_D,
            },
            // 2048x2048 BC7 -> 512x512x16 = 4 MiB (> PARALLEL_MIN_BYTES, SW_64KB_D)
            Desc {
                dimension: Dimension::Tex2d,
                width: 2048,
                height: 2048,
                depth: 1,
                mips: 1,
                element_bytes: 16,
                block: 4,
                samples: 1,
                swizzle: Swizzle::SW_64KB_D,
            },
            // NPOT 2056x1544 BC7 -> 514x386 blocks (3 MiB full body > PARALLEL_MIN_BYTES + remainders, SW_64KB_S)
            Desc {
                dimension: Dimension::Tex2d,
                width: 2056,
                height: 1544,
                depth: 1,
                mips: 1,
                element_bytes: 16,
                block: 4,
                samples: 1,
                swizzle: Swizzle::SW_64KB_S,
            },
            // 1024x1024 R8 -> 1 MiB inline (SW_4KB_D)
            Desc {
                dimension: Dimension::Tex2d,
                width: 1024,
                height: 1024,
                depth: 1,
                mips: 1,
                element_bytes: 1,
                block: 1,
                samples: 1,
                swizzle: Swizzle::SW_4KB_D,
            },
        ];

        for d in cases {
            let layout = Layout::new(&d).unwrap();
            let tiled_src = fill_pattern(
                layout.size as usize,
                u64::from(d.width) * 17 + u64::from(d.element_bytes),
            );
            let m = layout.mips[0];
            let active = m.width as usize * d.element_bytes as usize;
            let row_pitch = active + 11;
            let plane = row_pitch * m.height as usize + 7;

            let mut ref_lin = vec![0xEEu8; plane];
            detile_baseline_walk(&layout, &tiled_src, 0, 0, &mut ref_lin, row_pitch, plane)
                .unwrap();

            // Run 4 concurrent threads calling detile + tile simultaneously to
            // exercise both pool dispatch and `try_lock` inline fallback.
            std::thread::scope(|s| {
                for tid in 0..4 {
                    let layout = &layout;
                    let tiled_src = &tiled_src;
                    let ref_lin = &ref_lin;
                    s.spawn(move || {
                        let mut opt_lin = vec![0xEEu8; plane];
                        layout
                            .detile(tiled_src, 0, 0, &mut opt_lin, row_pitch, plane)
                            .unwrap();
                        assert_eq!(
                            opt_lin, *ref_lin,
                            "parallel detile mismatch tid={tid} {d:?}"
                        );

                        let mut opt_tiled = vec![0x11u8; layout.size as usize];
                        let mut ref_tiled = vec![0x11u8; layout.size as usize];
                        layout
                            .tile(&mut opt_tiled, 0, 0, &opt_lin, row_pitch, plane)
                            .unwrap();
                        tile_scalar_ref(layout, &mut ref_tiled, 0, 0, &opt_lin, row_pitch, plane)
                            .unwrap();
                        assert_eq!(
                            opt_tiled, ref_tiled,
                            "parallel tile mismatch tid={tid} {d:?}"
                        );
                    });
                }
            });
        }
    }

    /// Release-mode CPU detile throughput benchmark comparing:
    /// - `scalar`: pure element-by-element `Layout::offset` reference
    /// - `baseline`: pre-change chunked `walk` run-coalesced loop
    /// - `optimized`: AVX2/SSE2 micro-tile + LUT + persistent parallel pool
    ///
    /// Run with:
    /// `cargo test --release -p victory-formats --locked -- --ignored --nocapture bench_detile_throughput`
    #[test]
    #[ignore]
    fn bench_detile_throughput() {
        use std::hint::black_box;
        use std::time::Instant;

        struct Case {
            name: &'static str,
            desc: Desc,
            full_chain: bool,
            iters: usize,
        }

        let cases = [
            Case {
                name: "4K BC7 SW_64KB_D (16 MiB)",
                desc: Desc {
                    dimension: Dimension::Tex2d,
                    width: 4096,
                    height: 4096,
                    depth: 1,
                    mips: 1,
                    element_bytes: 16,
                    block: 4,
                    samples: 1,
                    swizzle: Swizzle::SW_64KB_D,
                },
                full_chain: false,
                iters: 20,
            },
            Case {
                name: "4K BC7 SW_64KB_S (16 MiB)",
                desc: Desc {
                    dimension: Dimension::Tex2d,
                    width: 4096,
                    height: 4096,
                    depth: 1,
                    mips: 1,
                    element_bytes: 16,
                    block: 4,
                    samples: 1,
                    swizzle: Swizzle::SW_64KB_S,
                },
                full_chain: false,
                iters: 20,
            },
            Case {
                name: "4K BC1 SW_64KB_D (8 MiB)",
                desc: Desc {
                    dimension: Dimension::Tex2d,
                    width: 4096,
                    height: 4096,
                    depth: 1,
                    mips: 1,
                    element_bytes: 8,
                    block: 4,
                    samples: 1,
                    swizzle: Swizzle::SW_64KB_D,
                },
                full_chain: false,
                iters: 30,
            },
            Case {
                name: "4K RGBA8 SW_64KB_D (64 MiB)",
                desc: Desc {
                    dimension: Dimension::Tex2d,
                    width: 4096,
                    height: 4096,
                    depth: 1,
                    mips: 1,
                    element_bytes: 4,
                    block: 1,
                    samples: 1,
                    swizzle: Swizzle::SW_64KB_D,
                },
                full_chain: false,
                iters: 10,
            },
            Case {
                name: "2K BC7 SW_64KB_D (4 MiB)",
                desc: Desc {
                    dimension: Dimension::Tex2d,
                    width: 2048,
                    height: 2048,
                    depth: 1,
                    mips: 1,
                    element_bytes: 16,
                    block: 4,
                    samples: 1,
                    swizzle: Swizzle::SW_64KB_D,
                },
                full_chain: false,
                iters: 60,
            },
            Case {
                name: "1K BC7 SW_64KB_D (1 MiB)",
                desc: Desc {
                    dimension: Dimension::Tex2d,
                    width: 1024,
                    height: 1024,
                    depth: 1,
                    mips: 1,
                    element_bytes: 16,
                    block: 4,
                    samples: 1,
                    swizzle: Swizzle::SW_64KB_D,
                },
                full_chain: false,
                iters: 160,
            },
            Case {
                name: "256 BC7 SW_64KB_D (64 KiB, ST)",
                desc: Desc {
                    dimension: Dimension::Tex2d,
                    width: 256,
                    height: 256,
                    depth: 1,
                    mips: 1,
                    element_bytes: 16,
                    block: 4,
                    samples: 1,
                    swizzle: Swizzle::SW_64KB_D,
                },
                full_chain: false,
                iters: 2000,
            },
            Case {
                name: "4K BC7 MipChain (13 mips, 21.3 MiB)",
                desc: Desc {
                    dimension: Dimension::Tex2d,
                    width: 4096,
                    height: 4096,
                    depth: 1,
                    mips: 13,
                    element_bytes: 16,
                    block: 4,
                    samples: 1,
                    swizzle: Swizzle::SW_64KB_D,
                },
                full_chain: true,
                iters: 16,
            },
        ];

        println!(
            "\n{:<36} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Case", "Base ms", "Base MB/s", "Opt ms", "Opt MB/s", "vs Base", "vs Scal"
        );
        println!("{}", "-".repeat(108));

        for c in cases {
            let layout = Layout::new(&c.desc).unwrap();
            let tiled = fill_pattern(layout.size as usize, 0x1234_5678);
            let mips: Vec<u32> = if c.full_chain {
                (0..c.desc.mips).collect()
            } else {
                vec![0]
            };
            let mut linear_bufs: Vec<(u32, usize, usize, Vec<u8>)> = mips
                .iter()
                .map(|&mip| {
                    let m = layout.mips[mip as usize];
                    let pitch =
                        (m.width as usize * c.desc.element_bytes as usize).next_multiple_of(256);
                    let len = pitch * m.height as usize;
                    (mip, pitch, len, vec![0u8; len])
                })
                .collect();
            let total_bytes: usize = mips
                .iter()
                .map(|&mip| {
                    let m = layout.mips[mip as usize];
                    m.width as usize * m.height as usize * c.desc.element_bytes as usize
                })
                .sum();

            // Verify bit-exact match before timing.
            let mut ref_bufs: Vec<Vec<u8>> = linear_bufs.iter().map(|b| vec![0u8; b.2]).collect();
            for (i, &(mip, pitch, len, _)) in linear_bufs.iter().enumerate() {
                detile_baseline_walk(&layout, &tiled, mip, 0, &mut ref_bufs[i], pitch, len)
                    .unwrap();
            }
            for (mip, pitch, len, buf) in &mut linear_bufs {
                layout.detile(&tiled, *mip, 0, buf, *pitch, *len).unwrap();
            }
            for (i, (_, _, _, buf)) in linear_bufs.iter().enumerate() {
                assert_eq!(*buf, ref_bufs[i], "benchmark parity failed for {}", c.name);
            }

            // Warmup both paths.
            for (mip, pitch, len, buf) in &mut linear_bufs {
                detile_baseline_walk(&layout, &tiled, *mip, 0, buf, *pitch, *len).unwrap();
                layout.detile(&tiled, *mip, 0, buf, *pitch, *len).unwrap();
            }

            // Time scalar reference (fewer iters for huge textures so bench stays quick).
            let scalar_iters = (c.iters / 4).max(2);
            let t0 = Instant::now();
            for _ in 0..scalar_iters {
                for (mip, pitch, len, buf) in &mut linear_bufs {
                    detile_scalar_ref(
                        &layout,
                        black_box(&tiled),
                        *mip,
                        0,
                        black_box(buf),
                        *pitch,
                        *len,
                    )
                    .unwrap();
                }
            }
            let scalar_ms = t0.elapsed().as_secs_f64() * 1000.0 / scalar_iters as f64;

            // Time pre-change baseline (`walk`).
            let t1 = Instant::now();
            for _ in 0..c.iters {
                for (mip, pitch, len, buf) in &mut linear_bufs {
                    detile_baseline_walk(
                        &layout,
                        black_box(&tiled),
                        *mip,
                        0,
                        black_box(buf),
                        *pitch,
                        *len,
                    )
                    .unwrap();
                }
            }
            let base_ms = t1.elapsed().as_secs_f64() * 1000.0 / c.iters as f64;
            let base_mbs = (total_bytes as f64 / (1024.0 * 1024.0)) / (base_ms / 1000.0);

            // Time optimized `Layout::detile`.
            let t2 = Instant::now();
            for _ in 0..c.iters {
                for (mip, pitch, len, buf) in &mut linear_bufs {
                    layout
                        .detile(black_box(&tiled), *mip, 0, black_box(buf), *pitch, *len)
                        .unwrap();
                }
            }
            let opt_ms = t2.elapsed().as_secs_f64() * 1000.0 / c.iters as f64;
            let opt_mbs = (total_bytes as f64 / (1024.0 * 1024.0)) / (opt_ms / 1000.0);

            println!(
                "{:<36} | {:>10.3} | {:>10.1} | {:>10.3} | {:>10.1} | {:>7.2}x | {:>7.2}x",
                c.name,
                base_ms,
                base_mbs,
                opt_ms,
                opt_mbs,
                base_ms / opt_ms,
                scalar_ms / opt_ms,
            );
        }
    }
}
