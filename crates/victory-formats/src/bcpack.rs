//! BCPack: lossless packing of BC1-BC7 block streams, as the console's
//! DirectStorage hardware unpacks them. Decode only; BC2 is left out (no
//! title data uses it).
//!
//! A packed stream has no header. It is 12 or 16 interleaved rANS *lanes*;
//! each lane is a run of fragments (header byte, length, payload) coded with
//! one of 128 static models, and a fixed schedule decides which lane takes
//! the next input byte. The lanes' symbols form one record per block: two
//! back references (to one of the last 256 blocks, or to a per-format
//! dictionary that stands in for blocks -1 to -255) and the residuals of the
//! endpoints and indices against them. The models, the index rank tables,
//! the BC7 control codes and the dictionaries are constants of the format
//! supplied by the caller ([`Tables::parse`]).

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub &'static str);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BCPack: {}", self.0)
    }
}

impl std::error::Error for Error {}

const MODELS: usize = 128;
const BUCKETS: usize = 9;
const HISTORY: usize = 256;
const DICTIONARY: usize = 255;
/// Blocks decoded per L1-resident strip.
const STRIP: usize = 256;
/// rANS: decode while the state is above this, read while below.
const LOW: u32 = 1 << 24;
/// Input bytes per schedule round.
const ROUND_BYTES: usize = 8;
/// Sentinel for an absent byte in `Lane::latch` and `Lane::out`.
const NONE: i16 = -1;

/// Bytes of the table file (see [`Tables::parse`]).
pub const TABLES_SIZE: usize = MODELS * BUCKETS * 2
    + 256
    + 64
    + 256
    + 64
    + 64
    + 512 * 2
    + DICTIONARY * (6 + 16 + 8 + 32 + 2 + 16 + 4 + 32 + 25 + 33 + 50 + 66);

/// Spreads 4 bits into even bit positions (0, 2, 4, 6).
const SPREAD4: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut v = 0usize;
    while v < 16 {
        let mut out = 0u8;
        let mut k = 0;
        while k < 4 {
            out |= (((v >> k) & 1) as u8) << (2 * k);
            k += 1;
        }
        t[v] = out;
        v += 1;
    }
    t
};

/// Spreads 8 bits into even bit positions (0, 2, ..., 14).
const SPREAD8: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut v = 0usize;
    while v < 256 {
        let lo = SPREAD4[v & 15] as u16;
        let hi = SPREAD4[v >> 4] as u16;
        t[v] = lo | (hi << 8);
        v += 1;
    }
    t
};

/// For each byte, its `[even, odd]` bit-deinterleaved 4-bit zigzag-decoded residuals (`-8..=7`).
const NIBBLES: [[i8; 2]; 256] = {
    let mut t = [[0i8; 2]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut even = 0u8;
        let mut odd = 0u8;
        let mut k = 0;
        while k < 4 {
            even |= (((b >> (2 * k)) & 1) as u8) << k;
            odd |= (((b >> (2 * k + 1)) & 1) as u8) << k;
            k += 1;
        }
        let ze = ((even >> 1) as i8) ^ -((even & 1) as i8);
        let zo = ((odd >> 1) as i8) ^ -((odd & 1) as i8);
        t[b] = [ze, zo];
        b += 1;
    }
    t
};

/// SWAR masks for 16 packed 3-bit lanes in bits `0..48` of a `u64`.
const M3_LO2: u64 = 0o3333_3333_3333_3333;
const M3_HI1: u64 = 0o4444_4444_4444_4444;
const M3_LSB: u64 = 0o1111_1111_1111_1111;

/// Per-bucket rANS decode metadata (16 bytes, power-of-2 stride).
///
/// Bucket 0 is folded branchlessly into the general formula:
/// - `b == 0`: `eff_start = 1`, `eff_freq = freq[0] - 1`, `base_sym = 0`, `recip = 0`.
///   Even when `slot == 0`, `offset = slot.wrapping_sub(1) = 0xffff_ffff`,
///   `q = ((offset * 0) >> 32) = 0`, `sym = 0`, `rem = offset = slot.wrapping_sub(1)`.
/// - `b >= 1`: `eff_start = start[b]`, `eff_freq = freq[b]`, `base_sym = 1 << (b - 1)`,
///   `recip = ceil(2^32 / freq[b])`.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct BucketEntry {
    recip: u64,
    eff_start: u16,
    eff_freq: u16,
    /// `start[b + 1] - 1`, or `65535` for the last used bucket and unused tail.
    last_slot: u16,
    base_sym: u8,
    _pad: u8,
}

/// Cache-packed per-model rANS lookup table (512 bytes, cache-line aligned).
///
/// `buckets` has 16 entries (9 used + 7 sentinel tail) so `(b & 15)` eliminates
/// bounds checks and aligns the whole struct to 512 bytes.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct ModelTable {
    buckets: [BucketEntry; 16],
    /// Bucket index of the first slot of each 256-slot window.
    window: [u8; 256],
}

impl Default for ModelTable {
    fn default() -> Self {
        Self {
            buckets: [BucketEntry::default(); 16],
            window: [0; 256],
        }
    }
}

/// Precomputed BC7 control-word metadata (avoids per-block mode/rotation/anchor work).
#[derive(Clone, Copy, Default)]
struct Ctrl7Meta {
    ctrl: u16,
    header_bits: u16,
    anchor_mask: u16,
    header_len: u8,
    mode: u8,
    subsets: u8,
    columns: u8,
    has_pbits: bool,
    idx_bits: [u8; 2],
    prec: [u8; 4],
    order: [u8; 4],
}

/// Packed history/dictionary entries for plain formats.
#[derive(Clone, Copy, Default)]
struct Bc1Entry {
    ep: [u8; 6],
    /// 16 2-bit indices packed in bits `0..32` (`2 * p`).
    ix: u32,
}

#[derive(Clone, Copy, Default)]
struct Bc3Entry {
    ep: [u8; 8],
    /// 16 3-bit alpha indices packed in bits `0..48` (`3 * p`).
    ix_a: u64,
    /// 16 2-bit colour indices packed in bits `0..32` (`2 * p`).
    ix_c: u32,
}

#[derive(Clone, Copy, Default)]
struct Bc4Entry {
    ep: [u8; 2],
    /// 16 3-bit indices packed in bits `0..48` (`3 * p`).
    ix: u64,
}

#[derive(Clone, Copy, Default)]
struct Bc5Entry {
    ep: [u8; 4],
    /// 16 3-bit red indices packed in bits `0..48` (`3 * p`).
    ix0: u64,
    /// 16 3-bit green indices packed in bits `0..48` (`3 * p`).
    ix1: u64,
}

#[derive(Clone, Copy, Default)]
struct Bc6Ep {
    /// Endpoint 0 R G B, then endpoints 1-3, left-aligned and biased.
    e: [[u16; 3]; 4],
}

#[derive(Clone, Copy, Default)]
struct Bc6Ix {
    mode: u8,
    ix: [i16; 16],
}

#[derive(Clone, Copy, Default)]
struct Bc7Ep {
    ctrl: u16,
    /// Per subset: R0 R1 G0 G1 B0 B1 A0 A1, left-aligned.
    rows: [[u16; 8]; 3],
}

#[derive(Clone, Copy, Default)]
struct Bc7Ix {
    bits: [u8; 2],
    ix: [[i16; 16]; 2],
}

/// Allocates a boxed fixed-size array directly on the heap without placing `[T; N]` on the stack.
fn alloc_boxed_array<T: Default, const N: usize>() -> Box<[T; N]> {
    let mut v = Vec::with_capacity(N);
    v.resize_with(N, T::default);
    match v.into_boxed_slice().try_into() {
        Ok(boxed) => boxed,
        Err(_) => unreachable!(),
    }
}

/// The format's constant tables.
pub struct Tables {
    models: Box<[ModelTable; MODELS]>,
    /// Pre-positioned 2x2 quad masks (`bits 0..4` and `bits 8..12`) for BC1/BC3c.
    rank1_quad: [u16; 256],
    /// Pre-packed 2x3-bit pair masks (`bits 0..6`) for BC3a/BC4/BC5.
    rank3a_pair: [u8; 64],
    rank3c_quad: [u16; 256],
    rank4_pair: [u8; 64],
    rank5_pair: [u8; 64],
    ctrl7_meta: Box<[Ctrl7Meta; 512]>,
    bc1: Vec<Bc1Entry>,
    bc3: Vec<Bc3Entry>,
    bc4: Vec<Bc4Entry>,
    bc5: Vec<Bc5Entry>,
    bc6: Vec<(Bc6Ep, Bc6Ix)>,
    bc7: Vec<(Bc7Ep, Bc7Ix)>,
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let (head, rest) = self.0.split_at(N);
        self.0 = rest;
        head.try_into().unwrap_or([0; N])
    }
    fn u8(&mut self) -> u8 {
        self.bytes::<1>()[0]
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.bytes())
    }
}

#[inline]
fn pack_quad_table(rank: &[u8; 256]) -> [u16; 256] {
    let mut out = [0u16; 256];
    for i in 0..256 {
        let v = rank[i];
        out[i] = u16::from(v & 0x0f) | (u16::from(v & 0xf0) << 4);
    }
    out
}

#[inline]
fn pack_pair_table(rank: &[u8; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for i in 0..64 {
        let v = rank[i];
        out[i] = (v & 7) | ((v >> 4 & 7) << 3);
    }
    out
}

#[inline]
fn pack_ix2(ix: &[u8]) -> u32 {
    let mut out = 0u32;
    for (p, &v) in ix.iter().take(16).enumerate() {
        out |= u32::from(v & 3) << (2 * p);
    }
    out
}

#[inline]
fn pack_ix3(ix: &[u8]) -> u64 {
    let mut out = 0u64;
    for (p, &v) in ix.iter().take(16).enumerate() {
        out |= u64::from(v & 7) << (3 * p);
    }
    out
}

fn build_ctrl7_meta(ctrl: u16) -> Ctrl7Meta {
    if ctrl == 0 {
        return Ctrl7Meta::default();
    }
    let mode = ctrl & 7;
    let subsets = (ctrl & 0xff) >> 6;
    let part = usize::from((ctrl >> 8) & 63);
    let rot = rotation(ctrl);
    let prec = bc7_precision(ctrl);
    let columns = if prec[3] > 0 { 8 } else { 6 };
    let mut order = [0u8, 1, 2, 3];
    if rot != 0 {
        order.swap(3, rot - 1);
    }
    let anchor_mask = match subsets {
        2 => 1u16 | (1u16 << ANCHOR2[part]),
        3 => 1u16 | (1u16 << ANCHOR3A[part]) | (1u16 << ANCHOR3B[part]),
        _ => 1u16,
    };
    let mut header_bits = 1u16 << mode;
    let mut header_len = (mode + 1) as u8;
    match mode {
        0 => {
            header_bits |= ((part as u16) & 0xf) << header_len;
            header_len += 4;
        }
        1 | 2 | 3 | 7 => {
            header_bits |= ((part as u16) & 0x3f) << header_len;
            header_len += 6;
        }
        4 => {
            header_bits |= ((rot as u16) & 3) << header_len;
            header_len += 2;
            header_bits |= ((ctrl >> 5) & 1) << header_len;
            header_len += 1;
        }
        5 => {
            header_bits |= ((rot as u16) & 3) << header_len;
            header_len += 2;
        }
        _ => {}
    }
    Ctrl7Meta {
        ctrl,
        header_bits,
        anchor_mask,
        header_len,
        mode: mode as u8,
        subsets: subsets as u8,
        columns,
        has_pbits: pbits(mode),
        idx_bits: BC7_INDEX[usize::from(mode)],
        prec,
        order,
    }
}

impl Tables {
    /// Reads the table file: 128 x 9 u16 model frequencies (per bucket, the
    /// frequency of each of its symbols), the rank tables (BC1 256 B, BC3
    /// alpha 64 B and colour 256 B, BC4 64 B, BC5 64 B), 512 u16 BC7 control
    /// words, then 255 dictionary entries per format in format order: BC1
    /// 6 + 16 B, BC3 8 + 32 B, BC4 2 + 16 B, BC5 4 + 32 B, BC6H (u8 mode,
    /// 12 u16, u8 mode, 16 i16), BC7 (u16 control, 24 u16, 2 u8, 2 B,
    /// 16 + 15 i16). Little-endian, no padding. Every model must tile 65536.
    pub fn parse(bytes: &[u8]) -> Result<Tables, Error> {
        if bytes.len() != TABLES_SIZE {
            return Err(Error("table file has the wrong size"));
        }
        let mut r = Reader(bytes);
        let mut models = alloc_boxed_array::<ModelTable, MODELS>();

        for m in 0..MODELS {
            let mut f = [0u32; BUCKETS];
            for v in &mut f {
                *v = u32::from(r.u16());
            }
            // Model 16 g + shape has g + 2 buckets; bucket b > 0 holds 2^(b-1) symbols.
            let used = m / 16 + 2;
            let mut st = [0u32; BUCKETS + 1];
            let mut rec = [0u64; BUCKETS];
            let mut at = 0u32;
            for b in 0..used {
                if f[b] == 0 {
                    return Err(Error("a model has an empty bucket"));
                }
                st[b] = at;
                rec[b] = (1u64 << 32).div_ceil(u64::from(f[b]));
                at += f[b] << b.saturating_sub(1);
            }
            if at != 1 << 16 {
                return Err(Error("a model does not tile 65536"));
            }
            st[used..].fill(1 << 16);

            let mt = &mut models[m];
            // Bucket 0: branchless fold (`eff_start = 1`, `eff_freq = f[0] - 1`, `recip = 0`).
            mt.buckets[0] = BucketEntry {
                recip: 0,
                eff_start: 1,
                eff_freq: (f[0] - 1) as u16,
                last_slot: (st[1] - 1) as u16,
                base_sym: 0,
                _pad: 0,
            };
            for b in 1..used {
                mt.buckets[b] = BucketEntry {
                    recip: rec[b],
                    eff_start: st[b] as u16,
                    eff_freq: f[b] as u16,
                    last_slot: (st[b + 1] - 1) as u16,
                    base_sym: (1u32 << (b - 1)) as u8,
                    _pad: 0,
                };
            }
            for b in used..16 {
                mt.buckets[b].last_slot = u16::MAX;
            }
            for (window, b) in mt.window.iter_mut().enumerate() {
                let first = (window as u32) << 8;
                *b = st[1..].iter().filter(|&&s| s <= first).count() as u8;
            }
        }

        let rank1_quad = pack_quad_table(&r.bytes());
        let rank3a_pair = pack_pair_table(&r.bytes());
        let rank3c_quad = pack_quad_table(&r.bytes());
        let rank4_pair = pack_pair_table(&r.bytes());
        let rank5_pair = pack_pair_table(&r.bytes());

        let mut ctrl7_meta = alloc_boxed_array::<Ctrl7Meta, 512>();
        for meta in ctrl7_meta.iter_mut() {
            *meta = build_ctrl7_meta(r.u16());
        }

        let mut bc1 = Vec::with_capacity(DICTIONARY);
        for _ in 0..DICTIONARY {
            let ep: [u8; 6] = r.bytes();
            let ix: [u8; 16] = r.bytes();
            bc1.push(Bc1Entry {
                ep,
                ix: pack_ix2(&ix),
            });
        }

        let mut bc3 = Vec::with_capacity(DICTIONARY);
        for _ in 0..DICTIONARY {
            let ep: [u8; 8] = r.bytes();
            let ix: [u8; 32] = r.bytes();
            bc3.push(Bc3Entry {
                ep,
                ix_a: pack_ix3(&ix[..16]),
                ix_c: pack_ix2(&ix[16..]),
            });
        }

        let mut bc4 = Vec::with_capacity(DICTIONARY);
        for _ in 0..DICTIONARY {
            let ep: [u8; 2] = r.bytes();
            let ix: [u8; 16] = r.bytes();
            bc4.push(Bc4Entry {
                ep,
                ix: pack_ix3(&ix),
            });
        }

        let mut bc5 = Vec::with_capacity(DICTIONARY);
        for _ in 0..DICTIONARY {
            let ep: [u8; 4] = r.bytes();
            let ix: [u8; 32] = r.bytes();
            bc5.push(Bc5Entry {
                ep,
                ix0: pack_ix3(&ix[..16]),
                ix1: pack_ix3(&ix[16..]),
            });
        }

        let bc6 = (0..DICTIONARY)
            .map(|_| {
                // The endpoint entry's mode byte plays no part in decoding.
                r.u8();
                let mut ep = Bc6Ep::default();
                for v in ep.e.as_flattened_mut() {
                    *v = r.u16();
                }
                let mut ix = Bc6Ix {
                    mode: r.u8(),
                    ..Default::default()
                };
                for v in &mut ix.ix {
                    *v = r.u16() as i16;
                }
                (ep, ix)
            })
            .collect();

        let bc7 = (0..DICTIONARY)
            .map(|k| {
                let mut ep = Bc7Ep {
                    ctrl: r.u16(),
                    ..Default::default()
                };
                for v in ep.rows.as_flattened_mut() {
                    *v = r.u16();
                }
                let mut ix = Bc7Ix {
                    bits: [r.u8(), r.u8()],
                    ..Default::default()
                };
                // Entry 0 holds all 32 indices right after its widths; the
                // others repeat the widths (unused) and hold 31, as set 1
                // has only 15 slots, which no entry uses (0 bits).
                let n = if k == 0 {
                    32
                } else {
                    r.u16();
                    31
                };
                for v in &mut ix.ix.as_flattened_mut()[..n] {
                    *v = r.u16() as i16;
                }
                (ep, ix)
            })
            .collect();

        Ok(Tables {
            models,
            rank1_quad,
            rank3a_pair,
            rank3c_quad,
            rank4_pair,
            rank5_pair,
            ctrl7_meta,
            bc1,
            bc3,
            bc4,
            bc5,
            bc6,
            bc7,
        })
    }
}

/// Unpacks `src` (BC number `bc`, 1 and 3-7) into `dst`, whose length gives
/// the block count. The stream carries no size: bytes after the last one the
/// lanes need are ignored, a stream that runs out is an error.
pub fn decode(bc: u8, src: &[u8], dst: &mut [u8], tables: &Tables) -> Result<(), Error> {
    let block = match bc {
        1 | 4 => 8,
        3 | 5 | 6 | 7 => 16,
        _ => return Err(Error("unsupported BC format")),
    };
    if dst.is_empty() || !dst.len().is_multiple_of(block) {
        return Err(Error("destination is not a whole number of blocks"));
    }
    match bc {
        1 => decode_bc1(src, dst, tables),
        3 => decode_bc3(src, dst, tables),
        4 => decode_bc4(src, dst, tables),
        5 => decode_bc5(src, dst, tables),
        6 => decode_bc6(src, dst, tables),
        _ => decode_bc7(src, dst, tables),
    }
}

// Lanes.

const BC1_WIDTHS: [u8; 12] = [7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7];
const BC3_WIDTHS: [u8; 12] = [3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7];
const BC4_WIDTHS: [u8; 12] = [7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5];
const BC5_WIDTHS: [u8; 12] = [3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5];
const BC6_WIDTHS: [u8; 16] = [3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4];
const BC7_WIDTHS: [u8; 16] = [3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4];

/// Per-lane rANS state (16 bytes).
#[derive(Clone, Copy)]
#[repr(C)]
struct Lane {
    x: u32,
    left: u32,
    /// Latched input byte (`0..=255`), or `NONE` (`-1`).
    latch: i16,
    /// Decoded output symbol (`0..=255`), or `NONE` (`-1`).
    out: i16,
    model: u8,
    need: bool,
    stalled: bool,
    /// Bits `0..2`: phase (0 header, 1 len lo, 2 len hi, 3 payload);
    /// bit 4: reset; bit 5: long.
    state: u8,
}

impl Lane {
    #[inline(always)]
    const fn new() -> Self {
        Self {
            x: 0,
            left: 0,
            latch: NONE,
            out: NONE,
            model: 0,
            need: true,
            stalled: false,
            state: 0,
        }
    }

    #[inline(always)]
    fn phase(&self) -> u8 {
        self.state & 3
    }

    /// Branchless rANS symbol decode for `x > LOW`.
    #[inline(always)]
    fn decode_symbol(model: &ModelTable, x: u32) -> (u8, u32) {
        let slot16 = x as u16;
        let mut b = usize::from(model.window[(slot16 >> 8) as usize]) & 15;
        while model.buckets[b].last_slot < slot16 {
            b = (b + 1) & 15;
        }
        let entry = &model.buckets[b];
        let offset = u32::from(slot16).wrapping_sub(u32::from(entry.eff_start));
        let q = ((u64::from(offset) * entry.recip) >> 32) as u32;
        let sym = (u32::from(entry.base_sym) + q) as u8;
        let f = u32::from(entry.eff_freq);
        (
            sym,
            (x >> 16)
                .wrapping_sub(q)
                .wrapping_mul(f)
                .wrapping_add(offset),
        )
    }

    /// Full step handling all phases, fragment transitions, and resets.
    #[inline]
    fn step_slow(&mut self, width: u8, t: &Tables) {
        let phase = self.state & 3;
        if phase != 3 {
            if self.latch < 0 {
                return;
            }
            let b = self.latch as u8;
            self.latch = NONE;
            self.need = true;
            match phase {
                0 => {
                    // The top two bits shrink the alphabet; width >= 3.
                    let group = width - (b >> 6);
                    self.model = group * 16 + (b & 15);
                    self.left = 0;
                    // Store reset (bit 4), long (bit 5), and next phase 1.
                    self.state = (b & 0x30) | 1;
                }
                1 if self.state & 0x20 != 0 => {
                    self.left = u32::from(b);
                    self.state = (self.state & 0x30) | 2;
                }
                1 => {
                    self.left = u32::from(b) + 1;
                    self.state = (self.state & 0x30) | 3;
                }
                _ => {
                    self.left = (u32::from(b) << 8 | self.left) + 257;
                    self.state = (self.state & 0x30) | 3;
                }
            }
            return;
        }
        let mut x = self.x;
        if x < LOW && self.latch >= 0 {
            let b = self.latch as u32;
            self.latch = NONE;
            self.left = self.left.wrapping_sub(1);
            x = (x << 8) | b;
        }
        self.need = self.left != 0 && self.latch < 0;
        if x > LOW {
            let model = &t.models[usize::from(self.model & 127)];
            let (sym, next_x) = Self::decode_symbol(model, x);
            self.x = next_x;
            self.out = i16::from(sym);
        } else if self.left != 0 {
            // At exactly 2^24 the lane neither reads nor decodes again; only a
            // lane that has delivered its symbols may end up there.
            self.x = x;
        } else {
            // The fragment ends; a byte read in this step does not reach the carried state.
            self.state &= 0x30; // phase = 0
            self.need = self.latch < 0;
            if self.state & 0x10 != 0 {
                self.x = 0;
            }
        }
    }
}

/// Decodes one strip of `records.len()` blocks into `records`, advancing `lanes`
/// and `src_pos` in place so state carries seamlessly across strips.
fn lanes_strip<const LANES: usize, const PER: usize, const STRIDE: usize>(
    t: &Tables,
    widths: &[u8; LANES],
    lanes: &mut [Lane; LANES],
    src: &[u8],
    src_pos: &mut usize,
    records: &mut [[u8; STRIDE]],
) -> Result<(), Error> {
    debug_assert!(PER == 1 || PER == 2);
    debug_assert_eq!(STRIDE, LANES * PER);
    let strip_blocks = records.len();
    if strip_blocks == 0 {
        return Ok(());
    }
    let full: u32 = (1u32 << LANES) - 1;
    let mut block_idx = 0usize;
    let mut sub = 0usize;
    let mut done = 0u32;
    let mut idle = 0u32;

    'outer: loop {
        // Steady-state fast burst: when every lane is in payload phase with
        // `left >= 2` and at least `ROUND_BYTES` remain in `src`, we can run
        // `min_left - 1` rounds without fragment-end or EOF checks.
        let mut min_left = u32::MAX;
        let mut all_payload = true;
        for lane in lanes.iter() {
            if lane.phase() != 3 {
                all_payload = false;
                break;
            }
            if lane.left < min_left {
                min_left = lane.left;
            }
        }
        if all_payload && min_left > 1 {
            let mut fast_rounds = min_left - 1;
            while fast_rounds > 0 && src.len() - *src_pos >= ROUND_BYTES {
                let src_chunk: &[u8; ROUND_BYTES] =
                    src[*src_pos..*src_pos + ROUND_BYTES].try_into().unwrap();
                let rec = &mut records[block_idx];
                let mut taken = 0usize;
                let mut moved = false;

                for j in 0..LANES {
                    let lane = &mut lanes[j];
                    let bit = 1u32 << j;
                    let mut out = lane.out;
                    let mut need = lane.need;
                    if !lane.stalled {
                        let mut x = lane.x;
                        let mut latch = lane.latch;
                        if x < LOW && latch >= 0 {
                            lane.left -= 1;
                            x = (x << 8) | (latch as u32);
                            latch = NONE;
                            lane.latch = NONE;
                        }
                        need = latch < 0;
                        lane.need = need;
                        if x > LOW {
                            let model = &t.models[usize::from(lane.model & 127)];
                            let (sym, next_x) = Lane::decode_symbol(model, x);
                            lane.x = next_x;
                            out = i16::from(sym);
                            lane.out = out;
                        } else {
                            lane.x = x;
                        }
                    }
                    let waiting = out >= 0 && (done & bit) != 0;
                    let stalled = if waiting {
                        true
                    } else if need {
                        if taken < ROUND_BYTES {
                            lane.latch = i16::from(src_chunk[taken]);
                            taken += 1;
                            false
                        } else {
                            true
                        }
                    } else {
                        false
                    };
                    lane.stalled = stalled;
                    if !stalled && out >= 0 {
                        rec[j * PER + sub] = out as u8;
                        lane.out = NONE;
                        done |= bit;
                        moved = true;
                    }
                }

                *src_pos += taken;
                if done == full {
                    done = 0;
                    moved = true;
                    if PER == 1 {
                        block_idx += 1;
                        if block_idx == strip_blocks {
                            break 'outer;
                        }
                    } else if sub == 0 {
                        sub = 1;
                    } else {
                        sub = 0;
                        block_idx += 1;
                        if block_idx == strip_blocks {
                            break 'outer;
                        }
                    }
                }
                if taken > 0 || moved {
                    idle = 0;
                } else {
                    idle += 1;
                    if idle > 3 {
                        return Err(Error("the stream ends early or is corrupt"));
                    }
                }
                fast_rounds -= 1;
            }
        }

        // Slow round: handles fragment headers/lengths/resets and EOF tail.
        let rec = &mut records[block_idx];
        let mut taken = 0usize;
        let mut moved = false;
        for j in 0..LANES {
            let lane = &mut lanes[j];
            let bit = 1u32 << j;
            let before = taken;
            if !lane.stalled {
                lane.step_slow(widths[j], t);
            }
            let waiting = lane.out >= 0 && (done & bit) != 0;
            if lane.need && !waiting && taken < ROUND_BYTES && *src_pos < src.len() {
                lane.latch = i16::from(src[*src_pos]);
                *src_pos += 1;
                taken += 1;
            }
            lane.stalled = (lane.need && before >= ROUND_BYTES) || waiting;
            if !lane.stalled && lane.out >= 0 {
                rec[j * PER + sub] = lane.out as u8;
                lane.out = NONE;
                done |= bit;
                moved = true;
            }
        }
        if done == full {
            done = 0;
            moved = true;
            if PER == 1 {
                block_idx += 1;
                if block_idx == strip_blocks {
                    break 'outer;
                }
            } else if sub == 0 {
                sub = 1;
            } else {
                sub = 0;
                block_idx += 1;
                if block_idx == strip_blocks {
                    break 'outer;
                }
            }
        }
        if taken > 0 || moved {
            idle = 0;
        } else {
            idle += 1;
            if idle > 3 {
                return Err(Error("the stream ends early or is corrupt"));
            }
        }
    }

    Ok(())
}

// Block reconstruction helpers.

/// Two 4-bit symbols, the first on the even bits.
#[inline(always)]
fn interleave(a: u8, b: u8) -> u8 {
    SPREAD4[usize::from(a & 15)] | (SPREAD4[usize::from(b & 15)] << 1)
}

/// Two 8-bit symbols, the first on the even bits.
#[inline(always)]
fn wide(a: u8, b: u8) -> u16 {
    SPREAD8[usize::from(a)] | (SPREAD8[usize::from(b)] << 1)
}

/// The record's endpoint and index references.
#[inline(always)]
fn references_unpaired(s: &[u8]) -> (usize, usize) {
    let d = s[0];
    let ix = s[1];
    let delta = (d >> 1) ^ (d & 1).wrapping_neg();
    (usize::from(ix.wrapping_add(delta)), usize::from(ix))
}

#[inline(always)]
fn references_paired(s: &[u8]) -> (usize, usize) {
    let d = interleave(s[0], s[1]);
    let ix = interleave(s[2], s[3]);
    let delta = (d >> 1) ^ (d & 1).wrapping_neg();
    (usize::from(ix.wrapping_add(delta)), usize::from(ix))
}

/// Block `i`'s reference `r`: 0 none, else `r` blocks back, before the first
/// block into the dictionary.
#[inline(always)]
fn pick<'a, T>(
    i: usize,
    r: usize,
    history: &'a [T; HISTORY],
    dictionary: &'a [T],
    zero: &'a T,
) -> &'a T {
    if r == 0 {
        zero
    } else if r <= i {
        &history[(i - r) & (HISTORY - 1)]
    } else {
        &dictionary[r - i - 1]
    }
}

/// Signed zigzag byte `(0, -1, 1, -2, ...)` in two's complement `u8`.
#[inline(always)]
fn zz_u8(v: u8) -> u8 {
    (v >> 1) ^ (v & 1).wrapping_neg()
}

/// SWAR addition of 16 packed 2-bit lanes modulo 4.
#[inline(always)]
fn swar_add2(a: u32, b: u32) -> u32 {
    let sum_lo = (a & 0x5555_5555) + (b & 0x5555_5555);
    let sum_hi = (a ^ b) & 0xaaaa_aaaa;
    sum_lo ^ sum_hi
}

/// SWAR conversion of 16 packed 2-bit linear indices `[0,1,2,3]` -> BC1 codes `[0,2,3,1]`.
#[inline(always)]
fn swar_bc1_code(ix: u32) -> u32 {
    let lo = (ix >> 1) & 0x5555_5555;
    let hi = (ix ^ (ix << 1)) & 0xaaaa_aaaa;
    lo | hi
}

/// SWAR addition of 16 packed 3-bit lanes modulo 8 (in bits `0..48`).
#[inline(always)]
fn swar_add3(a: u64, b: u64) -> u64 {
    let sum_lo = (a & M3_LO2) + (b & M3_LO2);
    let sum_hi = (a ^ b) & M3_HI1;
    sum_lo ^ sum_hi
}

/// SWAR conversion of 16 packed 3-bit linear indices `[0..=7]` -> BC4 codes `[0,2,3,4,5,6,7,1]`.
#[inline(always)]
fn swar_bc4_code(ix: u64) -> u64 {
    let nz = (ix | (ix >> 1) | (ix >> 2)) & M3_LSB;
    let s7 = (ix & (ix >> 1) & (ix >> 2)) & M3_LSB;
    swar_add3(ix, nz + s7)
}

#[inline(always)]
fn quads_packed(rank_quad: &[u16; 256], s0: u8, s1: u8, s2: u8, s3: u8) -> u32 {
    u32::from(rank_quad[usize::from(s0)])
        | (u32::from(rank_quad[usize::from(s1)]) << 4)
        | (u32::from(rank_quad[usize::from(s2)]) << 16)
        | (u32::from(rank_quad[usize::from(s3)]) << 20)
}

#[inline(always)]
fn pairs_packed(rank_pair: &[u8; 64], s: &[u8]) -> u64 {
    u64::from(rank_pair[usize::from(s[0] & 63)])
        | (u64::from(rank_pair[usize::from(s[1] & 63)]) << 6)
        | (u64::from(rank_pair[usize::from(s[2] & 63)]) << 12)
        | (u64::from(rank_pair[usize::from(s[3] & 63)]) << 18)
        | (u64::from(rank_pair[usize::from(s[4] & 63)]) << 24)
        | (u64::from(rank_pair[usize::from(s[5] & 63)]) << 30)
        | (u64::from(rank_pair[usize::from(s[6] & 63)]) << 36)
        | (u64::from(rank_pair[usize::from(s[7] & 63)]) << 42)
}

#[inline(always)]
fn write_bc1_color(ep: &[u8; 6], ix: u32, out: &mut [u8]) {
    let c0 = (u16::from(ep[0]) << 11) | (u16::from(ep[1]) << 5) | u16::from(ep[2]);
    let c1 = (u16::from(ep[3]) << 11) | (u16::from(ep[4]) << 5) | u16::from(ep[5]);
    let bits = swar_bc1_code(ix);
    let block = u64::from(c0) | (u64::from(c1) << 16) | (u64::from(bits) << 32);
    out[..8].copy_from_slice(&block.to_le_bytes());
}

#[inline(always)]
fn write_bc4_alpha(e0: u8, e1: u8, ix: u64, out: &mut [u8]) {
    let bits = swar_bc4_code(ix);
    let block = u64::from(e0 ^ 0x80) | (u64::from(e1 ^ 0x80) << 8) | (bits << 16);
    out[..8].copy_from_slice(&block.to_le_bytes());
}

// Specialized strip-streaming decoders for BC1, BC3, BC4, BC5.

#[inline(never)]
fn decode_bc1(src: &[u8], dst: &mut [u8], t: &Tables) -> Result<(), Error> {
    let mut lanes = [Lane::new(); 12];
    let mut src_pos = 0usize;
    let mut strip_buf = [[0u8; 12]; STRIP];
    let mut history = [Bc1Entry::default(); HISTORY];
    let zero = Bc1Entry::default();
    let dict = &t.bc1;
    let rank1 = &t.rank1_quad;
    let mut block_base = 0usize;

    for out_strip in dst.chunks_mut(STRIP * 8) {
        let n = out_strip.len() / 8;
        let recs = &mut strip_buf[..n];
        lanes_strip::<12, 1, 12>(t, &BC1_WIDTHS, &mut lanes, src, &mut src_pos, recs)?;
        for (k, (s, out)) in recs.iter().zip(out_strip.chunks_exact_mut(8)).enumerate() {
            let i = block_base + k;
            let (ep_ref, ix_ref) = references_unpaired(s);
            let er = &pick(i, ep_ref, &history, dict, &zero).ep;
            let ir = pick(i, ix_ref, &history, dict, &zero).ix;
            let ep = [
                er[0].wrapping_add(zz_u8(s[2])) & 0x1f,
                er[1].wrapping_add(zz_u8(s[3])) & 0x3f,
                er[2].wrapping_add(zz_u8(s[4])) & 0x1f,
                er[3].wrapping_add(zz_u8(s[5])) & 0x1f,
                er[4].wrapping_add(zz_u8(s[6])) & 0x3f,
                er[5].wrapping_add(zz_u8(s[7])) & 0x1f,
            ];
            let ix = swar_add2(ir, quads_packed(rank1, s[8], s[9], s[10], s[11]));
            history[i & (HISTORY - 1)] = Bc1Entry { ep, ix };
            write_bc1_color(&ep, ix, out);
        }
        block_base += n;
    }
    Ok(())
}

#[inline(never)]
fn decode_bc3(src: &[u8], dst: &mut [u8], t: &Tables) -> Result<(), Error> {
    let mut lanes = [Lane::new(); 12];
    let mut src_pos = 0usize;
    let mut strip_buf = [[0u8; 24]; STRIP];
    let mut history = [Bc3Entry::default(); HISTORY];
    let zero = Bc3Entry::default();
    let dict = &t.bc3;
    let rank3a = &t.rank3a_pair;
    let rank3c = &t.rank3c_quad;
    let mut block_base = 0usize;

    for out_strip in dst.chunks_mut(STRIP * 16) {
        let n = out_strip.len() / 16;
        let recs = &mut strip_buf[..n];
        lanes_strip::<12, 2, 24>(t, &BC3_WIDTHS, &mut lanes, src, &mut src_pos, recs)?;
        for (k, (s, out)) in recs.iter().zip(out_strip.chunks_exact_mut(16)).enumerate() {
            let i = block_base + k;
            let (ep_ref, ix_ref) = references_paired(s);
            let er = &pick(i, ep_ref, &history, dict, &zero).ep;
            let ir = pick(i, ix_ref, &history, dict, &zero);
            let ep = [
                er[0].wrapping_add(zz_u8(s[14])) & 0x1f,
                er[1].wrapping_add(zz_u8(s[16])) & 0x3f,
                er[2].wrapping_add(zz_u8(s[18])) & 0x1f,
                er[3].wrapping_add(zz_u8(s[15])) & 0x1f,
                er[4].wrapping_add(zz_u8(s[17])) & 0x3f,
                er[5].wrapping_add(zz_u8(s[19])) & 0x1f,
                er[6].wrapping_add(zz_u8(s[4])),
                er[7].wrapping_add(zz_u8(s[5])),
            ];
            let ix_a = swar_add3(ir.ix_a, pairs_packed(rank3a, &s[6..14]));
            let ix_c = swar_add2(ir.ix_c, quads_packed(rank3c, s[20], s[21], s[22], s[23]));
            history[i & (HISTORY - 1)] = Bc3Entry { ep, ix_a, ix_c };
            write_bc4_alpha(ep[6], ep[7], ix_a, &mut out[..8]);
            let ep_c = [ep[0], ep[1], ep[2], ep[3], ep[4], ep[5]];
            write_bc1_color(&ep_c, ix_c, &mut out[8..16]);
        }
        block_base += n;
    }
    Ok(())
}

#[inline(never)]
fn decode_bc4(src: &[u8], dst: &mut [u8], t: &Tables) -> Result<(), Error> {
    let mut lanes = [Lane::new(); 12];
    let mut src_pos = 0usize;
    let mut strip_buf = [[0u8; 12]; STRIP];
    let mut history = [Bc4Entry::default(); HISTORY];
    let zero = Bc4Entry::default();
    let dict = &t.bc4;
    let rank4 = &t.rank4_pair;
    let mut block_base = 0usize;

    for out_strip in dst.chunks_mut(STRIP * 8) {
        let n = out_strip.len() / 8;
        let recs = &mut strip_buf[..n];
        lanes_strip::<12, 1, 12>(t, &BC4_WIDTHS, &mut lanes, src, &mut src_pos, recs)?;
        for (k, (s, out)) in recs.iter().zip(out_strip.chunks_exact_mut(8)).enumerate() {
            let i = block_base + k;
            let (ep_ref, ix_ref) = references_unpaired(s);
            let er = &pick(i, ep_ref, &history, dict, &zero).ep;
            let ir = pick(i, ix_ref, &history, dict, &zero).ix;
            let ep = [
                er[0].wrapping_add(zz_u8(s[2])),
                er[1].wrapping_add(zz_u8(s[3])),
            ];
            let ix = swar_add3(ir, pairs_packed(rank4, &s[4..12]));
            history[i & (HISTORY - 1)] = Bc4Entry { ep, ix };
            write_bc4_alpha(ep[0], ep[1], ix, out);
        }
        block_base += n;
    }
    Ok(())
}

#[inline(never)]
fn decode_bc5(src: &[u8], dst: &mut [u8], t: &Tables) -> Result<(), Error> {
    let mut lanes = [Lane::new(); 12];
    let mut src_pos = 0usize;
    let mut strip_buf = [[0u8; 24]; STRIP];
    let mut history = [Bc5Entry::default(); HISTORY];
    let zero = Bc5Entry::default();
    let dict = &t.bc5;
    let rank5 = &t.rank5_pair;
    let mut block_base = 0usize;

    for out_strip in dst.chunks_mut(STRIP * 16) {
        let n = out_strip.len() / 16;
        let recs = &mut strip_buf[..n];
        lanes_strip::<12, 2, 24>(t, &BC5_WIDTHS, &mut lanes, src, &mut src_pos, recs)?;
        for (k, (s, out)) in recs.iter().zip(out_strip.chunks_exact_mut(16)).enumerate() {
            let i = block_base + k;
            let (ep_ref, ix_ref) = references_paired(s);
            let er = &pick(i, ep_ref, &history, dict, &zero).ep;
            let ir = pick(i, ix_ref, &history, dict, &zero);
            let ep = [
                er[0].wrapping_add(zz_u8(s[4])),
                er[1].wrapping_add(zz_u8(s[5])),
                er[2].wrapping_add(zz_u8(s[6])),
                er[3].wrapping_add(zz_u8(s[7])),
            ];
            let ix0 = swar_add3(ir.ix0, pairs_packed(rank5, &s[8..16]));
            let ix1 = swar_add3(ir.ix1, pairs_packed(rank5, &s[16..24]));
            history[i & (HISTORY - 1)] = Bc5Entry { ep, ix0, ix1 };
            write_bc4_alpha(ep[0], ep[1], ix0, &mut out[..8]);
            write_bc4_alpha(ep[2], ep[3], ix1, &mut out[8..16]);
        }
        block_base += n;
    }
    Ok(())
}

/// Fast two-word little-endian bit writer for one 128-bit block.
#[derive(Default)]
struct Bits {
    lo: u64,
    hi: u64,
    at: u32,
}

impl Bits {
    #[inline(always)]
    fn put(&mut self, value: u64, n: u32) {
        if n > 0 {
            let masked = value & ((1u64 << n) - 1);
            self.put_unmasked(masked, n);
        }
    }

    /// Writes `value` assuming all bits at and above `n` are already zero (`n < 64`).
    #[inline(always)]
    fn put_unmasked(&mut self, value: u64, n: u32) {
        let at = self.at;
        if at < 64 {
            self.lo |= value << at;
            self.hi |= (value >> 1) >> (63 - at);
        } else if at < 128 {
            self.hi |= value << (at - 64);
        }
        self.at = at + n;
    }

    #[inline(always)]
    fn write_to(&self, out: &mut [u8]) {
        out[..8].copy_from_slice(&self.lo.to_le_bytes());
        out[8..16].copy_from_slice(&self.hi.to_le_bytes());
    }

    #[cfg(test)]
    fn to_le_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        self.write_to(&mut out);
        out
    }
}

/// Reconstructs 16 signed index values from 8 packed nibble bytes (first 6 bytes
/// odd-first, last 2 bytes even-first) plus the reference indices.
#[inline(always)]
fn decode_index_set(
    bytes: [u8; 8],
    bits: u8,
    ref_ix: &[i16; 16],
    ref_bits: u8,
    dst: &mut [i16; 16],
) {
    let sh = 16 - u32::from(bits);
    let mut raw = [0i16; 16];
    for k in 0..6 {
        let [e, o] = NIBBLES[usize::from(bytes[k])];
        raw[2 * k] = i16::from(o);
        raw[2 * k + 1] = i16::from(e);
    }
    for k in 6..8 {
        let [e, o] = NIBBLES[usize::from(bytes[k])];
        raw[2 * k] = i16::from(e);
        raw[2 * k + 1] = i16::from(o);
    }
    if ref_bits == 0 || ref_bits > 16 {
        for p in 0..16 {
            dst[p] = ((raw[p] as u16) << sh) as i16 >> sh;
        }
    } else if ref_bits == bits {
        for p in 0..16 {
            let sum = (raw[p] as u16).wrapping_add(ref_ix[p] as u16) << sh;
            dst[p] = (sum as i16) >> sh;
        }
    } else {
        let ref_sh = 16 - u32::from(ref_bits);
        for p in 0..16 {
            let sum = ((raw[p] as u16) << sh).wrapping_add((ref_ix[p] as u16) << ref_sh);
            dst[p] = (sum as i16) >> sh;
        }
    }
}

// BC7.

/// Public BC7 anchor indices: second subset of two; second and third of three.
const ANCHOR2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
];
const ANCHOR3A: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5,
    15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5,
    10, 8, 13, 15, 12, 3, 3,
];
const ANCHOR3B: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6,
    10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];

/// Endpoint bits of R G B A per mode (public), before p-bits and rotation.
const BC7_BITS: [[u8; 4]; 8] = [
    [4, 4, 4, 0],
    [6, 6, 6, 0],
    [5, 5, 5, 0],
    [7, 7, 7, 0],
    [5, 5, 5, 6],
    [7, 7, 7, 8],
    [7, 7, 7, 7],
    [5, 5, 5, 5],
];
/// Index bits of the primary and secondary sets per mode.
const BC7_INDEX: [[u8; 2]; 8] = [
    [3, 0],
    [3, 0],
    [2, 0],
    [2, 0],
    [3, 2],
    [2, 2],
    [4, 0],
    [2, 0],
];

const fn pbits(mode: u16) -> bool {
    matches!(mode, 0 | 1 | 3 | 6 | 7)
}

/// Channel rotation bits; only BC7 modes 4 and 5 have a rotation field.
const fn rotation(ctrl: u16) -> usize {
    if matches!(ctrl & 7, 4 | 5) {
        ((ctrl >> 3) & 3) as usize
    } else {
        0
    }
}

/// Channel precisions: the p-bit is red's lowest bit; rotation swaps alpha
/// with the rotated channel.
fn bc7_precision(ctrl: u16) -> [u8; 4] {
    let (mode, rot) = (ctrl & 7, rotation(ctrl));
    let mut p = BC7_BITS[usize::from(mode)];
    if rot != 0 {
        p.swap(3, rot - 1);
    } else if pbits(mode) {
        p[0] += 1;
    }
    p
}

#[inline(always)]
fn zz_u16(v: u8) -> u16 {
    u16::from(v >> 1) ^ u16::from(v & 1).wrapping_neg()
}

#[inline(never)]
fn decode_bc7(src: &[u8], dst: &mut [u8], t: &Tables) -> Result<(), Error> {
    let mut lanes = [Lane::new(); 16];
    let mut src_pos = 0usize;
    let mut strip_buf = [[0u8; 32]; STRIP];
    let mut history = alloc_boxed_array::<(Bc7Ep, Bc7Ix), HISTORY>();
    let zero = (Bc7Ep::default(), Bc7Ix::default());
    let dict = &t.bc7;
    let ctrl7_meta = &t.ctrl7_meta;
    let mut block_base = 0usize;

    for out_strip in dst.chunks_mut(STRIP * 16) {
        let n = out_strip.len() / 16;
        let recs = &mut strip_buf[..n];
        lanes_strip::<16, 2, 32>(t, &BC7_WIDTHS, &mut lanes, src, &mut src_pos, recs)?;
        for (k, (s, block)) in recs.iter().zip(out_strip.chunks_exact_mut(16)).enumerate() {
            let i = block_base + k;
            let (ep_ref, ix_ref) = references_paired(s);
            let meta = &ctrl7_meta[usize::from(s[31] & 15) << 5 | usize::from(s[30] & 31)];
            if meta.ctrl == 0 {
                return Err(Error("unused BC7 control code"));
            }
            let subsets = usize::from(meta.subsets);
            let prec = meta.prec;
            let columns = usize::from(meta.columns);
            let r = &pick(i, ep_ref, &history, dict, &zero).0;
            let mut ep = Bc7Ep {
                ctrl: meta.ctrl,
                rows: [[0; 8]; 3],
            };
            // Rows the reference lacks build on its first row.
            let common = subsets.min(usize::from((r.ctrl & 0xff) >> 6));

            match subsets {
                1 => {
                    for c in 0..columns {
                        let p = prec[c / 2];
                        if p == 0 {
                            continue;
                        }
                        let sh = 16 - u32::from(p);
                        let mask = 0xffffu16 << sh;
                        ep.rows[0][c] = (r.rows[0][c] & mask).wrapping_add(zz_u16(s[4 + c]) << sh);
                    }
                }
                2 => {
                    let r1 = usize::from(common >= 2);
                    for c in 0..columns {
                        let p = prec[c / 2];
                        if p == 0 {
                            continue;
                        }
                        let sh = 16 - u32::from(p);
                        let mask = 0xffffu16 << sh;
                        ep.rows[0][c] = (r.rows[0][c] & mask).wrapping_add(zz_u16(s[4 + c]) << sh);
                        ep.rows[1][c] =
                            (r.rows[r1][c] & mask).wrapping_add(zz_u16(s[12 + c]) << sh);
                    }
                }
                3 => {
                    let r1 = usize::from(common >= 2);
                    let r2 = if common >= 3 { 2 } else { 0 };
                    let s0 = [s[4], s[5], s[6], s[7], s[8], s[9], 0, 0];
                    let s1 = [s[12], s[13], s[14], s[15], s[16], s[17], 0, 0];
                    let s2 = [s[20], s[21], s[10], s[11], s[18], s[19], 0, 0];
                    for c in 0..columns {
                        let p = prec[c / 2];
                        if p == 0 {
                            continue;
                        }
                        let sh = 16 - u32::from(p);
                        let mask = 0xffffu16 << sh;
                        ep.rows[0][c] = (r.rows[0][c] & mask).wrapping_add(zz_u16(s0[c]) << sh);
                        ep.rows[1][c] = (r.rows[r1][c] & mask).wrapping_add(zz_u16(s1[c]) << sh);
                        ep.rows[2][c] = (r.rows[r2][c] & mask).wrapping_add(zz_u16(s2[c]) << sh);
                    }
                }
                _ => {}
            }

            let bits = meta.idx_bits;
            let rx = &pick(i, ix_ref, &history, dict, &zero).1;
            let mut ix = Bc7Ix {
                bits,
                ix: [[0; 16]; 2],
            };
            if bits[0] != 0 {
                decode_index_set(
                    [s[22], s[23], s[24], s[25], s[26], s[27], s[28], s[29]],
                    bits[0],
                    &rx.ix[0],
                    rx.bits[0],
                    &mut ix.ix[0],
                );
            }
            if bits[1] != 0 {
                decode_index_set(
                    [s[12], s[13], s[14], s[15], s[16], s[17], s[20], s[21]],
                    bits[1],
                    &rx.ix[1],
                    rx.bits[1],
                    &mut ix.ix[1],
                );
            }
            let dst_slot = &mut history[i & (HISTORY - 1)];
            dst_slot.0.ctrl = ep.ctrl;
            match subsets {
                0 => dst_slot.0.rows[0] = [0; 8],
                1 => dst_slot.0.rows[0] = ep.rows[0],
                2 => {
                    dst_slot.0.rows[0] = ep.rows[0];
                    dst_slot.0.rows[1] = ep.rows[1];
                }
                _ => dst_slot.0.rows = ep.rows,
            }
            dst_slot.1.bits = ix.bits;
            if bits[0] != 0 {
                dst_slot.1.ix[0] = ix.ix[0];
            }
            if bits[1] != 0 {
                dst_slot.1.ix[1] = ix.ix[1];
            }
            bc7_pack(&ep, &ix, meta, block);
        }
        block_base += n;
    }
    Ok(())
}

#[inline(always)]
fn bc7_pack_indices(w: &mut Bits, ix: &[i16; 16], bits: u32, anchor_mask: u16) {
    let mask_full = (1u64 << bits) - 1;
    let mask_short = mask_full >> 1;
    if w.at >= 64 && w.at + 15 * bits < 128 {
        let mut hi = w.hi;
        let mut shift = w.at - 64;
        for p in 0..16usize {
            let short = (anchor_mask >> p) & 1;
            let n = bits - u32::from(short);
            let mask = if short != 0 { mask_short } else { mask_full };
            hi |= (u64::from(ix[p] as u16) & mask) << shift;
            shift += n;
        }
        w.hi = hi;
        w.at = shift + 64;
    } else {
        for p in 0..16usize {
            let short = (anchor_mask >> p) & 1;
            let n = bits - u32::from(short);
            let mask = if short != 0 { mask_short } else { mask_full };
            w.put_unmasked(u64::from(ix[p] as u16) & mask, n);
        }
    }
}

#[inline(always)]
fn bc7_pack(ep: &Bc7Ep, ix: &Bc7Ix, meta: &Ctrl7Meta, out: &mut [u8]) {
    let subsets = usize::from(meta.subsets);
    let mut w = Bits {
        lo: u64::from(meta.header_bits),
        hi: 0,
        at: u32::from(meta.header_len),
    };
    for &ch_u8 in &meta.order {
        let ch = usize::from(ch_u8);
        let p = u32::from(meta.prec[ch]);
        if p == 0 {
            continue;
        }
        let drop = u32::from(ch == 0 && meta.has_pbits);
        let shift = 16 - p + drop;
        let n = p - drop;
        let col = 2 * ch;
        for row in 0..subsets {
            w.put_unmasked(u64::from(ep.rows[row][col] >> shift), n);
            w.put_unmasked(u64::from(ep.rows[row][col + 1] >> shift), n);
        }
    }
    if meta.mode == 1 {
        let sh0 = 16 - u32::from(meta.prec[0]);
        for row in 0..subsets {
            w.put_unmasked(u64::from((ep.rows[row][0] >> sh0) & 1), 1);
        }
    } else if meta.has_pbits {
        let sh0 = 16 - u32::from(meta.prec[0]);
        for row in 0..subsets {
            w.put_unmasked(u64::from((ep.rows[row][0] >> sh0) & 1), 1);
            w.put_unmasked(u64::from((ep.rows[row][1] >> sh0) & 1), 1);
        }
    }
    let anchor_mask = meta.anchor_mask;
    match meta.mode {
        4 => {
            bc7_pack_indices(&mut w, &ix.ix[1], u32::from(ix.bits[1]), anchor_mask);
            bc7_pack_indices(&mut w, &ix.ix[0], u32::from(ix.bits[0]), anchor_mask);
        }
        5 => {
            bc7_pack_indices(&mut w, &ix.ix[0], u32::from(ix.bits[0]), anchor_mask);
            bc7_pack_indices(&mut w, &ix.ix[1], u32::from(ix.bits[1]), anchor_mask);
        }
        _ => {
            bc7_pack_indices(&mut w, &ix.ix[0], u32::from(ix.bits[0]), anchor_mask);
        }
    }
    w.write_to(out);
}

// BC6H.

/// Bits of endpoint 0, then of the other endpoints' R, G, B, per mode 1-14.
const BC6_BITS: [[u8; 4]; 15] = [
    [0; 4],
    [10, 5, 5, 5],
    [7, 6, 6, 6],
    [11, 5, 4, 4],
    [11, 4, 5, 4],
    [11, 4, 4, 5],
    [9, 5, 5, 5],
    [8, 6, 5, 5],
    [8, 5, 6, 5],
    [8, 5, 5, 6],
    [6, 6, 6, 6],
    [10, 10, 10, 10],
    [11, 9, 9, 9],
    [12, 8, 8, 8],
    [16, 4, 4, 4],
];
/// Mode code and its width, modes 1-14.
const BC6_CODE: [(u64, u32); 15] = [
    (0, 0),
    (0, 2),
    (1, 2),
    (2, 5),
    (6, 5),
    (10, 5),
    (14, 5),
    (18, 5),
    (22, 5),
    (26, 5),
    (30, 5),
    (3, 5),
    (7, 5),
    (11, 5),
    (15, 5),
];

/// A run of the public BC6H layout: `len` bits of channel `ch` (0-2; 3 is
/// the partition) of endpoint `ep`, from bit `first` up.
#[derive(Clone, Copy)]
struct Run {
    ch: u8,
    ep: u8,
    first: u8,
    len: u8,
}

const fn r(ch: u8, ep: u8, first: u8, len: u8) -> Run {
    Run { ch, ep, first, len }
}

const R: u8 = 0;
const G: u8 = 1;
const B: u8 = 2;
const D: Run = r(3, 0, 0, 5);

/// The header after the mode bits, per mode 1-14. Modes 13 and 14 write the
/// high base bits in ascending order (what the console's packer does).
#[rustfmt::skip]
const BC6_LAYOUT: [&[Run]; 15] = [
    &[],
    &[
        r(G, 2, 4, 1), r(B, 2, 4, 1), r(B, 3, 4, 1), r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10),
        r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4), r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4),
        r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5),
        r(B, 3, 3, 1), D,
    ],
    &[
        r(G, 2, 5, 1), r(G, 3, 4, 1), r(G, 3, 5, 1), r(R, 0, 0, 7), r(B, 3, 0, 1), r(B, 3, 1, 1),
        r(B, 2, 4, 1), r(G, 0, 0, 7), r(B, 2, 5, 1), r(B, 3, 2, 1), r(G, 2, 4, 1), r(B, 0, 0, 7),
        r(B, 3, 3, 1), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 6),
        r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 6), r(R, 3, 0, 6), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 5), r(R, 0, 10, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 4), r(G, 0, 10, 1), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 4), r(B, 0, 10, 1),
        r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 1), r(G, 3, 4, 1),
        r(G, 2, 0, 4), r(G, 1, 0, 5), r(G, 0, 10, 1), r(G, 3, 0, 4), r(B, 1, 0, 4), r(B, 0, 10, 1),
        r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 4), r(B, 3, 0, 1), r(B, 3, 2, 1), r(R, 3, 0, 4),
        r(G, 2, 4, 1), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 1), r(B, 2, 4, 1),
        r(G, 2, 0, 4), r(G, 1, 0, 4), r(G, 0, 10, 1), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 5),
        r(B, 0, 10, 1), r(B, 2, 0, 4), r(R, 2, 0, 4), r(B, 3, 1, 1), r(B, 3, 2, 1), r(R, 3, 0, 4),
        r(B, 3, 4, 1), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 9), r(B, 2, 4, 1), r(G, 0, 0, 9), r(G, 2, 4, 1), r(B, 0, 0, 9), r(B, 3, 4, 1),
        r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4), r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4),
        r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5),
        r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 8), r(G, 3, 4, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(B, 3, 2, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(B, 3, 3, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 5),
        r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 6),
        r(R, 3, 0, 6), D,
    ],
    &[
        r(R, 0, 0, 8), r(B, 3, 0, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(G, 2, 5, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(G, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 6), r(G, 3, 0, 4), r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5),
        r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 8), r(B, 3, 1, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(B, 2, 5, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 5),
        r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 6), r(G, 3, 4, 1), r(B, 3, 0, 1), r(B, 3, 1, 1), r(B, 2, 4, 1), r(G, 0, 0, 6),
        r(G, 2, 5, 1), r(B, 2, 5, 1), r(B, 3, 2, 1), r(G, 2, 4, 1), r(B, 0, 0, 6), r(G, 3, 5, 1),
        r(B, 3, 3, 1), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 6),
        r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 6), r(R, 3, 0, 6), D,
    ],
    &[r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 10), r(G, 1, 0, 10), r(B, 1, 0, 10)],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 9), r(R, 0, 10, 1), r(G, 1, 0, 9),
        r(G, 0, 10, 1), r(B, 1, 0, 9), r(B, 0, 10, 1),
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 8), r(R, 0, 10, 2), r(G, 1, 0, 8),
        r(G, 0, 10, 2), r(B, 1, 0, 8), r(B, 0, 10, 2),
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 6), r(G, 1, 0, 4),
        r(G, 0, 10, 6), r(B, 1, 0, 4), r(B, 0, 10, 6),
    ],
];

#[inline(always)]
const fn bc6_index_bits(mode: u8) -> u8 {
    match mode {
        1..=10 => 3,
        11..=14 => 4,
        _ => 0,
    }
}

#[inline(never)]
fn decode_bc6(src: &[u8], dst: &mut [u8], t: &Tables) -> Result<(), Error> {
    let mut lanes = [Lane::new(); 16];
    let mut src_pos = 0usize;
    let mut strip_buf = [[0u8; 32]; STRIP];
    let mut history = alloc_boxed_array::<(Bc6Ep, Bc6Ix), HISTORY>();
    let zero = (
        Bc6Ep::default(),
        Bc6Ix {
            mode: 11,
            ix: [0; 16],
        },
    );
    let dict = &t.bc6;
    let z = |v: u16| (v >> 1) ^ (v & 1).wrapping_neg();
    let mut block_base = 0usize;

    for out_strip in dst.chunks_mut(STRIP * 16) {
        let n = out_strip.len() / 16;
        let recs = &mut strip_buf[..n];
        lanes_strip::<16, 2, 32>(t, &BC6_WIDTHS, &mut lanes, src, &mut src_pos, recs)?;
        for (k, (s, block)) in recs.iter().zip(out_strip.chunks_exact_mut(16)).enumerate() {
            let i = block_base + k;
            let (ep_ref, ix_ref) = references_paired(s);
            let mode = s[30].wrapping_add(1);
            if !(1..=14).contains(&mode) {
                return Err(Error("BC6H mode out of range"));
            }
            let two = mode < 11;
            let shift = BC6_BITS[usize::from(mode)].map(|b| 16 - u32::from(b));
            let e0 = [wide(s[4], s[5]), wide(s[6], s[7]), wide(s[8], s[9])].map(z);
            let e1 = [wide(s[10], s[11]), wide(s[12], s[13]), wide(s[14], s[15])].map(z);
            let r = &pick(i, ep_ref, &history, dict, &zero).0;
            let mut ep = Bc6Ep::default();
            let m0 = 0xffffu16 << shift[0];
            let m1 = [
                0xffffu16 << shift[1],
                0xffffu16 << shift[2],
                0xffffu16 << shift[3],
            ];
            for c in 0..3 {
                ep.e[0][c] = r.e[0][c].wrapping_add(e0[c] << shift[0]) & m0;
                ep.e[1][c] = r.e[1][c].wrapping_add(e1[c] << shift[1 + c]) & m1[c];
            }
            if two {
                let e2 = [
                    z(u16::from(s[16])),
                    z(u16::from(s[18])),
                    z(u16::from(s[20])),
                ];
                let e3 = [
                    z(u16::from(s[17])),
                    z(u16::from(s[19])),
                    z(u16::from(s[21])),
                ];
                for c in 0..3 {
                    ep.e[2][c] = r.e[2][c].wrapping_add(e2[c] << shift[1 + c]) & m1[c];
                    ep.e[3][c] = r.e[3][c].wrapping_add(e3[c] << shift[1 + c]) & m1[c];
                }
            } else {
                for c in 0..3 {
                    ep.e[2][c] = (ep.e[0][c] >> shift[0]) << shift[1];
                }
                ep.e[3] = ep.e[1];
            }

            let rx = &pick(i, ix_ref, &history, dict, &zero).1;
            let bits = bc6_index_bits(mode);
            let mut ix = Bc6Ix { mode, ix: [0; 16] };
            decode_index_set(
                [s[22], s[23], s[24], s[25], s[26], s[27], s[28], s[29]],
                bits,
                &rx.ix,
                bc6_index_bits(rx.mode),
                &mut ix.ix,
            );
            history[i & (HISTORY - 1)] = (ep, ix);

            // Precompute shifted & unbiased endpoint fields.
            let f = [
                [
                    u64::from((ep.e[0][0] ^ 0x8000) >> shift[0]),
                    u64::from((ep.e[0][1] ^ 0x8000) >> shift[0]),
                    u64::from((ep.e[0][2] ^ 0x8000) >> shift[0]),
                ],
                [
                    u64::from((ep.e[1][0] ^ 0x8000) >> shift[1]),
                    u64::from((ep.e[1][1] ^ 0x8000) >> shift[2]),
                    u64::from((ep.e[1][2] ^ 0x8000) >> shift[3]),
                ],
                [
                    u64::from((ep.e[2][0] ^ 0x8000) >> shift[1]),
                    u64::from((ep.e[2][1] ^ 0x8000) >> shift[2]),
                    u64::from((ep.e[2][2] ^ 0x8000) >> shift[3]),
                ],
                [
                    u64::from((ep.e[3][0] ^ 0x8000) >> shift[1]),
                    u64::from((ep.e[3][1] ^ 0x8000) >> shift[2]),
                    u64::from((ep.e[3][2] ^ 0x8000) >> shift[3]),
                ],
            ];
            let (code, code_len) = BC6_CODE[usize::from(mode)];
            let mut w = Bits {
                lo: code,
                hi: 0,
                at: code_len,
            };
            let part = s[31] & 31;
            for run in BC6_LAYOUT[usize::from(mode)] {
                if run.ch == 3 {
                    w.put_unmasked(u64::from(part), 5);
                } else {
                    w.put(
                        f[usize::from(run.ep)][usize::from(run.ch)] >> run.first,
                        u32::from(run.len),
                    );
                }
            }
            // In all BC6H modes 1..=14, `w.at` is 82 (two-subset) or 65 (one-subset),
            // so every index lands strictly in `w.hi`.
            let anchor_mask = if two {
                1u16 | (1u16 << ANCHOR2[usize::from(part)])
            } else {
                1u16
            };
            let bias = 1u16 << (bits - 1);
            let bits_u32 = u32::from(bits);
            let mut hi = w.hi;
            let mut sh = w.at - 64;
            for p in 0..16usize {
                let n = bits_u32 - u32::from((anchor_mask >> p) & 1);
                let v = u64::from((ix.ix[p] as u16).wrapping_add(bias)) & ((1u64 << n) - 1);
                hi |= v << sh;
                sh += n;
            }
            w.hi = hi;
            w.write_to(block);
        }
        block_base += n;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Original reference implementation kept verbatim as a bit-exact test oracle.
    mod ref_oracle {
        use super::*;

        #[derive(Clone, Copy, Default)]
        pub(super) struct Plain {
            pub(super) ep: [u8; 8],
            pub(super) ix: [u8; 32],
        }

        pub(super) struct RefTables {
            pub(super) freq: [[u32; BUCKETS]; MODELS],
            pub(super) start: [[u32; BUCKETS + 1]; MODELS],
            pub(super) bucket: Box<[[u8; 256]; MODELS]>,
            pub(super) recip: [[u64; BUCKETS]; MODELS],
            pub(super) rank1: [u8; 256],
            pub(super) rank3a: [u8; 64],
            pub(super) rank3c: [u8; 256],
            pub(super) rank4: [u8; 64],
            pub(super) rank5: [u8; 64],
            pub(super) ctrl7: [u16; 512],
            pub(super) bc1: Vec<Plain>,
            pub(super) bc3: Vec<Plain>,
            pub(super) bc4: Vec<Plain>,
            pub(super) bc5: Vec<Plain>,
            pub(super) bc6: Vec<(Bc6Ep, Bc6Ix)>,
            pub(super) bc7: Vec<(Bc7Ep, Bc7Ix)>,
        }

        fn plain_dict(r: &mut Reader<'_>, ep_len: usize, ix_len: usize) -> Vec<Plain> {
            (0..DICTIONARY)
                .map(|_| {
                    let mut p = Plain::default();
                    let (ep, rest) = r.0.split_at(ep_len);
                    let (ix, rest) = rest.split_at(ix_len);
                    r.0 = rest;
                    p.ep[..ep_len].copy_from_slice(ep);
                    p.ix[..ix_len].copy_from_slice(ix);
                    p
                })
                .collect()
        }

        impl RefTables {
            pub(super) fn parse(bytes: &[u8]) -> Result<RefTables, Error> {
                if bytes.len() != TABLES_SIZE {
                    return Err(Error("table file has the wrong size"));
                }
                let mut r = Reader(bytes);
                let mut freq = [[0u32; BUCKETS]; MODELS];
                let mut start = [[0u32; BUCKETS + 1]; MODELS];
                let mut bucket: Box<[[u8; 256]; MODELS]> = vec![[0u8; 256]; MODELS]
                    .into_boxed_slice()
                    .try_into()
                    .unwrap();
                let mut recip = [[0u64; BUCKETS]; MODELS];
                for m in 0..MODELS {
                    for f in &mut freq[m] {
                        *f = u32::from(r.u16());
                    }
                    let used = m / 16 + 2;
                    let mut at = 0u32;
                    for b in 0..used {
                        if freq[m][b] == 0 {
                            return Err(Error("a model has an empty bucket"));
                        }
                        start[m][b] = at;
                        recip[m][b] = (1u64 << 32).div_ceil(u64::from(freq[m][b]));
                        at += freq[m][b] << b.saturating_sub(1);
                    }
                    if at != 1 << 16 {
                        return Err(Error("a model does not tile 65536"));
                    }
                    start[m][used..].fill(1 << 16);
                    for (window, b) in bucket[m].iter_mut().enumerate() {
                        let first = (window as u32) << 8;
                        *b = start[m][1..].iter().filter(|&&s| s <= first).count() as u8;
                    }
                }
                let rank1 = r.bytes();
                let rank3a = r.bytes();
                let rank3c = r.bytes();
                let rank4 = r.bytes();
                let rank5 = r.bytes();
                let mut ctrl7 = [0u16; 512];
                for c in &mut ctrl7 {
                    *c = r.u16();
                }
                let bc1 = plain_dict(&mut r, 6, 16);
                let bc3 = plain_dict(&mut r, 8, 32);
                let bc4 = plain_dict(&mut r, 2, 16);
                let bc5 = plain_dict(&mut r, 4, 32);
                let bc6 = (0..DICTIONARY)
                    .map(|_| {
                        r.u8();
                        let mut ep = Bc6Ep::default();
                        for v in ep.e.as_flattened_mut() {
                            *v = r.u16();
                        }
                        let mut ix = Bc6Ix {
                            mode: r.u8(),
                            ..Default::default()
                        };
                        for v in &mut ix.ix {
                            *v = r.u16() as i16;
                        }
                        (ep, ix)
                    })
                    .collect();
                let bc7 = (0..DICTIONARY)
                    .map(|k| {
                        let mut ep = Bc7Ep {
                            ctrl: r.u16(),
                            ..Default::default()
                        };
                        for v in ep.rows.as_flattened_mut() {
                            *v = r.u16();
                        }
                        let mut ix = Bc7Ix {
                            bits: [r.u8(), r.u8()],
                            ..Default::default()
                        };
                        let n = if k == 0 {
                            32
                        } else {
                            r.u16();
                            31
                        };
                        for v in &mut ix.ix.as_flattened_mut()[..n] {
                            *v = r.u16() as i16;
                        }
                        (ep, ix)
                    })
                    .collect();
                Ok(RefTables {
                    freq,
                    start,
                    bucket,
                    recip,
                    rank1,
                    rank3a,
                    rank3c,
                    rank4,
                    rank5,
                    ctrl7,
                    bc1,
                    bc3,
                    bc4,
                    bc5,
                    bc6,
                    bc7,
                })
            }
        }

        #[derive(Clone, Copy, Default)]
        pub(super) struct RefLane {
            pub(super) width: u8,
            pub(super) phase: u8,
            pub(super) long: bool,
            pub(super) reset: bool,
            pub(super) model: u8,
            pub(super) need: bool,
            pub(super) latch: Option<u8>,
            pub(super) out: Option<u8>,
            pub(super) stalled: bool,
            pub(super) x: u32,
            pub(super) left: u32,
        }

        impl RefLane {
            pub(super) fn step(&mut self, t: &RefTables) -> Result<(), Error> {
                if self.phase != 3 {
                    let Some(b) = self.latch.take() else {
                        return Ok(());
                    };
                    self.need = true;
                    match self.phase {
                        0 => {
                            let group = self.width - (b >> 6);
                            self.model = group * 16 + (b & 15);
                            self.reset = b & 0x10 != 0;
                            self.long = b & 0x20 != 0;
                            self.left = 0;
                            self.phase = 1;
                        }
                        1 if self.long => {
                            self.left = b.into();
                            self.phase = 2;
                        }
                        1 => {
                            self.left = u32::from(b) + 1;
                            self.phase = 3;
                        }
                        _ => {
                            self.left = (u32::from(b) << 8 | self.left) + 257;
                            self.phase = 3;
                        }
                    }
                    return Ok(());
                }
                let mut x = self.x;
                if x < LOW
                    && let Some(b) = self.latch.take()
                {
                    self.left = self.left.wrapping_sub(1);
                    x = x << 8 | u32::from(b);
                }
                self.need = self.left != 0 && self.latch.is_none();
                if x > LOW {
                    let freq = &t.freq[usize::from(self.model)];
                    let start = &t.start[usize::from(self.model)];
                    let slot = x & 0xffff;
                    let b = bucket_of(t, self.model, slot);
                    let (sym, f, rem) = if b == 0 {
                        (0, freq[0] - 1, slot.wrapping_sub(1))
                    } else {
                        let offset = slot - start[b];
                        let q = ((u64::from(offset) * t.recip[usize::from(self.model)][b]) >> 32)
                            as u32;
                        ((1u32 << (b - 1)) + q, freq[b], offset - q * freq[b])
                    };
                    self.x = (x >> 16).wrapping_mul(f).wrapping_add(rem);
                    self.out = Some(sym as u8);
                } else if self.left != 0 {
                    self.x = x;
                } else {
                    self.phase = 0;
                    self.need = self.latch.is_none();
                    if self.reset {
                        self.x = 0;
                    }
                }
                Ok(())
            }
        }

        pub(super) fn bucket_of(t: &RefTables, model: u8, slot: u32) -> usize {
            let start = &t.start[usize::from(model)];
            let mut b = usize::from(t.bucket[usize::from(model)][(slot >> 8) as usize]);
            while start[b + 1] <= slot {
                b += 1;
            }
            b
        }

        pub(super) fn lanes(
            t: &RefTables,
            widths: &[u8],
            per: usize,
            blocks: usize,
            mut fetch: impl FnMut(usize) -> Option<u8>,
            records: &mut [u8],
        ) -> Result<(), Error> {
            let mut lanes = [RefLane::default(); 16];
            let lanes = &mut lanes[..widths.len()];
            for (lane, &width) in lanes.iter_mut().zip(widths) {
                lane.width = width;
                lane.need = true;
            }
            let stride = widths.len() * per;
            let full = (1u32 << widths.len()) - 1;
            let (mut pos, mut done, mut idle) = (0, 0u32, 0);
            let mut at = 0;
            while pos < blocks * per {
                let (mut taken, mut moved) = (0, false);
                for (j, lane) in lanes.iter_mut().enumerate() {
                    let bit = 1 << j;
                    let before = taken;
                    if !lane.stalled {
                        lane.step(t)?;
                    }
                    let waiting = lane.out.is_some() && done & bit != 0;
                    if lane.need
                        && !waiting
                        && taken < ROUND_BYTES
                        && let Some(b) = fetch(j)
                    {
                        lane.latch = Some(b);
                        taken += 1;
                    }
                    lane.stalled = (lane.need && before >= ROUND_BYTES) || waiting;
                    if !lane.stalled
                        && let Some(sym) = lane.out.take()
                    {
                        records[at + j * per] = sym;
                        done |= bit;
                        moved = true;
                    }
                }
                if done == full {
                    pos += 1;
                    at += if pos.is_multiple_of(per) {
                        stride - per + 1
                    } else {
                        1
                    };
                    done = 0;
                    moved = true;
                }
                idle = if taken > 0 || moved { 0 } else { idle + 1 };
                if idle > 3 {
                    return Err(Error("the stream ends early or is corrupt"));
                }
            }
            Ok(())
        }

        fn interleave_ref(a: u8, b: u8) -> u8 {
            let spread = |v: u8| (0..4).fold(0, |acc, k| acc | (v >> k & 1) << (2 * k));
            spread(a) | spread(b) << 1
        }

        fn references(s: &[u8], paired: bool) -> (usize, usize) {
            let (d, ix) = if paired {
                (interleave_ref(s[0], s[1]), interleave_ref(s[2], s[3]))
            } else {
                (s[0], s[1])
            };
            let delta = (d >> 1) ^ (d & 1).wrapping_neg();
            (usize::from(ix.wrapping_add(delta)), usize::from(ix))
        }

        fn pick_ref<'a, T>(
            i: usize,
            r: usize,
            history: &'a [T],
            dictionary: &'a [T],
            zero: &'a T,
        ) -> &'a T {
            match r {
                0 => zero,
                r if r <= i => &history[(i - r) % HISTORY],
                r => &dictionary[r - i - 1],
            }
        }

        fn zz(v: u8, bits: u32) -> u8 {
            let mask = ((1u32 << bits) - 1) as u8;
            ((v >> 1) ^ if v & 1 != 0 { mask } else { 0 }) & mask
        }

        struct Widths {
            ep: [u8; 8],
            ix: [u8; 32],
            paired: bool,
        }

        const fn widths(ep: [u8; 8], ix3: usize, ix2: usize, paired: bool) -> Widths {
            let mut ix = [0; 32];
            let mut k = 0;
            while k < ix3 + ix2 {
                ix[k] = if k < ix3 { 3 } else { 2 };
                k += 1;
            }
            Widths { ep, ix, paired }
        }

        const BC1: Widths = widths([5, 6, 5, 5, 6, 5, 0, 0], 0, 16, false);
        const BC3: Widths = widths([5, 6, 5, 5, 6, 5, 8, 8], 16, 16, true);
        const BC4: Widths = widths([8, 8, 0, 0, 0, 0, 0, 0], 16, 0, false);
        const BC5: Widths = widths([8, 8, 8, 8, 0, 0, 0, 0], 32, 0, true);

        const BC1_CODE: [u32; 4] = [0, 2, 3, 1];
        pub(super) const BC4_CODE: [u64; 8] = [0, 2, 3, 4, 5, 6, 7, 1];

        fn plain<'a>(
            records: impl Iterator<Item = &'a [u8]>,
            out: impl Iterator<Item = &'a mut [u8]>,
            dictionary: &[Plain],
            w: &Widths,
            residual: impl Fn(&[u8]) -> Plain,
            pack: fn(&Plain, &mut [u8]),
        ) -> Result<(), Error> {
            let mut history = vec![Plain::default(); HISTORY];
            let zero = Plain::default();
            for (i, (s, block)) in records.zip(out).enumerate() {
                let (ep_ref, ix_ref) = references(s, w.paired);
                let ep_ref = &pick_ref(i, ep_ref, &history, dictionary, &zero).ep;
                let ix_ref = &pick_ref(i, ix_ref, &history, dictionary, &zero).ix;
                let mut v = residual(s);
                for ((v, r), bits) in v.ep.iter_mut().zip(ep_ref).zip(w.ep) {
                    *v = v.wrapping_add(*r) & ((1u16 << bits) - 1) as u8;
                }
                for ((v, r), bits) in v.ix.iter_mut().zip(ix_ref).zip(w.ix) {
                    *v = v.wrapping_add(*r) & ((1u8 << bits) - 1);
                }
                history[i % HISTORY] = v;
                pack(&v, block);
            }
            Ok(())
        }

        fn quads(rank: &[u8; 256], s: &[u8], ix: &mut [u8]) {
            for (q, &sym) in s.iter().enumerate() {
                let v = rank[usize::from(sym)];
                let base = (q >> 1) * 8 + (q & 1) * 2;
                for (k, p) in [0, 1, 4, 5].into_iter().enumerate() {
                    ix[base + p] = v >> (2 * k) & 3;
                }
            }
        }

        fn pairs(rank: &[u8; 64], s: &[u8], ix: &mut [u8]) {
            for (k, &sym) in s.iter().enumerate() {
                let v = rank[usize::from(sym & 63)];
                ix[2 * k] = v & 7;
                ix[2 * k + 1] = v >> 4 & 7;
            }
        }

        fn bc1_residual(t: &RefTables, s: &[u8]) -> Plain {
            let mut v = Plain::default();
            for (k, bits) in [5, 6, 5, 5, 6, 5].into_iter().enumerate() {
                v.ep[k] = zz(s[2 + k], bits);
            }
            quads(&t.rank1, &s[8..12], &mut v.ix);
            v
        }

        fn bc3_residual(t: &RefTables, s: &[u8]) -> Plain {
            let mut v = Plain {
                ep: [
                    zz(s[14], 5),
                    zz(s[16], 6),
                    zz(s[18], 5),
                    zz(s[15], 5),
                    zz(s[17], 6),
                    zz(s[19], 5),
                    zz(s[4], 8),
                    zz(s[5], 8),
                ],
                ..Default::default()
            };
            pairs(&t.rank3a, &s[6..14], &mut v.ix[..16]);
            quads(&t.rank3c, &s[20..24], &mut v.ix[16..]);
            v
        }

        fn bc4_residual(t: &RefTables, s: &[u8]) -> Plain {
            let mut v = Plain::default();
            v.ep[..2].copy_from_slice(&[zz(s[2], 8), zz(s[3], 8)]);
            pairs(&t.rank4, &s[4..12], &mut v.ix[..16]);
            v
        }

        fn bc5_residual(t: &RefTables, s: &[u8]) -> Plain {
            let mut v = Plain::default();
            for k in 0..4 {
                v.ep[k] = zz(s[4 + k], 8);
            }
            pairs(&t.rank5, &s[8..24], &mut v.ix);
            v
        }

        fn bc1_color(ep: &[u8], ix: &[u8], out: &mut [u8]) {
            let c0 = u16::from(ep[0]) << 11 | u16::from(ep[1]) << 5 | u16::from(ep[2]);
            let c1 = u16::from(ep[3]) << 11 | u16::from(ep[4]) << 5 | u16::from(ep[5]);
            let bits = (0..16).fold(0, |acc, p| {
                acc | BC1_CODE[usize::from(ix[p] & 3)] << (2 * p)
            });
            out[..2].copy_from_slice(&c0.to_le_bytes());
            out[2..4].copy_from_slice(&c1.to_le_bytes());
            out[4..8].copy_from_slice(&bits.to_le_bytes());
        }

        fn bc4_alpha(e0: u8, e1: u8, ix: &[u8], out: &mut [u8]) {
            let bits = (0..16).fold(0u64, |acc, p| {
                acc | BC4_CODE[usize::from(ix[p] & 7)] << (16 + 3 * p)
            });
            let v = u64::from(e0 ^ 0x80) | u64::from(e1 ^ 0x80) << 8 | bits;
            out[..8].copy_from_slice(&v.to_le_bytes());
        }

        fn bc1_pack(v: &Plain, out: &mut [u8]) {
            bc1_color(&v.ep, &v.ix, out);
        }

        fn bc3_pack(v: &Plain, out: &mut [u8]) {
            bc4_alpha(v.ep[6], v.ep[7], &v.ix[..16], &mut out[..8]);
            bc1_color(&v.ep, &v.ix[16..], &mut out[8..]);
        }

        fn bc4_pack(v: &Plain, out: &mut [u8]) {
            bc4_alpha(v.ep[0], v.ep[1], &v.ix, out);
        }

        fn bc5_pack(v: &Plain, out: &mut [u8]) {
            bc4_alpha(v.ep[0], v.ep[1], &v.ix[..16], &mut out[..8]);
            bc4_alpha(v.ep[2], v.ep[3], &v.ix[16..], &mut out[8..]);
        }

        fn nibbles(b: u8, odd_first: bool) -> [i16; 2] {
            let half = |v: u8| (0..4).fold(0u8, |acc, k| acc | (v >> (2 * k) & 1) << k);
            let z = |u: u8| i16::from(u >> 1) ^ -i16::from(u & 1);
            let (even, odd) = (z(half(b)), z(half(b >> 1)));
            if odd_first { [odd, even] } else { [even, odd] }
        }

        fn add_index(cur: i16, bits: u8, refv: i16, ref_bits: u8) -> i16 {
            let up = |v: i16, b: u8| {
                if b == 0 || b > 16 {
                    0
                } else {
                    (v as u16) << (16 - b)
                }
            };
            let sum = up(cur, bits).wrapping_add(up(refv, ref_bits)) as i16;
            if bits == 0 { 0 } else { sum >> (16 - bits) }
        }

        #[derive(Default)]
        struct RefBits {
            v: u128,
            at: u32,
        }

        impl RefBits {
            fn put(&mut self, value: u64, n: u32) {
                if n > 0 && self.at < 128 {
                    self.v |= (u128::from(value) & ((1u128 << n) - 1)) << self.at;
                }
                self.at += n;
            }
        }

        fn bc7<'a>(
            records: impl Iterator<Item = &'a [u8]>,
            out: impl Iterator<Item = &'a mut [u8]>,
            t: &RefTables,
        ) -> Result<(), Error> {
            let mut history = vec![(Bc7Ep::default(), Bc7Ix::default()); HISTORY];
            let zero = (Bc7Ep::default(), Bc7Ix::default());
            for (i, (s, block)) in records.zip(out).enumerate() {
                let (ep_ref, ix_ref) = references(s, true);
                let ctrl = t.ctrl7[usize::from(s[31] & 15) << 5 | usize::from(s[30] & 31)];
                if ctrl == 0 {
                    return Err(Error("unused BC7 control code"));
                }
                let mode = usize::from(ctrl & 7);
                let subsets = usize::from((ctrl & 0xff) >> 6);
                let prec = bc7_precision(ctrl);
                let three = subsets == 3;
                let or0 = |v: u8| if three { 0 } else { v };
                let if3 = |v: u8| if three { v } else { 0 };
                let stored = [
                    [s[4], s[5], s[6], s[7], s[8], s[9], or0(s[10]), or0(s[11])],
                    [
                        s[12],
                        s[13],
                        s[14],
                        s[15],
                        s[16],
                        s[17],
                        or0(s[18]),
                        or0(s[19]),
                    ],
                    [
                        s[20],
                        s[21],
                        if3(s[10]),
                        if3(s[11]),
                        if3(s[18]),
                        if3(s[19]),
                        0,
                        0,
                    ],
                ];
                let r = &pick_ref(i, ep_ref, &history, &t.bc7, &zero).0;
                let mut ep = Bc7Ep {
                    ctrl,
                    ..Default::default()
                };
                let common = subsets.min(usize::from((r.ctrl & 0xff) >> 6));
                let columns = if prec[3] > 0 { 8 } else { 6 };
                for c in 0..columns {
                    let p = prec[c / 2];
                    if p == 0 {
                        continue;
                    }
                    let mask = 0xffffu16 << (16 - p);
                    for (row, st) in stored.iter().enumerate().take(subsets) {
                        let v = st[c];
                        let residual =
                            (u16::from(v >> 1) ^ (v & 1).wrapping_neg() as i8 as u16) << (16 - p);
                        let base = r.rows[if row < common { row } else { 0 }][c] & mask;
                        ep.rows[row][c] = base.wrapping_add(residual);
                    }
                }
                let bits = BC7_INDEX[mode];
                let rx = &pick_ref(i, ix_ref, &history, &t.bc7, &zero).1;
                let mut ix = Bc7Ix {
                    bits,
                    ..Default::default()
                };
                for (set, offsets) in [[22, 24, 26, 28], [12, 14, 16, 20]].into_iter().enumerate() {
                    if bits[set] == 0 {
                        continue;
                    }
                    for (k, o) in offsets.into_iter().enumerate() {
                        for (b, &byte) in s[o..o + 2].iter().enumerate() {
                            let pair = nibbles(byte, k < 3);
                            for (h, &v) in pair.iter().enumerate() {
                                let p = 4 * k + 2 * b + h;
                                ix.ix[set][p] =
                                    add_index(v, bits[set], rx.ix[set][p], rx.bits[set]);
                            }
                        }
                    }
                }
                history[i % HISTORY] = (ep, ix);
                bc7_pack(&ep, &ix, prec, block);
            }
            Ok(())
        }

        fn bc7_pack(ep: &Bc7Ep, ix: &Bc7Ix, prec: [u8; 4], out: &mut [u8]) {
            let ctrl = ep.ctrl;
            let mode = ctrl & 7;
            let subsets = usize::from((ctrl & 0xff) >> 6);
            let part = usize::from(ctrl >> 8 & 63);
            let rot = rotation(ctrl);
            let mut w = RefBits::default();
            w.put(1 << mode, u32::from(mode) + 1);
            match mode {
                0 => w.put(part as u64, 4),
                1 | 2 | 3 | 7 => w.put(part as u64, 6),
                4 => {
                    w.put(rot as u64, 2);
                    w.put(u64::from(ctrl >> 5 & 1), 1);
                }
                5 => w.put(rot as u64, 2),
                _ => {}
            }
            let ends = 2 * subsets;
            let field = |row: usize, col: usize, p: u8| u64::from(ep.rows[row][col] >> (16 - p));
            let mut order = [0, 1, 2, 3];
            if rot != 0 {
                order.swap(3, rot - 1);
            }
            for ch in order {
                let p = prec[ch];
                if p == 0 {
                    continue;
                }
                let drop = u32::from(ch == 0 && pbits(mode));
                for e in 0..ends {
                    w.put(field(e / 2, 2 * ch + e % 2, p) >> drop, u32::from(p) - drop);
                }
            }
            match mode {
                1 => (0..subsets).for_each(|row| w.put(field(row, 0, prec[0]) & 1, 1)),
                m if pbits(m) => (0..ends).for_each(|e| w.put(field(e / 2, e % 2, prec[0]) & 1, 1)),
                _ => {}
            }
            let anchors: &[u8] = match subsets {
                2 => &[0, ANCHOR2[part]],
                3 => &[0, ANCHOR3A[part], ANCHOR3B[part]],
                _ => &[0],
            };
            let mut indices = |set: usize| {
                let bits = ix.bits[set];
                for (p, &v) in ix.ix[set].iter().enumerate() {
                    let n = u32::from(bits) - u32::from(anchors.contains(&(p as u8)));
                    w.put(u64::from(v as u16), n);
                }
            };
            match mode {
                4 => {
                    indices(1);
                    indices(0);
                }
                5 => {
                    indices(0);
                    indices(1);
                }
                _ => indices(0),
            }
            out.copy_from_slice(&w.v.to_le_bytes());
        }

        fn bc6<'a>(
            records: impl Iterator<Item = &'a [u8]>,
            out: impl Iterator<Item = &'a mut [u8]>,
            dictionary: &[(Bc6Ep, Bc6Ix)],
        ) -> Result<(), Error> {
            let mut history = vec![(Bc6Ep::default(), Bc6Ix::default()); HISTORY];
            let zero = (
                Bc6Ep::default(),
                Bc6Ix {
                    mode: 11,
                    ..Default::default()
                },
            );
            let z = |v: u16| (v >> 1) ^ (v & 1).wrapping_neg();
            let wide_ref = |a: u8, b: u8| {
                (0..8).fold(0u16, |acc, k| {
                    acc | u16::from(a >> k & 1) << (2 * k) | u16::from(b >> k & 1) << (2 * k + 1)
                })
            };
            for (i, (s, block)) in records.zip(out).enumerate() {
                let (ep_ref, ix_ref) = references(s, true);
                let mode = s[30].wrapping_add(1);
                if !(1..=14).contains(&mode) {
                    return Err(Error("BC6H mode out of range"));
                }
                let two = mode < 11;
                let shift = BC6_BITS[usize::from(mode)].map(|b| 16 - u32::from(b));
                let e0 = [
                    wide_ref(s[4], s[5]),
                    wide_ref(s[6], s[7]),
                    wide_ref(s[8], s[9]),
                ]
                .map(z);
                let e1 = [
                    wide_ref(s[10], s[11]),
                    wide_ref(s[12], s[13]),
                    wide_ref(s[14], s[15]),
                ]
                .map(z);
                let (e2, e3) = if two {
                    (
                        [s[16], s[18], s[20]].map(|v| z(v.into())),
                        [s[17], s[19], s[21]].map(|v| z(v.into())),
                    )
                } else {
                    (e0, e1)
                };
                let r = &pick_ref(i, ep_ref, &history, dictionary, &zero).0;
                let mut ep = Bc6Ep::default();
                let mask = |sh: u32| 0xffffu16 << sh;
                for ((e, &r), res) in ep.e[0].iter_mut().zip(&r.e[0]).zip(e0) {
                    *e = r.wrapping_add(res << shift[0]) & mask(shift[0]);
                }
                for (k, res) in [e1, e2, e3]
                    .iter()
                    .enumerate()
                    .take(if two { 3 } else { 1 })
                {
                    for c in 0..3 {
                        ep.e[k + 1][c] =
                            r.e[k + 1][c].wrapping_add(res[c] << shift[1 + c]) & mask(shift[1 + c]);
                    }
                }
                if !two {
                    for c in 0..3 {
                        ep.e[2][c] = (ep.e[0][c] >> shift[0]) << shift[1];
                    }
                    ep.e[3] = ep.e[1];
                }
                let rx = &pick_ref(i, ix_ref, &history, dictionary, &zero).1;
                let bits = bc6_index_bits(mode);
                let mut ix = Bc6Ix {
                    mode,
                    ..Default::default()
                };
                for k in 0..8 {
                    for (h, v) in nibbles(s[22 + k], k < 6).into_iter().enumerate() {
                        let p = 2 * k + h;
                        ix.ix[p] = add_index(v, bits, rx.ix[p], bc6_index_bits(rx.mode));
                    }
                }
                history[i % HISTORY] = (ep, ix);
                let field = |k: usize, c: usize| {
                    let sh = if k == 0 { shift[0] } else { shift[1 + c] };
                    u64::from((ep.e[k][c] ^ 0x8000) >> sh)
                };
                let mut w = RefBits::default();
                let (code, n) = BC6_CODE[usize::from(mode)];
                w.put(code, n);
                let part = s[31] & 31;
                for run in BC6_LAYOUT[usize::from(mode)] {
                    if run.ch == 3 {
                        w.put(part.into(), 5);
                    } else {
                        w.put(
                            field(run.ep.into(), run.ch.into()) >> run.first,
                            run.len.into(),
                        );
                    }
                }
                let anchor = if two { ANCHOR2[usize::from(part)] } else { 0 };
                for (p, &v) in ix.ix.iter().enumerate() {
                    let short = p == 0 || p == usize::from(anchor);
                    let v = (v as u16).wrapping_add(1 << (bits - 1));
                    w.put(v.into(), u32::from(bits) - u32::from(short));
                }
                block.copy_from_slice(&w.v.to_le_bytes());
            }
            Ok(())
        }

        pub(super) fn decode(
            bc: u8,
            src: &[u8],
            dst: &mut [u8],
            tables: &RefTables,
        ) -> Result<(), Error> {
            let block = match bc {
                1 | 4 => 8,
                3 | 5 | 6 | 7 => 16,
                _ => return Err(Error("unsupported BC format")),
            };
            if dst.is_empty() || !dst.len().is_multiple_of(block) {
                return Err(Error("destination is not a whole number of blocks"));
            }
            let blocks = dst.len() / block;
            let (widths, per): (&[u8], usize) = match bc {
                1 => (&BC1_WIDTHS, 1),
                3 => (&BC3_WIDTHS, 2),
                4 => (&BC4_WIDTHS, 1),
                5 => (&BC5_WIDTHS, 2),
                6 => (&BC6_WIDTHS, 2),
                _ => (&BC7_WIDTHS, 2),
            };
            let stride = widths.len() * per;
            let mut records = vec![0u8; blocks * stride];
            let mut next = src.iter().copied();
            lanes(tables, widths, per, blocks, |_| next.next(), &mut records)?;
            let records = records.chunks_exact(stride);
            let out = dst.chunks_exact_mut(block);
            let t = tables;
            match bc {
                1 => plain(records, out, &t.bc1, &BC1, |s| bc1_residual(t, s), bc1_pack),
                3 => plain(records, out, &t.bc3, &BC3, |s| bc3_residual(t, s), bc3_pack),
                4 => plain(records, out, &t.bc4, &BC4, |s| bc4_residual(t, s), bc4_pack),
                5 => plain(records, out, &t.bc5, &BC5, |s| bc5_residual(t, s), bc5_pack),
                6 => bc6(records, out, &t.bc6),
                _ => bc7(records, out, t),
            }
        }
    }

    fn local_table_bytes() -> Option<Vec<u8>> {
        let path = std::env::var_os("VICTORY_BCPACK_TABLES")?;
        Some(std::fs::read(path).expect("read the local BCPack tables"))
    }

    fn parse_test_tables(bytes: &[u8]) -> (Tables, ref_oracle::RefTables) {
        (
            Tables::parse(bytes).unwrap(),
            ref_oracle::RefTables::parse(bytes).unwrap(),
        )
    }

    /// Tables we made: uniform models, identity ranks, a BC7 code per mode,
    /// empty dictionaries.
    fn tables() -> Vec<u8> {
        let mut t = Vec::new();
        for m in 0..MODELS {
            let g = m / 16;
            for b in 0..BUCKETS {
                let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
                t.extend_from_slice(&f.to_le_bytes());
            }
        }
        t.extend((0..=255u8).map(|v| v & 0x77)); // rank1 (unused by the tests beyond 2-bit fields)
        t.extend(0..64u8); // rank3a
        t.extend(0..=255u8); // rank3c
        t.extend((0..64u8).map(|v| (v & 7) | (v >> 3) << 4)); // rank4: low 3 bits, next 3 bits
        t.extend(0..64u8); // rank5
        for code in 0..512u16 {
            // Code c: mode 6 (one subset), the only one the tests use.
            let ctrl: u16 = if code == 0 { 0x46 } else { 0 };
            t.extend_from_slice(&ctrl.to_le_bytes());
        }
        t.resize(TABLES_SIZE, 0);
        t
    }

    /// Encodes one fragment for any model in `t` (`model = (width - shrink) * 16 + shape`),
    /// ending at state `LOW` with optional `reset`.
    fn encode_fragment(
        t: &ref_oracle::RefTables,
        width: u8,
        shrink: u8,
        shape: u8,
        reset: bool,
        symbols: &[u8],
    ) -> Vec<u8> {
        let group = width - shrink;
        let model = usize::from(group * 16 + (shape & 15));
        let freqs = &t.freq[model];
        let starts = &t.start[model];
        let mut x = LOW;
        let mut bytes = Vec::new();
        for &s in symbols.iter().rev() {
            let (freq, start) = if s == 0 {
                (freqs[0] - 1, 1u32)
            } else {
                let k = (8 - s.leading_zeros()) as usize;
                let f = freqs[k];
                (f, starts[k] + (u32::from(s) - (1u32 << (k - 1))) * f)
            };
            while x >= (freq << 16) {
                bytes.push(x as u8);
                x >>= 8;
            }
            x = ((x / freq) << 16) + x % freq + start;
        }
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.reverse();
        let mut header = (shrink << 6) | (shape & 15);
        if reset {
            header |= 0x10;
        }
        let mut out = vec![header];
        if bytes.len() > 256 {
            out[0] |= 0x20;
            let n = bytes.len() - 257;
            out.extend_from_slice(&[n as u8, (n >> 8) as u8]);
        } else {
            out.push((bytes.len() - 1) as u8);
        }
        out.extend(bytes);
        out
    }

    /// A lane's symbols as one fragment: uniform model, rANS from 2^24, no
    /// reset, so the lane keeps 2^24 when it ends.
    fn fragment(width: u8, symbols: &[u8]) -> Vec<u8> {
        let f = 1u32 << (15 - width);
        let mut x = LOW;
        let mut bytes = Vec::new();
        for &s in symbols.iter().rev() {
            let (freq, start) = if s == 0 {
                (f - 1, 1)
            } else {
                (f, u32::from(s) * f)
            };
            while x >= (freq << 16) {
                bytes.push(x as u8);
                x >>= 8;
            }
            x = ((x / freq) << 16) + x % freq + start;
        }
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.reverse();
        let mut out = vec![0];
        if bytes.len() > 256 {
            out[0] |= 0x20;
            let n = bytes.len() - 257;
            out.extend_from_slice(&[n as u8, (n >> 8) as u8]);
        } else {
            out.push((bytes.len() - 1) as u8);
        }
        out.extend(bytes);
        out
    }

    /// Interleaves per-lane fragments the way the decoder takes bytes. A lane
    /// that has finished still asks: a filler fragment (two payload bytes)
    /// brings its state to 2^24 with a byte latched, which parks it for good.
    fn stream(
        t: &ref_oracle::RefTables,
        widths: &[u8],
        per: usize,
        records: &[Vec<u8>],
    ) -> Vec<u8> {
        let stride = widths.len() * per;
        let mut queues: Vec<std::collections::VecDeque<u8>> = (0..widths.len())
            .map(|j| {
                let syms: Vec<u8> = records
                    .iter()
                    .flat_map(|r| r[j * per..j * per + per].to_vec())
                    .collect();
                let mut bytes = fragment(widths[j], &syms);
                bytes.extend([0, 1, 0, 0]);
                bytes.into()
            })
            .collect();
        let mut out = Vec::new();
        let mut check = vec![0; records.len() * stride];
        let fetch = |j: usize| {
            let b = queues[j]
                .pop_front()
                .expect("a parked lane asked for input");
            out.push(b);
            Some(b)
        };
        ref_oracle::lanes(t, widths, per, records.len(), fetch, &mut check).unwrap();
        assert_eq!(check, records.concat());
        out
    }

    /// Interleaves multi-fragment per-lane streams using any `Tables` (including
    /// local format tables), splitting each lane into chunks of `chunk_syms` symbols
    /// with `reset = true` between chunks and varying model shapes.
    fn stream_multi_fragment(
        t: &ref_oracle::RefTables,
        widths: &[u8],
        per: usize,
        records: &[Vec<u8>],
        chunk_syms: usize,
    ) -> Vec<u8> {
        let stride = widths.len() * per;
        let mut queues: Vec<std::collections::VecDeque<u8>> = (0..widths.len())
            .map(|j| {
                let syms: Vec<u8> = records
                    .iter()
                    .flat_map(|r| r[j * per..j * per + per].to_vec())
                    .collect();
                let mut bytes = Vec::new();
                if syms.is_empty() {
                    bytes.extend(encode_fragment(t, widths[j], 0, 0, true, &[]));
                } else {
                    for (ci, chunk) in syms.chunks(chunk_syms).enumerate() {
                        let shape = ((j + ci * 3) & 15) as u8;
                        // Shrink when all symbols in this chunk fit a smaller alphabet.
                        let max_sym = chunk.iter().copied().max().unwrap_or(0);
                        let shrink = if ci % 3 == 1 && max_sym < (1u8 << (widths[j] - 1)) {
                            1
                        } else {
                            0
                        };
                        bytes.extend(encode_fragment(t, widths[j], shrink, shape, true, chunk));
                    }
                }
                // Park the lane after all fragments.
                bytes.extend(encode_fragment(t, widths[j], 0, 0, false, &[]));
                bytes.extend([0, 1, 0, 0]);
                bytes.into()
            })
            .collect();
        let mut out = Vec::new();
        let mut check = vec![0; records.len() * stride];
        let fetch = |j: usize| {
            let b = queues[j]
                .pop_front()
                .expect("a parked lane asked for input");
            out.push(b);
            Some(b)
        };
        ref_oracle::lanes(t, widths, per, records.len(), fetch, &mut check).unwrap();
        assert_eq!(check, records.concat());
        out
    }

    const BC4_LANES: [u8; 12] = BC4_WIDTHS;

    fn zigzag(v: i32) -> u8 {
        if v >= 0 {
            (2 * v) as u8
        } else {
            (-2 * v - 1) as u8
        }
    }

    #[test]
    fn bc4_blocks_with_and_without_references() {
        let raw = tables();
        let t = Tables::parse(&raw).unwrap();
        let rt = ref_oracle::RefTables::parse(&raw).unwrap();
        // Block 0: endpoints 10, -3 (+128 stored), indices p -> p & 7.
        let mut b0 = vec![0, 0, zigzag(10), zigzag(-3)];
        b0.extend((0..8).map(|k| ((2 * k) & 7) | ((2 * k + 1) & 7) << 3));
        // Block 1 refers to block 0 for both: residual endpoints +1, +1; indices +1.
        let mut b1 = vec![0, 1, zigzag(1), zigzag(1)];
        b1.extend((0..8).map(|_| 1 | 1 << 3));
        let src = stream(&rt, &BC4_LANES, 1, &[b0, b1]);
        let mut dst = [0u8; 16];
        decode(4, &src, &mut dst, &t).unwrap();
        let block = |e0: u8, e1: u8, ix: &dyn Fn(usize) -> usize| {
            let bits = (0..16).fold(0u64, |a, p| a | ref_oracle::BC4_CODE[ix(p)] << (16 + 3 * p));
            (u64::from(e0) | u64::from(e1) << 8 | bits).to_le_bytes()
        };
        assert_eq!(dst[..8], block(138, 125, &|p| p & 7));
        assert_eq!(dst[8..], block(139, 126, &|p| (p + 1) & 7));
    }

    #[test]
    fn lanes_take_bytes_in_schedule_order_and_long_fragments() {
        let raw = tables();
        let t = Tables::parse(&raw).unwrap();
        let rt = ref_oracle::RefTables::parse(&raw).unwrap();
        // Many blocks: long fragments (over 256 payload bytes) on the wide lanes.
        let records: Vec<Vec<u8>> = (0..400u32)
            .map(|i| {
                let mut r = vec![0, 0, (i * 7) as u8, (i * 13) as u8];
                r.extend((0..8).map(|k| ((i + k) * 5 % 64) as u8));
                r
            })
            .collect();
        let src = stream(&rt, &BC4_LANES, 1, &records);
        let mut got = vec![[0u8; 12]; records.len()];
        let mut lanes = [Lane::new(); 12];
        let mut src_pos = 0;
        lanes_strip::<12, 1, 12>(&t, &BC4_LANES, &mut lanes, &src, &mut src_pos, &mut got).unwrap();
        assert_eq!(got.as_flattened(), records.concat());
    }

    #[test]
    fn bc7_mode6_block() {
        let raw = tables();
        let t = Tables::parse(&raw).unwrap();
        let rt = ref_oracle::RefTables::parse(&raw).unwrap();
        // Control code 0 -> mode 6; endpoints R 8 bits (p-bit low), G B A 7 bits.
        let mut s = vec![0u8; 32];
        s[4] = zigzag(0x55); // R0 = 0x55: field 0x2a, p-bit 1
        s[5] = zigzag(-1); // R1 = 0xff (8 bits)
        s[6] = zigzag(3);
        s[10] = zigzag(0x7f >> 1);
        let src = stream(&rt, &BC7_WIDTHS, 2, &[s]);
        let mut dst = [0u8; 16];
        decode(7, &src, &mut dst, &t).unwrap();
        let mut w = Bits::default();
        w.put(1 << 6, 7);
        for (v, n) in [
            (0x2a, 7),
            (0x7f, 7),
            (3, 7),
            (0, 7),
            (0, 7),
            (0, 7),
            (0x3f, 7),
            (0, 7),
        ] {
            w.put(v, n);
        }
        w.put(1, 1);
        w.put(1, 1);
        assert_eq!(dst, w.to_le_bytes());
    }

    #[test]
    fn bc7_rotation_applies_only_to_modes_4_and_5() {
        const CTRL7_OFFSET: usize = MODELS * BUCKETS * 2 + 256 + 64 + 256 + 64 + 64;
        let tables_for = |ctrl: u16| {
            let mut raw = tables();
            raw[CTRL7_OFFSET..CTRL7_OFFSET + 2].copy_from_slice(&ctrl.to_le_bytes());
            let t = Tables::parse(&raw).unwrap();
            let rt = ref_oracle::RefTables::parse(&raw).unwrap();
            assert_eq!(t.ctrl7_meta[0].ctrl, ctrl);
            assert_eq!(rt.ctrl7[0], ctrl);
            (t, rt)
        };

        let mut record = vec![0u8; 32];
        for (i, &val) in [11, 23, 5, 19, 13, 29, 7, 17, 3, 25, 9, 21, 15, 31, 1, 12]
            .iter()
            .enumerate()
        {
            record[4 + i] = zigzag(val);
        }
        record[20] = zigzag(6);
        record[21] = zigzag(14);

        // Modes without a BC7 rotation field must ignore ctrl bits 4:3 instead of
        // panicking on shift overflow (modes 0/1/3 with rot=1) or corrupting channels.
        for (mode, subsets) in [(0u16, 3u16), (1, 2), (2, 3), (3, 2), (6, 1), (7, 2)] {
            let base_ctrl = (subsets << 6) | mode;
            let (base_tables, base_ref) = tables_for(base_ctrl);
            let base_src = stream(&base_ref, &BC7_WIDTHS, 2, std::slice::from_ref(&record));
            let mut want = [0u8; 16];
            decode(7, &base_src, &mut want, &base_tables).unwrap();

            for rot in 1..=3u16 {
                let ctrl = base_ctrl | (rot << 3);
                let (t, rt) = tables_for(ctrl);
                let src = stream(&rt, &BC7_WIDTHS, 2, std::slice::from_ref(&record));
                let mut got = [0u8; 16];
                decode(7, &src, &mut got, &t).unwrap();
                assert_eq!(got, want, "mode {mode} rot {rot}");
                assert_eq!(
                    bc7_precision(ctrl),
                    bc7_precision(base_ctrl),
                    "mode {mode} rot {rot}"
                );
            }
        }

        // Modes 4 and 5 do have a 2-bit rotation field: rotation swaps alpha with channel rot-1.
        for (mode, base_bits) in [(4u16, [5u8, 5, 5, 6]), (5u16, [7u8, 7, 7, 8])] {
            for rot in 1..=3usize {
                let ctrl = (1 << 6) | ((rot as u16) << 3) | mode;
                let mut expected_prec = base_bits;
                expected_prec.swap(3, rot - 1);
                assert_eq!(bc7_precision(ctrl), expected_prec, "mode {mode} rot {rot}");
            }

            // Decode mode 4/5 with rot=1 (swap R and A) and check the packed BC7 block.
            let ctrl = (1 << 6) | (1 << 3) | mode;
            let (t, rt) = tables_for(ctrl);
            let mut s = vec![0u8; 32];
            s[4] = zigzag(0x21); // stored R0 (holds alpha precision after rot=1)
            s[5] = zigzag(0x12); // stored R1
            s[6] = zigzag(0x05); // stored G0
            s[7] = zigzag(0x06); // stored G1
            s[8] = zigzag(0x09); // stored B0
            s[9] = zigzag(0x0a); // stored B1
            s[10] = zigzag(0x0d); // stored A0 (holds color precision after rot=1)
            s[11] = zigzag(0x0e); // stored A1
            let src = stream(&rt, &BC7_WIDTHS, 2, &[s]);
            let mut dst = [0u8; 16];
            decode(7, &src, &mut dst, &t).unwrap();

            let color_bits = u32::from(base_bits[0]);
            let alpha_bits = u32::from(base_bits[3]);
            let mut w = Bits::default();
            w.put(1 << mode, u32::from(mode) + 1);
            w.put(1, 2); // rotation = 1
            if mode == 4 {
                w.put(0, 1); // indexSelectionBit = 0
            }
            // Endpoint order after rot=1 swaps indices 3 and 0: [3 (R), 1 (G), 2 (B), 0 (A)].
            for (v, n) in [
                (0x0d, color_bits),
                (0x0e, color_bits),
                (0x05, color_bits),
                (0x06, color_bits),
                (0x09, color_bits),
                (0x0a, color_bits),
                (0x21, alpha_bits),
                (0x12, alpha_bits),
            ] {
                w.put(v, n);
            }
            assert_eq!(dst, w.to_le_bytes(), "mode {mode} rot 1");
        }
    }

    #[test]
    fn malformed_input_is_an_error() {
        let raw = tables();
        let t = Tables::parse(&raw).unwrap();
        let rt = ref_oracle::RefTables::parse(&raw).unwrap();
        let records: Vec<Vec<u8>> = (0..64u8)
            .map(|i| vec![0, 0, i, i, 1, 2, 3, 4, 5, 6, 7, 8])
            .collect();
        let src = stream(&rt, &BC4_LANES, 1, &records);
        let mut dst = vec![0u8; 64 * 8];
        decode(4, &src, &mut dst, &t).unwrap();
        // Truncated.
        assert!(decode(4, &src[..src.len() / 2], &mut dst, &t).is_err());
        // Headers promising more payload than the stream holds. (Other corrupt
        // headers decode to garbage: the format has no redundancy.)
        let headers = [[0x30; 8], [0xff; 8]].concat();
        assert!(decode(4, &headers, &mut dst[..8], &t).is_err());
        assert!(decode(4, &[], &mut dst, &t).is_err());
        // Destination not a whole number of blocks, empty, more blocks than coded.
        assert!(decode(4, &src, &mut dst[..12], &t).is_err());
        assert!(decode(4, &src, &mut [], &t).is_err());
        assert!(decode(4, &src, &mut vec![0u8; 65 * 8], &t).is_err());
        // Unsupported formats, bad tables.
        assert!(decode(2, &src, &mut dst, &t).is_err());
        assert!(Tables::parse(&[0; 16]).is_err());
        let mut no_tiling = tables();
        no_tiling[0] = 1;
        assert!(Tables::parse(&no_tiling).is_err());
    }

    #[test]
    fn noise_never_panics() {
        let t = Tables::parse(&tables()).unwrap();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for round in 0..300 {
            let len = round * 7 % 900;
            let src: Vec<u8> = (0..len)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed as u8
                })
                .collect();
            for bc in [1, 3, 4, 5, 6, 7] {
                let mut dst = vec![0u8; 16 * (1 + round % 40)];
                let _ = decode(bc, &src, &mut dst, &t);
            }
        }
    }

    /// The window table finds the bucket a scan of the starts finds, for
    /// every slot of every model, using wide and narrow buckets and any local tables.
    #[test]
    fn bucket_lookup_matches_a_scan_of_the_starts() {
        let mut narrow = tables();
        for m in 0..MODELS {
            // Buckets 1.. of one slot each where they fit; bucket 0 takes the rest.
            let used = m / 16 + 2;
            let rest = (1u32 << 16) - (1..used).map(|b| 1u32 << (b - 1)).sum::<u32>();
            for b in 0..used {
                let f = if b == 0 { rest } else { 1 };
                narrow[(m * BUCKETS + b) * 2..][..2].copy_from_slice(&(f as u16).to_le_bytes());
            }
        }
        let synth_raw = tables();
        let local = local_table_bytes();
        for raw in [synth_raw.as_slice(), narrow.as_slice()]
            .into_iter()
            .chain(local.as_deref())
        {
            let t = Tables::parse(raw).unwrap();
            let rt = ref_oracle::RefTables::parse(raw).unwrap();
            for model in 0..MODELS as u8 {
                let start = &rt.start[usize::from(model)];
                let mt = &t.models[usize::from(model)];
                for slot in 0..1u32 << 16 {
                    let scan = start[1..].iter().filter(|&&s| s <= slot).count();
                    assert_eq!(
                        ref_oracle::bucket_of(&rt, model, slot),
                        scan,
                        "ref model {model} slot {slot}"
                    );
                    let slot16 = slot as u16;
                    let mut b = usize::from(mt.window[(slot16 >> 8) as usize]) & 15;
                    while mt.buckets[b].last_slot < slot16 {
                        b = (b + 1) & 15;
                    }
                    assert_eq!(b, scan, "fast model {model} slot {slot}");
                }
            }
        }
    }

    /// One rANS decode as first written: scan for the bucket, divide for the
    /// symbol, rebuild its frequency and start from the symbol.
    fn reference_decode(t: &ref_oracle::RefTables, model: u8, x: u32) -> (u8, u32) {
        let (freq, start) = (&t.freq[usize::from(model)], &t.start[usize::from(model)]);
        let slot = x & 0xffff;
        let b = start[1..BUCKETS].iter().filter(|&&s| s <= slot).count();
        let sym = match b {
            0 => 0,
            1 => 1,
            _ => (1u32 << (b - 1)) + (slot - start[b]) / freq[b],
        } as u8;
        let (f, c) = if sym == 0 {
            (freq[0] - 1, 1)
        } else {
            let k = (8 - sym.leading_zeros()) as usize;
            let f = freq[k];
            (
                f,
                ((u32::from(sym) - (1 << (k - 1))) * f + start[k]) & 0xffff,
            )
        };
        (
            sym,
            (x >> 16).wrapping_mul(f).wrapping_sub(c).wrapping_add(slot),
        )
    }

    /// A payload step decodes the symbol and state the first version did, for
    /// every model and slot of the synthetic edge tables and any local tables
    /// (including `slot == 0` and `freq[0] == 1`).
    #[test]
    fn payload_step_matches_the_reference_decode() {
        // Table where bucket 0 has freq == 1 (only slot 0 lands in bucket 0) and
        // bucket 1 takes the bulk, plus a narrow table where buckets 1.. have freq == 1.
        let mut freq0_one = tables();
        let mut narrow = tables();
        for m in 0..MODELS {
            let used = m / 16 + 2;
            let rest = (1u32 << 16) - (1..used).map(|b| 1u32 << (b - 1)).sum::<u32>();
            for b in 0..used {
                let f_narrow = if b == 0 { rest } else { 1 };
                narrow[(m * BUCKETS + b) * 2..][..2]
                    .copy_from_slice(&(f_narrow as u16).to_le_bytes());

                let f_one = if b == 1 { rest } else { 1 };
                freq0_one[(m * BUCKETS + b) * 2..][..2]
                    .copy_from_slice(&(f_one as u16).to_le_bytes());
            }
        }
        let synth_raw = tables();
        let local = local_table_bytes();
        for raw in [
            synth_raw.as_slice(),
            narrow.as_slice(),
            freq0_one.as_slice(),
        ]
        .into_iter()
        .chain(local.as_deref())
        {
            let t = Tables::parse(raw).unwrap();
            let rt = ref_oracle::RefTables::parse(raw).unwrap();
            for model in 0..MODELS as u8 {
                let mt = &t.models[usize::from(model)];
                for high in [0x100u32, 0x1234, 0xffff] {
                    for slot in 0..1u32 << 16 {
                        let x = high << 16 | slot;
                        if x <= LOW {
                            continue;
                        }
                        // Check branchless ModelTable decode directly.
                        let got_fast = Lane::decode_symbol(mt, x);
                        let (sym, next) = reference_decode(&rt, model, x);
                        assert_eq!(got_fast, (sym, next), "fast model {model} x {x:#x}");

                        // Also check Lane::step_slow on representative slots.
                        if slot < 4 || slot == 0xffff {
                            let mut lane = Lane {
                                state: 3,
                                model,
                                x,
                                left: 1,
                                latch: NONE,
                                out: NONE,
                                need: false,
                                stalled: false,
                            };
                            lane.step_slow(7, &t);
                            assert_eq!(
                                (lane.out, lane.x),
                                (i16::from(sym), next),
                                "slow model {model} x {x:#x}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Pseudo-random records that fit each lane's alphabet; BC7 keeps control
    /// code 0 (mode 6), the only one the synthetic tables define.
    fn random_records(widths: &[u8], per: usize, blocks: usize, bc7: bool) -> Vec<Vec<u8>> {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        (0..blocks)
            .map(|_| {
                let mut r: Vec<u8> = (0..widths.len() * per)
                    .map(|k| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        let alphabet = 1u32 << (widths[k / per] + 1);
                        (seed as u32 % alphabet) as u8
                    })
                    .collect();
                if bc7 {
                    r[30] = 0;
                    r[31] = 0;
                }
                r
            })
            .collect()
    }

    /// Pseudo-random valid records for any BC format (1, 3, 4, 5, 6, 7) and any `Tables`.
    fn valid_records_for(bc: u8, t: &Tables, blocks: usize, seed_init: u64) -> Vec<Vec<u8>> {
        let (widths, per): (&[u8], usize) = match bc {
            1 => (&BC1_WIDTHS, 1),
            3 => (&BC3_WIDTHS, 2),
            4 => (&BC4_WIDTHS, 1),
            5 => (&BC5_WIDTHS, 2),
            6 => (&BC6_WIDTHS, 2),
            _ => (&BC7_WIDTHS, 2),
        };
        let valid_ctrl7: Vec<u16> = (0..512u16)
            .filter(|&c| t.ctrl7_meta[usize::from(c)].ctrl != 0)
            .collect();
        let mut seed = seed_init;
        (0..blocks)
            .map(|bi| {
                let mut r: Vec<u8> = (0..widths.len() * per)
                    .map(|k| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        let alphabet = 1u32 << (widths[k / per] + 1);
                        (seed as u32 % alphabet) as u8
                    })
                    .collect();
                if bc == 6 {
                    // Valid BC6H modes are 1..=14 -> stored s[30] in 0..14.
                    r[30] = (bi % 14) as u8;
                } else if bc == 7 {
                    let code = valid_ctrl7[bi % valid_ctrl7.len()];
                    r[30] = (code & 31) as u8;
                    r[31] = ((code >> 5) & 15) as u8;
                }
                r
            })
            .collect()
    }

    /// The fast paths decode exactly what one lane step at a time does, on
    /// every format's lane layout, with short and long fragments.
    #[test]
    fn every_layout_round_trips_through_the_lanes() {
        let raw = tables();
        let t = Tables::parse(&raw).unwrap();
        let rt = ref_oracle::RefTables::parse(&raw).unwrap();
        for blocks in [0, 1, 2, 17, 255, 256, 257, 300, 1000] {
            macro_rules! check_layout {
                ($widths:expr, $L:literal, $P:literal, $S:literal) => {{
                    let records = random_records(&$widths, $P, blocks, false);
                    let src = stream(&rt, &$widths, $P, &records);
                    let mut got = vec![[0u8; $S]; blocks];
                    let mut lanes = [Lane::new(); $L];
                    let mut src_pos = 0;
                    for chunk in got.chunks_mut(STRIP) {
                        lanes_strip::<$L, $P, $S>(
                            &t,
                            &$widths,
                            &mut lanes,
                            &src,
                            &mut src_pos,
                            chunk,
                        )
                        .unwrap();
                    }
                    assert_eq!(
                        got.as_flattened(),
                        records.concat(),
                        "{:?} x{}, {blocks} blocks",
                        $widths,
                        $P
                    );
                }};
            }
            check_layout!(BC1_WIDTHS, 12, 1, 12);
            check_layout!(BC3_WIDTHS, 12, 2, 24);
            check_layout!(BC4_WIDTHS, 12, 1, 12);
            check_layout!(BC5_WIDTHS, 12, 2, 24);
            check_layout!(BC6_WIDTHS, 16, 2, 32);
            check_layout!(BC7_WIDTHS, 16, 2, 32);
        }
    }

    /// Bit-exact oracle parity across all 6 formats (`BC1`, `BC3`, `BC4`, `BC5`, `BC6H`, `BC7`),
    /// both synthetic and locally supplied tables, at strip-boundary block counts and with
    /// short, long, and reset multi-fragment streams.
    #[test]
    fn oracle_parity_all_formats_and_strip_boundaries() {
        let local = local_table_bytes().as_deref().map(parse_test_tables);
        let (synth, synth_ref) = parse_test_tables(&tables());

        for (label, t, rt, use_multi) in [
            ("synthetic", &synth, &synth_ref, false),
            ("synthetic-multi", &synth, &synth_ref, true),
        ]
        .into_iter()
        .chain(local.as_ref().map(|(t, rt)| ("local", t, rt, true)))
        {
            for (bc, widths, per, block_bytes) in [
                (1u8, &BC1_WIDTHS[..], 1usize, 8usize),
                (3, &BC3_WIDTHS[..], 2, 16),
                (4, &BC4_WIDTHS[..], 1, 8),
                (5, &BC5_WIDTHS[..], 2, 16),
                (6, &BC6_WIDTHS[..], 2, 16),
                (7, &BC7_WIDTHS[..], 2, 16),
            ] {
                for blocks in [0usize, 1, 2, 255, 256, 257, 512, 513, 1024] {
                    if blocks == 0 {
                        let mut got = [];
                        let mut want = [];
                        assert_eq!(
                            decode(bc, &[], &mut got, t),
                            ref_oracle::decode(bc, &[], &mut want, rt),
                            "{label} BC{bc} 0 blocks"
                        );
                        continue;
                    }
                    // Keep runtime brisk while hitting every boundary count for all 6 formats.
                    if label == "synthetic-multi" && blocks > 513 {
                        continue;
                    }
                    let recs = valid_records_for(
                        bc,
                        t,
                        blocks,
                        0x1234_5678_9abc_def0 ^ u64::from(bc) ^ (blocks as u64).rotate_left(17),
                    );
                    let src = if use_multi {
                        // Mix short (<256 B) and long (>256 B) reset fragments.
                        let chunk = if blocks >= 512 { 380 } else { 37 };
                        stream_multi_fragment(rt, widths, per, &recs, chunk)
                    } else {
                        stream(rt, widths, per, &recs)
                    };
                    let mut got = vec![0u8; blocks * block_bytes];
                    let mut want = vec![0u8; blocks * block_bytes];
                    decode(bc, &src, &mut got, t).unwrap();
                    ref_oracle::decode(bc, &src, &mut want, rt).unwrap();
                    assert_eq!(got, want, "{label} BC{bc} {blocks} blocks");
                }
            }
        }
    }

    /// Bit-exact oracle parity for ALL BC7 modes (0..=7), rotations (0..=3), subsets (1..=3),
    /// index selection bits, and all 64 partitions, with dictionary and history references.
    #[test]
    fn oracle_parity_all_bc7_modes_rotations_subsets() {
        const CTRL7_OFFSET: usize = MODELS * BUCKETS * 2 + 256 + 64 + 256 + 64 + 64;
        let mut raw = tables();
        // Populate ctrl7 with all 8 modes, all rotations (0..=3), indexSel (0/1), and partitions.
        let mut codes: Vec<u16> = Vec::new();
        for mode in 0u16..8 {
            let subsets = match mode {
                0 | 2 => 3u16,
                1 | 3 | 7 => 2u16,
                _ => 1u16,
            };
            match mode {
                0 => {
                    for part in 0..16u16 {
                        codes.push((part << 8) | (subsets << 6) | mode);
                    }
                }
                1 | 2 | 3 | 7 => {
                    for part in 0..64u16 {
                        codes.push((part << 8) | (subsets << 6) | mode);
                    }
                }
                4 => {
                    for rot in 0..4u16 {
                        for idx_sel in 0..2u16 {
                            codes.push((subsets << 6) | (idx_sel << 5) | (rot << 3) | mode);
                        }
                    }
                }
                5 => {
                    for rot in 0..4u16 {
                        codes.push((subsets << 6) | (rot << 3) | mode);
                    }
                }
                6 => {
                    for rot in 0..4u16 {
                        // Also include non-zero ignored rotation bits on mode 6.
                        codes.push((subsets << 6) | (rot << 3) | mode);
                    }
                }
                _ => unreachable!(),
            }
        }
        assert!(codes.len() <= 512);
        for (i, &ctrl) in codes.iter().enumerate() {
            let off = CTRL7_OFFSET + i * 2;
            raw[off..off + 2].copy_from_slice(&ctrl.to_le_bytes());
        }
        let t = Tables::parse(&raw).unwrap();
        let rt = ref_oracle::RefTables::parse(&raw).unwrap();

        // Decode 600 blocks cycling through every defined ctrl code (crossing strip boundaries).
        let blocks = 600usize;
        let mut recs = valid_records_for(7, &t, blocks, 0xfeed_face_cafe_babe);
        for (i, r) in recs.iter_mut().enumerate() {
            let code = (i % codes.len()) as u16;
            r[30] = (code & 31) as u8;
            r[31] = ((code >> 5) & 15) as u8;
        }
        let src = stream(&rt, &BC7_WIDTHS, 2, &recs);
        let mut got = vec![0u8; blocks * 16];
        let mut want = vec![0u8; blocks * 16];
        decode(7, &src, &mut got, &t).unwrap();
        ref_oracle::decode(7, &src, &mut want, &rt).unwrap();
        assert_eq!(got, want);

        let Some(bytes) = local_table_bytes() else {
            return;
        };
        let (local, local_ref) = parse_test_tables(&bytes);
        let blocks = 512usize;
        let recs_b = valid_records_for(7, &local, blocks, 0x0123_4567_89ab_cdef);
        let src_b = stream_multi_fragment(&local_ref, &BC7_WIDTHS, 2, &recs_b, 128);
        let mut got_b = vec![0u8; blocks * 16];
        let mut want_b = vec![0u8; blocks * 16];
        decode(7, &src_b, &mut got_b, &local).unwrap();
        ref_oracle::decode(7, &src_b, &mut want_b, &local_ref).unwrap();
        assert_eq!(got_b, want_b);
    }

    /// Bit-exact oracle parity for ALL BC6H modes (1..=14) and all 32 partitions,
    /// across synthetic and locally supplied tables.
    #[test]
    fn oracle_parity_all_bc6_modes() {
        let local = local_table_bytes().as_deref().map(parse_test_tables);
        let (synth, synth_ref) = parse_test_tables(&tables());
        // 14 modes * 32 partitions = 448 blocks (> 1 strip, exercises every mode x partition pair).
        let blocks = 14 * 32;
        for (label, t, rt, use_multi) in [
            ("synthetic", &synth, &synth_ref, false),
            ("synthetic-multi", &synth, &synth_ref, true),
        ]
        .into_iter()
        .chain(local.as_ref().map(|(t, rt)| ("local", t, rt, true)))
        {
            let mut recs = valid_records_for(6, t, blocks, 0xabcd_ef01_2345_6789);
            for (i, r) in recs.iter_mut().enumerate() {
                r[30] = (i % 14) as u8; // mode = 1..=14
                r[31] = ((i / 14) % 32) as u8; // partition = 0..=31
            }
            let src = if use_multi {
                stream_multi_fragment(rt, &BC6_WIDTHS, 2, &recs, 150)
            } else {
                stream(rt, &BC6_WIDTHS, 2, &recs)
            };
            let mut got = vec![0u8; blocks * 16];
            let mut want = vec![0u8; blocks * 16];
            decode(6, &src, &mut got, t).unwrap();
            ref_oracle::decode(6, &src, &mut want, rt).unwrap();
            assert_eq!(got, want, "{label} BC6H all modes");
        }
    }

    /// Identical error behavior on truncated, corrupt, invalid mode/ctrl, and noise inputs.
    #[test]
    fn oracle_parity_errors_and_noise() {
        let local = local_table_bytes().as_deref().map(parse_test_tables);
        let (synth, synth_ref) = parse_test_tables(&tables());
        let (t, rt) = local
            .as_ref()
            .map_or((&synth, &synth_ref), |(t, rt)| (t, rt));

        // 1. Truncated valid streams across strip boundaries for all 6 formats.
        for (bc, widths, per, block_bytes) in [
            (1u8, &BC1_WIDTHS[..], 1usize, 8usize),
            (3, &BC3_WIDTHS[..], 2, 16),
            (4, &BC4_WIDTHS[..], 1, 8),
            (5, &BC5_WIDTHS[..], 2, 16),
            (6, &BC6_WIDTHS[..], 2, 16),
            (7, &BC7_WIDTHS[..], 2, 16),
        ] {
            let blocks = 300usize;
            let recs = valid_records_for(bc, t, blocks, 0x7777_1111_2222_3333 ^ u64::from(bc));
            let src = stream_multi_fragment(rt, widths, per, &recs, 100);
            for cut in [0, 1, 7, 8, 9, src.len() / 3, src.len() / 2, src.len() - 1] {
                let mut got = vec![0u8; blocks * block_bytes];
                let mut want = vec![0u8; blocks * block_bytes];
                let r_got = decode(bc, &src[..cut], &mut got, t);
                let r_want = ref_oracle::decode(bc, &src[..cut], &mut want, rt);
                assert_eq!(r_got, r_want, "BC{bc} truncated at {cut}/{}", src.len());
                if r_got.is_ok() {
                    assert_eq!(got, want, "BC{bc} ok at {cut}");
                }
            }
        }

        // 2. Invalid BC6H mode (mode 15, 16, 32) in strip 0 and strip 1.
        for bad_block in [0usize, 5, 255, 256, 260] {
            for bad_s30 in [14u8, 15, 31] {
                let blocks = 270;
                let mut recs = valid_records_for(6, &synth, blocks, 0x4242);
                recs[bad_block][30] = bad_s30;
                let src = stream(&synth_ref, &BC6_WIDTHS, 2, &recs);
                let mut got = vec![0u8; blocks * 16];
                let mut want = vec![0u8; blocks * 16];
                assert_eq!(
                    decode(6, &src, &mut got, &synth),
                    ref_oracle::decode(6, &src, &mut want, &synth_ref),
                    "BC6H bad mode s30={bad_s30} at block {bad_block}"
                );
                assert_eq!(
                    decode(6, &src, &mut got, &synth),
                    Err(Error("BC6H mode out of range"))
                );
            }
        }

        // 3. Unused BC7 control code (`ctrl == 0`) in strip 0 and strip 1.
        for bad_block in [0usize, 5, 255, 256, 260] {
            let blocks = 270;
            let mut recs = valid_records_for(7, &synth, blocks, 0x9999);
            recs[bad_block][30] = 1; // synthetic tables only define code 0
            recs[bad_block][31] = 0;
            let src = stream(&synth_ref, &BC7_WIDTHS, 2, &recs);
            let mut got = vec![0u8; blocks * 16];
            let mut want = vec![0u8; blocks * 16];
            assert_eq!(
                decode(7, &src, &mut got, &synth),
                ref_oracle::decode(7, &src, &mut want, &synth_ref),
                "BC7 unused ctrl at block {bad_block}"
            );
            assert_eq!(
                decode(7, &src, &mut got, &synth),
                Err(Error("unused BC7 control code"))
            );
        }

        // 4. Random noise parity across all 6 formats and both table sets.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for round in 0..200 {
            let len = (round * 13) % 900;
            let src: Vec<u8> = (0..len)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed as u8
                })
                .collect();
            let (t, rt) = if round % 2 == 0 {
                (t, rt)
            } else {
                (&synth, &synth_ref)
            };
            for bc in [1u8, 3, 4, 5, 6, 7] {
                let block_bytes = if matches!(bc, 1 | 4) { 8 } else { 16 };
                let blocks = 1 + (round % 40);
                let mut got = vec![0u8; blocks * block_bytes];
                let mut want = vec![0u8; blocks * block_bytes];
                let r_got = decode(bc, &src, &mut got, t);
                let r_want = ref_oracle::decode(bc, &src, &mut want, rt);
                assert_eq!(r_got, r_want, "noise round {round} BC{bc}");
                if r_got.is_ok() {
                    assert_eq!(got, want, "noise ok round {round} BC{bc}");
                }
            }
        }
    }

    /// Decode throughput on synthetic streams (no game data):
    /// `cargo test --release -p victory-formats bcpack_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn bcpack_throughput() {
        let raw = tables();
        let t = Tables::parse(&raw).unwrap();
        let rt = ref_oracle::RefTables::parse(&raw).unwrap();
        let cases: [(u8, &[u8], usize, usize); 3] = [
            (1, &BC1_WIDTHS, 1, 8),
            (4, &BC4_LANES, 1, 8),
            (7, &BC7_WIDTHS, 2, 16),
        ];
        for (bc, widths, per, block) in cases {
            // One fragment per lane holds at most 65792 bytes.
            let blocks = 1 << 14;
            let records = random_records(widths, per, blocks, bc == 7);
            let src = stream(&rt, widths, per, &records);
            let mut dst = vec![0u8; blocks * block];
            decode(bc, &src, &mut dst, &t).unwrap();
            let runs = 40;
            let started = std::time::Instant::now();
            for _ in 0..runs {
                decode(bc, std::hint::black_box(&src), &mut dst, &t).unwrap();
            }
            let secs = started.elapsed().as_secs_f64() / f64::from(runs);
            println!(
                "BC{bc}: {blocks} blocks, {:.1} MB out/s, {:.1} ns/block",
                (blocks * block) as f64 / secs / 1e6,
                secs * 1e9 / blocks as f64
            );
        }
    }

    /// Decode throughput across all 6 formats using locally supplied tables.
    /// Run `cargo test --release -p victory-formats bcpack_throughput_local -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn bcpack_throughput_local() {
        let bytes = local_table_bytes().expect("set VICTORY_BCPACK_TABLES");
        let (t, rt) = parse_test_tables(&bytes);
        let cases: [(u8, &[u8], usize, usize); 6] = [
            (1, &BC1_WIDTHS, 1, 8),
            (3, &BC3_WIDTHS, 2, 16),
            (4, &BC4_WIDTHS, 1, 8),
            (5, &BC5_WIDTHS, 2, 16),
            (6, &BC6_WIDTHS, 2, 16),
            (7, &BC7_WIDTHS, 2, 16),
        ];
        for (bc, widths, per, block) in cases {
            let blocks = 1 << 14;
            let records = valid_records_for(bc, &t, blocks, 0x9e37_79b9_7f4a_7c15 ^ u64::from(bc));
            let src = stream_multi_fragment(&rt, widths, per, &records, 4096);
            let mut dst = vec![0u8; blocks * block];
            decode(bc, &src, &mut dst, &t).unwrap();
            let runs = 40;
            let started = std::time::Instant::now();
            for _ in 0..runs {
                decode(bc, std::hint::black_box(&src), &mut dst, &t).unwrap();
            }
            let secs = started.elapsed().as_secs_f64() / f64::from(runs);
            println!(
                "local BC{bc}: {blocks} blocks, {:.1} MB out/s, {:.1} ns/block",
                (blocks * block) as f64 / secs / 1e6,
                secs * 1e9 / blocks as f64
            );
        }
    }

    /// Compares with reference outputs made outside the repository:
    /// `VICTORY_BCPACK_TABLES` (the table file) and `VICTORY_BCPACK_CASES`, a
    /// directory of `<name>.bc<n>` packed chunks with `<name>.out` beside them.
    #[test]
    #[ignore = "needs the format tables and reference chunks"]
    fn reference_chunks() {
        let tables = std::env::var_os("VICTORY_BCPACK_TABLES").expect("VICTORY_BCPACK_TABLES");
        let cases = std::env::var_os("VICTORY_BCPACK_CASES").expect("VICTORY_BCPACK_CASES");
        let t = Tables::parse(&std::fs::read(tables).unwrap()).unwrap();
        let mut n = 0;
        for entry in std::fs::read_dir(cases).unwrap() {
            let path = entry.unwrap().path();
            let Some(bc) = path
                .extension()
                .and_then(|e| e.to_str()?.strip_prefix("bc")?.parse().ok())
            else {
                continue;
            };
            let want = std::fs::read(path.with_extension("out")).unwrap();
            let mut got = vec![0u8; want.len()];
            decode(bc, &std::fs::read(&path).unwrap(), &mut got, &t).unwrap();
            assert!(got == want, "{}", path.display());
            n += 1;
        }
        assert!(n > 0);
    }

    /// A dictionary entry stands in for a block the decoder made: a set with
    /// 0 index bits holds no indices, and pixel 0 is the anchor (top bit clear).
    #[test]
    #[ignore = "needs locally supplied format tables"]
    fn bc7_dictionary_indices_line_up() {
        let bytes = local_table_bytes().expect("set VICTORY_BCPACK_TABLES");
        let t = Tables::parse(&bytes).unwrap();
        for (k, (_, ix)) in t.bc7.iter().enumerate() {
            let anchor = 0..1 << (ix.bits[0] - 1);
            let ok = (ix.bits[1] != 0 || ix.ix[1] == [0; 16]) && anchor.contains(&ix.ix[0][0]);
            assert!(ok, "entry {k}: {:?}", ix.ix);
        }
    }
}
