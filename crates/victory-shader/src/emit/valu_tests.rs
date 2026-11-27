//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, turn-range reduction for `v_sin_*`/`v_cos_*`, and the LDS
//! float atomics `ds_min_f32`/`ds_max_f32`.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SOPK `op sdst, simm16`; the compares read `sdst` as their operand.
const fn sopk(op: u8, sdst: u8, simm16: u16) -> u32 {
    0xB000_0000 | (op as u32 & 0x1F) << 23 | (sdst as u32 & 0x7F) << 16 | simm16 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    vop3_mod(op, vdst, src0, src1, src2, 0, 0, 0, false)
}

/// VOP3 `op vdst, src0, src1, src2` with explicit `abs`, `neg`, `omod`, `clamp`.
#[allow(clippy::too_many_arguments)]
const fn vop3_mod(
    op: u16,
    vdst: u8,
    src0: u16,
    src1: u16,
    src2: u16,
    abs: u8,
    neg: u8,
    omod: u8,
    clamp: bool,
) -> [u32; 2] {
    [
        0x35 << 26
            | (op as u32 & 0x3FF) << 16
            | (clamp as u32) << 15
            | (abs as u32 & 7) << 8
            | vdst as u32,
        (neg as u32 & 7) << 29
            | (omod as u32 & 3) << 27
            | (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    build_cs(body, 1, 0)
}

/// [`build_valu_cs`] with `threads` threads in x (thread index in `v0`
/// on entry, before `body`) and `lds_bytes` of LDS (a multiple of 512).
fn build_cs(body: &[u32], threads: u32, lds_bytes: u32) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (5 << 1) | ((lds_bytes / 512) << 15), // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// One GPU test at a time. With these tests on the hardware adapter and WARP
/// from several test threads at once, now and then none of their dispatches
/// ever completes and the test binary hangs; and two GPU tests at once on
/// WARP can end the test process with an access violation. Other GPU test
/// modules take this lock too.
pub(super) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    run_cases_on(shader, cases, &[("hardware", false), ("warp", true)], check);
}

fn run_cases_on(
    shader: &HostShader,
    cases: &[[u32; 4]],
    devices: &[(&str, bool)],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for &(label, warp) in devices {
        let host = match Host::new(warp) {
            Ok(host) => host,
            Err(e) => {
                eprintln!("no SM 6.6 {label} device ({e}); skipped");
                continue;
            }
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    if ran == 0 && devices.iter().all(|&(_, warp)| !warp) {
        // A case for the hardware adapter alone, on a machine without one.
        eprintln!("no SM 6.6 hardware device; a hardware-only case not run");
        return;
    }
    assert!(ran > 0, "no SM 6.6 device available ({devices:?})");
}

#[test]
fn scalar_quad_masks_pack_nibbles_and_set_scc() {
    let pack_words = |opcode: u32| {
        [
            0xBE90_0000 | opcode << 8,   // s_quadmask s16, s0 (pairs for b64)
            sop2(0x0a, 18, 129, 128),  // s_cselect_b32 s18, 1, 0
            sop2(0x1e, 18, 18, 144),   // s_lshl_b32 s18, s18, 16
            sop2(0x10, 16, 16, 18),    // s_or_b32 s16, s16, s18
            vop1(1, 0, sgpr(16)),      // v_mov_b32 v0, s16
        ]
    };
    let cases = [
        ([0, 0, 0, 0], [0, 0]),
        ([0x8000_0420, 0x0100_0001, 0, 0], [0x1_0086, 0x1_4186]),
        ([0, 0x8000_0000, 0, 0], [0, 0x1_8000]),
        ([u32::MAX, u32::MAX, 0, 0], [0x1_00ff, 0x1_ffff]),
    ];
    let inputs = cases.map(|(input, _)| input);
    for (wide, opcode) in [0x2c, 0x2d].into_iter().enumerate() {
        let shader = build_valu_cs(&pack_words(opcode));
        run_cases(&shader, &inputs, |backend, i, _, got| {
            assert_eq!(got, cases[i].1[wide], "{backend}, opcode {opcode:#x}");
        });
    }
    let high = build_valu_cs(&[
        0xBE90_2D00, // s_quadmask_b64 s[16:17], s[0:1]
        vop1(1, 0, sgpr(17)),
    ]);
    run_cases(&high, &inputs, |backend, _, _, got| {
        assert_eq!(got, 0, "{backend}: high dword");
    });
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

#[test]
fn a_branch_to_fallthrough_keeps_both_values_at_a_join() {
    let shader = build_valu_cs(&[
        0xBF06_0100,            // s_cmp_eq_u32 s0, s1
        0xBF85_0002,            // s_cbranch_scc1 -> the s2 = 1 path
        s_mov_b32(2, 128),      // s2 = 0
        0xBF82_0002,            // s_branch -> the join
        s_mov_b32(2, 129),      // s2 = 1
        0xBF85_0000,            // s_cbranch_scc1 -> the following instruction
        vop1(0x001, 0, sgpr(2)), // join: v0 = s2
    ]);
    run_cases(
        &shader,
        &[[7, 7, 0, 0], [7, 5, 0, 0]],
        |backend, _, inputs, actual| {
            assert_eq!(actual, u32::from(inputs[0] == inputs[1]), "{backend}");
        },
    );
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

/// MTBUF `op vdata, off, s[srsrc:srsrc+3], 0 format:format` (at byte 0).
const fn mtbuf(op: u8, vdata: u8, srsrc: u8, format: u8) -> [u32; 2] {
    [
        0x3A << 26 | (format as u32 & 0x7F) << 19 | (op as u32 & 7) << 16,
        0x80 << 24
            | (op as u32 >> 3 & 1) << 21
            | (srsrc as u32 >> 2 & 0x1F) << 16
            | (vdata as u32) << 8,
    ]
}

/// Stores `v1` to the buffer, then reads it back into `v0` with the tbuffer
/// load `op` in buffer format `format`.
fn tbuffer_load_of_v1(op: u8, format: u8) -> HostShader {
    let mut body = buffer_store_dword(1, 8, 0).to_vec();
    body.extend(mtbuf(op, 0, 8, format));
    build_valu_cs(&body)
}

#[test]
fn tbuffer_d16_loads_pack_two_channels_per_dword() {
    // tbuffer_load_format_d16_x and tbuffer_load_format_d16_xy.
    let (x, xy) = (8, 9);
    // (opcode, format, dword stored, dword loaded)
    let cases: &[(u8, u8, u32, u32)] = &[
        // 16_16_FLOAT: the f16 pair (1.0, 2.0) as it was stored.
        (xy, 29, 0x4000_3C00, 0x4000_3C00),
        // 16_16_UINT: the 16-bit integers as they were stored.
        (xy, 27, 0xBEEF_1234, 0xBEEF_1234),
        // 16_16_UNORM: 0.0 and 1.0, as f16.
        (xy, 23, 0xFFFF_0000, 0x3C00_0000),
        // 16_FLOAT, one channel: the high half is 0.
        (x, 13, 0x4000_3C00, 0x0000_3C00),
    ];
    for &(op, format, stored, want) in cases {
        let shader = tbuffer_load_of_v1(op, format);
        run_cases(&shader, &[[stored, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                got, want,
                "{backend}: d16 opcode {op} format {format} of {stored:#010x} = {got:#010x}, want {want:#010x}",
            );
        });
    }
}

#[test]
fn normalized_buffer_formats_load_their_ends_exactly() {
    // tbuffer_load_format_x.
    let x = 0;
    // (format, dword stored, value loaded)
    let cases: &[(u8, u32, f32)] = &[
        (1, 0xFF, 1.0),   // 8_UNORM
        (7, 0xFFFF, 1.0), // 16_UNORM
        (2, 0x7F, 1.0),   // 8_SNORM: the top, the one above the lowest, the lowest
        (2, 0x81, -1.0),
        (2, 0x80, -1.0),
        (8, 0x7FFF, 1.0), // 16_SNORM
        (8, 0x8001, -1.0),
        (8, 0x8000, -1.0),
    ];
    for &(format, stored, want) in cases {
        let shader = tbuffer_load_of_v1(x, format);
        run_cases(&shader, &[[stored, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                got,
                want.to_bits(),
                "{backend}: format {format} of {stored:#x} = {got:#010x}, want {want}",
            );
        });
    }
}

/// Codes of the normalized formats sampled below: every 8-bit code; for
/// 16 bits a stride, the RX 6700 XT's division misses (257, 261, ...) and
/// the ends.
fn normalized_codes(bits: u32) -> Vec<u32> {
    if bits <= 10 {
        return (0..1 << bits).collect();
    }
    let edges = [1, 2, 3, 255, 256, 257, 261, 265, 269, 511, 512, 1000, 0x7FFF, 0x8000, 0x8001, 0xFFFE];
    (0..=0xFFFF).step_by(257).chain(edges).collect()
}

/// Every normalized code loads as the float nearest its exact quotient,
/// as the console's buffer unit converts it, not as the host divides (the
/// RX 6700 XT gave 125 of the 8-bit codes one ulp high and 16-bit codes
/// such as 257 one ulp low): unsigned and signed, 8, 10 (the x channel of
/// 2_10_10_10) and 16 bits, the signed formats' two lowest codes -1.0.
#[test]
fn normalized_buffer_formats_load_correctly_rounded_quotients() {
    let x = 0;
    // (format, bits, signed): 8_UNORM, 16_UNORM, 8_SNORM, 16_SNORM,
    // 2_10_10_10_UNORM and _SNORM.
    for (format, bits, signed) in [(1u8, 8u32, false), (7, 16, false), (2, 8, true), (8, 16, true), (50, 10, false), (51, 10, true)] {
        let shader = tbuffer_load_of_v1(x, format);
        let cases: Vec<[u32; 4]> = normalized_codes(bits).iter().map(|&a| [a, 0, 0, 0]).collect();
        run_cases(&shader, &cases, |backend, _, [a, ..], got| {
            let want = if signed {
                let top = (1i32 << (bits - 1)) - 1;
                let s = ((a << (32 - bits)) as i32) >> (32 - bits);
                (s.max(-top) as f32 / top as f32).to_bits()
            } else {
                (a as f32 / ((1u32 << bits) - 1) as f32).to_bits()
            };
            let sign = if signed { "signed" } else { "unsigned" };
            assert_eq!(got, want, "{backend}: {bits}-bit {sign} normalized {a:#x} loads as {got:#010x}, want {want:#010x}");
        });
    }
}

/// A title truncates `x * (2^n - 1)` back to the code it stored (the
/// material index a mesh program reads from a vertex colour's alpha, as
/// `v_mul_legacy_f32` by 255.0 then `v_cvt_u32_f32`): every code comes back
/// as itself, on hardware and WARP, which a quotient one ulp low breaks.
#[test]
fn normalized_codes_truncate_back_to_themselves() {
    let x = 0;
    for (format, bits) in [(1u8, 8u32), (7, 16)] {
        let scale = (((1u32 << bits) - 1) as f32).to_bits();
        let mut body = buffer_store_dword(1, 8, 0).to_vec();
        body.extend(mtbuf(x, 0, 8, format));
        // v_mul_legacy_f32 v0, literal, v0; v_cvt_u32_f32 v0, v0.
        body.extend([0x07 << 25 | 0xFF, scale]);
        body.push(vop1(0x07, 0, vgpr(0)));
        let shader = build_valu_cs(&body);
        let cases: Vec<[u32; 4]> = normalized_codes(bits).iter().map(|&a| [a, 0, 0, 0]).collect();
        run_cases(&shader, &cases, |backend, _, [a, ..], got| {
            assert_eq!(got, a, "{backend}: {bits}-bit code {a} comes back as {got}");
        });
    }
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

/// The float `v_cvt_f32_f64` gives for `x`: Rust's rounding (to nearest
/// even), with results below the smallest normal float flushed to signed
/// zero as the console's default float mode does.
fn cvt_f32_f64_reference(x: f64) -> u32 {
    let r = x as f32;
    if r.is_subnormal() || (r == 0.0 && x != 0.0) {
        r.to_bits() & 0x8000_0000
    } else {
        r.to_bits()
    }
}

#[test]
#[ignore = "expects results below the smallest normal float flushed to zero; the IEEE lowering (v_cvt_f32_f64_matches_ieee754) is kept until the console's mode is checked"]
fn v_cvt_f32_f64_rounds_to_nearest_even() {
    // v_cvt_f32_f64 v0, v[1:2] (VOP1 opcode 0x0f)
    let shader = build_valu_cs(&[vop1(0x00f, 0, vgpr(1))]);
    let tie_even = 1.0f64 + f64::from(f32::EPSILON) / 2.0; // ties to 1.0
    let tie_odd = 1.0f64 + 1.5 * f64::from(f32::EPSILON); // ties up to 1 + 2 ulp
    let values = [
        0.0,
        -0.0,
        1.0,
        -2.5,
        std::f64::consts::PI,
        1.0e-3,
        123_456_789.123,
        tie_even,
        tie_odd,
        1.0 + f64::from(f32::EPSILON) * 0.75,
        f64::from(f32::MAX),
        f64::from(f32::MAX) * 1.000_000_1, // rounds to infinity
        1.0e300,
        -1.0e300,
        1.0e-40, // a float denormal: flushed
        -1.0e-300,
        f64::from(f32::MIN_POSITIVE),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let inputs: Vec<[u32; 4]> = values
        .iter()
        .map(|x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&shader, &inputs, |backend, i, _, got| {
        let x = values[i];
        if x.is_nan() {
            assert!(f32::from_bits(got).is_nan(), "{backend}: cvt({x}) = {got:#010x}, want NaN");
            return;
        }
        let want = cvt_f32_f64_reference(x);
        assert_eq!(got, want, "{backend}: cvt({x:e}) = {got:#010x}, want {want:#010x}");
    });
}

/// The lane `v_permlane16_b32` (`cross` false) or `v_permlanex16_b32` reads
/// for `lane`, with selector dwords `s1` (lanes 0-7 of a row) and `s2`.
fn permlane_source(lane: u32, s1: u32, s2: u32, cross: bool) -> u32 {
    let sel = u64::from(s1) | u64::from(s2) << 32;
    let pick = (sel >> (4 * (lane & 15)) & 15) as u32;
    let row = if cross {
        (lane & !31) | ((lane & 16) ^ 16)
    } else {
        lane & !15
    };
    row | pick
}

#[test]
fn v_permlane16_and_x16_read_the_lanes_their_selectors_pick() {
    // 32 lanes, each with its lane number in v0; v1 = permute(v0, s0, s1);
    // lane `read` of v1 into s12, then into v0 for the store.
    let selectors: [(u32, u32); 3] = [
        (0x7654_3210, 0xfedc_ba98), // identity within the row
        (0x89ab_cdef, 0x0123_4567), // reversed
        (0x1111_1111, 0xf0f0_f0f0), // repeats
    ];
    for (op, cross) in [(0x377u16, false), (0x378, true)] {
        for read in [0u8, 5, 15, 16, 22, 31] {
            let mut body = Vec::new();
            body.extend_from_slice(&vop3(op, 1, vgpr(0), sgpr(0), sgpr(1)));
            // v_readlane_b32 s12, v1, read (inline constant 128 + read)
            body.extend_from_slice(&vop3(0x360, 12, vgpr(1), 128 + u16::from(read), 0));
            body.push(vop1(0x001, 0, sgpr(12))); // v_mov_b32 v0, s12
            let shader = build_cs(&body, 32, 0);
            let inputs: Vec<[u32; 4]> = selectors.iter().map(|&(a, b)| [a, b, 0, 0]).collect();
            // Cross-lane code declares WaveSize(32), which WARP's pipeline
            // creation refuses: hardware only.
            run_cases_on(&shader, &inputs, &[("hardware", false)], |backend, i, inp, got| {
                let want = permlane_source(u32::from(read), inp[0], inp[1], cross);
                assert_eq!(
                    got, want,
                    "{backend}: op {op:#x} selectors {:08x?}, lane {read}",
                    selectors[i]
                );
            });
        }
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }
}

/// The f16 forms go through the same `max_min` helper, after widening to f32.
#[test]
fn f16_max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_u32);
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f16", 0x139, pz),
        ("v_min_f16", 0x13A, nz),
        ("v_max3_f16", 0x354, pz),
        ("v_min3_f16", 0x351, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            // The f16 result is the low half of v0.
            assert_eq!(got & 0xFFFF, want, "{backend}: {name} {inp:x?}");
        });
    }
}

#[test]
fn v_min3_max3_med3_integer_forms_match_rdna2() {
    // v_{min3,max3,med3}_{i32,u32} v0, v1, v2, v3 (VOP3 0x152..0x159).
    type Pick = fn([u32; 3]) -> u32;
    let ops: [(&str, u16, Pick); 6] = [
        ("v_min3_i32", 0x152, |t| {
            t.into_iter().min_by_key(|&x| x as i32).unwrap()
        }),
        ("v_min3_u32", 0x153, |t| t.into_iter().min().unwrap()),
        ("v_max3_i32", 0x155, |t| {
            t.into_iter().max_by_key(|&x| x as i32).unwrap()
        }),
        ("v_max3_u32", 0x156, |t| t.into_iter().max().unwrap()),
        ("v_med3_i32", 0x158, |mut t| {
            t.sort_by_key(|&x| x as i32);
            t[1]
        }),
        ("v_med3_u32", 0x159, |mut t| {
            t.sort_unstable();
            t[1]
        }),
    ];
    // Every ordering of three values, with the sign bit set on some so that
    // the signed and unsigned answers differ.
    let values = [0u32, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF];
    let mut cases = Vec::new();
    for a in values {
        for b in values {
            for c in values {
                cases.push([a, b, c, 0]);
            }
        }
    }
    for (name, op, pick) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let want = pick([inp[0], inp[1], inp[2]]);
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1], inp[2],
            );
        });
    }
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

#[test]
fn v_cvt_off_f32_i4_uses_the_signed_low_nibble() {
    // v_cvt_off_f32_i4 v0, v1 (VOP1 opcode 0x00e)
    let shader = build_valu_cs(&[vop1(0x00e, 0, vgpr(1))]);
    let nibble = |n: u32| (((n as i32) << 28) >> 28) as f32 / 16.0;

    let mut cases: Vec<u32> = (0..16).collect();
    // Bits above the low nibble are ignored.
    cases.extend([0xABCD_EF05, 0xFFFF_FFF8, 0x1234_5670, 0x8000_000F, 0x0000_00F7]);
    let inputs: Vec<[u32; 4]> = cases.iter().map(|&x| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = nibble(cases[i]).to_bits();
        assert_eq!(
            got, want,
            "{backend}: v_cvt_off_f32_i4({:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
            inp[0],
            f32::from_bits(got),
        );
    });
    // Spot-check the reference itself: 0x5 -> 0.3125, 0x8 -> -0.5, 0xF -> -0.0625.
    assert_eq!(nibble(0xABCD_EF05), 0.3125);
    assert_eq!(nibble(8), -0.5);
    assert_eq!(nibble(0xF), -0.0625);
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

#[test]
fn v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would
    // round before the rounding to an integer, ties, and the edges.
    let mut bits: Vec<u16> = (0..=u16::MAX).step_by(251).collect();
    bits.extend([
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
        0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0x3A9B, 0xBA9B, 0x2E66, 0x1400,
    ]);
    for (op, signed, what) in [
        (0x063u16, true, "v_cvt_norm_i16_f16"),
        (0x064, false, "v_cvt_norm_u16_f16"),
    ] {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = bits.iter().map(|&b| [u32::from(b), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = cvt_norm16_reference(bits[i], signed);
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                bits[i],
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    let cases: &[(u16, u16, u16)] = &[
        // (f16 input, u16 result, i16 result)
        (0x0000, 0, 0),          // 0
        (0x8000, 0, 0),          // -0
        (0x3E00, 1, 1),          // 1.5 -> 1 (toward zero)
        (0xBE00, 0, 0xFFFF),     // -1.5 -> 0 / -1
        (0x57D0, 125, 125),      // 125.0
        (0x7800, 32768, 32767),  // 32768 saturates i16
        (0x77FF, 32752, 32752),  // 32752, the largest f16 below 32768
        (0xF800, 0, 0x8000),     // -32768 fits i16
        (0xF801, 0, 0x8000),     // -32800 saturates i16
        (0x7BFF, 65504, 32767),  // max finite f16
        (0xFBFF, 0, 0x8000),     // -65504
        (0x7C00, 0xFFFF, 32767), // +inf
        (0xFC00, 0, 0x8000),     // -inf
        (0x7E00, 0, 0),          // NaN
        (0xFE00, 0, 0),          // -NaN
        (0x0001, 0, 0),          // smallest subnormal
    ];
    for (op, what, pick) in [
        (0x052u16, "v_cvt_u16_f16", 0usize),
        (0x053, "v_cvt_i16_f16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_cvt_f16_u16_and_i16_widen_the_integer_first() {
    // v_mov_b32 v0, 0; v_cvt_f16_u16 / v_cvt_f16_i16 v0, v1 (VOP1 0x050 / 0x051).
    // Only integers f16 holds exactly: how the others round is the narrowing's
    // business, not this conversion's. `None`: the result depends on rounding.
    let cases: &[(u16, Option<u16>, Option<u16>)] = &[
        // (u16 / i16 input bits, f16 of the unsigned value, f16 of the signed value)
        (0x0000, Some(0x0000), Some(0x0000)), // 0
        (0x0001, Some(0x3C00), Some(0x3C00)), // 1.0
        (0x07FF, Some(0x67FF), Some(0x67FF)), // 2047, the last integer below 2048 that f16 holds
        (0x8000, Some(0x7800), Some(0xF800)), // 32768 / -32768
        (0xFC00, Some(0x7BE0), Some(0xE400)), // 64512 / -1024
        (0xFFFE, None, Some(0xC000)),         // 65534 does not fit f16 / -2
        (0xFFFF, None, Some(0xBC00)),         // 65535 does not fit f16 / -1
    ];
    for (op, what, pick) in [
        (0x050u16, "v_cvt_f16_u16", 0usize),
        (0x051, "v_cvt_f16_i16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            let Some(want) = want else { return };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn s_cmpk_compares_against_the_extended_immediate() {
    // s_cmpk_<cond>_<type> s0, imm; s_cselect_b32 s6, 1, 0 (SOP2 0x0a, on SCC);
    // v_mov_b32 v0, s6. The signed forms sign-extend the 16-bit immediate, the
    // unsigned forms zero-extend it.
    type Cond = fn(i64, i64) -> bool;
    let conds: [(&str, Cond); 6] = [
        ("eq", |a, b| a == b),
        ("lg", |a, b| a != b),
        ("gt", |a, b| a > b),
        ("ge", |a, b| a >= b),
        ("lt", |a, b| a < b),
        ("le", |a, b| a <= b),
    ];
    // An immediate below and one at the sign bit; operands around each extension.
    let imms: [u16; 2] = [0x0005, 0x8000];
    let operands: [u32; 9] = [
        0,
        4,
        5,
        6,
        0x7FFF,
        0x8000,
        0xFFFF_8000,
        0xFFFF_FFFF,
        0x8000_0000,
    ];
    for (signed, first_op) in [(true, 0x003u8), (false, 0x009)] {
        for (k, (cond, holds)) in conds.iter().enumerate() {
            for imm in imms {
                let shader = build_valu_cs(&[
                    sopk(first_op + k as u8, 0, imm),
                    sop2(0x0a, 6, 129, 128),
                    vop1(0x001, 0, sgpr(6)),
                ]);
                let name = format!("s_cmpk_{cond}_{}32", if signed { "i" } else { "u" });
                let inputs: Vec<[u32; 4]> = operands.iter().map(|&v| [v, 0, 0, 0]).collect();
                run_cases(&shader, &inputs, |backend, i, _, got| {
                    let (a, b) = if signed {
                        (i64::from(operands[i] as i32), i64::from(imm as i16))
                    } else {
                        (i64::from(operands[i]), i64::from(imm))
                    };
                    let want = u32::from(holds(a, b));
                    assert_eq!(
                        got, want,
                        "{backend}: {name}({:#010x}, {imm:#06x}) = {got}, want {want}",
                        operands[i],
                    );
                });
            }
        }
    }
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// A 32-bit literal in a signed 64-bit operand is sign-extended (an
/// unsigned one's is zero-extended): -100 compared with `v[1:2]`, and
/// shifted by `s_ashr_i64`.
#[test]
fn signed_64_bit_operands_sign_extend_their_literal() {
    const MINUS_100: u32 = -100_i32 as u32;
    let check = |name: &str, body: &[u32], cases: &[([u32; 4], u32)]| {
        let shader = build_valu_cs(body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    };
    // v_cmp_*_i64 s6, v[1:2], -100 (VOP3, literal src1), then
    // v_cndmask_b32 v0, 0, 1, s6.
    let compare = |op: u16| {
        let [a, b] = vop3(op, 6, vgpr(1), 255, 0);
        let [c, d] = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
        [a, b, MINUS_100, c, d]
    };
    // An i64 as the two input dwords of `v[1:2]`.
    let pair = |n: i64| [n as u32, (n >> 32) as u32, 0, 0];
    let eq = [(pair(-100), 1), ([MINUS_100, 0, 0, 0], 0), (pair(0), 0)];
    check("v_cmp_eq_i64", &compare(0x0A2), &eq);
    let lt = [(pair(-200), 1), (pair(-50), 0), (pair(0), 0)];
    check("v_cmp_lt_i64", &compare(0x0A1), &lt);
    // s_ashr_i64 s[6:7], -100, 4; v_mov_b32 v0, s7: the high dword.
    let ashr = [sop2(0x23, 6, 255, 0x84), MINUS_100, vop1(0x001, 0, sgpr(7))];
    check("s_ashr_i64", &ashr, &[([0; 4], u32::MAX)]);
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);
    // s_bfe_u32 s6, s0, s1 (SOP2 opcode 0x27: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_unsigned = build_valu_cs(&[sop2(0x27, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);
    // SCC = (result != 0), consumed by s_cselect_b32 s7, s2, s3.
    let s_signed_scc = build_valu_cs(&[
        sop2(0x28, 6, 0, 1),
        sop2(0x0A, 7, 2, 3),
        vop1(0x001, 0, sgpr(7)),
    ]);
    let s_unsigned_scc = build_valu_cs(&[
        sop2(0x27, 6, 0, 1),
        sop2(0x0A, 7, 2, 3),
        vop1(0x001, 0, sgpr(7)),
    ]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_{i32,u32} width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    for (name, shader) in [("v_bfe_i32", &v_signed), ("v_bfe_u32", &v_unsigned)] {
        run_cases(shader, &[[0x8000_0000, 4, 32, 0]], |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: {name} width 32 should mask to 0");
        });
    }

    const WHEN_NZ: u32 = 0xAAAA_5555;
    const WHEN_ZERO: u32 = 0x1111_2222;
    let pack = |off: u32, width: u32| (off & 31) | ((width & 0x7F) << 16);

    // s_bfe_i32 width is 7-bit (src1[22:16]), so widths >= 32 keep all remaining bits
    // with sign extension from bit 31 (including width=64 where width & 31 == 0).
    let mut s_cases = signed_cases.to_vec();
    s_cases.extend([
        ([0x8000_0000, 4, 32, 0], 0xf800_0000),
        ([0x8000_0000, 0, 32, 0], 0x8000_0000),
        ([0x8000_0000, 4, 64, 0], 0xf800_0000),
        ([0xd2f9_0159, 28, 127, 0], 0xffff_fffd),
        ([0x0000_000f, 4, 32, 0], 0),
    ]);
    let mut s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], pack(inp[1], inp[2]), WHEN_NZ, WHEN_ZERO])
        .collect();
    // Unused bits src1[15:5] and src1[31:23] are ignored by hardware.
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    s_in.push([0x8000_0000, 0xFF80_FFE0 | pack(4, 32), WHEN_NZ, WHEN_ZERO]);
    s_cases.push(([0xffff_ffff, 4, 0, 0], 0));
    s_in.push([0xffff_ffff, 0xFF80_FFE4, WHEN_NZ, WHEN_ZERO]);
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
    run_cases(&s_signed_scc, &s_in, |backend, i, inp, got| {
        let want = if s_cases[i].1 != 0 {
            WHEN_NZ
        } else {
            WHEN_ZERO
        };
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32 SCC({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0x0000_0080, 4, 4, 0], 0x0000_0008),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    // s_bfe_u32 width is 7-bit (src1[22:16]): widths >= 32 shift without sign extension.
    let mut su_cases = unsigned_cases.to_vec();
    su_cases.extend([
        ([0x8000_0000, 4, 32, 0], 0x0800_0000),
        ([0x8000_0000, 0, 32, 0], 0x8000_0000),
        ([0x8000_0000, 4, 64, 0], 0x0800_0000),
        ([0xd2f9_0159, 28, 127, 0], 0x0000_000d),
        ([0x0000_000f, 4, 32, 0], 0),
    ]);
    let mut su_in: Vec<[u32; 4]> = su_cases
        .iter()
        .map(|&(inp, _)| [inp[0], pack(inp[1], inp[2]), WHEN_NZ, WHEN_ZERO])
        .collect();
    su_cases.push(([0x8000_0000, 4, 32, 0], 0x0800_0000));
    su_in.push([0x8000_0000, 0xFF80_FFE0 | pack(4, 32), WHEN_NZ, WHEN_ZERO]);
    su_cases.push(([0xffff_ffff, 4, 0, 0], 0));
    su_in.push([0xffff_ffff, 0xFF80_FFE4, WHEN_NZ, WHEN_ZERO]);
    run_cases(&s_unsigned, &su_in, |backend, i, inp, got| {
        let want = su_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_u32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
    run_cases(&s_unsigned_scc, &su_in, |backend, i, inp, got| {
        let want = if su_cases[i].1 != 0 {
            WHEN_NZ
        } else {
            WHEN_ZERO
        };
        assert_eq!(
            got, want,
            "{backend}: s_bfe_u32 SCC({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    // Literal src1 (`0xFF` followed by 32-bit immediate) for both s_bfe_u32 and s_bfe_i32:
    // offset=4 (with ignored bits [15:5]), width=32/64/0 (with ignored bits [31:23]).
    let literals: [(u8, u32, u32, u32, u32); 6] = [
        (0x27, 0x8020_FFE4, 0x8000_0000, 0x0800_0000, WHEN_NZ),
        (0x28, 0x8020_FFE4, 0x8000_0000, 0xf800_0000, WHEN_NZ),
        (0x27, 0x0040_001C, 0xd2f9_0159, 0x0000_000d, WHEN_NZ),
        (0x28, 0x0040_001C, 0xd2f9_0159, 0xffff_fffd, WHEN_NZ),
        (0x27, 0xFF80_FFE4, 0xffff_ffff, 0, WHEN_ZERO),
        (0x28, 0xFF80_FFE4, 0xffff_ffff, 0, WHEN_ZERO),
    ];
    for (op, lit, src0, want, want_scc) in literals {
        let val_cs = build_valu_cs(&[sop2(op, 6, 0, 255), lit, vop1(0x001, 0, sgpr(6))]);
        run_cases(&val_cs, &[[src0, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                got, want,
                "{backend}: literal sop2({op:#x}, {src0:#010x}, {lit:#010x}) = {got:#010x}, want {want:#010x}",
            );
        });
        let scc_cs = build_valu_cs(&[
            sop2(op, 6, 0, 255),
            lit,
            sop2(0x0A, 7, 2, 3),
            vop1(0x001, 0, sgpr(7)),
        ]);
        run_cases(
            &scc_cs,
            &[[src0, 0, WHEN_NZ, WHEN_ZERO]],
            |backend, _, _, got| {
                assert_eq!(
                    got, want_scc,
                    "{backend}: literal sop2({op:#x}) SCC({src0:#010x}, {lit:#010x}) = {got:#010x}",
                );
            },
        );
    }
}

#[test]
fn f16_omod_applies_before_clamp() {
    // VOP3 v_add_f16 (0x132 = 0x100 + VOP2 0x32): v0 = (v1 + v2) * omod [clamp]
    let mul2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 1, false));
    let mul4 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 2, false));
    let div2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, false));
    let div2_clamp = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, true));

    // f16 constants: 0.25=0x3400, 0.5=0x3800, 0.75=0x3A00, 1.0=0x3C00, 1.5=0x3E00, 2.0=0x4000, -0.5=0xB800
    run_cases(
        &mul2,
        &[[0x3400, 0x3800, 0, 0]], // (0.25 + 0.5) * 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 mul:2");
        },
    );
    run_cases(
        &mul4,
        &[[0x3400, 0x3400, 0, 0]], // (0.25 + 0.25) * 4 = 2.0 (0x4000)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x4000, "{backend}: v_add_f16 mul:4");
        },
    );
    run_cases(
        &div2,
        &[[0x3E00, 0x3E00, 0, 0]], // (1.5 + 1.5) / 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 div:2");
        },
    );
    // Ordering check: (0.75 + 0.75) = 1.5; div:2 before clamp gives 0.75 (0x3A00),
    // whereas clamp before div:2 would give 0.5 (0x3800).
    let clamp_cases: &[(u16, u16, u16)] = &[
        (0x3A00, 0x3A00, 0x3A00), // (0.75 + 0.75) / 2 = 0.75 -> 0.75
        (0x3E00, 0x3E00, 0x3C00), // (1.5 + 1.5) / 2 = 1.5 -> clamped to 1.0
        (0xB800, 0xB800, 0x0000), // (-0.5 + -0.5) / 2 = -0.5 -> clamped to 0.0
    ];
    let inputs: Vec<[u32; 4]> = clamp_cases
        .iter()
        .map(|&(a, b, _)| [u32::from(a), u32::from(b), 0, 0])
        .collect();
    run_cases(&div2_clamp, &inputs, |backend, i, _, got| {
        let want = clamp_cases[i].2;
        assert_eq!(
            got as u16, want,
            "{backend}: v_add_f16 div:2 clamp case {i}: got {:#06x}, want {want:#06x}",
            got as u16,
        );
    });
}

#[test]
fn cvt_16bit_int_from_f16_saturates_and_honors_modifiers() {
    // VOP3 v_cvt_u16_f16 = 0x1D2 (0x180 + VOP1 0x52)
    // VOP3 v_cvt_i16_f16 = 0x1D3 (0x180 + VOP1 0x53)
    let cvt_u16 = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_u16_abs = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 1, 0, 0, false));
    let cvt_i16 = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_i16_neg = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 1, 0, false));

    let u16_cases: &[(u16, u16)] = &[
        (0xBC00, 0),      // -1.0 -> 0
        (0xFC00, 0),      // -inf -> 0
        (0x7E00, 0),      // NaN -> 0
        (0x57BC, 123),    // 123.75 -> 123
        (0x7BFF, 65504),  // 65504.0 (max finite f16) -> 65504
        (0x7C00, 0xFFFF), // +inf -> 65535 (saturated)
    ];
    let u16_in: Vec<[u32; 4]> = u16_cases.iter().map(|&(x, _)| [u32::from(x), 0, 0, 0]).collect();
    run_cases(&cvt_u16, &u16_in, |backend, i, inp, got| {
        let want = u16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_u16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_u16_abs,
        &[[0xD7BC, 0, 0, 0]], // |-123.75| -> 123
        |backend, _, _, got| {
            assert_eq!(got as u16, 123, "{backend}: v_cvt_u16_f16 |src0|");
        },
    );

    let i16_cases: &[(u16, u16)] = &[
        (0xFC00, 0x8000),       // -inf -> -32768
        (0xF8E2, 0x8000),       // -40000.0 -> -32768 (saturated, not wrapped)
        (0xD7BC, (-123i16) as u16), // -123.75 -> -123
        (0x7E00, 0),            // NaN -> 0
        (0x57BC, 123),          // 123.75 -> 123
        (0x78E2, 0x7FFF),       // +40000.0 -> 32767 (saturated, not wrapped)
        (0x7C00, 0x7FFF),       // +inf -> 32767
    ];
    let i16_in: Vec<[u32; 4]> = i16_cases.iter().map(|&(x, _)| [u32::from(x), 0, 0, 0]).collect();
    run_cases(&cvt_i16, &i16_in, |backend, i, inp, got| {
        let want = i16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_i16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_i16_neg,
        &[[0x57BC, 0, 0, 0], [0x78E2, 0, 0, 0]], // -(123.75) -> -123; -(40000.0) -> -32768
        |backend, i, _, got| {
            let want = [(-123i16) as u16, 0x8000][i];
            assert_eq!(got as u16, want, "{backend}: v_cvt_i16_f16 -src0 case {i}");
        },
    );
}

const S_BARRIER: u32 = 0xBF8A_0000;

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// DS `op vdst, addr, data0` at offset 0.
const fn ds(op: u8, vdst: u8, addr: u8, data0: u8) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18,
        (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32,
    ]
}

const DS_WRITE_B32: u8 = 0x0D;
const DS_MIN_F32: u8 = 0x12;
const DS_MAX_F32: u8 = 0x13;
const DS_MIN_RTN_F32: u8 = 0x32;
const DS_MAX_RTN_F32: u8 = 0x33;
const DS_READ_B32: u8 = 0x36;

/// What RDNA2 writes for `ds_max_f32` / `ds_min_f32` over a dword holding
/// `old` with `data` (bits), as measured on gfx1036 with f32 denormals kept:
/// an sNaN, data first, then memory, is written quieted; otherwise a NaN
/// loses to a number (two quiet NaNs give the data); numbers compare as
/// floats with -0 below +0.
fn lds_min_max_ref(old: u32, data: u32, max: bool) -> u32 {
    let nan = |x: u32| x & 0x7FFF_FFFF > 0x7F80_0000;
    let snan = |x: u32| nan(x) && x & 0x40_0000 == 0;
    let key = |x: u32| {
        let b = x as i32;
        if b < 0 { b ^ 0x7FFF_FFFF } else { b }
    };
    if snan(data) {
        return data | 0x40_0000;
    }
    if snan(old) {
        return old | 0x40_0000;
    }
    match (nan(data), nan(old)) {
        (true, true) | (false, true) => return data,
        (true, false) => return old,
        (false, false) => {}
    }
    let take = if max {
        key(data) > key(old)
    } else {
        key(data) < key(old)
    };
    if take { data } else { old }
}

#[test]
fn ds_min_max_f32_order_floats_and_return_the_old_value() {
    let values = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.5,
        -2.5,
        1e-30,
        -1e-30,
        f32::from_bits(1), // denormal
        f32::from_bits(0x8000_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
    ];
    // Quiet and signalling NaNs of both signs, in memory and as data.
    let nans = [
        0x7FC0_0000u32,
        0xFFC0_0000,
        0x7F80_0001,
        0xFFA0_0000,
        0x7FBF_FFFF,
    ];
    let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).chain(nans).collect();
    let mut cases: Vec<[u32; 4]> = Vec::new();
    for &old in &bits {
        for &data in &bits {
            cases.push([old, data, 0, 0]);
        }
    }
    for (rtn, plain, max, what) in [
        (DS_MAX_RTN_F32, DS_MAX_F32, true, "ds_max"),
        (DS_MIN_RTN_F32, DS_MIN_F32, false, "ds_min"),
    ] {
        // v5 = 0; LDS[v5] = v1; the atomic with v2; v0 = the old value
        // (rtn) or the dword afterwards (read back).
        let mut returns_old = vec![vop1(0x001, 5, 0x80)];
        returns_old.extend(ds(DS_WRITE_B32, 0, 5, 1));
        returns_old.extend(ds(rtn, 0, 5, 2));
        let mut returns_new = vec![vop1(0x001, 5, 0x80)];
        returns_new.extend(ds(DS_WRITE_B32, 0, 5, 1));
        returns_new.extend(ds(plain, 0, 5, 2));
        returns_new.extend(ds(DS_READ_B32, 0, 5, 0));
        for (body, old_result) in [(returns_old, true), (returns_new, false)] {
            // One thread with LDS declares WaveSize(32), which WARP's
            // pipeline creation refuses: hardware only (the 64-thread test
            // below runs on WARP too).
            let shader = build_cs(&body, 1, 512);
            run_cases_on(
                &shader,
                &cases,
                &[("hardware", false)],
                |backend, _, inp, got| {
                    let (old, data) = (f32::from_bits(inp[0]), f32::from_bits(inp[1]));
                    let want = if old_result {
                        inp[0]
                    } else {
                        lds_min_max_ref(inp[0], inp[1], max)
                    };
                    assert_eq!(
                        got,
                        want,
                        "{backend}: {what}_f32 on {old:e} [{:#010x}] with {data:e} [{:#010x}] gave {got:#010x}, want {want:#010x} ({})",
                        inp[0],
                        inp[1],
                        if old_result {
                            "old value"
                        } else {
                            "dword after"
                        },
                    );
                },
            );
        }
    }
}

#[test]
fn ds_min_max_f32_are_atomic_across_waves() {
    // 64 threads (two host waves of 32 or one of 64): LDS[0] = v1, then
    // every thread applies (thread - 32) as a float; v0 = LDS[0].
    for (op, max, init, want) in [
        (DS_MAX_F32, true, f32::NEG_INFINITY, 31.0f32),
        (DS_MAX_F32, true, 100.0, 100.0),
        (DS_MIN_F32, false, f32::INFINITY, -32.0),
        (DS_MIN_F32, false, -0.0, -32.0),
    ] {
        let mut body = vec![vop1(0x001, 5, 0x80)];
        body.extend(ds(DS_WRITE_B32, 0, 5, 1));
        body.push(S_BARRIER);
        body.push(vop2(0x26, 6, vgpr(0), 3)); // v_sub_nc_u32 v6, v0, v3
        body.push(vop1(0x005, 6, vgpr(6))); // v_cvt_f32_i32
        body.extend(ds(op, 0, 5, 6));
        body.push(S_BARRIER);
        body.extend(ds(DS_READ_B32, 0, 5, 0));
        let shader = build_cs(&body, 64, 512);
        run_cases(
            &shader,
            &[[init.to_bits(), 0, 32, 0]],
            |backend, _, _, got| {
                assert_eq!(
                    f32::from_bits(got),
                    want,
                    "{backend}: {} from {init} over 64 threads",
                    if max { "max" } else { "min" },
                );
            },
        );
    }
}

#[test]
fn v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp_lo = build_valu_cs(&[
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1 << 11; // opsel[0] = 1 (high half)
    w1 |= 1 << 29; // neg[0] = 1
    let cmp_hi_neg = build_valu_cs(&[
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0), // sNaN (+sNaN, quiet bit 0x0200 clear)
        (0xFDFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7E00, 1), // +qNaN
        (0xFE01, 1), // -qNaN
        (0xFC00, 2), // -inf
        (0xBC00, 3), // -normal (-1.0)
        (0x8001, 4), // -denormal
        (0x8000, 5), // -0.0
        (0x0000, 6), // +0.0
        (0x0001, 7), // +denormal
        (0x3C00, 8), // +normal (+1.0)
        (0x7C00, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_lo, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16({:#06x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3);
    // high half 0x7C01 (+sNaN) negated -> still sNaN (class 0, not quieted).
    let hi_cases: &[([u32; 4], u32)] = &[
        ([0x3C00_0000, 1 << 3, 0, 0], 1),
        ([0x3C00_0000, 1 << 8, 0, 0], 0),
        ([0x7C01_FFFF, 1 << 0, 0, 0], 1),
        ([0x7C01_FFFF, 1 << 1, 0, 0], 0),
    ];
    let hi_in: Vec<[u32; 4]> = hi_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_hi_neg, &hi_in, |backend, i, inp, got| {
        let want = hi_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn v_cvt_f32_f64_matches_ieee754() {
    // v_cvt_f32_f64 v0, v[1:2] (VOP1 opcode 0x00f -> VOP3 opcode 0x18f).
    let plain = build_valu_cs(&[vop1(0x00f, 0, vgpr(1))]);
    // Same with abs + neg on src0 (VOP3: abs[0] = bit 8 of w0, neg[0] = bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x18f, 0, vgpr(1), 0, 0);
    w0 |= 1 << 8;
    w1 |= 1 << 29;
    let neg_abs = build_valu_cs(&[w0, w1]);

    let values: &[f64] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.5,
        -123456.75,
        // Tie-to-even: even mantissa stays, odd mantissa rounds up.
        f64::from_bits(0x3FF0_0000_1000_0000), // 1.0 + 0.5 ulp(f32) -> 1.0
        f64::from_bits(0x3FF0_0000_3000_0000), // (1.0 + 1 ulp) + 0.5 ulp -> 1.0 + 2 ulp
        f64::from_bits(0x3FF0_0000_1000_0001), // > 0.5 ulp -> rounds up
        // Largest finite f32 and overflow to +Inf / -Inf.
        f64::from(f32::MAX),
        f64::from(f32::MAX) * 1.000_001,
        -f64::from(f32::MAX) * 1.000_001,
        1e300,
        // Subnormal range (exp64 in [874, 896]), including carry into smallest normal.
        f64::from(f32::MIN_POSITIVE),
        f64::from(f32::MIN_POSITIVE) * 0.5,
        f64::from(f32::MIN_POSITIVE) * 0.75,
        f64::from(f32::from_bits(0x007F_FFFF)),
        f64::from_bits(0x380F_FFFF_F000_0000), // max subnormal + 0.5 ulp -> min normal
        // exp64 == 873: exact 2^-150 ties to 0; strictly greater rounds to min subnormal (1).
        f64::from_bits(0x3690_0000_0000_0000),
        f64::from_bits(0x3690_0000_0000_0001),
        f64::from_bits(0xB690_0000_0000_0001),
        // Deep underflow -> +/-0.0.
        1e-200,
        -1e-200,
        // Infinities and NaNs.
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7FF8_0000_0000_0000), // +qNaN
        f64::from_bits(0x7FF0_0000_0000_0001), // +sNaN -> quieted
    ];
    let inputs: Vec<[u32; 4]> = values
        .iter()
        .map(|&x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&plain, &inputs, |backend, i, _, got| {
        let x = values[i];
        let want = (x as f32).to_bits();
        if x.is_nan() {
            assert!(
                f32::from_bits(got).is_nan() && (got & 0x0040_0000) != 0,
                "{backend}: v_cvt_f32_f64({:#018x}) = {got:#010x}, expected qNaN",
                x.to_bits(),
            );
        } else {
            assert_eq!(
                got, want,
                "{backend}: v_cvt_f32_f64({x} / {:#018x}) = {got:#010x}, want {want:#010x}",
                x.to_bits(),
            );
        }
    });

    // -|x| via VOP3 abs + neg modifiers.
    let mod_values: &[f64] = &[2.5, -2.5, 0.0, f64::INFINITY];
    let mod_in: Vec<[u32; 4]> = mod_values
        .iter()
        .map(|&x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&neg_abs, &mod_in, |backend, i, _, got| {
        let x = mod_values[i];
        let want = (-(x.abs()) as f32).to_bits();
        assert_eq!(
            got, want,
            "{backend}: v_cvt_f32_f64(-|{x}|) = {got:#010x}, want {want:#010x}",
        );
    });
}

#[test]
fn whole_wave_f64_sum_and_cvt_f32_f64() {
    let s_or_saveexec_b32 =
        |sdst: u8, src0: u8| 0xBE80_0000 | (sdst as u32) << 16 | 0x3D << 8 | src0 as u32;
    let v_mov_dpp8 = |vdst: u8, vsrc: u8, sel: u32| [vop1(0x001, vdst, 0xE9), vsrc as u32 | (sel << 8)];
    let v_mov_row_mirror = |vdst: u8, vsrc: u8| {
        [
            vop1(0x001, vdst, 0xFA),
            vsrc as u32 | (0x140 << 8) | (1 << 19) | (0xF << 24) | (0xF << 28),
        ]
    };

    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80));
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Per-lane f64 input in v[6:7] = (s0, s1).
    words.push(vop1(0x001, 6, sgpr(0)));
    words.push(vop1(0x001, 7, sgpr(1)));
    // Whole-wave f64 sum over Wave32 with s_mov_b32 copy before EXEC restore.
    words.push(s_or_saveexec_b32(12, 0xC1));
    words.extend_from_slice(&vop3(0x101, 2, 0x80, vgpr(6), sgpr(12)));
    words.extend_from_slice(&vop3(0x101, 3, 0x80, vgpr(7), sgpr(12)));
    for sel in [14570689u32, 9658386, 2340] {
        words.extend_from_slice(&v_mov_dpp8(4, 2, sel));
        words.extend_from_slice(&v_mov_dpp8(5, 3, sel));
        words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0)); // v_add_f64
    }
    words.extend_from_slice(&v_mov_row_mirror(4, 2));
    words.extend_from_slice(&v_mov_row_mirror(5, 3));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x378, 4, vgpr(2), 0x80, 0x80)); // v_permlanex16_b32
    words.extend_from_slice(&vop3(0x378, 5, vgpr(3), 0x80, 0x80));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x360, 14, vgpr(2), 0x80, 0)); // v_readlane_b32 s14, v2, 0
    words.extend_from_slice(&vop3(0x360, 15, vgpr(3), 0x80, 0)); // v_readlane_b32 s15, v3, 0
    words.push(s_mov_b32(16, 14));
    words.push(s_mov_b32(17, 15));
    words.push(s_mov_b32(126, 12)); // restore exec_lo
    words.push(vop1(0x00f, 0, sgpr(16))); // v_cvt_f32_f64 v0, s[16:17]
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 12,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [4, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }

    let cases: &[(f64, f32)] = &[(1.25, 5.0), (-0.75, -3.0)];
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            // WARP does not support fixed WaveSize(32) on SM 6.6.
            Err(e) if warp => {
                eprintln!("{label}: compute_pipeline skipped ({e})");
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        for &(x, want) in cases {
            let b = x.to_bits();
            let inputs = [b as u32, (b >> 32) as u32, 0, 0];
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(
                out[0][0],
                want.to_bits(),
                "{label}: wave f64 sum({x}) = {:#010x}, want {:#010x}",
                out[0][0],
                want.to_bits(),
            );
        }
    }
}

/// Whether the hardware adapter is an AMD RDNA card's: what the translator
/// writes for one ([`super::rdna`]) holds there.
fn rdna_card() -> bool {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::*;
    let Ok(host) = Host::new(false) else { return false };
    let mut waves = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: plain queries.
    unsafe {
        let amd = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid()))
            .and_then(|a| a.GetDesc1())
            .is_ok_and(|d| d.VendorId == 0x1002);
        amd && host
            .device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut waves).cast(),
                size_of_val(&waves) as u32,
            )
            .is_ok()
            && waves.WaveLaneCountMin == 32
            && waves.WaveLaneCountMax == 64
    }
}

/// The forms written for the console's own kind of hardware give, on such a
/// card, what the ones spelled out for every host give; a legacy product's
/// zero may have the other sign.
#[test]
fn rdna_forms_give_what_the_spelled_out_ones_give() {
    if !rdna_card() {
        eprintln!("no RDNA card; skipped");
        return;
    }
    const ONE: u16 = 242;
    const MINUS_ONE: u16 = 243;
    let (a, b, c) = (vgpr(1), vgpr(2), vgpr(3));
    let saturated = vop3_mod(0x110, 5, vgpr(1), vgpr(1), 0, 0, 0, 0, true);
    let two = |x: [u32; 2], y: [u32; 2]| [x, y].concat();
    let bodies: [(&str, Vec<u32>, bool); 13] = [
        ("max", vop3(0x110, 0, a, b, 0).to_vec(), true),
        ("min", vop3(0x10F, 0, a, b, 0).to_vec(), true),
        ("max3", vop3(0x154, 0, a, b, c).to_vec(), true),
        ("min3", vop3(0x151, 0, a, b, c).to_vec(), true),
        ("med3 0 1", vop3(0x157, 0, a, 128, ONE).to_vec(), true),
        ("med3 1 0", vop3(0x157, 0, ONE, a, 128).to_vec(), true),
        ("med3 1 -1", vop3(0x157, 0, ONE, MINUS_ONE, a).to_vec(), true),
        ("fract", vec![vop1(0x020, 0, a)], true),
        ("mul_legacy", vop3(0x107, 0, a, b, 0).to_vec(), false),
        ("mul_legacy by itself", vop3(0x107, 0, a, a, 0).to_vec(), false),
        ("mul_legacy by a constant", vop3(0x107, 0, a, MINUS_ONE, 0).to_vec(), false),
        ("mul_legacy of a saturated", two(saturated, vop3(0x107, 0, vgpr(5), b, 0)), false),
        ("max of a mul_legacy", two(vop3(0x107, 5, a, b, 0), vop3(0x110, 0, vgpr(5), c, 0)), false),
    ];
    let numbers = [0.0f32, -0.0, 1.0, -1.0, 0.3, -2.5e-7, 3e38, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    let mut cases = Vec::new();
    for (i, x) in numbers.iter().enumerate() {
        for (j, y) in numbers.iter().enumerate() {
            let z = numbers[(3 * i + j) % numbers.len()];
            cases.push([x.to_bits(), y.to_bits(), z.to_bits(), 0]);
        }
    }
    for (name, body, exact) in bodies {
        let mut got: Vec<Vec<u32>> = Vec::new();
        let mut written = Vec::new();
        for bits in [0, super::rdna::ALL] {
            lower::TEST_RDNA.set(Some(bits));
            let shader = build_valu_cs(&body);
            lower::TEST_RDNA.set(None);
            written.push(shader.dxil.clone());
            let mut all = Vec::new();
            run_cases_on(&shader, &cases, &[("hardware", false)], |_, _, _, r| all.push(r));
            got.push(all);
        }
        assert_ne!(written[0], written[1], "{name}: written alike");
        for (case, (&was, &is)) in cases.iter().zip(got[0].iter().zip(&got[1])) {
            let (x, y) = (f32::from_bits(was), f32::from_bits(is));
            let same = was == is || x.is_nan() && y.is_nan() || !exact && x == 0.0 && y == 0.0;
            assert!(same, "{name} of {:?}: {x} ({was:#x}), now {y} ({is:#x})", case.map(f32::from_bits));
        }
    }
}

/// MUBUF `op vdata, vaddr, s[srsrc:srsrc+3], 0 idxen`.
const fn mubuf_idxen(op: u8, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | (op as u32) << 18 | 1 << 13,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], byte_offset` (soffset = null).
const fn s_load_dwordx4_off(dst: u8, base: u8, byte_offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        (125 << 25) | (byte_offset & 0x1F_FFFF),
    ]
}

/// Builds a compute shader whose descriptors live in a `RootType::Table` at
/// `s4` (like `compute_graph_ops` and `ped_standard_head_cs`).
fn build_table_typed_cs(body: &[u32], used: Vec<UsedResource>) -> HostShader {
    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80)); // s5 = 0
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [4, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// `compute_graph_ops` (facial lip-sync & wrapped submesh solver) binds
/// `R16G16B16A16_UINT` / `R16_UINT` UAVs via `RootType::Table` (which on NVIDIA
/// are snapshotted into dynamic descriptor slots) and reads/writes FP16 bit
/// patterns with `buffer_load_format_d16_xyzw` and `buffer_store_format_d16_xyzw`.
/// Verify bit-exact round-trips for UINT, SINT, and FLOAT views (including
/// non-zero table offsets matching dynamic descriptor snapshots) on hardware
/// GPU and WARP.
#[test]
fn typed_d16_table_uav_roundtrip_uint_sint_and_float() {
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_SINT,
        DXGI_FORMAT_R16G16B16A16_UINT,
    };
    let _serial = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut body = Vec::new();
    // s[8:11] = table[0] (src typed UAV), s[12:15] = table[1] (dst typed UAV),
    // s[16:19] = table[2] (raw UAV dump of loaded v[2:3]).
    body.extend_from_slice(&s_load_dwordx4_off(8, 4, 0));
    body.extend_from_slice(&s_load_dwordx4_off(12, 4, 32));
    body.extend_from_slice(&s_load_dwordx4_off(16, 4, 64));
    // buffer_load_format_d16_xyzw v[2:3], v0, s[8:11], 0 idxen (op 0x83)
    body.extend_from_slice(&mubuf_idxen(0x83, 2, 0, 8));
    // buffer_store_format_d16_xyzw v[2:3], v0, s[12:15], 0 idxen (op 0x87)
    body.extend_from_slice(&mubuf_idxen(0x87, 2, 0, 12));
    // Also dump thread 0's loaded v[2:3] into raw UAV (table[2]) to verify the load itself.
    // v4 = v0 << 3 (byte offset = thread * 8)
    body.push(0x3408_0083); // v_lshlrev_b32 v4, 3, v0
    // buffer_store_dwordx2 v[2:3], v4, s[16:19], 0 offen (op 0x1D, offen bit 12)
    body.extend_from_slice(&[
        0x38 << 26 | 0x1D << 18 | 1 << 12,
        0x80 << 24 | (16u32 >> 2) << 16 | (2u32 << 8) | 4u32,
    ]);
    let used = (0..3u32)
        .map(|k| UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: k * 8,
            lower_bound: k,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 0,
        })
        .collect();
    let shader = build_table_typed_cs(&body, used);
    assert!(
        shader.root_srvs.contains(&super::SLOT_TABLE_STRIDES),
        "table typed d16 shader should request SLOT_TABLE_STRIDES"
    );
    assert!(
        !shader.root_srvs.contains(&super::SLOT_STRIDES),
        "table-only shader must not trigger the bindless SLOT_STRIDES sweep"
    );

    // 4 elements x 4 u16 channels = 8 u32 dwords.
    // Includes FP16 subnormal (0x0001), -0.0 (0x8000), 1.0 (0x3C00), -1.0 (0xBC00),
    // NaN (0x7E00), max u16 (0xFFFF), and arbitrary patterns.
    let uint_bits: [u32; 8] = [
        0x3C00_0001, 0x7E00_BC00,
        0xFFFF_8000, 0xABCD_1234,
        0x0002_0000, 0x7BFF_FBFF,
        0xDEAD_BEEF, 0xCAFE_BABE,
    ];
    let sint_bits: [u32; 8] = [
        0xFFFF_8000, 0x7FFF_0001,
        0xC000_0000, 0x8001_7FFE,
        0xFFFF_FFFF, 0x0000_0000,
        0xFEDC_BA98, 0x7654_3210,
    ];
    // Normal finite FP16 values that round-trip cleanly through f32.
    let float_bits: [u32; 8] = [
        0x4000_3C00, 0x3800_BC00, // 1.0, 2.0, -1.0, 0.5
        0xC200_3400, 0x4500_0000, // 0.25, -3.0, 0.0, 5.0
        0x3E00_BE00, 0x4400_C400, // -1.5, 1.5, -4.0, 4.0
        0x3C00_0000, 0x4200_C000, // 0.0, 1.0, -2.0, 3.0
    ];
    let zeros = [0u32; 8];

    let cases = [
        ("R16G16B16A16_UINT @0", DXGI_FORMAT_R16G16B16A16_UINT, 8 | super::VIEW_INTEGER, &uint_bits, 0u32),
        ("R16G16B16A16_UINT @3 (dynamic offset)", DXGI_FORMAT_R16G16B16A16_UINT, 8 | super::VIEW_INTEGER, &uint_bits, 3u32),
        ("R16G16B16A16_SINT @0", DXGI_FORMAT_R16G16B16A16_SINT, 8 | super::VIEW_INTEGER | super::VIEW_SIGNED, &sint_bits, 0u32),
        ("R16G16B16A16_FLOAT @0", DXGI_FORMAT_R16G16B16A16_FLOAT, 8, &float_bits, 0u32),
    ];

    for (backend, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{backend}: compute_pipeline failed: {e}"));
        for &(label, format, stride_word, input, base_slot) in &cases {
            let mut views = Vec::new();
            let mut strides = Vec::new();
            for _ in 0..base_slot {
                views.push(View::RawUav(1));
                strides.push(0u32);
            }
            let uav_base = base_slot as usize;
            views.push(View::TypedUav { data: input, format, elem_bytes: 8 });
            strides.push(stride_word);
            views.push(View::TypedUav { data: &zeros, format, elem_bytes: 8 });
            strides.push(stride_word);
            views.push(View::RawUav(8));
            strides.push(1);

            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                if u.root == RootType::Table && u.slot == 0 {
                    ud[u.index as usize] = base_slot * 32;
                }
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (1, Root::View { srv: true, data: strides }),
                        ],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{backend} {label}: dispatch failed: {e}"));
            assert_eq!(
                out[uav_base + 2],
                input,
                "{backend} {label}: buffer_load_format_d16_xyzw mismatch"
            );
            assert_eq!(
                out[uav_base + 1],
                input,
                "{backend} {label}: buffer_store_format_d16_xyzw mismatch"
            );
        }
    }
}

#[test]
fn typed_store_guards_float_nan_and_infinity_but_keeps_integer_bits() {
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_SINT,
        DXGI_FORMAT_R32G32B32A32_UINT,
    };
    let _serial = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let input = [0x7fc0_0000, 0x7f80_0000, 0xff80_0000, 1.0f32.to_bits()];
    let mut body = s_load_dwordx4_off(8, 4, 0).to_vec();
    for (channel, bits) in input.into_iter().enumerate() {
        body.extend_from_slice(&[vop1(1, 2 + channel as u8, 0xff), bits]);
    }
    // buffer_store_format_xyzw v[2:5], v0, s[8:11], 0 idxen.
    body.extend_from_slice(&mubuf_idxen(0x07, 2, 0, 8));
    let shader = build_table_typed_cs(&body, vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 16,
        dwords: 4,
        user_data: 4,
        slot: 0,
    }]);
    assert!(shader.root_srvs.contains(&super::SLOT_TABLE_STRIDES));
    assert!(!shader.root_srvs.contains(&super::SLOT_STRIDES));

    let float_output = [0, 65504.0f32.to_bits(), (-65504.0f32).to_bits(), input[3]];
    let cases = [
        (DXGI_FORMAT_R32G32B32A32_UINT, 16 | super::VIEW_INTEGER, input),
        (DXGI_FORMAT_R32G32B32A32_SINT, 16 | super::VIEW_INTEGER | super::VIEW_SIGNED, input),
        (DXGI_FORMAT_R32G32B32A32_FLOAT, 16, float_output),
    ];
    for (backend, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.compute_pipeline(&shader).expect("compute pipeline");
        for (format, kind, expected) in cases {
            let output = host.dispatch(&root, &pso, &Dispatch {
                roots: vec![
                    (0, Root::Constants(vec![0; shader.user_data_dwords().max(1) as usize])),
                    (1, Root::View { srv: true, data: vec![kind] }),
                ],
                views: &[View::TypedUav { data: &[0; 16], format, elem_bytes: 16 }],
                samplers: &[],
                groups: [1, 1, 1],
            }).unwrap_or_else(|error| panic!("{backend} {format:?}: {error}"));
            assert_eq!(output[0], expected.repeat(4), "{backend} {format:?}");
        }
    }
}

#[test]
fn typed_store_preserves_bits_when_the_view_kind_is_unavailable() {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_UINT;
    let _serial = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let input = [0x7fc0_0000, 0x7f80_0000, 0xff80_0000, 0x4123_4567];
    let mut body = s_load_dwordx4_off(8, 4, 0).to_vec();
    body.extend_from_slice(&s_load_dwordx4_off(12, 4, 32));
    body.extend_from_slice(&[
        0xBF06_8004, // s_cmp_eq_u32 s4, 0
        0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
        0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
    ]);
    for (channel, bits) in input.into_iter().enumerate() {
        body.extend_from_slice(&[vop1(1, 2 + channel as u8, 0xff), bits]);
    }
    body.extend_from_slice(&mubuf_idxen(0x07, 2, 0, 16));
    let used = (0..2)
        .map(|slot| UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: slot * 8,
            lower_bound: slot,
            stride: 16,
            dwords: 4,
            user_data: 4,
            slot: 0,
        })
        .collect();
    let shader = build_table_typed_cs(&body, used);
    assert!(!shader.root_srvs.contains(&super::SLOT_TABLE_STRIDES));

    for (backend, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.compute_pipeline(&shader).expect("compute pipeline");
        let views = [(); 2].map(|()| View::TypedUav {
            data: &[0; 16],
            format: DXGI_FORMAT_R32G32B32A32_UINT,
            elem_bytes: 16,
        });
        let output = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(vec![0; shader.user_data_dwords().max(1) as usize]))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|error| panic!("{backend}: {error}"));
        assert_eq!(output[0], input.repeat(4), "{backend}");
        assert_eq!(output[1], [0; 16], "{backend}: unselected view");
    }
}

/// Non-`d16` typed loads (`buffer_load_format_x`) are used by
/// `CS_BlendshapeTranslationsHSSGraphOutputExtract` (reading `HSSGraphOut`
/// `R16_UINT`) and `CS_PedStandardSkinning_*` (reading bone indices). Verify
/// exact integer bits for `R16_UINT`, `R16_SINT`, and `R32_UINT`.
#[test]
fn typed_non_d16_table_load_preserves_integer_bits() {
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_UINT,
    };
    let _serial = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut body = Vec::new();
    // s[8:11] = table[0] (src typed SRV), s[12:15] = table[1] (raw UAV out)
    body.extend_from_slice(&s_load_dwordx4_off(8, 4, 0));
    body.extend_from_slice(&s_load_dwordx4_off(12, 4, 32));
    // buffer_load_format_x v2, v0, s[8:11], 0 idxen (op 0x00)
    body.extend_from_slice(&mubuf_idxen(0x00, 2, 0, 8));
    // v4 = v0 << 2 (byte offset = thread * 4)
    body.push(0x3408_0082); // v_lshlrev_b32 v4, 2, v0
    // buffer_store_dword v2, v4, s[12:15], 0 offen (op 0x1C, offen bit 12)
    body.extend_from_slice(&[
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | (12u32 >> 2) << 16 | (2u32 << 8) | 4u32,
    ]);
    let used = vec![
        UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 0,
        },
        UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 8,
            lower_bound: 1,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 0,
        },
    ];
    let shader = build_table_typed_cs(&body, used);
    // Non-d16 typed loads stay untouched (no SLOT_TABLE_STRIDES root SRV, no
    // CFG branches) so non-d16 pixel/mesh/particle/skinning shaders keep
    // identical DXIL and pipeline cache keys while still preserving raw bits.
    assert!(
        !shader.root_srvs.contains(&super::SLOT_TABLE_STRIDES),
        "non-d16 typed load should not add SLOT_TABLE_STRIDES"
    );

    let r16_data = [0x3C00_0001u32, 0xFFFF_BC00u32];
    let r32_data = [0xDEAD_BEEFu32, 0x0000_0001, 0x8000_0000, 0x7FFF_FFFF];
    let cases: [(&str, _, u32, &[u32], [u32; 4]); 3] = [
        (
            "R16_UINT",
            DXGI_FORMAT_R16_UINT,
            2,
            &r16_data,
            [0x0001, 0x3C00, 0xBC00, 0xFFFF],
        ),
        (
            "R16_SINT",
            DXGI_FORMAT_R16_SINT,
            2,
            &r16_data,
            [0x0001, 0x3C00, 0xFFFF_BC00, 0xFFFF_FFFF],
        ),
        (
            "R32_UINT",
            DXGI_FORMAT_R32_UINT,
            4,
            &r32_data,
            r32_data,
        ),
    ];

    for (backend, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{backend}: compute_pipeline failed: {e}"));
        for &(label, format, elem_bytes, input, want) in &cases {
            let views = [
                View::Buffer {
                    data: input,
                    stride: super::host::Stride::Format(format, elem_bytes),
                },
                View::RawUav(4),
            ];
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                if u.root == RootType::Table && u.slot == 0 {
                    ud[u.index as usize] = 0;
                }
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{backend} {label}: dispatch failed: {e}"));
            assert_eq!(out[0], want, "{backend} {label}: buffer_load_format_x mismatch");
        }
    }
}

#[test]
fn v_mul_legacy_f32_treats_denormals_as_zero() {
    // v_mul_legacy_f32 v0, v1, v2 (VOP2 opcode 0x007 in its VOP3 encoding)
    let shader = build_valu_cs(&vop3(0x107, 0, vgpr(1), vgpr(2), vgpr(0)));
    let f = f32::to_bits;
    let (denorm, neg_denorm, min_normal) = (0x0000_0001u32, 0x8040_0000u32, 0x0080_0000u32);

    let cases: &[(u32, u32, u32)] = &[
        // Zero or denormal times anything, non-finite included, is +0.
        (f(0.0), f(f32::INFINITY), f(0.0)),
        (f(-0.0), f(f32::NAN), f(0.0)),
        (f(f32::NAN), f(0.0), f(0.0)),
        (denorm, f(f32::INFINITY), f(0.0)),
        (f(f32::NAN), neg_denorm, f(0.0)),
        (denorm, f(2.0f32.powi(100)), f(0.0)),
        // The smallest normal is not zero.
        (min_normal, f(2.0), f(f32::MIN_POSITIVE * 2.0)),
        // Ordinary products.
        (f(1.5), f(-4.0), f(-6.0)),
        (f(3.0), f(0.5), f(1.5)),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(a, b, _)| [a, b, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].2;
        assert_eq!(
            got, want,
            "{backend}: v_mul_legacy_f32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

/// `v_mad_legacy_f32` and `v_mac_legacy_f32` as RDNA 2 runs them (fork PR #1026): fused,
/// as `v_fma_legacy_f32` and `v_fmac_legacy_f32`, with subnormal sources flushed and the
/// legacy rule that a zero factor makes the product +0.
#[test]
fn legacy_multiply_adds_fuse_their_product_and_flush_their_sources() {
    let mad = build_valu_cs(&vop3(0x140, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_mac_legacy_f32 v3, v1, v2; v_mov_b32 v0, v3
    let mac = build_valu_cs(&[vop2(0x06, 3, vgpr(1), 2), vop1(0x001, 0, vgpr(3))]);
    let cases = [
        // (1 + 2^-23) * (1 - 2^-23) - 1 is -2^-46 exactly: kept by one rounding,
        // lost when the product is rounded before the add.
        [0x3f80_0001, 0x3f7f_fffe, 0xbf80_0000, 0],
        // A subnormal source flushes to zero: subnormal * 1.0 + 0.0 == 0.0.
        [0x0000_0001, 0x3f80_0000, 0x0000_0000, 0],
        // A flushed subnormal factor is a zero factor, even against +inf.
        [0x0000_0001, 0x7f80_0000, 0x3f80_0000, 0],
        // A zero factor gives a +0 product even with an infinite factor.
        [0, 0x7f80_0000, 0x3f80_0000, 0],
        // A zero factor's product is +0 whatever the other factor's sign: +0 + -0 is +0.
        [0, 0xbf80_0000, 0x8000_0000, 0],
    ];
    let want = [0xa880_0000u32, 0x0000_0000, 0x3f80_0000, 0x3f80_0000, 0x0000_0000];
    for (name, shader) in [("v_mad_legacy_f32", &mad), ("v_mac_legacy_f32", &mac)] {
        run_cases(shader, &cases, |backend, i, _, got| {
            if backend == "warp" && i == 0 {
                // WARP rounds the product before the add; the hardware's single
                // rounding is the reference for the fused form.
                eprintln!("{backend}: fused rounding differs on WARP; skipped");
                return;
            }
            assert_eq!(got, want[i], "{backend}: {name} case {i}");
        });
    }
}

/// `v_fmac_f32` in the VOP3 encoding takes `neg[2]` (`-acc`) and `abs[2]` (`|acc|`) on its
/// accumulator, the destination register (upstream #3158). The rows are measured on RDNA2
/// (gfx1031, IEEE mode off, f32 denormals flushed) as published by the AnyPS5 project
/// (`VFmacF32AccumulatorModifiers.cpp`, 85129d854d); a NaN is compared as any NaN, as hosts
/// give NaNs bits of their own.
#[test]
fn v_fmac_f32_takes_the_accumulator_modifiers() {
    // ([a, b, acc], with neg[2], with abs[2])
    const ROWS: [([u32; 3], u32, u32); 16] = [
        ([0x3F80_0000, 0x4000_0000, 0x3F80_0000], 0x3F80_0000, 0x4040_0000),
        ([0x3F80_0000, 0x4000_0000, 0xBF80_0000], 0x4040_0000, 0x4040_0000),
        ([0x4000_0000, 0x4040_0000, 0xC0C0_0000], 0x4140_0000, 0x4140_0000),
        ([0x4000_0000, 0xC040_0000, 0xC0C0_0000], 0x0000_0000, 0x0000_0000),
        ([0x0000_0000, 0xBF80_0000, 0x0000_0000], 0x8000_0000, 0x0000_0000),
        ([0x0000_0000, 0xBF80_0000, 0x8000_0000], 0x0000_0000, 0x0000_0000),
        ([0x3F80_0000, 0x3F80_0000, 0x7F80_0000], 0xFF80_0000, 0x7F80_0000),
        ([0x3F80_0000, 0x3F80_0000, 0xFF80_0000], 0x7F80_0000, 0x7F80_0000),
        ([0x7F80_0000, 0x3F80_0000, 0xFF80_0000], 0x7F80_0000, 0x7F80_0000),
        ([0x3F80_0000, 0x3F80_0000, 0x7FC0_0001], 0xFFC0_0001, 0x7FC0_0001),
        ([0x3F80_0000, 0x3F80_0000, 0xFFA0_0000], 0x7FA0_0000, 0x7FA0_0000),
        ([0x3F80_0000, 0x3F80_0000, 0x0000_0001], 0x3F80_0000, 0x3F80_0000),
        ([0x0080_0000, 0x3F80_0000, 0x8080_0001], 0x0100_0000, 0x0100_0000),
        ([0x4040_0000, 0x3EAA_AAAB, 0xBF80_0000], 0x4000_0000, 0x4000_0000),
        ([0x7F7F_FFFF, 0x4000_0000, 0xFF7F_FFFF], 0x7F80_0000, 0x7F80_0000),
        ([0xBF80_0000, 0x4000_0000, 0x4000_0000], 0xC080_0000, 0x0000_0000),
    ];
    let nan = |x: u32| x & 0x7FFF_FFFF > 0x7F80_0000;
    let cases: Vec<[u32; 4]> = ROWS.iter().map(|&([a, b, acc], _, _)| [a, b, acc, 0]).collect();
    for (label, abs, neg) in [("neg[2]", 0, 4), ("abs[2]", 4, 0)] {
        // v_mov_b32 v5, v3; v_fmac_f32 v5, v1, v2 (VOP3, the modifier on src2); v_mov_b32 v0, v5
        let mut body = vec![vop1(0x001, 5, vgpr(3))];
        body.extend(vop3_mod(0x12B, 5, vgpr(1), vgpr(2), 0, abs, neg, 0, false));
        body.push(vop1(0x001, 0, vgpr(5)));
        let shader = build_valu_cs(&body);
        run_cases(&shader, &cases, |backend, i, _, got| {
            let want = if neg != 0 { ROWS[i].1 } else { ROWS[i].2 };
            assert!(
                got == want || nan(got) && nan(want),
                "{backend}: {label} row {i}: {got:#010x}, want {want:#010x}"
            );
        });
    }
}
