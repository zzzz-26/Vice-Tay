use super::{Lowered, Numeric, Parts, buffer_format, half_shift, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

#[test]
fn early_depth_requires_the_switch_and_a_compatible_pixel_program() {
    let translate = |db_shader_control, enabled| {
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 1,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control,
                patches: vec![],
            },
            code: [0x7E00_02F2, 0xF800_180F, 0, S_ENDPGM]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        };
        let p = ir::build(&native.code, false, 1).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let parts = Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![super::SigElement {
                name: "SV_Target".into(),
                index: 0,
                kind: 16,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 15,
                half: false,
            }],
            prims: vec![],
            mesh_stride: 0,
        };
        super::Lower::new(&parts, &std::collections::HashMap::new(), enabled)
            .run(false)
            .ok()
            .expect("lower pixel shader")
    };

    const EARLY: u32 = 1 << 12;
    let enabled = translate(EARLY, true);
    assert!(enabled.early_depth);
    assert!(!translate(EARLY, false).early_depth);
    assert!(!translate(0, true).early_depth);
    for export in [1, 2, 4] {
        assert!(!translate(EARLY | export, true).early_depth);
    }
    #[cfg(windows)]
    if let Ok(mut validator) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        validator.validate(&mut enabled.dxil.clone()).unwrap();
    }
}

#[test]
fn shifted_thread_ids_are_uniform_only_when_every_lane_agrees() {
    use crate::ir::Reg;
    for shift in 0..=64u32 {
        // v_lshrrev_b32 v7, shift, v0; s_endpgm
        let words = [0x2C0E_0080 + shift, S_ENDPGM];
        let code: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let p = ir::build(&code, true, 8).unwrap();
        let st = structure::structure(&p).unwrap();
        let inst = &p.blocks[0].insts[0];
        let tid = inst.read(Reg::V(0)).unwrap();
        let shifted = inst.writes.iter().find(|w| w.0 == Reg::V(7)).unwrap().1;
        let uniform = super::lane_uniform(&p, &st, &[true], Some(tid), true);
        let expected = (0..64u32).all(|lane| lane.wrapping_shr(shift) == 0);
        assert_eq!(uniform.contains(&shifted), expected, "inline shift {shift}");
    }
}

#[test]
fn a_swap_also_depends_on_its_old_destination() {
    use crate::ir::Reg;
    let words = [
        0x7E02_0280, // v_mov_b32 v1, 0
        0x7E00_CB01, // v_swap_b32 v0, v1
        S_ENDPGM,
    ];
    let code: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    let p = ir::build(&code, true, 8).unwrap();
    let st = structure::structure(&p).unwrap();
    let swap = &p.blocks[0].insts[1];
    assert_eq!(swap.name(), "v_swap_b32");
    let old_v0 = swap.read(Reg::V(0)).unwrap();
    let new_v1 = swap.write(Reg::V(1)).unwrap();
    let uniform = super::lane_uniform(&p, &st, &[true], Some(old_v0), true);
    assert!(!uniform.contains(&new_v1));
}

#[test]
fn packed_pixel_inputs_find_their_second_row() {
    use super::super::container::SigElement;
    use super::half_input_partners;
    use crate::native::{PsInput, Usage};
    let entry = |row, row2, pack2| PsInput {
        usage: Usage::Generic,
        row,
        mask: 0xF,
        attr: row,
        flat: false,
        fp16: true,
        pack2,
        row2,
    };
    let element = |row| SigElement {
        name: "TEXCOORD".into(),
        index: row,
        kind: 0,
        comp: 9,
        interp: 2,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0,
        half: true,
    };
    // Rows 0 and 3 pack and leave their second row out; 5 names its
    // own (6). Unnamed 16-bit rows: 1, 4, 6, 7.
    let io = StageIo::Ps {
        inputs: vec![entry(0, 0, true), entry(3, 3, true), entry(5, 6, true)],
        input_addr: 0,
        db_shader_control: 0,
        patches: vec![],
    };
    let sig: Vec<_> = [0, 1, 3, 4, 5, 6, 7].map(element).into();
    let p = half_input_partners(&io, &sig);
    assert_eq!(p.get(&0), Some(&1));
    assert_eq!(p.get(&3), Some(&4));
    assert_eq!(p.get(&5), None);
}

/// The packed pedestrian layouts of the title: an element
/// behind another one in its register follows in the first parameter
/// with a dword left, and is not a half of its register's row.
#[test]
fn packed_elements_behind_another_take_the_first_free_dword() {
    use super::super::container::SigElement;
    use super::packed_half;
    let element = |index, row, col, cols| SigElement {
        name: "TEXCOORD".into(),
        index,
        kind: 0,
        comp: 9,
        interp: 2,
        row: Some(row),
        col,
        cols,
        usage: 0,
        half: true,
    };
    // MS_VS_PedTransformPreSkinD_Packed 005fe88ec1bdc0c7 and its pixel
    // shader 93667197d3d05e1c: TEXCOORD6 is one value at register 4
    // column 3; parameter 3 (rows 6 and 7) carries it in dword 3.
    let sig = vec![
        element(1, 2, 0, 4),
        element(2, 3, 0, 4),
        element(4, 4, 0, 3),
        element(6, 4, 3, 1),
        element(3, 5, 0, 4),
        element(5, 6, 0, 4),
        element(7, 7, 0, 2),
        element(9, 9, 0, 3),
        element(11, 10, 0, 3),
        element(15, 12, 0, 4),
    ];
    let packs = [
        (2, Some(3)),
        (4, Some(5)),
        (6, Some(7)),
        (9, Some(10)),
        (12, None),
    ];
    let at = |a, h| packed_half(&sig, &packs, a, h);
    // Parameter 2: TEXCOORD4's three values, a zero half, TEXCOORD3.
    assert_eq!(at(4, 2), Some((4, 2)));
    assert_eq!(at(4, 3), None);
    assert_eq!(at(4, 4), Some((5, 0)));
    assert_eq!(at(4, 7), Some((5, 3)));
    // Parameter 3: TEXCOORD5, TEXCOORD7, then TEXCOORD6 alone.
    assert_eq!(at(6, 3), Some((6, 3)));
    assert_eq!(at(6, 5), Some((7, 1)));
    assert_eq!(at(6, 6), Some((4, 3)));
    assert_eq!(at(6, 7), None);
    // Parameters without such an element are as before.
    assert_eq!(at(9, 3), None);
    assert_eq!(at(9, 4), Some((10, 0)));
    assert_eq!(at(12, 3), Some((12, 3)));
    assert_eq!(at(12, 4), None);
    // The common layout (29805d23a495cdf8): TEXCOORD7 is two values at
    // register 1 columns 2-3; parameter 0 is TEXCOORD0, TEXCOORD1,
    // TEXCOORD7.
    let sig = vec![
        element(0, 1, 0, 2),
        element(7, 1, 2, 2),
        element(1, 2, 0, 4),
        element(2, 3, 0, 4),
        element(4, 4, 0, 3),
    ];
    let packs = [(3, Some(4)), (1, Some(2))];
    let at = |a, h| packed_half(&sig, &packs, a, h);
    assert_eq!(at(1, 1), Some((1, 1)));
    assert_eq!(at(1, 2), Some((2, 0)));
    assert_eq!(at(1, 5), Some((2, 3)));
    assert_eq!(at(1, 6), Some((1, 2)));
    assert_eq!(at(1, 7), Some((1, 3)));
    assert_eq!(at(3, 6), Some((4, 2)));
    assert_eq!(at(3, 7), None);
    // A row no element holds gives nothing, as before.
    assert_eq!(packed_half(&sig, &[(8, None)], 8, 0), None);
}

/// Whether a synthetic compute program translates with these input registers.
fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .map(|_| ())
}

/// Lowers a compute program with the root parameters `layout` (in user
/// SGPRs from s0 on) and the used resources `used`.
fn lower_with(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<Lowered, crate::ShaderError> {
    lower_wave(words, layout, used, false)
}

/// [`lower_with`] in wave64 if `wave64`.
fn lower_wave(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    wave64: bool,
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 256,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            // The layout's user SGPRs (at least two), 512 bytes of LDS.
            rsrc2: 1 << 15 | sgprs.max(2) << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, wave64, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
}

/// Root parameter `slot` in user SGPRs from `user_data` on.
fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: if root == RootType::Descriptor { 2 } else { 1 },
        base_register: 0,
        user_data,
        slot,
    }
}

/// Lowers a one-thread compute program that copies four root constants
/// (s0..s3) into v1..v4, runs `body`, and stores v0 to a raw buffer from a
/// descriptor table, so the vector instructions in `body` have an effect.
fn lowers_valu(body: &[u32]) -> Result<(), crate::ShaderError> {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xFA00_0000,
    ];
    // v_mov_b32 v1..v4, s0..s3
    words.extend((0..4u32).map(|i| 0x7E00_0200 | (i + 1) << 17 | i));
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0xE070_0000, // buffer_store_dword v0, off, s[8:11], 0
        0x8002_0000,
        S_ENDPGM,
    ]);
    let constants = RootParam {
        sgprs: 4,
        ..root(RootType::Constants, 0, 0)
    };
    let buffer = UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    lower_with(
        &words,
        vec![constants, root(RootType::Table, 4, 1)],
        vec![buffer],
    )
    .map(|_| ())
}

#[test]
fn the_24_bit_multiplies_lower_both_halves() {
    // VOP2 `op v0, v1, v2`: v_mul_i32_i24 and v_mul_u32_u24 (the low
    // dword) and v_mul_hi_i32_i24 and v_mul_hi_u32_u24 (the high one).
    for op in [0x09u32, 0x0A, 0x0B, 0x0C] {
        let word = op << 25 | 2 << 9 | 0x101;
        lowers_valu(&[word]).unwrap_or_else(|e| panic!("VOP2 {op:#x}: {e}"));
    }
}

#[test]
fn buffer_cache_invalidations_lower_as_no_ops() {
    // MUBUF opcode 0x071/0x072 with no data operand: the title asks the
    // host to invalidate its cached view of a buffer.
    assert!(lowers(&[0xE1C4_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl0_inv");
    assert!(lowers(&[0xE1C8_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl1_inv");
}

#[test]
fn sums_of_absolute_differences_lower() {
    // VOP3 `op v0, v1, v2, v3`: v_sad_u8, v_sad_hi_u8, v_sad_u16,
    // v_sad_u32 and v_msad_u8.
    for op in [0x15Au32, 0x15B, 0x15C, 0x15D, 0x171] {
        let words = [0xD400_0000 | op << 16, 0x101 | 0x102 << 9 | 0x103 << 18];
        lowers_valu(&words).unwrap_or_else(|e| panic!("VOP3 {op:#x}: {e}"));
    }
}

#[test]
fn a_scalar_carry_chain_reaches_the_select() {
    // The high dword of a 64-bit add or subtract, then a select on the
    // carry (borrow) the high half leaves in SCC, and a branch on it
    // (only code with an effect needs its operands).
    let add = [
        0xBE80_03C1, // s_mov_b32 s0, -1
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8000_8100, // s_add_u32 s0, s0, 1
        0x8201_8001, // s_addc_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    let sub = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8080_8100, // s_sub_u32 s0, s0, 1
        0x8281_8001, // s_subb_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];

    for words in [add, sub] {
        lowers(&words, 0).unwrap();
    }
}

fn perm_features(selector: u32) -> u64 {
    // Synthetic compute shader:
    // v0 is the thread ID and can supply a dynamic selector.
    // Store the result in LDS to keep the permutation live.
    let words = [
        0x7E02_02FFu32,
        0x817F_34AB,
        0x7E04_02FF,
        0xFE01_56CD,
        0xD744_0003,
        257 | (258 << 9) | (selector << 18),
        0x3408_0082,
        0xD834_0000,
        0x0000_0304,
        S_ENDPGM,
    ];

    let code: Vec<u8> = words.iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15,
            threads: [64, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, false, 256).unwrap();

    assert!(p.blocks.iter()
        .flat_map(|b| &b.insts)
        .any(|i| i.name() == "v_perm_b32"));

    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

    let io = io::raise(&p, &native).unwrap();

    let emitted = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    }).unwrap();

    let container =
        crate::Container::parse(&emitted.dxil).unwrap();

    let features =
        container.part(crate::FourCC::SFI0).unwrap();

    u64::from_le_bytes(features.data.try_into().unwrap())
}

#[test]
fn perm_constant_selector_needs_no_int64() {
    assert_eq!(perm_features(128) & 0x8000, 0);
}

#[test]
fn perm_dynamic_selector_needs_no_int64() {
    assert_eq!(
        perm_features(256) & 0x8000,
        0,
        "dynamic v_perm_b32 must not require INT64"
    );
}

#[test]
fn a_descriptor_selected_by_scc_picks_its_heap_index() {
    // Two buffer descriptors loaded from the table at s0, one picked by
    // a root constant (s1) with s_cselect_b64 per half, and loaded
    // through by index and stored to LDS so the load is needed.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xF408_0302,
        0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
        0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
        0xE030_2000,
        0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    // Both are structured buffers of dwords in the table's range.
    let used = || {
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }]
    };
    lower_with(&words, layout(), used()).unwrap();
}

/// Tables of 2D textures at s0 and samplers at s1.
fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let range = |kind, user_data, slot| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
        user_data,
        slot,
    };
    (
        vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
        vec![
            range(ResourceType::Srv, 0, 0),
            range(ResourceType::Sampler, 1, 1),
        ],
    )
}

#[test]
fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
    use crate::resource::tests::{
        WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
    };
    for (words, wave64) in [
        (WATERFALL_FETCHED_WAVE32, false),
        (WATERFALL_FETCHED_WAVE64, true),
        (WATERFALL_PICKED_WAVE32, false),
    ] {
        let (layout, used) = texture_tables();
        let l = lower_wave(words, layout, used, wave64).unwrap();
        // The loop's reads of the first lane became each lane's own
        // dwords: no wave operation is left.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

/// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let (layout, mut used) = texture_tables();
    used.push(UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 16,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    });
    (layout, used)
}

#[test]
fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
    use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
    for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
        let (layout, used) = texture_and_uav_tables();
        let l = lower_wave(words, layout, used, false).unwrap();
        // One image handle, from the join's per-lane heap index.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

#[test]
fn scalar_bit_tests_lower() {
    // Each tests one bit of a constant into SCC and branches on it (only
    // code with an effect needs its operands).
    let tail = [
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let bit_tests: [&[u32]; 4] = [
        &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
        &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
        ],
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
        ],
    ];
    for head in bit_tests {
        let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
        lowers(&words, 0).unwrap();
    }
}

#[test]
#[ignore = "ours marks joins read through joins only with VICTORY_STRICT_JOINS (40 programs stop otherwise)"]
fn a_missing_value_stays_missing_through_two_live_joins() {
    let mut words = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0001, // skip the first overwrite
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
        0xBF85_0001, // skip the second overwrite
        0xBE86_0381, // s_mov_b32 s6, 1
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // use the result to branch
        0xBE83_0380,
        S_ENDPGM,
    ];
    let group_ids_xy = 3 << 7;
    // Compare to a direct read so future timestamp support works in both cases.
    let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let direct = [&words[..2], &words[8..]].concat();
    assert_eq!(refusal(&words), refusal(&direct));
    words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
    lowers(&words, group_ids_xy).expect("dead timestamp");
}

#[test]
fn buffer_formats_by_number() {
    // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
    // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
    assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
    assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
    assert_eq!(
        buffer_format(50),
        Some((&[10, 10, 10, 2][..], Numeric::Unorm))
    );
    assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
    assert_eq!(
        buffer_format(77),
        Some((&[32, 32, 32, 32][..], Numeric::Float))
    );
    // Invalid, past the table.
    assert_eq!(buffer_format(0), None);
    assert_eq!(buffer_format(78), None);
}

#[test]
fn packed_buffer_formats_by_number() {
    // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
    // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
    // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
    assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
    assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
    assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
    assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
    assert_eq!(
        buffer_format(44),
        Some((&[2, 10, 10, 10][..], Numeric::Unorm))
    );
    assert_eq!(
        buffer_format(45),
        Some((&[2, 10, 10, 10][..], Numeric::Snorm))
    );
    assert_eq!(
        buffer_format(46),
        Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
    );
    assert_eq!(
        buffer_format(47),
        Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
    );
    assert_eq!(
        buffer_format(48),
        Some((&[2, 10, 10, 10][..], Numeric::Uint))
    );
    assert_eq!(
        buffer_format(49),
        Some((&[2, 10, 10, 10][..], Numeric::Sint))
    );
}

/// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
/// `legacyF16ToF32` widens it.
fn half(h: u32) -> f32 {
    let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
    let v = match e {
        0 => m as f32 * 2f32.powi(-24),
        31 if m == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
    };
    if s == 1 { -v } else { v }
}

#[test]
fn small_floats_widen_as_halves() {
    // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
    // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
    let f11 = |x: u32| half(x << half_shift(11));
    let f10 = |x: u32| half(x << half_shift(10));
    assert_eq!(f11(0), 0.0);
    assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
    assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
    assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
    assert_eq!(f11(0x3C0), 1.0);
    assert_eq!(f11(0x3E0), 1.5);
    assert_eq!(f11(0x7BF), 65024.0); // largest
    assert_eq!(f11(0x7C0), f32::INFINITY);
    assert!(f11(0x7C1).is_nan());
    assert_eq!(f10(0x001), 2f32.powi(-19));
    assert_eq!(f10(0x020), 2f32.powi(-14));
    assert_eq!(f10(0x1E0), 1.0);
    assert_eq!(f10(0x3DF), 64512.0);
    assert_eq!(f10(0x3E0), f32::INFINITY);
    assert!(f10(0x3FF).is_nan());
    // A 16-bit float is a half already, sign included.
    assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
}

#[cfg(windows)]
#[test]
fn pops_wait_loop_exits_immediately_for_inactive_lanes() {
    use super::super::HostShader;
    use super::super::container::SigElement;
    use super::super::host::{Draw, Host};
    use super::super::validator::Validator;
    use crate::native::{Usage, VsInput};

    let _one = super::super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(true) else {
        return;
    };
    let pos_sig = |name: &str, kind: u8, interp: u8| SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    };
    let to_host = |native: &NativeProgram, inputs: Vec<SigElement>, outputs: Vec<SigElement>| {
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource");
        let io = io::raise(&p, native).expect("io");
        let l = lower(&Parts {
            native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs,
            outputs,
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .expect("lower");
        if let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) {
            let mut copy = l.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        HostShader {
            stage: native.stage,
            dxil: l.dxil,
            user_data: l.user_data,
            samplers: l.samplers,
            static_samplers: Vec::new(),
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            root_indexed_views: Vec::new(),
        }
    };

    // Pass-through full-screen vertex shader (attributes in v4..v7; v0..v3 hold hardware ids).
    let vs_code: [u32; 3] = [
        0xF800_08CF,
        0x0706_0504, // exp pos0, v4, v5, v6, v7 done
        S_ENDPGM,
    ];
    let vs_native = NativeProgram {
        stage: HwStage::Vs,
        wave64: true,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Vs {
            inputs: vec![VsInput {
                usage: Usage::Generic,
                row: 0,
                mask: 0xF,
                vgpr: 4,
            }],
            outputs: Vec::new(),
        },
        code: vs_code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let vs = to_host(
        &vs_native,
        vec![pos_sig("POSITION", 0, 0)],
        vec![pos_sig("SV_Position", 3, 4)],
    );

    // Pixel shader with narrowed EXEC before the title's wrap-around POPS wait
    // loop: lanes with SV_Position.x >= 2.0 have EXEC cleared at s_cbranch_vccnz
    // so they fall into the wait loop, then restore EXEC and export the loop
    // iteration count in s12. If the collision id has wave_id == collision_id
    // (0 == 0), wave_id + ~collision_id is 0xffff_ffff so `s_cmp_lt_u32 s5, s0`
    // never holds and the loop runs to the pixel loop cap (1 << 7) instead of
    // exiting after one pass.
    let ps_code: [u32; 32] = [
        0xBE8C_0380, // s_mov_b32 s12, 0
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x7C28_00F4, // v_cmpx_gt_f32 exec, 2.0, v0  (active for x < 2.0)
        0x9390_FF09,
        0x0002_001C, // s_bfe_u32 s16, s9, lit(0x0002001c) (packer)
        0x9382_FF09,
        0x000A_0000, // s_bfe_u32 s2, s9, lit(0x000a0000) (collision id)
        0x9385_FF09,
        0x000A_0010, // s_bfe_u32 s5, s9, lit(0x000a0010) (wave id)
        0xBE84_0702, // s_not_b32 s4, s2
        0x9000_9F09, // s_lshr_b32 s0, s9, 31 (overlap bit)
        0x8105_0405, // s_add_i32 s5, s5, s4
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xB990_1019, // s_setreg_b32 hwreg(HW_REG_POPS_PACKER, 0, 3), s16
        0x8580_807E, // s_cselect_b64 s[0:1], exec, 0
        0x8102_04EF, // s_add_i32 s2, src_pops_exiting_wave_id, s4
        0xBF08_0502, // s_cmp_gt_u32 s2, s5
        0x8582_807E, // s_cselect_b64 s[2:3], exec, 0
        0x8880_0200, // s_or_b64 s[0:1], s[0:1], s[2:3]
        0x87EA_007E, // s_and_b64 vcc, exec, s[0:1]
        0xBF87_0005, // s_cbranch_vccnz +5
        0xBF8E_FFFF, // s_sleep 0xffff
        0x810C_810C, // s_add_i32 s12, s12, 1
        0x8100_04EF, // s_add_i32 s0, src_pops_exiting_wave_id, s4
        0xBF0A_0005, // s_cmp_lt_u32 s5, s0
        0xBF84_FFFB, // s_cbranch_scc0 -5
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        0x7E08_0C0C, // v_cvt_f32_u32 v4, s12
        0xF800_180F,
        0x0404_0404, // exp mrt0, v4, v4, v4, v4 done vm
        S_ENDPGM,
    ];
    let ps_native = NativeProgram {
        stage: HwStage::Ps,
        wave64: true,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 1 << 8, // SV_Position.x in v0
            db_shader_control: 1 << 16,
            patches: Vec::new(),
        },
        code: ps_code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let ps = to_host(
        &ps_native,
        vec![pos_sig("SV_Position", 3, 4)],
        vec![pos_sig("SV_Target", 16, 0)],
    );

    let (root, pso) = host.pipeline(&vs, &ps).expect("pipeline");
    let pixels = host
        .draw(
            &root,
            &pso,
            &Draw {
                roots: Vec::new(),
                textures: &[],
                samplers: &[],
                vertices: &[
                    -1.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0,
                ],
                stride: 16,
                size: (4, 1),
                mesh_groups: 0,
            },
        )
        .expect("draw");
    // Active pixels (x < 2.0) skip the wait loop (0 iterations); inactive
    // pixels (x >= 2.0) fall through once and exit immediately (1 iteration),
    // never spinning to the pixel loop cap.
    assert_eq!(pixels, vec![[0.0; 4], [0.0; 4], [1.0; 4], [1.0; 4]]);
}
