//! End-to-end tests of what a program tells about a view by how it uses it:
//! descriptors read for sizes (the dwords of a view no instruction uses) and
//! counters, lowered to DXIL, validated by `dxil.dll` when present, and run
//! on the host GPU and WARP.

use super::host::{Stride, View};
use super::waterfall_tests::run;
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

/// Lowers a wave32 compute program of one thread with the table of views at
/// s0 and the view heap at s1; `srv`: the dwords and the stride of the used
/// resource at the table's first entry (the third is a raw UAV).
fn build(words: &[u32], srv: (u8, u32)) -> Result<HostShader, ShaderError> {
    let param = |root, user_data| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: 0,
    };
    let range = |kind, table_offset_dwords, dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![param(RootType::Table, 0), param(RootType::ResourceHeap, 1)],
        used: vec![
            range(ResourceType::Srv, 0, srv.0, srv.1),
            range(ResourceType::Uav, 16, 4, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
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
    })?;
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
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    Ok(shader)
}

/// The table at s[2:3], its raw UAV's descriptor in s[20:23].
const TABLE: &[u32] = &[
    0xBE82_0300, // s_mov_b32 s2, s0
    0xBE83_0380, // s_mov_b32 s3, 0
    0xF408_0501, // s_load_dwordx4 s[20:23], s[2:3], 64
    0xFA00_0040,
];

/// Stores s4 and s5 at bytes 0 and 4 of the UAV.
const STORE: &[u32] = &[
    0x7E08_0204, // v_mov_b32 v4, s4
    0x7E0A_0205, // v_mov_b32 v5, s5
    0xE070_0000, // buffer_store_dword v4, off, s[20:23], 0
    0x8005_0400,
    0xE070_0004, // buffer_store_dword v5, off, s[20:23], 0 offset:4
    0x8005_0500,
    0xBF81_0000, // s_endpgm
];

/// `GetDimensions` of a texture as the console compiler expands it, from
/// the descriptor in s[8:15]: width in s4, height in s5, base level in s6.
const SIZES: &[u32] = &[
    0x9004_9E09, // s_lshr_b32 s4, s9, 30
    0x8F05_820A, // s_lshl_b32 s5, s10, 2
    0x8804_0504, // s_or_b32 s4, s4, s5
    0x8704_FF04, // s_and_b32 s4, s4, 0xffff
    0x0000_FFFF,
    0x9385_FF0A, // s_bfe_u32 s5, s10, 0x10000e
    0x0010_000E,
    0x9386_FF0B, // s_bfe_u32 s6, s11, 0x4000c
    0x0004_000C,
    0x8104_8104, // s_add_i32 s4, s4, 1
    0x8105_8105, // s_add_i32 s5, s5, 1
    0x9004_0604, // s_lshr_b32 s4, s4, s6
    0x9005_0605, // s_lshr_b32 s5, s5, s6
];

/// s_load_dwordx8 s[8:15], s[6:7], s12: entry 1 of the view heap.
const FROM_HEAP: &[u32] = &[
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xBE8C_03A0, // s_mov_b32 s12, 32
    0xF40C_0203,
    0x1800_0000,
];

/// `s_load_dwordx4 s[8:11], s[2:3], 0`, then the record count in s4 and 0
/// in s5.
const COUNT: &[u32] = &[
    0xF408_0201,
    0xFA00_0000,
    0xBE84_030A, // s_mov_b32 s4, s10
    0xBE85_0380, // s_mov_b32 s5, 0
];

/// A puddle pass of the title picks three textures from the view heap,
/// reads the first one's size and samples the other two.
#[test]
fn the_size_of_a_heap_texture_nothing_reads() {
    let words = [TABLE, FROM_HEAP, SIZES, STORE].concat();
    let shader = build(&words, (8, 0)).expect("translates");
    let (one, texels) = ([[0.0; 4]; 1], [[0.0; 4]; 15]);
    let views = [
        View::Texture(1, 1, &one),
        View::Texture(5, 3, &texels),
        View::RawUav(2),
    ];
    for (label, got) in run(&shader, &views) {
        assert_eq!(got[..2], [5, 3], "{label}");
    }
}

/// Without the height nothing tells a 2D texture from the others: taken for
/// one, as most are. A volume's depth or an array's layers are no 2D
/// texture's.
#[test]
fn sizes_of_a_volume_or_an_array_stay_untranslated() {
    let width = [
        0x9004_9E09, // s_lshr_b32 s4, s9, 30
        0x8F05_820A, // s_lshl_b32 s5, s10, 2
        0x8804_0504, // s_or_b32 s4, s4, s5
    ];
    let words = [TABLE, FROM_HEAP, &width, STORE].concat();
    assert!(build(&words, (8, 0)).is_ok());
    let depth = [
        0x8805_0C05, // s_or_b32 s5, s5, s12
    ];
    let words = [TABLE, FROM_HEAP, SIZES, &depth, STORE].concat();
    assert!(build(&words, (8, 0)).is_err());
}

/// A water simulation step runs only while a structured buffer is bound:
/// it tests the record count of a view it never reads.
#[test]
fn the_record_count_of_a_structured_buffer_nothing_reads() {
    let words = [TABLE, COUNT, STORE].concat();
    let shader = build(&words, (4, 16)).expect("translates");
    let (data, texel) = ([0u32; 12], [[0.0; 4]; 1]);
    // The runtime makes structured views raw.
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::Texture(1, 1, &texel),
        View::RawUav(2),
    ];
    for (label, got) in run(&shader, &views) {
        assert_eq!(got[..2], [3, 0], "{label}");
    }
    // A stride of 4 or less may be a typed view's component count.
    assert!(build(&words, (4, 4)).is_err());
    assert!(build(&words, (4, 0)).is_err());
}

/// As [`the_record_count_of_a_structured_buffer_nothing_reads`], of one the
/// program also reads by index.
#[test]
fn the_record_count_of_a_structured_buffer_read_by_index() {
    let load = [
        0x7E06_0280, // v_mov_b32 v3, 0
        0xE030_2000, // buffer_load_dword v5, v3, s[8:11], 0 idxen
        0x8002_0503,
        0x7E0A_0505, // v_readfirstlane_b32 s5, v5
    ];
    let words = [TABLE, COUNT, &load, STORE].concat();
    let shader = build(&words, (4, 16)).expect("translates");
    let (mut data, texel) = ([0u32; 12], [[0.0; 4]; 1]);
    data[0] = 77;
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::Texture(1, 1, &texel),
        View::RawUav(2),
    ];
    for (label, got) in run(&shader, &views) {
        assert_eq!(got[..2], [3, 77], "{label}");
    }
}

/// An upscaler counts its groups in a texture with atomics and sets the
/// counter back with a store of one channel, 16 bits wide.
#[test]
fn a_counter_texture_is_set_by_a_store_of_one_channel() {
    let program = |store: [u32; 2], value: u32| {
        let words = [
            0xF40C_0201u32, // s_load_dwordx8 s[8:15], s[2:3], 0
            0xFA00_0000,
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_0280, // v_mov_b32 v1, 0
            0x7E04_0283, // v_mov_b32 v2, 3
            0xF044_3108, // image_atomic_add v2, v[0:1], s[8:15] dmask:0x1 2D glc
            0x0002_0200,
            value, // v_mov_b32 v3, ..
            store[0],
            store[1],
            0x7E08_0281, // v_mov_b32 v4, 1
            0xF044_3108, // image_atomic_add v4, v[0:1], s[8:15] dmask:0x1 2D glc
            0x0002_0400,
            0xBE84_0380, // s_mov_b32 s4, 0
            0xBE85_0380, // s_mov_b32 s5, 0
        ];
        build(&[TABLE, &words, STORE].concat(), (8, 0))
    };
    // image_store v3, v[0:1], s[8:15] dmask:0x1 2D, with and without d16
    for store in [[0xF020_1108, 0x0002_0300], [0xF020_1108, 0x8002_0300]] {
        let shader = program(store, 0x7E06_0289).expect("translates"); // v3 = 9
        let texel = [[0.0; 4]; 1];
        let views = [
            View::UintTextureUav(1, 1),
            View::Texture(1, 1, &texel),
            View::RawUav(2),
        ];
        for (label, got) in run(&shader, &views) {
            assert_eq!(got[0], 10, "{label}");
        }
    }
    // Sixteen bits of a number only the card knows may be signed.
    assert!(program([0xF020_1108, 0x8002_0300], 0x7E06_0302).is_err()); // v3 = v2
    // A store of two channels says nothing about the view.
    assert!(program([0xF020_1308, 0x0002_0300], 0x7E06_0289).is_err());
}

/// A load past the wave's cache (GLC) of a buffer other groups write: its
/// view is globally coherent on the host, as is every access to it.
#[test]
fn a_buffer_loaded_past_the_cache_is_coherent() {
    let program = |load: u32| {
        let words = [
            0x7E04_028Bu32, // v_mov_b32 v2, 11
            0xE070_0008,    // buffer_store_dword v2, off, s[20:23], 0 offset:8
            0x8005_0200,
            load, // buffer_load_dword v3, off, s[20:23], 0 offset:8
            0x8005_0300,
            0x7E08_0503, // v_readfirstlane_b32 s4, v3
            0xBE85_0380, // s_mov_b32 s5, 0
        ];
        build(&[TABLE, &words, STORE].concat(), (8, 0)).expect("translates")
    };
    let (plain, coherent) = (program(0xE030_0008), program(0xE030_C008));
    // the annotations of the store's and the load's handles
    assert_ne!(plain.dxil, coherent.dxil);
    let texel = [[0.0; 4]; 1];
    let views = [
        View::Texture(1, 1, &texel),
        View::Texture(1, 1, &texel),
        View::RawUav(3),
    ];
    for shader in [&plain, &coherent] {
        for (label, got) in run(shader, &views) {
            assert_eq!(got[..3], [11, 0, 11], "{label}");
        }
    }
}

/// A buffer picked from the view heap and read as raw bytes is read at the
/// host's raw twin of its slot, when the host keeps some; an index past the
/// title's slots reads the last twin, not a slot of something else.
#[test]
fn a_picked_buffer_is_read_at_its_raw_twin() {
    let program = |at: u32| {
        let words = [
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xBE8C_03FF, // s_mov_b32 s12, at
            at,
            0xF408_0203, // s_load_dwordx4 s[8:11], s[6:7], s12
            0x1800_0000,
            0xE030_0004, // buffer_load_dword v5, off, s[8:11], 0 offset:4
            0x8002_0500,
            0x7E0A_0505, // v_readfirstlane_b32 s5, v5
            0xBE84_0380, // s_mov_b32 s4, 0
        ];
        [TABLE, &words, STORE].concat()
    };
    let (title, twin, last, texel) = ([1u32, 2, 3, 4], [5u32, 6, 7, 8], [9u32, 10, 11, 12], [[0.0; 4]; 1]);
    let raw = |data| View::Buffer { data, stride: Stride::Raw };
    // (The store goes through the table's third entry: its twin, too.)
    let views = |twins: bool| {
        let uav = |here: bool| if here { View::RawUav(2) } else { View::Texture(1, 1, &texel) };
        [
            View::Texture(1, 1, &texel),
            raw(&title),
            uav(!twins),
            View::Texture(1, 1, &texel),
            View::Texture(1, 1, &texel),
            raw(&twin),
            uav(twins),
            raw(&last),
        ]
    };
    for (first, at, expected) in [(0, 32, 2), (4, 32, 6), (4, 32 * 1000, 10)] {
        super::shader::RAW_VIEWS_HERE.set(Some(first));
        let shader = build(&program(at), (8, 0)).expect("translates");
        super::shader::RAW_VIEWS_HERE.set(None);
        for (label, got) in run(&shader, &views(first != 0)) {
            assert_eq!(got[..2], [0, expected], "{label}: twins from {first}, byte {at}");
        }
    }
}

/// A table's buffer read as raw bytes is read at its raw twin too.
#[test]
fn a_tables_raw_buffer_is_read_at_its_raw_twin() {
    let load = [
        0xF408_0201u32, // s_load_dwordx4 s[8:11], s[2:3], 0
        0xFA00_0000,
        0xE030_0004, // buffer_load_dword v5, off, s[8:11], 0 offset:4
        0x8002_0500,
        0x7E0A_0505, // v_readfirstlane_b32 s5, v5
        0xBE84_0380, // s_mov_b32 s4, 0
    ];
    let words = [TABLE, &load, STORE].concat();
    let (title, twin, texel) = ([1u32, 2, 3, 4], [5u32, 6, 7, 8], [[0.0; 4]; 1]);
    let raw = |data| View::Buffer { data, stride: Stride::Raw };
    super::shader::RAW_VIEWS_HERE.set(Some(4));
    let shader = build(&words, (4, 0)).expect("translates");
    super::shader::RAW_VIEWS_HERE.set(None);
    let other = || View::Texture(1, 1, &texel);
    let views = [raw(&title), other(), other(), other(), raw(&twin), other(), View::RawUav(2)];
    for (label, got) in run(&shader, &views) {
        assert_eq!(got[..2], [0, 6], "{label}");
    }
}

/// A texel a program stores to a texture it picks from the view heap and then loads back is
/// the one it stored: the load reads through the texture's UAV, as the store writes it. (A
/// read through another view of it need not see the dispatch's own writes: an RTX 3070 read
/// the old value, xcl-dev/xcl ec356bc5c8.)
#[test]
fn a_heap_texture_loaded_after_a_store_reads_the_store() {
    let words = [
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0x7E04_02FF, // v_mov_b32 v2, 7.0
        0x40E0_0000,
        0x7E06_0302, // v_mov_b32 v3, v2
        0x7E08_0302, // v_mov_b32 v4, v2
        0x7E0A_0302, // v_mov_b32 v5, v2
        0xF020_1F08, // image_store v[2:5], v[0:1], s[8:15] dmask:0xf 2D unorm
        0x0002_0200,
        0xF000_1F08, // image_load v[6:9], v[0:1], s[8:15] dmask:0xf 2D unorm
        0x0002_0600,
        0x7E08_0506, // v_readfirstlane_b32 s4, v6
        0x7E0A_0509, // v_readfirstlane_b32 s5, v9
    ];
    let shader = build(&[TABLE, FROM_HEAP, &words, STORE].concat(), (8, 0)).expect("translates");
    let texel = [[0.0; 4]; 1];
    let views = [View::Texture(1, 1, &texel), View::TextureUav(1, 1), View::RawUav(2)];
    for (label, got) in run(&shader, &views) {
        assert_eq!(got[..2], [7.0f32.to_bits(); 2], "{label}");
    }
}

/// Lowers `words` for `stage` (compute or pixel) without discarding lowering
/// counters, validating the DXIL when `dxil.dll` is present.
fn build_stage(
    words: &[u32],
    srv: (u8, u32),
    stage: HwStage,
) -> Result<(HostShader, lower::Lowered), ShaderError> {
    let param = |root, user_data| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: 0,
    };
    let range = |kind, table_offset_dwords, dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 0,
        slot: 0,
    };
    let io = match stage {
        HwStage::Ps => StageIo::Ps {
            inputs: vec![],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        },
        _ => StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
    };
    let native = NativeProgram {
        stage,
        wave64: false,
        vgprs: 32,
        layout: vec![param(RootType::Table, 0), param(RootType::ResourceHeap, 1)],
        used: vec![
            range(ResourceType::Srv, 0, srv.0, srv.1),
            range(ResourceType::Uav, 16, 4, 0),
        ],
        io,
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
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
    })?;
    let shader = HostShader {
        stage,
        dxil: lowered.dxil.clone(),
        user_data: lowered.user_data.clone(),
        samplers: lowered.samplers.clone(),
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs.clone(),
        root_srvs: lowered.root_srvs.clone(),
        root_uavs: lowered.root_uavs.clone(),
        root_indexed_views: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    Ok((shader, lowered))
}

/// Runs `f` with thread-local test overrides for shader opt, view-heap clamp
/// limit, and raw-twin base, restoring all three even on unwind.
fn with_emit_overrides<T>(
    opt: Option<u32>,
    view_limit: Option<u32>,
    raw_views: Option<u32>,
    f: impl FnOnce() -> T,
) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            super::TEST_SHADER_OPT.set(None);
            super::shader::VIEW_LIMIT_HERE.set(None);
            super::shader::RAW_VIEWS_HERE.set(None);
        }
    }
    let _reset = Reset;
    super::TEST_SHADER_OPT.set(opt);
    super::shader::VIEW_LIMIT_HERE.set(view_limit);
    super::shader::RAW_VIEWS_HERE.set(raw_views);
    f()
}

/// Disassembles `dxil` with `dxc.exe -dumpbin` when available on the machine.
fn dxc_disasm(dxil: &[u8], tag: &str) -> Option<String> {
    let dxc = match std::env::var_os("VICTORY_PC_DXC") {
        Some(p) => std::path::PathBuf::from(p),
        None => validator::Validator::default_path()?.with_file_name("dxc.exe"),
    };
    if !dxc.exists() {
        return None;
    }
    let file = std::env::temp_dir().join(format!(
        "victory-size-{tag}-{}.dxil",
        std::process::id()
    ));
    std::fs::write(&file, dxil).expect("write temp dxil");
    let out = std::process::Command::new(dxc)
        .arg("-dumpbin")
        .arg(&file)
        .output()
        .expect("run dxc -dumpbin");
    let _ = std::fs::remove_file(&file);
    assert!(
        out.status.success(),
        "dxc -dumpbin failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Extracts the SSA value right after `before` in `line`.
fn ssa_after(line: &str, before: &str) -> String {
    let at = line
        .find(before)
        .unwrap_or_else(|| panic!("no {before:?} in {line:?}"));
    line[at + before.len()..]
        .split([',', ')'])
        .next()
        .unwrap()
        .trim()
        .to_owned()
}

/// A buffer's record-count query (`descriptor_data`) uses its own non-uniform
/// heap handle and its own index clamp, distinct from the buffer's loads,
/// across both unoptimised and optimised DXIL, with and without a view-heap
/// clamp (`VICTORY_VIEW_LIMIT`) and raw twins, without adding `wave_ops` or
/// incrementing `nonuniform_handles`.
#[test]
fn a_buffer_size_query_uses_its_own_nonuniform_handle_and_clamps() {
    let load = [
        0x7E06_0280, // v_mov_b32 v3, 0
        0xE030_2000, // buffer_load_dword v5, v3, s[8:11], 0 idxen
        0x8002_0503,
        0x7E0A_0505, // v_readfirstlane_b32 s5, v5
    ];
    let words = [TABLE, COUNT, &load, STORE].concat();
    let count_only = [TABLE, COUNT, STORE].concat();
    let (mut data, mut other_data, texel) = ([0u32; 12], [0u32; 4], [[0.0; 4]; 1]);
    data[0] = 77;
    other_data[0] = 11;

    // A count-only program (no `v_readfirstlane_b32`) emits zero wave ops for
    // the buffer size handle in both compute and pixel stages (helper-lane safe).
    for (stage_label, stage) in [("compute", HwStage::Cs), ("pixel", HwStage::Ps)] {
        for (opt_label, opt) in [("opt0", 0), ("default", super::bitcode::opt::DEFAULT)] {
            let (_, l) = with_emit_overrides(Some(opt), Some(2_097_152), Some(0), || {
                build_stage(&count_only, (4, 16), stage).expect("translates")
            });
            assert!(
                !l.wave_ops,
                "{stage_label}/{opt_label}: no wave ops for buffer size handle"
            );
            assert_eq!(
                l.nonuniform_handles, 0,
                "{stage_label}/{opt_label}: uniform-index size handle is not counted as non-uniform"
            );
        }
    }

    for (label, opt, limit, raw_base) in [
        ("opt0_unclamped", 0, 0, 0),
        ("default_unclamped", super::bitcode::opt::DEFAULT, 0, 0),
        ("default_clamped", super::bitcode::opt::DEFAULT, 2_097_152, 0),
        ("all_clamped", super::bitcode::opt::ALL, 2_097_152, 0),
        ("default_raw_twins", super::bitcode::opt::DEFAULT, 2_097_152, 4),
    ] {
        let (shader, lowered) = with_emit_overrides(
            Some(opt),
            Some(limit),
            Some(raw_base),
            || build_stage(&words, (4, 16), HwStage::Cs).expect("translates"),
        );
        assert_eq!(
            lowered.nonuniform_handles, 0,
            "{label}: uniform-index size handle is not counted as non-uniform"
        );
        if let Some(text) = dxc_disasm(&shader.dxil, label) {
            let def = |id: &str| {
                text.lines()
                    .find(|l| l.trim_start().starts_with(&format!("{id} = ")))
                    .unwrap_or_else(|| panic!("{label}: missing def for {id}"))
            };
            let dims_line = text
                .lines()
                .find(|l| l.contains("@dx.op.getDimensions"))
                .unwrap_or_else(|| panic!("{label}: missing @dx.op.getDimensions"));
            let load_line = text
                .lines()
                .find(|l| l.contains("@dx.op.rawBufferLoad"))
                .unwrap_or_else(|| panic!("{label}: missing @dx.op.rawBufferLoad"));
            let size_h = ssa_after(dims_line, "%dx.types.Handle ");
            let load_h = ssa_after(load_line, "%dx.types.Handle ");
            assert_ne!(
                size_h, load_h,
                "{label}: getDimensions and rawBufferLoad must not share a handle"
            );
            let size_made = def(&ssa_after(
                def(&size_h),
                "annotateHandle(i32 216, %dx.types.Handle ",
            ));
            let load_made = def(&ssa_after(
                def(&load_h),
                "annotateHandle(i32 216, %dx.types.Handle ",
            ));
            assert!(
                size_made.contains("i1 false, i1 true)"),
                "{label}: size handle must be marked nonUniform: {size_made}"
            );
            assert!(
                load_made.contains("i1 false, i1 false)"),
                "{label}: load handle must stay uniform: {load_made}"
            );
            if limit != 0 && raw_base == 0 {
                let size_idx = ssa_after(size_made, "createHandleFromHeap(i32 218, i32 ");
                let load_idx = ssa_after(load_made, "createHandleFromHeap(i32 218, i32 ");
                assert_ne!(
                    size_idx, load_idx,
                    "{label}: size and load handles must keep distinct Tag::Clamp indices"
                );
                assert!(
                    def(&size_idx).contains("@dx.op.binary.i32(i32 40,"),
                    "{label}: expected UMin clamp on size handle: {}",
                    def(&size_idx)
                );
                assert!(
                    def(&load_idx).contains("@dx.op.binary.i32(i32 40,"),
                    "{label}: expected UMin clamp on load handle: {}",
                    def(&load_idx)
                );
            }
        }
        if raw_base == 0 {
            let views = [
                View::Buffer {
                    data: &data,
                    stride: Stride::Raw,
                },
                View::Texture(1, 1, &texel),
                View::RawUav(2),
            ];
            for (backend, got) in run(&shader, &views) {
                assert_eq!(got[..2], [3, 77], "{label}/{backend}");
            }
        } else {
            // Raw twins start at slot 4: slot 0 holds a different buffer; both
            // the size query and the load must target twin slot 4.
            let raw = |d| View::Buffer {
                data: d,
                stride: Stride::Raw,
            };
            let other = || View::Texture(1, 1, &texel);
            let views = [
                raw(&other_data),
                other(),
                other(),
                other(),
                raw(&data),
                other(),
                View::RawUav(2),
            ];
            for (backend, got) in run(&shader, &views) {
                assert_eq!(got[..2], [3, 77], "{label}/{backend}");
            }
        }
    }
}

/// A buffer descriptor loaded inside a branch whose record count is read as a
/// number and used as an offset (the reproducer shape from fxceless PR #1213):
/// compiles cleanly with `VICTORY_VIEW_LIMIT` clamping and keeps separate
/// handles and clamps.
#[test]
fn branched_buffer_size_read_as_number_compiles_with_view_clamp() {
    let words: [u32; 19] = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0402,
        0xFA00_0020, // s_load_dwordx4 s[16:19], s[4:5], 0x20 (result UAV)
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7D88_0081, // v_cmp_gt_u32 vcc_lo, 1, v0 (lane 0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0xBF88_0008, // s_cbranch_execz +8
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0 (SRV)
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E00_020A, // v_mov_b32 v0, s10: record count as a number
        0xE030_1000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x887E_147E, // s_or_b32 exec_lo, exec_lo, s20
        0xE070_1000,
        0x8004_0100, // buffer_store_dword v1, v0, s[16:19], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: [(ResourceType::Srv, 0), (ResourceType::Uav, 8)]
            .into_iter()
            .map(|(kind, table_offset_dwords)| UsedResource {
                kind,
                root: RootType::Table,
                table_offset_dwords,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            })
            .collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 0,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let shader = with_emit_overrides(
        Some(super::bitcode::opt::DEFAULT),
        Some(2_097_152),
        Some(0),
        || {
            let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
            let st = structure::structure(&p).expect("structure");
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
            let io = io::raise(&p, &native).expect("io::raise");
            let l = lower::lower(&lower::Parts {
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
            assert!(!l.wave_ops);
            assert_eq!(l.nonuniform_handles, 0);
            HostShader {
                stage: HwStage::Cs,
                dxil: l.dxil,
                user_data: l.user_data,
                samplers: l.samplers,
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs,
                root_srvs: l.root_srvs,
                root_uavs: l.root_uavs,
                root_indexed_views: Vec::new(),
            }
        },
    );
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    if let Some(text) = dxc_disasm(&shader.dxil, "branched_dims") {
        let dims_line = text
            .lines()
            .find(|l| l.contains("@dx.op.getDimensions"))
            .expect("missing @dx.op.getDimensions");
        let load_line = text
            .lines()
            .find(|l| l.contains("@dx.op.rawBufferLoad"))
            .expect("missing @dx.op.rawBufferLoad");
        assert_ne!(
            ssa_after(dims_line, "%dx.types.Handle "),
            ssa_after(load_line, "%dx.types.Handle ")
        );
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for warp in [false, true] {
        let Ok(host) = super::host::Host::new(warp) else {
            continue;
        };
        host.compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("warp={warp}: compute_pipeline failed: {e}"));
    }
}

/// Two distinct `image_get_resinfo` size queries on the same texture across a
/// dominating/dominated block never share one `dx.op.getDimensions` call,
/// both without and with DXIL optimisation (Belobug PR #324).
#[test]
fn two_image_resinfo_queries_stay_distinct_with_and_without_opt() {
    let queries = [
        0x7E02_0280u32, // v_mov_b32 v1, 0 (mip 0)
        0xF038_0308,    // image_get_resinfo v[6:7], v1, s[8:15] dmask:0x3 2D
        0x0002_0601,
        0x7E10_0280,    // v_mov_b32 v8, 0
        0x7E12_0280,    // v_mov_b32 v9, 0
        0x7D88_0081,    // v_cmp_gt_u32 vcc_lo, 1, v0 (thread 0; non-constant)
        0xBE90_3C6A,    // s_and_saveexec_b32 s16, vcc_lo
        0xBF88_0002,    // s_cbranch_execz +2
        0xF038_0308,    // image_get_resinfo v[8:9], v1, s[8:15] dmask:0x3 2D
        0x0002_0801,
        0x887E_107E,    // s_or_b32 exec_lo, exec_lo, s16
        0x4A0C_1106,    // v_add_nc_u32 v6, v6, v8
        0x4A0E_1307,    // v_add_nc_u32 v7, v7, v9
        0x7E08_0506,    // v_readfirstlane_b32 s4, v6
        0x7E0A_0507,    // v_readfirstlane_b32 s5, v7
    ];
    let words = [TABLE, FROM_HEAP, &queries, STORE].concat();
    let (one, texels) = ([[0.0; 4]; 1], [[0.0; 4]; 15]);
    let views = [
        View::Texture(1, 1, &one),
        View::Texture(5, 3, &texels),
        View::RawUav(2),
    ];
    for (label, opt, expected_dims) in [
        // Without DEAD, the two unused descriptor_data queries on TABLE and
        // FROM_HEAP remain beside the two live image_get_resinfo calls.
        ("resinfo_opt0", 0, 4),
        // With DEAD + REUSE, the two dead descriptor_data queries are dropped
        // while both live image_get_resinfo queries stay distinct.
        ("resinfo_default", super::bitcode::opt::DEFAULT, 2),
        ("resinfo_all", super::bitcode::opt::ALL, 2),
    ] {
        let (shader, _) = with_emit_overrides(
            Some(opt),
            Some(2_097_152),
            Some(0),
            || build_stage(&words, (8, 0), HwStage::Cs).expect("translates"),
        );
        if let Some(text) = dxc_disasm(&shader.dxil, label) {
            let count = text
                .lines()
                .filter(|l| l.contains("call %dx.types.Dimensions @dx.op.getDimensions("))
                .count();
            assert_eq!(
                count, expected_dims,
                "{label}: expected {expected_dims} @dx.op.getDimensions calls, got {count}"
            );
        }
        for (backend, got) in run(&shader, &views) {
            assert_eq!(got[..2], [10, 6], "{label}/{backend}");
        }
    }
}
