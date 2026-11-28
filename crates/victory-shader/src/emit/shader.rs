//! A DXIL vertex, pixel, compute or mesh shader under construction: the module,
//! the `dx.op` declarations it calls, its signatures and bound resources,
//! and the metadata and container that wrap it up (SM 6.6, validator 1.6).

use super::bitcode::{Attr, Md, Module, Ty, V};
use super::container::{self, Binding, Psv, SigElement};

/// Where the shader reads its user data: `b0` in this space.
pub const USER_DATA_SPACE: u32 = 0;
/// Where immediate samplers are bound: `s<n>` in this space.
pub const SAMPLER_SPACE: u32 = 0;
/// Where root CBVs and SRVs are bound: `b<root parameter index>` and
/// `t<root parameter index>` in this space.
pub const ROOT_CBV_SPACE: u32 = 1;

/// The root SRV `t<SLOT_STRIDES>` in [`ROOT_CBV_SPACE`] of a shader that
/// reads buffers through descriptors it picks from the view heap at an
/// index it computes (SM 6.6 dynamic resources): a raw buffer of
/// [`SLOT_STRIDES_COUNT`] dwords the host keeps, dword `i` the bytes per
/// element of the buffer view at heap slot `i` (0: none or not a buffer).
/// The console's descriptor holds that stride and its hardware multiplies
/// the element index by it; the program's code does not say it, and the
/// used-resource list only names what tables hold. A shader that has this
/// root SRV also tells the host that the buffer views of the whole heap
/// must be ready when it runs, not only those its tables name.
pub const SLOT_STRIDES: u8 = 255;
/// The same table of strides and format kinds as [`SLOT_STRIDES`], bound for
/// shaders that only inspect views from descriptor tables (`RootType::Table`).
/// Kept distinct from [`SLOT_STRIDES`] so the host binds the table without
/// running the full bindless heap buffer sweep.
pub const SLOT_TABLE_STRIDES: u8 = 252;
/// The bits of a [`SLOT_STRIDES`] dword that hold the stride.
pub const VIEW_STRIDE_BITS: u32 = (1 << 30) - 1;
/// In a [`SLOT_STRIDES`] dword: a typed buffer view of an integer format
/// (`UINT` or `SINT`), whose 16-bit typed accesses (`_d16`) pass bits.
/// (After VIctory 4243a96a.)
pub const VIEW_INTEGER: u32 = 1 << 31;
/// With [`VIEW_INTEGER`]: a signed (`SINT`) format.
pub const VIEW_SIGNED: u32 = 1 << 30;
/// Dwords of the [`SLOT_STRIDES`] buffer's strides: heap indices are
/// clamped to it.
pub const SLOT_STRIDES_COUNT: u32 = 1 << 21;
/// Where the [`SLOT_STRIDES`] buffer's second table begins (bytes): two
/// dwords per view heap slot, the title address of the buffer view (or
/// ray-tracing structure) there, low 32 bits first; 0 for anything else.
/// Read by a program that takes the address out of a descriptor it picks
/// at a computed index and walks the memory behind it.
pub const VIEW_ADDRESSES: u32 = SLOT_STRIDES_COUNT * 4;
/// Bytes of the whole [`SLOT_STRIDES`] buffer.
pub const SLOT_STRIDES_BYTES: u32 = VIEW_ADDRESSES + SLOT_STRIDES_COUNT * 8;

/// Title memory by address (`resource::MEMORY`): console code reads and
/// writes through raw 40-bit addresses it takes from root constants,
/// descriptors and other memory (the ray-tracing structure builder and the
/// programs that walk its structures). The host keeps copies of the title
/// memory shaders use; a shader with such an access finds the copy by the
/// address:
///
/// - the root SRV `t<SLOT_MEMORY>` in [`ROOT_CBV_SPACE`] is a directory of
///   [`MEMORY_PAGES`] entries of two dwords, one per 64 KiB of title
///   address space: the number of the host copy holding that page (0: none
///   yet; with [`MEMORY_READ_ONLY`]: stores there are dropped) and the page's
///   byte offset in that copy;
/// - copy `n` is the raw buffer view at view-heap slot `base + n`, `base`
///   the user-data dword of kind [`USER_MEMORY_BASE`]; slot `base` itself
///   reads zeros and drops writes;
/// - the root UAV `u<SLOT_FAULTS>` takes the page numbers of accesses that
///   found no copy ([`MEMORY_FAULTS`] dwords, dword 0 a sink), which the
///   host reads afterwards to make those copies for the frames to come.
pub const SLOT_MEMORY: u8 = 254;
/// See [`SLOT_MEMORY`].
pub const SLOT_FAULTS: u8 = 253;
/// Entries of the [`SLOT_MEMORY`] directory: title addresses below 128 GiB.
pub const MEMORY_PAGES: u32 = 1 << 21;
/// In a directory entry's first dword: shaders do not store there by address
/// (a title buffer the structure builder has not been given to write).
pub const MEMORY_READ_ONLY: u32 = 1 << 31;
/// Dwords of the [`SLOT_FAULTS`] buffer.
pub const MEMORY_FAULTS: u32 = 4096;
/// `RootType::Other` code of a user-data dword the host computes: the view
/// heap slot of the memory copies' views ([`SLOT_MEMORY`]).
pub const USER_MEMORY_BASE: u32 = 0x100;
/// `RootType::Other` code of a user-data dword the host computes: the low
/// 32 bits of the title address of the buffer view at dword `dword` of the
/// descriptor table of root parameter `slot`; [`USER_VIEW_ADDRESS`] + 1:
/// its bits from 32 up. (Console code reads the address out of a buffer
/// descriptor and uses it as a pointer.)
pub const USER_VIEW_ADDRESS: u32 = 0x101;
/// `RootType::Other` code of a user-data dword the host computes: the
/// bytes (0 to 15) by which the raw view the host binds for the root view
/// of root parameter `slot` begins before the title's address. A raw view
/// begins on 16 bytes of its host buffer; the console's root views begin
/// anywhere (the structure builder's counters sit at addresses ending in
/// 4 and c). The shader adds the dword to every byte offset through that
/// view. Asked for every root UAV.
pub const USER_VIEW_BIAS: u32 = 0x103;
/// `RootType::Other` code of a user-data dword the host computes: the low
/// 32 bits of the title address in the root view of root parameter `slot`;
/// [`USER_ROOT_ADDRESS`] + 1: its bits from 32 up. Programs with a spilled
/// user-data table (the ray-tracing structure builder's) reach what their
/// root views name as title memory by address ([`SLOT_MEMORY`]): a root view
/// has no size, and the host's copy of what it names may end in the middle of
/// it (a structure of megabytes crosses the 4 MiB windows).
pub const USER_ROOT_ADDRESS: u32 = 0x104;
/// `RootType::Other` code of a user-data dword the host sets at a dispatch
/// counted in threads (the console's `DispatchX`): how many of the threads
/// numbered along X run (0: all of every group). The metadata's `dword` is
/// the group's width. Asked for by the structure builder's programs
/// ([`crate::spill::is_spilled`]).
pub const USER_DISPATCH_THREADS_X: u32 = 0x106;

/// `ShaderFlags` bits (metadata) and their `SFI0` feature bits.
pub(crate) mod flags {
    /// ForceEarlyDepthStencil, the DXIL equivalent of HLSL's earlydepthstencil.
    pub const EARLY_DEPTH_STENCIL: u64 = 1 << 3;
    pub const RAW_BUFFERS: u64 = 1 << 4;
    pub const DOUBLES: u64 = 1 << 2;
    pub const STENCIL_REF: u64 = 1 << 11;
    pub const TYPED_UAV_LOADS: u64 = 1 << 13;
    /// Unordered views in a vertex, geometry or mesh program.
    pub const UAVS_EVERY_STAGE: u64 = 1 << 16;
    pub const ROVS: u64 = 1 << 18;
    pub const WAVE_OPS: u64 = 1 << 19;
    pub const BARYCENTRICS: u64 = 1 << 22;
    pub const INT64: u64 = 1 << 20;
    pub const ATOMIC_TYPED64: u64 = 1 << 27;
    pub const ATOMIC_HEAP64: u64 = 1 << 32;
    pub const RESOURCE_HEAP: u64 = 1 << 30;
    pub const SAMPLER_HEAP: u64 = 1 << 31;
}

pub(crate) struct Types {
    pub void: Ty,
    pub i1: Ty,
    pub i32: Ty,
    pub f32: Ty,
    pub handle: Ty,
    pub props: Ty,
    pub bind: Ty,
    pub cbuf_ret: Ty,
    pub res_ret: Ty,
}

/// What a mesh shader declares beyond its thread-group size.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Mesh {
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// 1 lines, 2 triangles.
    pub topology: u8,
}

/// What a geometry shader declares: triangles in, triangle strips out on
/// stream 0, one instance.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Geometry {
    pub max_vertices: u32,
}

/// The null slot every host view heap keeps at its end (0: no limit). Shaders translated while
/// it is set read that slot for any view index at or past it. A read past a heap's end is
/// undefined: a host that caps its heap (NVIDIA at 1 000 000 views; a retail game's heap is
/// larger) hung its GPU (GPU-based validation: "descriptor heap index out of bounds"), and AMD
/// RDNA3 took a page fault and removed the device.
static VIEW_LIMIT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[cfg(test)]
thread_local! {
    /// For a test's thread alone: [`view_heap_limit`].
    pub(crate) static VIEW_LIMIT_HERE: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// See [`VIEW_LIMIT`]: `null_slot` is the last slot of the host's view heap, kept null.
pub fn set_view_heap_limit(null_slot: u32) {
    VIEW_LIMIT.store(null_slot, std::sync::atomic::Ordering::Relaxed);
}

/// See [`VIEW_LIMIT`]: the null slot shaders translated now clamp view indices to (0: none).
pub fn view_heap_limit() -> u32 {
    #[cfg(test)]
    if let Some(n) = VIEW_LIMIT_HERE.get() {
        return n;
    }
    VIEW_LIMIT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Where the host's raw twins of its buffer views start in its view heaps (0: there are none).
///
/// A buffer a shader picks from the heap by an index it computes is read as raw bytes whatever
/// view the title has there, and the index comes from data: a record of a mesh long gone, a list
/// entry nothing wrote. On AMD a raw load through a view that is not raw (typed, structured, a
/// constant buffer's) has no end: it reads whatever lies there, up to 4 GiB on, and loses the
/// device where nothing does; and a texture's descriptor read as a buffer's is a wild address.
/// So the host keeps, for slot `i`, a raw view of the same bytes at this place + `i` (a constant
/// buffer view of a constant buffer in a title slot, a null raw view of everything else), and
/// shaders translated while this is set read picked buffers there, and tables' buffers that they
/// read as raw bytes.
/// A power of two above every slot index.
static RAW_VIEWS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[cfg(test)]
thread_local! {
    /// For a test's thread alone.
    pub(crate) static RAW_VIEWS_HERE: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// See [`RAW_VIEWS`].
pub fn set_raw_views(first: u32) {
    debug_assert!(first == 0 || first.is_power_of_two());
    RAW_VIEWS.store(first, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
thread_local! {
    /// For a test's thread alone: [`constants_raw`].
    pub(crate) static CONSTANTS_RAW_HERE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Renderer port 5 (`VICTORY_PORT_CBV=1`; off by default): constant buffers are read as raw
/// bytes, a table's through the raw twin of its slot ([`raw_views`]), a root one as a root SRV,
/// so the host can give the card the title's bytes where they are (16-byte aligned, which a
/// constant buffer view cannot take) instead of a copy made at every submission. The host's
/// views must follow: the twin of every constant buffer's slot a raw view of its bytes. Part of
/// the translation: the shader cache keys on the switch.
pub fn constants_raw() -> bool {
    #[cfg(test)]
    if let Some(on) = CONSTANTS_RAW_HERE.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_CBV").is_ok_and(|v| v.trim() == "1"))
}

/// See [`RAW_VIEWS`]: where shaders translated now read the buffers they pick (0: in place).
pub fn raw_views() -> u32 {
    #[cfg(test)]
    if let Some(first) = RAW_VIEWS_HERE.get() {
        return first;
    }
    RAW_VIEWS.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) struct Shader {
    pub m: Module,
    pub t: Types,
    pub pixel: bool,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    /// A mesh shader's per-primitive outputs.
    pub prims: Vec<SigElement>,
    /// A mesh shader (`threads` is its thread-group size).
    pub mesh: Option<Mesh>,
    /// A geometry shader (the rectangle stage, `rect.rs`); `pixel` is false.
    pub geometry: Option<Geometry>,
    /// Dwords of user data the shader reads (0: no user-data buffer).
    pub user_data: u32,
    /// Immediate samplers, bound as `s0..` in [`SAMPLER_SPACE`].
    pub samplers: u32,
    /// Bit n: immediate sampler n is a comparison sampler.
    pub cmp_samplers: u64,
    /// Root CBVs read, by root parameter index, with their handles.
    pub root_cbvs: Vec<(u8, V)>,
    /// Root SRVs (raw buffers) read, likewise.
    pub root_srvs: Vec<(u8, V)>,
    /// Root UAVs (raw buffers) read or written, likewise.
    pub root_uavs: Vec<(u8, V)>,
    pub flags: u64,
    /// Heap handles made with a non-uniform index.
    pub nonuniform_handles: u32,
    /// A compute shader with this thread-group size.
    pub threads: Option<[u32; 3]>,
    /// The wave size a compute shader requires (`WaveSize`).
    pub wave_size: Option<u32>,
    /// Thread-group shared memory, in dwords.
    pub lds_dwords: u32,
    lds: Option<V>,
    main: V,
    pub entry: usize,
}

impl Shader {
    pub fn new(pixel: bool) -> Self {
        let mut m = Module::default();
        let void = m.void();
        let i1 = m.int(1);
        let i8 = m.int(8);
        let i32 = m.int(32);
        let f32 = m.float();
        let i8p = m.ptr(i8);
        let handle = m.named_struct("dx.types.Handle", vec![i8p]);
        let props = m.named_struct("dx.types.ResourceProperties", vec![i32, i32]);
        let bind = m.named_struct("dx.types.ResBind", vec![i32, i32, i32, i8]);
        let cbuf_ret = m.named_struct("dx.types.CBufRet.i32", vec![i32, i32, i32, i32]);
        let res_ret = m.named_struct("dx.types.ResRet.f32", vec![f32, f32, f32, f32, i32]);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        let main = m.function("main", main_ty, None, true);
        let entry = m.new_block();
        Shader {
            m,
            t: Types {
                void,
                i1,
                i32,
                f32,
                handle,
                props,
                bind,
                cbuf_ret,
                res_ret,
            },
            pixel,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh: None,
            geometry: None,
            user_data: 0,
            samplers: 0,
            cmp_samplers: 0,
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            flags: 0,
            nonuniform_handles: 0,
            threads: None,
            wave_size: None,
            lds_dwords: 0,
            lds: None,
            main,
            entry,
        }
    }

    /// The thread-group shared memory: `[lds_dwords x i32]` in address
    /// space 3.
    pub fn lds(&mut self) -> V {
        if let Some(g) = self.lds {
            return g;
        }
        let arr = self.m.array(self.lds_dwords, self.t.i32);
        let g = self.m.global("lds", arr, 3);
        self.lds = Some(g);
        g
    }

    /// Calls `dx.op.<name>` (class and overload) with `opcode` and `args`.
    pub fn op(
        &mut self,
        block: usize,
        name: &str,
        attr: Attr,
        ret: Ty,
        opcode: u32,
        args: &[V],
    ) -> V {
        let mut params = vec![self.t.i32];
        params.extend(args.iter().map(|&a| self.m.type_of(a)));
        let fty = self.m.func_ty(ret, params);
        self.m.ptr(fty);
        let f = self
            .m
            .function(&format!("dx.op.{name}"), fty, Some(attr), false);
        let mut all = vec![self.m.i32(opcode)];
        all.extend(args);
        self.m.call(block, f, all)
    }

    /// The handle of the user-data constant buffer, made in the entry block.
    pub fn user_data_handle(&mut self) -> V {
        let bind = [
            self.m.i32(0),
            self.m.i32(0),
            self.m.i32(USER_DATA_SPACE),
            self.m.i8(2), // CBV
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let block = self.entry;
        let (i, f) = (self.m.i32(0), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        // The size is known in `finish`, which fixes this placeholder.
        self.annotate(block, h, 13, u32::MAX)
    }

    /// The handle of the root CBV of root parameter `slot`, made in the
    /// entry block.
    pub fn root_cbv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_cbvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(2)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 13, 65536);
        self.root_cbvs.push((slot, h));
        h
    }

    /// The handle of the root SRV (a raw buffer) of root parameter `slot`.
    pub fn root_srv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_srvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_srvs.push((slot, h));
        h
    }

    /// The handle of the root UAV (a raw buffer) of root parameter `slot`:
    /// `u<slot>` in [`ROOT_CBV_SPACE`].
    pub fn root_uav_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_uavs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(1)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11 | 0x1000, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_uavs.push((slot, h));
        h
    }

    /// The handle of immediate sampler `n` (a comparison sampler when its
    /// bit in `cmp_samplers` is set).
    pub fn sampler_handle(&mut self, block: usize, n: u32) -> V {
        let bind = [
            self.m.i32(n),
            self.m.i32(n),
            self.m.i32(SAMPLER_SPACE),
            self.m.i8(3), // sampler
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let (i, f) = (self.m.i32(n), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        let cmp = if self.cmp_samplers >> n & 1 != 0 {
            0x8000
        } else {
            0
        };
        self.annotate(block, h, 14 | cmp, 0)
    }

    pub fn annotate(&mut self, block: usize, h: V, props0: u32, props1: u32) -> V {
        let p = [self.m.i32(props0), self.m.i32(props1)];
        let p = self.m.aggregate(self.t.props, &p);
        self.op(
            block,
            "annotateHandle",
            Attr::ReadNone,
            self.t.handle,
            216,
            &[h, p],
        )
    }

    /// A descriptor heap handle, annotated with `props`; `nonuniform` when
    /// the index may differ between lanes (`NonUniformResourceIndex`).
    pub fn heap_handle(
        &mut self,
        block: usize,
        index: V,
        sampler: bool,
        nonuniform: bool,
        props: (u32, u32),
    ) -> V {
        self.heap_handle_at(block, index, sampler, nonuniform, props, false)
    }

    /// [`Shader::heap_handle`]; `bounded`: the index is inside the host's heap already
    /// ([`raw_views`]).
    pub fn heap_handle_at(
        &mut self,
        block: usize,
        index: V,
        sampler: bool,
        nonuniform: bool,
        props: (u32, u32),
        bounded: bool,
    ) -> V {
        self.nonuniform_handles += u32::from(nonuniform);
        self.marked_heap_handle_at(block, index, sampler, nonuniform, props, bounded)
    }

    /// The handle a buffer's size is asked through (`descriptor_data`):
    /// marked non-uniform for AMD's RDNA4 compiler (`amdxc64.dll+0x861b9f`),
    /// which reads a null pointer compiling `getDimensions` through the
    /// uniform heap handle the buffer's loads use, and not counted in
    /// `nonuniform_handles`, its index being the same in every lane.
    pub fn size_handle_at(
        &mut self,
        block: usize,
        index: V,
        props: (u32, u32),
        bounded: bool,
    ) -> V {
        self.marked_heap_handle_at(block, index, false, true, props, bounded)
    }

    fn marked_heap_handle_at(
        &mut self,
        block: usize,
        index: V,
        sampler: bool,
        nonuniform: bool,
        props: (u32, u32),
        bounded: bool,
    ) -> V {
        // a view index past the host heap (`set_view_heap_limit`) reads its null slot instead
        // (the raw twins of buffer views lie past it: such an index comes bounded)
        let limit = if bounded { 0 } else { view_heap_limit() };
        // (as `dx.op.binary` UMin: a compare and select emitted here made the bitcode invalid)
        let index = if !sampler && limit != 0 {
            let null = self.m.i32(limit);
            let i32t = self.t.i32;
            let v = self.op(block, "binary.i32", Attr::ReadNone, i32t, 40, &[index, null]);
            self.m.tag(v, super::bitcode::Tag::Clamp);
            v
        } else if sampler {
            // A sampler heap has at most 2,048 samplers on any host: an
            // index past that (a number read from memory the host has not
            // got yet) stays inside the largest heap there is.
            // shortcut: not the bound heap's own count, which a title
            // heap smaller than 2,048 would need.
            let last = self.m.i32(2047);
            let i32t = self.t.i32;
            let v = self.op(block, "binary.i32", Attr::ReadNone, i32t, 40, &[index, last]);
            self.m.tag(v, super::bitcode::Tag::Clamp);
            v
        } else {
            index
        };
        self.flags |= if sampler {
            flags::SAMPLER_HEAP
        } else {
            flags::RESOURCE_HEAP
        };
        let (s, n) = (self.m.bool(sampler), self.m.bool(nonuniform));
        let h = self.op(
            block,
            "createHandleFromHeap",
            Attr::ReadNone,
            self.t.handle,
            218,
            &[index, s, n],
        );
        self.annotate(block, h, props.0, props.1)
    }

    /// The finished container.
    pub fn finish(mut self) -> Vec<u8> {
        if self.pixel {
            // Reading a per-vertex value with attributeAtVertex needs a
            // constant input, even when the source declaration was smooth.
            // D3D requires all components packed into the same register to
            // share a mode, including the ones the shader never reads.
            for index in 0..self.inputs.len() {
                let unused = &self.inputs[index];
                if unused.kind != 0 || unused.usage != 0 || unused.row.is_none() {
                    continue;
                }
                let mut modes = self.inputs.iter()
                    .filter(|e| e.row == unused.row && e.usage != 0)
                    .map(|e| e.interp);
                if let Some(mode) = modes.next()
                    && (mode == 1 || unused.comp == 9)
                    && modes.all(|other| other == mode)
                {
                    self.inputs[index].interp = mode;
                }
            }
        }
        let ud_size = (self.user_data * 4).next_multiple_of(16);
        self.fix_user_data_size(ud_size);
        // The clean-up, before anything is declared about the code: a
        // feature whose last use it removed must not be declared (the
        // validator compares the flags with what the code uses).
        let passes = super::shader_opt() & super::bitcode::opt::MODULE;
        if passes != 0 {
            self.m.optimise(passes);
            let used = self.m.usage();
            for (flag, used) in [
                (flags::ROVS, used.rovs),
                (flags::WAVE_OPS, used.wave_ops),
                (flags::INT64, used.int64),
                (flags::TYPED_UAV_LOADS, used.typed_uav_loads),
                (flags::RESOURCE_HEAP, used.heap_views),
                (flags::SAMPLER_HEAP, used.heap_samplers),
                (flags::ATOMIC_TYPED64 | flags::ATOMIC_HEAP64, used.atomic64),
            ] {
                if !used {
                    self.flags &= !flag;
                }
            }
        }
        // Unordered views outside pixel and compute programs (memory by
        // address), as DXC counts them: ranges declared below, not views
        // picked from the heap (VIctory a1a2077a).
        let compute = self.threads.is_some() && self.mesh.is_none() && self.geometry.is_none();
        if !self.root_uavs.is_empty() && !self.pixel && !compute {
            self.flags |= flags::UAVS_EVERY_STAGE;
        }
        let mut resources = None;
        let mut bindings = Vec::new();
        let i32c = |s: &mut Self, v: u32| {
            let c = s.m.i32(v);
            s.m.md_value(c)
        };
        if self.user_data > 0
            || self.samplers > 0
            || !self.root_cbvs.is_empty()
            || !self.root_srvs.is_empty()
            || !self.root_uavs.is_empty()
        {
            let empty = self.m.md_str("");
            let mut cbvs = None;
            let mut cbv_list = Vec::new();
            if self.user_data > 0 {
                let arr = self.m.array(self.user_data.next_multiple_of(4), self.t.i32);
                let ty = self.m.named_struct("UserData", vec![arr]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let ops = vec![
                    Some(i32c(&mut self, 0)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, USER_DATA_SPACE)),
                    Some(i32c(&mut self, 0)),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, ud_size)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: USER_DATA_SPACE,
                    register: 0,
                    dxil_kind: 13,
                });
            }
            let blob = self.m.array(16384, self.t.i32);
            let ty = self.m.named_struct("RootCbv", vec![blob]);
            let p = self.m.ptr(ty);
            let u = self.m.undef(p);
            for (slot, _) in self.root_cbvs.clone() {
                let ops = vec![
                    Some(i32c(&mut self, cbv_list.len() as u32)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, ROOT_CBV_SPACE)),
                    Some(i32c(&mut self, u32::from(slot))),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, 65536)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: ROOT_CBV_SPACE,
                    register: u32::from(slot),
                    dxil_kind: 13,
                });
            }
            if !cbv_list.is_empty() {
                cbvs = Some(self.m.md_node(cbv_list));
            }
            let mut srvs = None;
            if !self.root_srvs.is_empty() {
                let ty = self
                    .m
                    .named_struct("struct.ByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let mut list = Vec::new();
                for (n, (slot, _)) in self.root_srvs.clone().into_iter().enumerate() {
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, ROOT_CBV_SPACE)),
                        Some(i32c(&mut self, u32::from(slot))),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(i32c(&mut self, 0)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 4,
                        space: ROOT_CBV_SPACE,
                        register: u32::from(slot),
                        dxil_kind: 11,
                    });
                }
                srvs = Some(self.m.md_node(list));
            }
            let mut uavs = None;
            if !self.root_uavs.is_empty() {
                let ty = self
                    .m
                    .named_struct("struct.RWByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let no = self.m.bool(false);
                let no = self.m.md_value(no);
                let mut list = Vec::new();
                for (n, (slot, _)) in self.root_uavs.clone().into_iter().enumerate() {
                    // Id, variable, name, space, lower bound, range size,
                    // shape (11: raw buffer), globally coherent, counter,
                    // rasterizer ordered, no extra properties.
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, ROOT_CBV_SPACE)),
                        Some(i32c(&mut self, u32::from(slot))),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(no),
                        Some(no),
                        Some(no),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 7,
                        space: ROOT_CBV_SPACE,
                        register: u32::from(slot),
                        dxil_kind: 11,
                    });
                }
                uavs = Some(self.m.md_node(list));
            }
            let mut samplers = None;
            if self.samplers > 0 {
                let ty = self.m.named_struct("struct.SamplerState", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let mut list = Vec::new();
                for n in 0..self.samplers {
                    let kind = (self.cmp_samplers >> n & 1) as u32;
                    let ops = vec![
                        Some(i32c(&mut self, n)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, SAMPLER_SPACE)),
                        Some(i32c(&mut self, n)),
                        Some(i32c(&mut self, 1)),
                        // Sampler kind: 1 comparison.
                        Some(i32c(&mut self, kind)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 1,
                        space: SAMPLER_SPACE,
                        register: n,
                        dxil_kind: 14,
                    });
                }
                samplers = Some(self.m.md_node(list));
            }
            resources = Some(self.m.md_node(vec![srvs, uavs, cbvs, samplers]));
            // PSV0 lists them as the module does: CBVs, samplers, SRVs, UAVs.
            bindings.sort_by_key(|b| match b.kind {
                2 => 0,
                1 => 1,
                7 => 3,
                _ => 2,
            });
        }
        if let Some(r) = resources {
            self.m.named_md("dx.resources", vec![r]);
        }

        let graphics = self.threads.is_none() || self.mesh.is_some();
        if graphics {
            // shortcut: every output is declared to depend on every input; the
            // exact sets need a dependence pass. Drivers read this state.
            let scalars = |els: &[SigElement]| {
                els.iter()
                    .filter_map(|e| e.row.map(|r| r * 4 + u32::from(e.col + e.cols)))
                    .max()
                    .unwrap_or(0)
            };
            let (ni, no) = (scalars(&self.inputs), scalars(&self.outputs));
            let mut view = vec![ni, no];
            // Mesh shaders have no inputs: then the primitive outputs' count.
            if self.mesh.is_some() {
                view.push(scalars(&self.prims));
            }
            for _ in 0..ni {
                for k in 0..no.div_ceil(32) {
                    let bits = (no - 32 * k).min(32);
                    view.push(if bits == 32 {
                        u32::MAX
                    } else {
                        (1 << bits) - 1
                    });
                }
            }
            // A geometry shader lists four streams: the other three have
            // no outputs.
            if self.geometry.is_some() {
                view.extend([0, 0, 0]);
            }
            let arr = self.m.array(view.len() as u32, self.t.i32);
            let elems: Vec<V> = view.iter().map(|&v| self.m.i32(v)).collect();
            let arr = self.m.aggregate(arr, &elems);
            let arr = self.m.md_value(arr);
            let view_node = self.m.md_node(vec![Some(arr)]);
            self.m.named_md("dx.viewIdState", vec![view_node]);
        }

        if self.pixel {
            container::same_mode_per_row(&mut self.inputs, true);
        } else {
            container::same_mode_per_row(&mut self.outputs, false);
        }
        let sigs = if graphics {
            let ins = self.signature_md(self.inputs.clone());
            let outs = self.signature_md(self.outputs.clone());
            let prims = self.signature_md(self.prims.clone());
            Some(self.m.md_node(vec![ins, outs, prims]))
        } else {
            None
        };
        // A pixel shader with an SV_StencilRef output (as DXC collects it).
        if self.pixel && self.outputs.iter().any(|o| o.kind == 20) {
            self.flags |= flags::STENCIL_REF;
        }
        // Entry properties: shader flags (tag 0), geometry state (1),
        // thread-group size (4), wave size (11).
        let mut props = Vec::new();
        if self.flags != 0 {
            props.push(i32c(&mut self, 0));
            let i64t = self.m.int(64);
            let f = self.m.int_const(i64t, self.flags as i64);
            props.push(self.m.md_value(f));
        }
        if let Some(g) = self.geometry {
            // Input primitive (3: triangles), maximum vertices, stream
            // mask, output topology (5: triangle strips), instances.
            props.push(i32c(&mut self, 1));
            let state = [3, g.max_vertices, 1, 5, 1].map(|n| Some(i32c(&mut self, n)));
            props.push(self.m.md_node(state.to_vec()));
        }
        if let Some(threads) = self.threads {
            let t = threads.map(|n| Some(i32c(&mut self, n)));
            let t = self.m.md_node(t.to_vec());
            match self.mesh {
                // Mesh state (tag 9): threads, maximum vertices and
                // primitives, topology, payload bytes.
                Some(mesh) => {
                    props.push(i32c(&mut self, 9));
                    let state = [
                        mesh.max_vertices,
                        mesh.max_primitives,
                        mesh.topology.into(),
                        0,
                    ]
                    .map(|n| Some(i32c(&mut self, n)));
                    let mut node = vec![Some(t)];
                    node.extend(state);
                    props.push(self.m.md_node(node));
                }
                None => {
                    props.push(i32c(&mut self, 4));
                    props.push(t);
                }
            }
        }
        if let Some(n) = self.wave_size {
            props.push(i32c(&mut self, 11));
            let n = Some(i32c(&mut self, n));
            props.push(self.m.md_node(vec![n]));
        }
        let props =
            (!props.is_empty()).then(|| self.m.md_node(props.into_iter().map(Some).collect()));
        let main = self.m.md_value(self.main);
        let name = self.m.md_str("main");
        let entry = self
            .m
            .md_node(vec![Some(main), Some(name), sigs, resources, props]);

        let ident = self.m.md_str("victory");
        let ident = self.m.md_node(vec![Some(ident)]);
        self.m.named_md("llvm.ident", vec![ident]);
        let one = i32c(&mut self, 1);
        let six = i32c(&mut self, 6);
        let version = self.m.md_node(vec![Some(one), Some(six)]);
        self.m.named_md("dx.version", vec![version]);
        self.m.named_md("dx.valver", vec![version]);
        // Program kinds: 0 pixel, 1 vertex, 2 geometry, 5 compute, 13 mesh.
        let (name, kind) = match (self.threads, self.pixel) {
            _ if self.mesh.is_some() => ("ms", 13u32),
            _ if self.geometry.is_some() => ("gs", 2),
            (Some(_), _) => ("cs", 5u32),
            (None, true) => ("ps", 0),
            (None, false) => ("vs", 1),
        };
        let name = self.m.md_str(name);
        let sm = self.m.md_node(vec![Some(name), Some(six), Some(six)]);
        self.m.named_md("dx.shaderModel", vec![sm]);
        self.m.named_md("dx.entryPoints", vec![entry]);

        self.m.single_latches();
        let bitcode = self.m.write();
        super::bitcode::census::note(&self.m, &bitcode);
        let mut dxil = Vec::with_capacity(24 + bitcode.len());
        dxil.extend((kind << 16 | 0x66).to_le_bytes());
        dxil.extend((((24 + bitcode.len()) / 4) as u32).to_le_bytes());
        dxil.extend(b"DXIL");
        dxil.extend(0x106u32.to_le_bytes());
        dxil.extend(16u32.to_le_bytes());
        dxil.extend((bitcode.len() as u32).to_le_bytes());
        dxil.extend(&bitcode);

        let mut features = 0u64;
        if self.flags & flags::DOUBLES != 0 {
            features |= 0x1;
        }
        if self.flags & flags::UAVS_EVERY_STAGE != 0 {
            features |= 0x4;
        }
        if self.flags & flags::STENCIL_REF != 0 {
            features |= 0x200;
        }
        if self.flags & flags::TYPED_UAV_LOADS != 0 {
            features |= 0x800;
        }
        if self.flags & flags::ROVS != 0 {
            features |= 0x1000;
        }
        if self.flags & flags::WAVE_OPS != 0 {
            features |= 0x4000;
        }
        if self.flags & flags::INT64 != 0 {
            features |= 0x8000;
        }
        if self.flags & flags::BARYCENTRICS != 0 {
            features |= 0x2_0000;
        }
        if self.flags & flags::RESOURCE_HEAP != 0 {
            features |= 0x200_0000;
        }
        if self.flags & flags::SAMPLER_HEAP != 0 {
            features |= 0x400_0000;
        }
        if self.flags & flags::ATOMIC_TYPED64 != 0 {
            features |= 0x40_0000;
        }
        if self.flags & flags::ATOMIC_HEAP64 != 0 {
            features |= 0x1000_0000;
        }
        let sfi0 = features.to_le_bytes();
        let isg1 = container::signature(&self.inputs, true);
        let osg1 = container::signature(&self.outputs, false);
        let flag = if self.pixel {
            self.outputs.iter().any(|o| (17..=19).contains(&o.kind))
        } else {
            self.outputs.iter().any(|o| o.kind == 3)
        };
        let psv0 = container::psv(&Psv {
            stage: kind as u8,
            flag: flag && self.threads.is_none(),
            inputs: &self.inputs,
            outputs: &self.outputs,
            prims: &self.prims,
            mesh: self.mesh.map(|m| (m, self.lds_dwords * 4)),
            geometry: self.geometry.map(|g| g.max_vertices),
            bindings: &bindings,
            threads: self.threads.unwrap_or_default(),
            wave_size: self.wave_size,
        });
        let hash = [0u8; 20];
        let psg1 = container::signature(&self.prims, false);
        let mut parts: Vec<(&[u8; 4], &[u8])> =
            vec![(b"SFI0", &sfi0), (b"ISG1", &isg1), (b"OSG1", &osg1)];
        if self.mesh.is_some() {
            parts.push((b"PSG1", &psg1));
        }
        parts.extend([(b"PSV0", &psv0[..]), (b"HASH", &hash), (b"DXIL", &dxil)]);
        container::build(&parts)
    }

    /// The user-data buffer's annotation carries its size, known only now.
    fn fix_user_data_size(&mut self, size: u32) {
        let old = [self.m.i32(13), self.m.i32(u32::MAX)];
        let old = self.m.aggregate(self.t.props, &old);
        let new = [self.m.i32(13), self.m.i32(size)];
        let new = self.m.aggregate(self.t.props, &new);
        self.m.replace_const(old, new);
    }

    fn signature_md(&mut self, elements: Vec<SigElement>) -> Option<Md> {
        if elements.is_empty() {
            return None;
        }
        let mut list = Vec::new();
        for (id, e) in elements.iter().enumerate() {
            let c32 = |s: &mut Self, v: u32| {
                let c = s.m.i32(v);
                Some(s.m.md_value(c))
            };
            let c8 = |s: &mut Self, v: u8| {
                let c = s.m.i8(v);
                Some(s.m.md_value(c))
            };
            let idx = c32(self, e.index);
            let indices = Some(self.m.md_node(vec![idx]));
            let usage = if e.usage != 0 {
                let (tag, mask) = (c32(self, 3), c32(self, u32::from(e.usage)));
                Some(self.m.md_node(vec![tag, mask]))
            } else {
                None
            };
            let ops = vec![
                c32(self, id as u32),
                Some(self.m.md_str(&e.name)),
                c8(self, e.comp),
                c8(self, e.kind),
                indices,
                c8(self, e.interp),
                c32(self, 1),
                c8(self, e.cols),
                c32(self, e.row.unwrap_or(u32::MAX)),
                c8(self, if e.row.is_some() { e.col } else { u8::MAX }),
                usage,
            ];
            list.push(Some(self.m.md_node(ops)));
        }
        Some(self.m.md_node(list))
    }
}
