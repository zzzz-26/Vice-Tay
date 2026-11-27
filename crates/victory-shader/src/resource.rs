//! Resource tracking (decision 0003): where each descriptor a program uses
//! comes from, in terms of the root signature.
//!
//! Root parameters arrive in user SGPRs ([`RootParam`]): a descriptor table
//! as the 32-bit heap address of its range (the code sets the high half to
//! 0), a root descriptor as a 64-bit buffer address, root constants as
//! themselves. A root signature with dynamic resources (SM 6.6) adds the
//! 32-bit base addresses of both heaps; a descriptor is then loaded from
//! one of them at an index the code computes (from a root constant, say). The code loads table descriptors from heap memory with
//! `s_load_dword*`, and builds a buffer descriptor around a root
//! descriptor's address itself. Each descriptor operand and each scalar
//! memory address is traced through the SSA values back to those
//! parameters, by data flow: copies, phis that agree, and additions.

use crate::ShaderError;
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, RootParam, RootType};
use crate::structure::{Bool, LaneExit, Meets, Structured};

/// An instruction: block and index in [`crate::ir::Block::insts`].
pub type Site = (usize, usize);

/// A byte address: a root parameter's, plus a constant and at most one
/// value the program computes (an index times a stride, say).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Address {
    /// Index into the layout, or [`CODE`].
    pub param: usize,
    /// Added modulo 2^32 (with carry into the high half of a root
    /// descriptor's address).
    pub offset: u32,
    pub dynamic: Option<Value>,
}

/// [`Address::param`] of the program's own code: a constant table the
/// compiler appends after the code and reads through `s_getpc_b64`; the
/// offset is then the byte offset in the code.
pub const CODE: usize = usize::MAX;

/// [`Address::param`] of title memory at an address no root parameter
/// gives: one the code computes from root constants, reads out of a
/// descriptor or loads from other memory (compute programs only: the
/// ray-tracing structure builder and the programs that walk its
/// structures). Offset and computed term are not used: lowering takes the
/// 64-bit address from the instruction's operands.
pub const MEMORY: usize = usize::MAX - 1;

/// The reasons an address traces to no root parameter for which the
/// address is then taken as it stands ([`MEMORY`]).
const RAW: [&str; 5] = [
    "an address not from a root parameter",
    "an address from a root parameter that is no address",
    "an address in an SGPR that is no root parameter",
    "an address loaded from memory",
    "an address from a phi",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Dwords `first..` of what the scalar load at `load` reads (see
    /// [`Resources::scalar`] for its address).
    Loaded { load: Site, first: u8 },
    /// A buffer descriptor built around an address: `bits` OR-ed into
    /// dword 1 (stride, swizzle), dwords 2 and 3 as given.
    Built {
        address: Address,
        bits: u32,
        words: [u32; 2],
    },
    /// Fixed in the code (an immediate sampler, say).
    Constant(Vec<u32>),
    /// Chosen by control flow: one source per predecessor of `block`, in
    /// [`crate::ir::Block::preds`] order.
    Phi { block: usize, args: Vec<Source> },
    /// Chosen by SCC (`scc`, the value an `s_cselect` or `s_cmov` of each
    /// dword reads): the source when it is set, then when it is clear.
    Select { scc: Value, args: Vec<Source> },
    /// Dwords `first..` of what the vector load at `load` (`global_load_*`)
    /// reads, each lane at its own address (see [`Resources::scalar`]); the
    /// dwords past that load come from loads of the bytes that follow. A
    /// descriptor that differs between lanes.
    Fetched { load: Site, first: u8 },
    /// Picked per lane by `v_cndmask_b32` reading the lane mask whose low
    /// dword is `mask`: the source where the lane's bit is set, then where it
    /// is clear.
    Lanes { mask: Value, args: Vec<Source> },
    /// Per-lane dwords chosen by control flow (VGPRs joined where lanes
    /// that took different paths meet): one source per predecessor of
    /// `block`, in [`crate::ir::Block::preds`] order, `None` on an edge no
    /// lane that accesses the descriptor takes. A descriptor that differs
    /// between lanes.
    LanePhi {
        block: usize,
        args: Vec<Option<Source>>,
    },
}

impl Source {
    /// The sources this one picks from (itself unless a phi).
    pub fn leaves(&self) -> Vec<&Source> {
        match self {
            Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } => {
                args.iter().flat_map(Source::leaves).collect()
            }
            Source::LanePhi { args, .. } => {
                args.iter().flatten().flat_map(Source::leaves).collect()
            }
            s => vec![s],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Image, buffer or `s_buffer_load` resource.
    Resource,
    Sampler,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descriptor {
    pub site: Site,
    pub role: Role,
    pub source: Source,
}

/// A view-heap descriptor whose byte address can be evaluated when binding
/// the draw, using a table base and a root constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootIndexedView {
    pub table_slot: u8,
    pub constant_slot: u8,
    pub constant_dword: u32,
    pub byte_offset: u32,
    pub shift: u8,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resources {
    /// Scalar values saved to a constant VGPR lane and restored within one
    /// block, even if that VGPR is also used in ordinary vector operations.
    pub(crate) restored_lanes: std::collections::HashMap<Site, Moved>,
    /// Every scalar memory access by address (`s_load_*`, `s_store_*`), and
    /// every global memory access (`global_*`, the dynamic term per lane).
    pub scalar: Vec<(Site, Address)>,
    pub descriptors: Vec<Descriptor>,
    /// Direct descriptor sources with a draw-evaluable index.
    pub root_indexed_views: Vec<RootIndexedView>,
    /// User SGPRs holding root parameters: value on entry, layout index,
    /// dword within the parameter.
    pub params: Vec<(Value, usize, u32)>,
    /// Scalar loads of one descriptor the code keeps per lane in VGPRs
    /// ([`Tracer::lane_copies`]): they read one address, so their dwords
    /// stand in for each other.
    pub lane_copied: Vec<Site>,
    pub waterfall: Waterfall,
    /// Every `image_bvh_intersect_ray` and `image_bvh64_intersect_ray`: where
    /// its nodes are.
    pub bvh: Vec<(Site, BvhNodes)>,
}

/// Where a ray-tracing node test ([`crate::bvh`]) reads its nodes. Its
/// descriptor operand names a range of memory, not a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvhNodes {
    /// A buffer of the host: the descriptor is loaded from a table (it is
    /// among [`Resources::descriptors`]), and node pointers count from the
    /// buffer's start. Its flags are not known to the shader: boxes are
    /// sorted and triangles return barycentrics, as every descriptor
    /// the title builds asks.
    Resource,
    /// Title memory by address: the code makes the descriptor from numbers
    /// (the title: the whole address space, or one structure's address in
    /// dword 0). Dwords 1 and 3, which hold its flags.
    Memory { word1: u32, word3: u32 },
}

/// Waterfall loops over descriptors that differ between lanes: each pass
/// reads the first active lane's descriptor with `v_readfirstlane_b32`,
/// compares it with every lane's own (`v_cmp_eq`), and runs the access with
/// the lanes that match. Their accesses take each lane's own descriptor
/// ([`Source::Fetched`], [`Source::Lanes`]), so the first lane's dwords are
/// each lane's own too: the compares hold in every lane and one pass serves
/// all lanes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Waterfall {
    /// The compares, true in every lane.
    pub compares: Vec<Site>,
    /// The `v_readfirstlane_b32` results they compare, each the lane's own
    /// source dword.
    pub reads: Vec<Value>,
    /// Descriptor loads at a table offset among `reads` (the console
    /// compiler's form): each lane loads its own entry.
    pub loads: Vec<Site>,
}

/// Traces every descriptor and scalar address of `p`, which runs as
/// `stage`, to the root parameters in `layout`. Fails on the first one it
/// cannot trace. `lanes`, `p` as one lane runs it, tells which paths the
/// lanes that access a per-lane descriptor took and which writes they ran;
/// without it, per-lane descriptors chosen by control flow are refused and
/// per-lane writes are taken to run in every lane.
pub fn track(
    p: &Program,
    layout: &[RootParam],
    stage: HwStage,
    lanes: Option<&Structured>,
) -> Result<Resources, ShaderError> {
    let t = Tracer::new(p, layout, stage, lanes)?;
    let mut out = Resources {
        params: p
            .inputs
            .iter()
            .filter_map(|&(reg, v)| {
                let (param, dword) = t.param_of(reg)?;
                Some((v, param, dword))
            })
            .collect(),
        ..Resources::default()
    };
    // Group accesses in the same loop so a shared mask can select several
    // descriptors together.
    let mut first_lane_at = std::collections::BTreeMap::<usize, (Vec<usize>, Vec<Value>)>::new();
    // The console compiler's waterfall reads the first lane's table offset
    // and loads that entry: accesses, offset reads and loads by block.
    let mut offset_at =
        std::collections::BTreeMap::<usize, (Vec<usize>, Vec<Value>, Vec<Site>)>::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            t.at.set(Some(site));
            if let Some(value) = t.restored(inst, Some(b)) {
                out.restored_lanes.insert(site, value);
            }
            let fail = |what| ShaderError::Unsupported {
                offset: inst.machine.offset,
                what,
            };
            let mut first_lane = Vec::new();
            let mut offset = Vec::new();
            let mut descriptor = |op: &Operand, role| -> Result<(), ShaderError> {
                let source = t.descriptor(inst, op, site).map_err(fail)?;
                // A descriptor the code keeps per lane ([`Tracer::lane_copies`])
                // is the same in every lane: its loop is lowered as it stands
                // and needs no proof.
                if !t.kept_per_lane(inst, op) {
                    first_lane.extend(t.first_lane_words(inst, op).unwrap_or_default());
                }
                if let Source::Loaded { load, .. } = source
                    && out.scalar.iter().any(|s| s.0 == load)
                    && let Some(v) = t.first_lane_offset(load)
                {
                    offset.push((v, load));
                }
                out.descriptors.push(Descriptor { site, role, source });
                Ok(())
            };
            match &inst.fields {
                Fields::Image { rsrc, .. } if inst.name().starts_with("image_bvh") => {
                    // The ray and node test names no texture. A descriptor
                    // whose flag dwords the code sets as numbers and whose
                    // type is a ray-tracing structure is a range of title
                    // memory (the title: all of it, or from one structure's
                    // address on), and the node's address is among the
                    // operands; any other descriptor must be a resource of
                    // the host.
                    let word = |k: u8| match *rsrc {
                        Operand::Scalar { code, dwords } if k < dwords => {
                            inst.read(Reg::S(code + k)).map(|v| t.origin(v))
                        }
                        _ => None,
                    };
                    match (word(1), word(3)) {
                        (Some(Term::Const(word1)), Some(Term::Const(word3)))
                            if word3 >> 28 == crate::bvh::DESCRIPTOR_TYPE =>
                        {
                            out.bvh.push((site, BvhNodes::Memory { word1, word3 }));
                        }
                        _ => {
                            descriptor(rsrc, Role::Resource)?;
                            out.bvh.push((site, BvhNodes::Resource));
                        }
                    }
                }
                Fields::Image { rsrc, sampler, .. } => {
                    descriptor(rsrc, Role::Resource)?;
                    if let Some(s) = sampler {
                        descriptor(s, Role::Sampler)?;
                    }
                }
                // Cache invalidations name a buffer's address space but do
                // not read its descriptor; the host has no cache to drop.
                Fields::Buffer { rsrc, .. } if !inst.name().ends_with("_inv") => {
                    descriptor(rsrc, Role::Resource)?
                }
                Fields::Buffer { .. } => {}
                Fields::Smem { base, .. } if inst.name().starts_with("s_buffer_") => {
                    descriptor(base, Role::Resource)?;
                }
                Fields::Smem {
                    base,
                    offset,
                    soffset,
                    ..
                } if inst.name().contains("_load_") || inst.name().contains("_store_") => {
                    let address = t
                        .smem_address(inst, base, *offset, soffset)
                        .or_else(|why| t.by_address(why))
                        .map_err(fail)?;
                    out.scalar.push((site, address));
                }
                Fields::Flat {
                    addr,
                    saddr,
                    offset,
                    ..
                } if inst.name().starts_with("global_") => {
                    let address = t
                        .flat_address(inst, addr.as_ref(), saddr.as_ref(), *offset)
                        .or_else(|why| t.by_address(why))
                        .map_err(fail)?;
                    out.scalar.push((site, address));
                }
                _ => {}
            }
            if !first_lane.is_empty() {
                let (accesses, reads) = first_lane_at.entry(b).or_default();
                accesses.push(i);
                reads.extend(first_lane);
            }
            if !offset.is_empty() {
                let (accesses, reads, loads) = offset_at.entry(b).or_default();
                accesses.push(i);
                for (v, load) in offset {
                    reads.push(v);
                    loads.push(load);
                }
            }
        }
    }
    // What follows traces no access of one block.
    t.at.set(None);
    // The proof needs the per-lane form. Without it nothing is emitted, and
    // its failure is the one to report, so the reads stay unproven here.
    if t.lanes.is_none() {
        return Ok(out);
    }
    for (block, (accesses, mut reads)) in first_lane_at {
        reads.sort_unstable();
        reads.dedup();
        let proof = t
            .waterfall(block, &accesses, &reads)
            .map_err(|what| ShaderError::Unsupported {
                offset: t.inst((block, accesses[0])).machine.offset,
                what,
            })?;
        out.waterfall.compares.extend(proof.compares);
        out.waterfall.reads.extend(proof.reads);
    }
    // Unproven, an offset read stays the first lane's (one uniform entry),
    // which is what the hardware reads outside a waterfall loop.
    for (block, (accesses, mut reads, loads)) in offset_at {
        reads.sort_unstable();
        reads.dedup();
        if let Ok(proof) = t.waterfall(block, &accesses, &reads) {
            out.waterfall.compares.extend(proof.compares);
            out.waterfall.reads.extend(proof.reads);
            out.waterfall.loads.extend(loads);
        }
    }
    out.lane_copied = t.lane_copied.take();
    out.lane_copied.sort_unstable();
    out.lane_copied.dedup();
    out.root_indexed_views = out
        .descriptors
        .iter()
        .filter_map(|d| t.root_indexed_view(d, &out.scalar))
        .collect();
    Ok(out)
}

/// A dword after copy propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Term {
    Const(u32),
    Value(Value),
}

/// A 32-bit sum: optional root parameter (as its first dword), constant,
/// optional computed term.
struct Sum {
    base: Option<Value>,
    offset: u32,
    dynamic: Option<Value>,
}

struct Tracer<'a> {
    p: &'a Program,
    layout: &'a [RootParam],
    stage: HwStage,
    /// Defining instruction per value (phis and inputs have none).
    defs: Vec<Option<Site>>,
    /// Block and index of each phi value.
    phis: Vec<Option<(usize, usize)>>,
    /// Register per input value.
    inputs: Vec<Option<Reg>>,
    origin: Vec<Term>,
    /// See [`Resources::lane_copied`].
    lane_copied: std::cell::RefCell<Vec<Site>>,
    lanes: Option<&'a Structured>,
    /// While set, [`Self::sum`] keeps the first-lane read of an index as
    /// the computed term instead of what the lanes hold.
    read_first_lane: std::cell::Cell<bool>,
    /// What the conditions asked of `lanes` so far came down to: the proofs
    /// of one program ask about the same blocks and masks again and again.
    meets: std::cell::RefCell<Meets>,
    /// Reverse-postorder index per block.
    rpo: Vec<usize>,
    /// The instruction site of the access being traced ([`Self::live`]).
    at: std::cell::Cell<Option<Site>>,
}

/// What lanes are known to satisfy: each [`Bool`] where its flag is set,
/// its negation where not.
type Known = Vec<(Bool, bool)>;

const MAX_ROUNDS: usize = 64;
const MAX_JOINS: usize = 8;
/// VGPR values followed for one dword of a descriptor kept per lane.
const MAX_COPIES: usize = 64;

impl<'a> Tracer<'a> {
    fn new(
        p: &'a Program,
        layout: &'a [RootParam],
        stage: HwStage,
        lanes: Option<&'a Structured>,
    ) -> Result<Self, ShaderError> {
        let all = || {
            p.inputs
                .iter()
                .map(|i| i.1)
                .chain(p.blocks.iter().flat_map(|b| {
                    b.phis
                        .iter()
                        .map(|phi| phi.value)
                        .chain(b.insts.iter().flat_map(|i| i.writes.iter().map(|w| w.1)))
                }))
        };
        let n = all().map(|v| v.0 as usize + 1).max().unwrap_or(0);
        let mut defs = vec![None; n];
        let mut phis = vec![None; n];
        let mut inputs = vec![None; n];
        for &(reg, v) in &p.inputs {
            inputs[v.0 as usize] = Some(reg);
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phis[phi.value.0 as usize] = Some((b, i));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    defs[v.0 as usize] = Some((b, i));
                }
            }
        }
        let mut rpo = vec![usize::MAX; p.blocks.len()];
        for (n, b) in crate::ir::reverse_postorder(&p.blocks)
            .into_iter()
            .enumerate()
        {
            rpo[b] = n;
        }
        let mut t = Tracer {
            p,
            layout,
            stage,
            defs,
            phis,
            inputs,
            origin: Vec::new(),
            lane_copied: Default::default(),
            lanes,
            read_first_lane: Default::default(),
            meets: Default::default(),
            rpo,
            at: Default::default(),
        };
        t.origin = t.propagate_copies(n)?;
        Ok(t)
    }

    /// Optimistic copy propagation: a move's result is its source's
    /// origin, a phi's the one origin its arguments agree on.
    fn propagate_copies(&self, n: usize) -> Result<Vec<Term>, ShaderError> {
        let mut o: Vec<Option<Term>> = vec![None; n];
        for &(_, v) in &self.p.inputs {
            o[v.0 as usize] = Some(Term::Value(v));
        }
        let order = crate::ir::reverse_postorder(&self.p.blocks);
        for _ in 0..MAX_ROUNDS {
            let mut changed = false;
            for &b in &order {
                let block = &self.p.blocks[b];
                for phi in &block.phis {
                    let own = Term::Value(phi.value);
                    let mut args = phi
                        .args
                        .iter()
                        .filter_map(|a| o[a.0 as usize])
                        .filter(|&t| t != own);
                    let new = args
                        .next()
                        .map(|first| if args.all(|t| t == first) { first } else { own });
                    if new.is_some() && new != o[phi.value.0 as usize] {
                        o[phi.value.0 as usize] = new;
                        changed = true;
                    }
                }
                for inst in &block.insts {
                    for (k, &(_, v)) in inst.writes.iter().enumerate() {
                        let new = match copy_source(inst, k).or_else(|| self.restored(inst, None)) {
                            Some(Moved::Const(c)) => Some(Term::Const(c)),
                            Some(Moved::Value(s)) => o[s.0 as usize],
                            None => Some(Term::Value(v)),
                        };
                        if new.is_some() && new != o[v.0 as usize] {
                            o[v.0 as usize] = new;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                return Ok(o
                    .into_iter()
                    .enumerate()
                    .map(|(v, t)| t.unwrap_or(Term::Value(Value(v as u32))))
                    .collect());
            }
        }
        Err(ShaderError::Unsupported {
            offset: 0,
            what: "copies that do not settle",
        })
    }

    /// Whether some lane can satisfy all of `conditions` ([`Bools::meet`]).
    ///
    /// [`Bools::meet`]: crate::structure::Bools::meet
    fn meet(&self, lanes: &Structured, conditions: &[(Bool, bool)]) -> bool {
        lanes
            .bools
            .meet_seen(conditions, &mut self.meets.borrow_mut())
    }

    fn inst(&self, (b, i): Site) -> &'a Inst {
        &self.p.blocks[b].insts[i]
    }

    fn def(&self, v: Value) -> Option<&'a Inst> {
        self.defs[v.0 as usize].map(|s| self.inst(s))
    }

    fn origin(&self, v: Value) -> Term {
        self.origin[v.0 as usize]
    }

    /// The origin of a one-dword operand as `inst` reads it.
    fn term(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
            Operand::Scalar { code, .. } => Some(self.origin(inst.read(Reg::S(code))?)),
            Operand::Const(c) => inline(c).map(Term::Const),
            Operand::Literal(x) => Some(Term::Const(x)),
            _ => None,
        }
    }

    /// Layout index and dword of the root parameter `reg` holds on entry.
    fn param_of(&self, reg: Reg) -> Option<(usize, u32)> {
        let Reg::S(code) = reg else { return None };
        self.layout.iter().enumerate().find_map(|(i, p)| {
            let first = self.stage.user_sgpr(p.user_data);
            let dword = u32::from(code).checked_sub(first.into())?;
            (dword < p.sgprs).then_some((i, dword))
        })
    }

    /// The SOP2 operands of `v` if an instruction with one of `names` wrote it.
    fn sop2(&self, v: Value, names: &[&str]) -> Option<(&'a Inst, [Term; 2])> {
        let inst = self.def(v)?;
        if !names.contains(&inst.name()) || inst.writes.first()?.1 != v {
            return None;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        Some((
            inst,
            [
                self.term(inst, src.first()?)?,
                self.term(inst, src.get(1)?)?,
            ],
        ))
    }

    fn root_indexed_view(
        &self,
        descriptor: &Descriptor,
        scalar: &[(Site, Address)],
    ) -> Option<RootIndexedView> {
        if descriptor.role != Role::Resource {
            return None;
        }
        let Source::Loaded { load, first } = descriptor.source else {
            return None;
        };
        let i = scalar.binary_search_by_key(&load, |(site, _)| *site).ok()?;
        let address = scalar[i].1;
        let base = self.layout.get(address.param)?;
        if base.root != RootType::Table {
            return None;
        }
        let dynamic = address.dynamic?;
        let (input, shift) = match self.sop2(dynamic, &["s_lshl_b32"]) {
            Some((_, [Term::Value(input), Term::Const(shift)])) => (input, (shift & 31) as u8),
            _ => (dynamic, 0),
        };
        let (param, constant_dword) = self.param_of(self.inputs[input.0 as usize]?)?;
        let constant = self.layout.get(param)?;
        if constant.root != RootType::Constants {
            return None;
        }
        Some(RootIndexedView {
            table_slot: base.slot,
            constant_slot: constant.slot,
            constant_dword,
            byte_offset: address.offset.wrapping_add(u32::from(first) * 4),
            shift,
        })
    }

    fn sum(&self, t: Term) -> Sum {
        let v = match t {
            Term::Const(c) => {
                return Sum {
                    base: None,
                    offset: c,
                    dynamic: None,
                };
            }
            Term::Value(v) => v,
        };
        let dynamic = Sum {
            base: None,
            offset: 0,
            dynamic: Some(v),
        };
        if let Some(reg) = self.inputs[v.0 as usize] {
            return match self.param_of(reg) {
                Some((p, 0)) if self.layout[p].root != RootType::Constants => Sum {
                    base: Some(v),
                    offset: 0,
                    dynamic: None,
                },
                _ => dynamic,
            };
        }
        if self.getpc(v).is_some() {
            return Sum {
                base: Some(v),
                offset: 0,
                dynamic: None,
            };
        }
        // Paths that each add an index to a table in the code and meet before one
        // load: the join itself is the computed term. Lowering gives every
        // `s_getpc_b64` its own offset in the code as its value, so the join is the
        // byte in the code whichever table a path took ([`Self::address`] adds no
        // base to it).
        if let Some(pc) = self.code_join(v) {
            return Sum {
                base: Some(pc),
                offset: 0,
                dynamic: Some(v),
            };
        }
        // An address computed in VGPRs and taken from the first lane (a
        // table in the code indexed by a value held per lane, or a table
        // offset held per lane, the console's waterfall): the sum of what
        // the lanes hold. Loads at one offset in different loops (each with
        // its own `v_readfirstlane_b32`) then have one address, which is
        // what tells lowering that they read one descriptor. With
        // `read_first_lane` a first-lane read of an index alone stays the
        // computed term ([`Self::first_lane_offset`]).
        if let Some(lane) = self.first_lane(v) {
            let per_lane = self.sum(self.vorigin(lane));
            return if per_lane.base.is_some() || !self.read_first_lane.get() {
                per_lane
            } else {
                dynamic
            };
        }
        let Some([a, b]) = self
            .sop2(v, &["s_add_u32", "s_add_i32"])
            .map(|(_, t)| t)
            .or_else(|| self.vadd(v))
        else {
            return dynamic;
        };
        let (x, y) = (self.sum(a), self.sum(b));
        if x.base.is_some() && y.base.is_some() || x.dynamic.is_some() && y.dynamic.is_some() {
            return dynamic;
        }
        Sum {
            base: x.base.or(y.base),
            offset: x.offset.wrapping_add(y.offset),
            dynamic: x.dynamic.or(y.dynamic),
        }
    }

    /// For a join `v` whose every incoming value adds to an `s_getpc_b64` low
    /// dword: the first of those (one level of joins only). The paths may add
    /// to different ones: two tables in the code, one picked by control flow
    /// (the title's fog reconstruction reads its sample offsets from
    /// one of two tables, by the sample's number; refused, the lit frame was
    /// multiplied by a fog transmittance nothing had written).
    fn code_join(&self, v: Value) -> Option<Value> {
        let (b, i) = self.phis[v.0 as usize]?;
        let mut base = None;
        for &a in &self.p.blocks[b].phis[i].args {
            let Term::Value(x) = self.vorigin(a) else {
                return None;
            };
            if x == v {
                continue;
            }
            let pc = self.code_base(x, 4)?;
            base = base.or(Some(pc));
        }
        base
    }

    /// `t` is the high dword of an `s_getpc_b64`, or `s_addc_u32`s of one:
    /// the carry may be added in steps (the counter's high dword plus 0
    /// where the table's offset is added, then plus the index's high dword).
    fn code_high(&self, t: Term) -> bool {
        self.code_high_in(t, 4)
    }

    /// [`Self::code_high`], at most `depth` additions deep.
    fn code_high_in(&self, t: Term, depth: u32) -> bool {
        let Term::Value(x) = t else { return false };
        if self.def(x).is_some_and(|i| {
            i.name() == "s_getpc_b64" && i.writes.get(1).is_some_and(|w| w.1 == x)
        }) {
            return true;
        }
        depth > 0
            && self.sop2(x, &["s_addc_u32"]).is_some_and(|(_, [p, q])| {
                self.code_high_in(p, depth - 1) || self.code_high_in(q, depth - 1)
            })
    }

    /// The `s_getpc_b64` low dword among the terms `v` adds up (`depth`
    /// additions deep; joins are not followed).
    fn code_base(&self, v: Value, depth: u32) -> Option<Value> {
        if self.getpc(v).is_some() {
            return Some(v);
        }
        // Computed in VGPRs and taken from the first lane.
        if let Some(lane) = self.first_lane(v) {
            return match self.vorigin(lane) {
                Term::Value(x) => self.code_base(x, depth),
                Term::Const(_) => None,
            };
        }
        if depth == 0 || self.phis[v.0 as usize].is_some() {
            return None;
        }
        let terms = self
            .sop2(v, &["s_add_u32", "s_add_i32"])
            .map(|(_, t)| t)
            .or_else(|| self.vadd(v))?;
        terms.into_iter().find_map(|t| match t {
            Term::Value(x) => self.code_base(x, depth - 1),
            Term::Const(_) => None,
        })
    }

    /// `t` is `high` or an `s_addc_u32` of it.
    fn is_high(&self, t: Term, high: Value) -> bool {
        match t {
            Term::Value(x) => {
                x == high
                    || self
                        .sop2(x, &["s_addc_u32"])
                        .is_some_and(|(_, [p, q])| p == Term::Value(high) || q == Term::Value(high))
            }
            Term::Const(_) => false,
        }
    }

    /// The VGPR value `v_readfirstlane_b32` reads to write the scalar `v`.
    fn first_lane(&self, v: Value) -> Option<Value> {
        let inst = self.def(v)?;
        if inst.name() != "v_readfirstlane_b32" || inst.writes.first()?.1 != v {
            return None;
        }
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if f.abs != 0 || f.neg != 0 || f.sdwa.is_some() || f.dpp.is_some() {
            return None;
        }
        match *f.src.first()? {
            Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
            _ => None,
        }
    }

    /// The `s_getpc_b64` writing `lo` as its low dword.
    fn getpc(&self, lo: Value) -> Option<&'a Inst> {
        let inst = self.def(lo)?;
        (inst.name() == "s_getpc_b64" && inst.writes.first().is_some_and(|w| w.1 == lo))
            .then_some(inst)
    }

    /// The operands of a VALU addition (`v_add_co_u32`, `v_add_nc_u32`)
    /// writing `v`, through VGPR copies of scalars.
    fn vadd(&self, v: Value) -> Option<[Term; 2]> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if !matches!(inst.name(), "v_add_co_u32" | "v_add_nc_u32")
            || f.dst.is_none_or(|d| !matches!(d, Operand::Vector { .. }))
            || inst.write(Reg::V(match f.dst? {
                Operand::Vector { reg, .. } => reg,
                _ => return None,
            })) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        Some([
            self.vterm(inst, f.src.first()?)?,
            self.vterm(inst, f.src.get(1)?)?,
        ])
    }

    /// The origin of a VALU operand.
    fn vterm(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Vector { reg, .. } => Some(self.vorigin(inst.read(Reg::V(reg))?)),
            op => self.term(inst, &op),
        }
    }

    /// A VGPR value's origin through `v_mov_b32` copies.
    fn vorigin(&self, v: Value) -> Term {
        if let Some(inst) = self.def(v)
            && inst.name() == "v_mov_b32"
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && let Some(t) = f.src.first().and_then(|s| self.vterm(inst, s))
        {
            return t;
        }
        self.origin(v)
    }

    /// The address of an access that traces to no root parameter (`why`
    /// says how it failed): title memory at the address as it stands, in a
    /// compute program.
    fn by_address(&self, why: &'static str) -> Result<Address, &'static str> {
        if self.stage == HwStage::Cs && RAW.contains(&why) {
            Ok(Address {
                param: MEMORY,
                offset: 0,
                dynamic: None,
            })
        } else {
            Err(why)
        }
    }

    /// A global memory address: an SGPR pair plus a VGPR offset, or a VGPR
    /// pair.
    fn flat_address(
        &self,
        inst: &Inst,
        addr: Option<&Operand>,
        saddr: Option<&Operand>,
        offset: i16,
    ) -> Result<Address, &'static str> {
        let Some(&Operand::Vector { reg, .. }) = addr else {
            return Err("a global address not in VGPRs");
        };
        let off = offset as i32 as u32;
        match saddr {
            Some(&Operand::Scalar { code, .. }) if code != NULL => {
                let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("an address not in SGPRs");
                };
                let v = inst.read(Reg::V(reg)).ok_or("a global address")?;
                let mut extra = self.sum(self.vorigin(v));
                if extra.base.is_some() {
                    return Err("an offset from a root parameter");
                }
                extra.offset = extra.offset.wrapping_add(off);
                self.address(lo, hi, extra)
            }
            _ => {
                let word = |k| inst.read(Reg::V(reg + k)).map(|v| self.vorigin(v));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("a global address");
                };
                self.address(
                    lo,
                    hi,
                    Sum {
                        base: None,
                        offset: off,
                        dynamic: None,
                    },
                )
            }
        }
    }

    /// The address `lo` and `hi` hold, plus `extra` (an instruction's
    /// immediate and offset register).
    fn address(&self, lo: Term, hi: Term, extra: Sum) -> Result<Address, &'static str> {
        // The high half as the lanes that read through the address hold
        // it, where that is a constant or what a register held on entry
        // (the code clears a table address's high half inside a branch
        // and uses the address after the join).
        let hi = match self.live(hi) {
            t @ Term::Const(_) => t,
            Term::Value(v) if self.inputs[v.0 as usize].is_some() => Term::Value(v),
            _ => hi,
        };
        let s = self.sum(lo);
        let base = s.base.ok_or_else(|| self.why_no_base(s.dynamic))?;
        if s.dynamic.is_some() && extra.dynamic.is_some() {
            return Err("an address with two computed terms");
        }
        if let Some(pc) = self.getpc(base) {
            // A table in the code: the base is the next instruction's offset.
            let high = pc.writes.get(1).map(|w| w.1);
            if !self.high_half(hi, high) && !high.is_some_and(|h| self.carried(hi, h)) {
                return Err("an address with an untracked high half");
            }
            // A join of paths into tables in the code is the byte in the code
            // already: each path's `s_getpc_b64` counts as its own offset.
            let joined = s.dynamic.is_some_and(|d| self.code_join(d).is_some());
            let first = if joined { 0 } else { pc.machine.offset as u32 + 4 };
            return Ok(Address {
                param: CODE,
                offset: first.wrapping_add(s.offset).wrapping_add(extra.offset),
                dynamic: s.dynamic.or(extra.dynamic),
            });
        }
        let reg = self.inputs[base.0 as usize].ok_or("an address not from a root parameter")?;
        let (param, _) = self
            .param_of(reg)
            .ok_or("an address not from a root parameter")?;
        let high = match self.layout[param].root {
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap => None,
            RootType::Descriptor => {
                let high = Reg::S(self.stage.user_sgpr(self.layout[param].user_data) + 1);
                let input = self.p.inputs.iter().find(|i| i.0 == high);
                Some(input.ok_or("a root descriptor without its high half")?.1)
            }
            _ => return Err("an address from a root parameter that is no address"),
        };
        // shortcut: with a computed term, a high half that adds anything to the
        // base's is taken as the carry of that term (offsets below 4 GB).
        let dynamic = s.dynamic.or(extra.dynamic);
        if !self.high_half(hi, high)
            && !(dynamic.is_some() && high.is_some_and(|h| self.carried(hi, h)))
        {
            return Err("an address with an untracked high half");
        }
        Ok(Address {
            param,
            offset: s.offset.wrapping_add(extra.offset),
            dynamic: s.dynamic.or(extra.dynamic),
        })
    }

    /// `hi` is `high` (0 for a table), possibly plus a carry.
    fn high_half(&self, hi: Term, high: Option<Value>) -> bool {
        let want = high.map_or(Term::Const(0), Term::Value);
        hi == want
            || matches!(hi, Term::Value(v) if self.sop2(v, &["s_addc_u32"]).is_some_and(
                |(_, [a, b])| a == want && b == Term::Const(0) || b == want && a == Term::Const(0)
            ))
    }

    /// `hi` adds something (with carry) to `high`: `s_addc_u32` or
    /// `v_add_co_ci_u32` with `high` as an operand.
    fn carried(&self, hi: Term, high: Value) -> bool {
        let Term::Value(v) = hi else { return false };
        if v == high {
            return true;
        }
        // A join ([`Self::code_join`]): every path carries into the table's high half.
        if let Some((b, i)) = self.phis[v.0 as usize] {
            // A path may carry into another table's high half ([`Self::code_join`]).
            let high_of = |t: Term| self.is_high(t, high) || self.code_high(t);
            return self.p.blocks[b].phis[i].args.iter().all(|&a| {
                let t = self.vorigin(a);
                if t == Term::Value(v) || high_of(t) {
                    return true;
                }
                let Term::Value(x) = t else {
                    return false;
                };
                // Computed in VGPRs and taken from the first lane.
                let x = match self.first_lane(x).map(|lane| self.vorigin(lane)) {
                    Some(Term::Value(y)) => y,
                    Some(Term::Const(_)) => return false,
                    None => x,
                };
                let Some(inst) = self.def(x) else {
                    return false;
                };
                let Fields::Valu(f) = &inst.fields else {
                    return false;
                };
                inst.name() == "v_add_co_ci_u32"
                    && f.src[..2.min(f.src.len())]
                        .iter()
                        .any(|o| self.vterm(inst, o).is_some_and(high_of))
            });
        }
        if let Some(lane) = self.first_lane(v) {
            return self.carried(self.vorigin(lane), high);
        }
        if let Some((_, [a, b])) = self.sop2(v, &["s_addc_u32"]) {
            return self.carried(a, high) || self.carried(b, high);
        }
        let Some(inst) = self.def(v) else {
            return false;
        };
        let Fields::Valu(f) = &inst.fields else {
            return false;
        };
        inst.name() == "v_add_co_ci_u32"
            && f.src[..2.min(f.src.len())]
                .iter()
                .any(|o| self.vterm(inst, o).is_some_and(|t| self.carried(t, high)))
    }

    fn why_no_base(&self, dynamic: Option<Value>) -> &'static str {
        let Some(v) = dynamic else {
            return "a constant address";
        };
        if let Some(reg) = self.inputs[v.0 as usize] {
            return if self.param_of(reg).is_some() {
                "an address from a root parameter that is no address"
            } else {
                "an address in an SGPR that is no root parameter"
            };
        }
        if self
            .leaves(v)
            .any(|l| self.def(l).is_some_and(|i| i.name() == "s_getpc_b64"))
        {
            return "an address relative to the program counter";
        }
        match self.def(v).map(|i| i.name()) {
            Some(n) if n.starts_with("s_load_") => "an address loaded from memory",
            None => "an address from a phi",
            _ => "an address not from a root parameter",
        }
    }

    /// The terms of the additions that compute `v`.
    fn leaves(&self, v: Value) -> Box<dyn Iterator<Item = Value> + '_> {
        match self.sop2(v, &["s_add_u32", "s_add_i32"]) {
            Some((_, terms)) => Box::new(terms.into_iter().flat_map(move |t| match t {
                Term::Value(x) => self.leaves(x),
                Term::Const(_) => Box::new(std::iter::empty()),
            })),
            None => Box::new(std::iter::once(v)),
        }
    }

    fn smem_address(
        &self,
        inst: &Inst,
        base: &Operand,
        offset: i32,
        soffset: &Operand,
    ) -> Result<Address, &'static str> {
        let Operand::Scalar { code, .. } = *base else {
            return Err("an address not in SGPRs");
        };
        let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
        let (Some(lo), Some(hi)) = (word(0), word(1)) else {
            return Err("an address not in SGPRs");
        };
        let mut extra = self.sum(
            self.term(inst, soffset)
                .ok_or("an unknown offset operand")?,
        );
        if extra.base.is_some() {
            return Err("an offset from a root parameter");
        }
        extra.offset = extra.offset.wrapping_add(offset as u32);
        self.address(lo, hi, extra)
    }

    /// The source of the descriptor operand `op` of `inst`, at `at`.
    fn descriptor(&self, inst: &Inst, op: &Operand, at: Site) -> Result<Source, &'static str> {
        let Operand::Scalar { code, dwords } = *op else {
            return Err("a descriptor not in SGPRs");
        };
        let mut words = (0..dwords)
            .map(|k| inst.read(Reg::S(code + k)).map(|v| self.origin(v)))
            .collect::<Option<Vec<_>>>()
            .ok_or("a descriptor not in SGPRs")?;
        // A byte access through a buffer descriptor made raw reads what the
        // descriptor's view does with a raw host view (VIctory upstream 0b35b7f).
        if !inst.name().contains("format")
            && let Some(raw) = self.made_raw(&words)
        {
            words = raw.to_vec();
        }
        // Only the lanes that reach and run the access read the descriptor.
        let mut known = Vec::new();
        if let Some(l) = self.lanes {
            known.push((l.reach[at.0], true));
            if let Some(guard) = l.guards[at.0][at.1] {
                known.push((guard, true));
            }
        }
        self.classify(&words, &mut Vec::new(), &mut known)
    }

    /// The source of a descriptor's words; `joins` are the phi blocks
    /// being resolved, `known` holds in every lane that reads them.
    fn classify(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Source, &'static str> {
        if let Some(source) = self.phi(words, joins, known)? {
            return Ok(source);
        }
        if let Some(source) = self.select(words, joins, known)? {
            return Ok(source);
        }
        let read = |w: &Term| match *w {
            Term::Value(v) => self.first_lane_source(v),
            Term::Const(_) => None,
        };
        if let Some(sources) = words.iter().map(read).collect::<Option<Vec<_>>>() {
            // One descriptor the code keeps per lane: the lanes agree.
            if let Some(source) = self.lane_copies(words) {
                return Ok(source);
            }
            // Read from the first active lane, as a waterfall loop does: the
            // dwords each lane holds.
            let lanes: Vec<Term> = sources.into_iter().map(Term::Value).collect();
            return self.per_lane(&lanes, joins, known);
        }
        if words.iter().any(|w| read(w).is_some()) {
            return Err("a descriptor partly read from the first lane");
        }
        if let Some(consts) = words
            .iter()
            .map(|w| match *w {
                Term::Const(c) => Some(c),
                Term::Value(_) => None,
            })
            .collect::<Option<Vec<_>>>()
        {
            return Ok(Source::Constant(consts));
        }
        if let Some(source) = self.loaded(words) {
            return Ok(source);
        }
        if let [lo, hi, Term::Const(w2), Term::Const(w3)] = *words {
            let (mut hi, mut bits) = (hi, 0);
            while let Term::Value(v) = hi
                && let Some((x, more)) = self.set_bits(v)
            {
                (hi, bits) = (x, bits | more);
            }
            let address = self.address(
                lo,
                hi,
                Sum {
                    base: None,
                    offset: 0,
                    dynamic: None,
                },
            )?;
            return Ok(Source::Built {
                address,
                bits,
                words: [w2, w3],
            });
        }
        let loaded = |w: &Term| {
            matches!(*w, Term::Value(v)
                if self.def(v).is_some_and(|i| i.name().starts_with("s_load_")))
        };
        Err(if words.iter().any(loaded) {
            "a descriptor changed after loading"
        } else {
            "a descriptor neither loaded nor built"
        })
    }

    /// The one edge into the join `block` that lanes satisfying `known`
    /// can have taken, if the lanes' paths ([`Structured`]) rule out every
    /// other: the console compiler loads a descriptor (or clears an
    /// address's high half) inside a branch that skips code when no lane
    /// is left for it, and uses it after the join, where only lanes that
    /// ran the branch are left (the title's
    /// `CS_Alpha_Schedule`: the registers' values from before the branch,
    /// which the join also sees, are what a wave with no such lane holds).
    /// None at loop headers and without the lanes' paths.
    fn only_edge(&self, block: usize, known: &mut Known) -> Option<usize> {
        let lanes = self.lanes?;
        let preds = &self.p.blocks[block].preds;
        if preds.len() < 2 || preds.iter().any(|&q| self.rpo[q] >= self.rpo[block]) {
            return None;
        }
        let mut only = None;
        for (j, &q) in preds.iter().enumerate() {
            let n = known.len();
            known.push((lanes.reach[q], true));
            if let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = lanes.exits[q]
                && taken != not_taken
            {
                known.push((when, taken == block));
            }
            let possible = self.meet(lanes, known);
            known.truncate(n);
            if possible {
                if only.is_some() {
                    return None;
                }
                only = Some(j);
            }
        }
        only
    }

    /// Whether lowering can make a handle for `source` in any block: its
    /// heap index comes from root parameters and constants alone (a table
    /// offset the code computes exists only where the load is).
    fn anywhere(&self, source: &Source) -> bool {
        match source {
            Source::Constant(_) => true,
            Source::Built { address, .. } => address.dynamic.is_none(),
            Source::Loaded { load, .. } => {
                let inst = self.inst(*load);
                match &inst.fields {
                    Fields::Smem {
                        base,
                        offset,
                        soffset,
                        ..
                    } => self
                        .smem_address(inst, base, *offset, soffset)
                        .is_ok_and(|a| a.dynamic.is_none()),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// `t` as the lanes that reach the access being traced hold it: through
    /// joins they can have entered by one edge only ([`Self::only_edge`]).
    fn live(&self, mut t: Term) -> Term {
        let (Some(at), Some(lanes)) = (self.at.get(), self.lanes) else {
            return t;
        };
        let mut known = vec![(lanes.reach[at.0], true)];
        if let Some(guard) = lanes.guards[at.0][at.1] {
            known.push((guard, true));
        }
        for _ in 0..MAX_JOINS {
            let Term::Value(v) = t else { break };
            let Some((block, i)) = self.phis[v.0 as usize] else {
                break;
            };
            let Some(j) = self.only_edge(block, &mut known) else {
                break;
            };
            t = self.origin(self.p.blocks[block].phis[i].args[j]);
        }
        t
    }

    /// Words that phis of one block join, traced per predecessor (words
    /// defined before the join hold on every edge).
    fn phi(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let phi_of = |w: &Term| match *w {
            Term::Value(v) => self.phis[v.0 as usize],
            Term::Const(_) => None,
        };
        let Some((block, _)) = words.iter().find_map(phi_of) else {
            return Ok(None);
        };
        if words.iter().filter_map(phi_of).any(|(b, _)| b != block) {
            return Err("a descriptor joined in several blocks");
        }
        if joins.contains(&block) {
            return Err("a descriptor that changes in a loop");
        }
        // Each level multiplies the sources by the predecessor count.
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(block);
        let phis = &self.p.blocks[block].phis;
        let edge = |j: usize| -> Vec<Term> {
            words
                .iter()
                .map(|w| match phi_of(w) {
                    Some((_, i)) => self.origin(phis[i].args[j]),
                    None => *w,
                })
                .collect()
        };
        let args = (0..self.p.blocks[block].preds.len())
            .map(|j| self.classify(&edge(j), joins, known))
            .collect::<Result<_, _>>();
        let source = match args {
            Ok(args) => Ok(Source::Phi { block, args }),
            // An edge holds no descriptor. With only one edge the lanes
            // that read the descriptor can have come by, it is what the
            // registers hold on that one (a wave that took another edge
            // has no lane left that reads it), if that can be named where
            // it is read.
            Err(e) => match self.only_edge(block, known) {
                Some(j) => match self.classify(&edge(j), joins, known) {
                    Ok(source) if self.anywhere(&source) => Ok(source),
                    _ => Err(e),
                },
                None => Err(e),
            },
        };
        joins.pop();
        source.map(Some)
    }

    /// Words that `s_cselect` (or `s_cmov`) picks by one SCC value, traced
    /// per outcome (other words are the same for both).
    fn select(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let picks: Vec<_> = words
            .iter()
            .map(|w| match *w {
                Term::Value(v) => self.picked(v),
                Term::Const(_) => None,
            })
            .collect();
        let Some(scc) = picks.iter().flatten().map(|p| p.0).next() else {
            return Ok(None);
        };
        if picks.iter().flatten().any(|p| p.0 != scc) {
            return Err("a descriptor chosen by several conditions");
        }
        // Each level doubles the sources; selects count like joins (a
        // marker no block index reaches).
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(usize::MAX);
        let side = |set: bool| -> Vec<Term> {
            words
                .iter()
                .zip(&picks)
                .map(|(w, p)| match p {
                    Some((_, t, f)) => {
                        if set {
                            *t
                        } else {
                            *f
                        }
                    }
                    None => *w,
                })
                .collect()
        };
        let args = [true, false]
            .into_iter()
            .map(|set| self.classify(&side(set), joins, known))
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Select { scc, args: args? }))
    }

    /// For `v` written by `s_cselect_b32`/`_b64` or `s_cmov_b32`/`_b64`:
    /// the SCC it reads and the dword it is when SCC is set, then clear.
    fn picked(&self, v: Value) -> Option<(Value, Term, Term)> {
        let inst = self.def(v)?;
        let Fields::Sop { dst, src, .. } = &inst.fields else {
            return None;
        };
        let k = inst.writes.iter().position(|w| w.1 == v)? as u8;
        let wide = match inst.name() {
            "s_cselect_b32" | "s_cmov_b32" => false,
            "s_cselect_b64" | "s_cmov_b64" => true,
            _ => return None,
        };
        // A dword of a 32- or 64-bit operand: 64-bit inline integers are
        // sign-extended; 64-bit literals and float constants are not
        // followed.
        let dword = |op: &Operand| match *op {
            Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
            Operand::Scalar { code, .. } => Some(self.origin(inst.read(Reg::S(code + k))?)),
            Operand::Const(c) if !wide => inline(c).map(Term::Const),
            Operand::Const(c @ 128..=208) => {
                let low = inline(c)?;
                Some(Term::Const(if k == 0 {
                    low
                } else {
                    ((low as i32) >> 31) as u32
                }))
            }
            Operand::Literal(x) if !wide => Some(Term::Const(x)),
            _ => None,
        };
        let set = dword(src.first()?)?;
        let clear = if inst.name().starts_with("s_cmov") {
            dword(dst.as_ref()?)?
        } else {
            dword(src.get(1)?)?
        };
        Some((inst.read(Reg::Scc)?, set, clear))
    }

    /// `v` as some other dword with constant bits set.
    fn set_bits(&self, v: Value) -> Option<(Term, u32)> {
        if let Some((_, [x, Term::Const(bits)] | [Term::Const(bits), x])) =
            self.sop2(v, &["s_or_b32"])
        {
            return Some((x, bits));
        }
        let inst = self.def(v)?;
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            src,
            ..
        } = &inst.fields
        else {
            return None;
        };
        match (inst.name(), self.term(inst, src.first()?)?) {
            ("s_bitset1_b32", Term::Const(bit)) => {
                Some((self.origin(inst.read(Reg::S(*code))?), 1 << (bit & 31)))
            }
            _ => None,
        }
    }

    /// The words of a buffer descriptor that `words` makes raw, as the
    /// FidelityFX upscaler's compiler does before byte accesses: the stride
    /// (dword 1 bits 16-29) cleared, the record count times max(stride, 1),
    /// out-of-bounds checking by bytes (dword 3 bits 28-29 set). The bounds
    /// are then the view's size in bytes either way.
    fn made_raw(&self, words: &[Term]) -> Option<[Term; 4]> {
        let [w0, Term::Value(w1), Term::Value(w2), Term::Value(w3)] = *words else {
            return None;
        };
        // `x op c` with `c` either side.
        let with = |v: Value, name: &str, c: u32| match self.sop2(v, &[name])?.1 {
            [x, Term::Const(k)] | [Term::Const(k), x] if k == c => Some(x),
            _ => None,
        };
        let d1 = with(w1, "s_and_b32", 0xC000_FFFF)?;
        let Term::Value(m3) = with(w3, "s_or_b32", 0x3000_0000)? else {
            return None;
        };
        let d3 = with(m3, "s_and_b32", 0xCFFF_FFFF)?;
        let [a, b] = self.sop2(w2, &["s_mul_i32"])?.1;
        let max = |t: Term| match t {
            Term::Value(s) => with(s, "s_max_u32", 1),
            Term::Const(_) => None,
        };
        let (d2, field) = match (max(a), max(b)) {
            (_, Some(f)) => (a, f),
            (Some(f), None) => (b, f),
            (None, None) => return None,
        };
        let Term::Value(field) = field else {
            return None;
        };
        // s_bfe_u32: offset 16, width 14.
        let (_, [x, Term::Const(0x000E_0010)]) = self.sop2(field, &["s_bfe_u32"])? else {
            return None;
        };
        (x == d1).then_some([w0, d1, d2, d3])
    }


    /// For a `v_readfirstlane_b32` result: the VGPR value it reads.
    fn first_lane_source(&self, v: Value) -> Option<Value> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_readfirstlane_b32" || f.sdwa.is_some() || f.dpp.is_some() {
            return None;
        }
        let Some(&Operand::Vector { reg, .. }) = f.src.first() else {
            return None;
        };
        inst.read(Reg::V(reg))
    }

    /// The `v_readfirstlane_b32` result the scalar load at `load` adds to
    /// its address as a table offset (the console's waterfall,
    /// `s_load_dwordx8 .., s[4:5], s20`): `track` proves the waterfall from
    /// it, and lowering reads it per lane there.
    fn first_lane_offset(&self, load: Site) -> Option<Value> {
        let inst = self.inst(load);
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            return None;
        };
        self.read_first_lane.set(true);
        let address = self.smem_address(inst, base, *offset, soffset);
        self.read_first_lane.set(false);
        match self.origin(address.ok()?.dynamic?) {
            Term::Value(v) if self.first_lane_source(v).is_some() => Some(v),
            _ => None,
        }
    }

    /// Whether the descriptor operand `op` of `inst` is one the code keeps
    /// per lane ([`Self::lane_copies`]).
    fn kept_per_lane(&self, inst: &Inst, op: &Operand) -> bool {
        let Operand::Scalar { code, dwords } = *op else {
            return false;
        };
        (0..dwords)
            .map(|k| inst.read(Reg::S(code + k)).map(|v| self.origin(v)))
            .collect::<Option<Vec<_>>>()
            .is_some_and(|words| self.lane_copies(&words).is_some())
    }

    /// The `v_readfirstlane_b32` results the descriptor operand `op` of
    /// `inst` is, when every dword is one.
    fn first_lane_words(&self, inst: &Inst, op: &Operand) -> Option<Vec<Value>> {
        let Operand::Scalar { code, dwords } = *op else {
            return None;
        };
        (0..dwords)
            .map(|k| match self.origin(inst.read(Reg::S(code + k))?) {
                Term::Value(v) => self.first_lane_source(v).map(|_| v),
                Term::Const(_) => None,
            })
            .collect()
    }

    /// `t` through VGPR copies (of VGPRs or of scalars) and copies.
    fn lane_term(&self, mut t: Term) -> Term {
        for _ in 0..MAX_JOINS * 2 {
            let Term::Value(v) = t else { break };
            let next = self.vorigin(v);
            if next == t {
                break;
            }
            t = next;
        }
        t
    }

    /// Whether `t` is a value each lane holds its own of (one in VGPRs).
    fn varies(&self, t: Term) -> bool {
        let Term::Value(v) = t else { return false };
        let reg = match (self.inputs[v.0 as usize], self.phis[v.0 as usize]) {
            (Some(reg), _) => Some(reg),
            (None, Some((b, i))) => Some(self.p.blocks[b].phis[i].reg),
            (None, None) => self
                .def(v)
                .and_then(|i| i.writes.iter().find(|w| w.1 == v))
                .map(|w| w.0),
        };
        matches!(reg, Some(Reg::V(_)))
    }

    /// [`Self::lane_term`], where every lane in `known` runs each VGPR copy
    /// it follows.
    fn lane_value(&self, mut t: Term, known: &Known) -> Result<Term, &'static str> {
        for _ in 0..MAX_JOINS * 2 {
            let Term::Value(v) = t else { break };
            let next = self.vstep(v);
            if next == t {
                break;
            }
            if self.varies(t)
                && let Some(site) = self.defs[v.0 as usize]
            {
                self.runs(site, known)?;
            }
            t = next;
        }
        Ok(t)
    }

    /// One copy of [`Self::vorigin`]: what the `v_mov_b32` writing `v`
    /// reads, or `v`'s origin.
    fn vstep(&self, v: Value) -> Term {
        if let Some(inst) = self.def(v)
            && inst.name() == "v_mov_b32"
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && let Some(t) = f.src.first().and_then(|s| self.raw(inst, s))
        {
            return t;
        }
        self.origin(v)
    }

    /// A VALU operand as `inst` reads it: a VGPR's value itself, other
    /// operands' origins.
    fn raw(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Vector { reg, .. } => Some(Term::Value(inst.read(Reg::V(reg))?)),
            op => self.term(inst, &op),
        }
    }

    /// Fails unless every lane in `known` that reaches the vector
    /// instruction at `site` runs it: one with its EXEC bit clear there
    /// keeps what the registers held before.
    fn runs(&self, site: Site, known: &Known) -> Result<(), &'static str> {
        let Some(lanes) = self.lanes else {
            return Ok(());
        };
        let Some(&Some(guard)) = lanes.guards.get(site.0).and_then(|g| g.get(site.1)) else {
            return Ok(());
        };
        let mut skips = known.clone();
        skips.extend([(lanes.reach[site.0], true), (guard, false)]);
        if self.meet(lanes, &skips) {
            return Err("a per-lane descriptor written where some lanes may keep the old one");
        }
        Ok(())
    }

    /// The source of descriptor dwords each lane holds its own of (a
    /// waterfall loop reads them from the first active lane): the same in
    /// every lane, loaded per lane from memory a heap address points to, or
    /// picked per lane between such sources, by a lane mask or by the path
    /// the lane took. `known` holds in every lane that accesses them.
    fn per_lane(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Source, &'static str> {
        let words = words
            .iter()
            .map(|&w| self.lane_value(w, known))
            .collect::<Result<Vec<_>, _>>()?;
        let words = &words[..];
        let Some(&lane) = words.iter().find(|&&w| self.varies(w)) else {
            return self.classify(words, joins, known);
        };
        if let Some(source) = self.fetched(words, known)? {
            return Ok(source);
        }
        if let Some(source) = self.lane_select(words, joins, known)? {
            return Ok(source);
        }
        if let Some(source) = self.lane_phi(words, joins, known)? {
            return Ok(source);
        }
        let Term::Value(v) = lane else { unreachable!() };
        let name = self.def(v).map(|i| i.name());
        Err(match name {
            _ if self.inputs[v.0 as usize].is_some() => {
                "a descriptor read from the first lane of an input VGPR"
            }
            None => "a descriptor read from the first lane of a phi",
            Some(n) if n.starts_with("ds_read") => {
                "a descriptor read from the first lane of an LDS read"
            }
            Some(n) if n.starts_with("buffer_load") => {
                "a descriptor read from the first lane of a buffer load"
            }
            Some(n) if n.starts_with("global_load") || n.starts_with("flat_load") => {
                "a descriptor read from the first lane of a partial or mixed load"
            }
            Some(_) => "a descriptor read from the first lane of computed VGPRs",
        })
    }

    /// Dwords of `global_load_dword*` that line up: `first..` of one load,
    /// then the bytes that follow (an image descriptor takes two
    /// `global_load_dwordx4`).
    fn fetched(&self, words: &[Term], known: &Known) -> Result<Option<Source>, &'static str> {
        let at = |w: &Term| -> Option<(Site, usize)> {
            let Term::Value(v) = *w else { return None };
            let site = self.defs[v.0 as usize]?;
            let inst = self.inst(site);
            if !inst.name().starts_with("global_load_dword") {
                return None;
            }
            let k = inst
                .writes
                .iter()
                .filter(|w| matches!(w.0, Reg::V(_)))
                .position(|w| w.1 == v)?;
            Some((site, k))
        };
        let Some(sites) = words.iter().map(at).collect::<Option<Vec<_>>>() else {
            return Ok(None);
        };
        let address = |site: Site| {
            let inst = self.inst(site);
            let Fields::Flat {
                addr,
                saddr,
                offset,
                ..
            } = &inst.fields
            else {
                return Err("a global load");
            };
            self.flat_address(inst, addr.as_ref(), saddr.as_ref(), *offset)
        };
        let (load, first) = sites[0];
        let a0 = address(load)?;
        for (k, &(site, dword)) in sites.iter().enumerate() {
            self.runs(site, known)?;
            let a = address(site)?;
            let want = a0.offset.wrapping_add(4 * (first + k) as u32);
            if (a.param, a.dynamic) != (a0.param, a0.dynamic)
                || a.offset.wrapping_add(4 * dword as u32) != want
            {
                return Err("a descriptor read from the first lane of loads that do not line up");
            }
        }
        Ok(Some(Source::Fetched {
            load,
            first: first as u8,
        }))
    }

    /// Words that `v_cndmask_b32` picks by one lane mask, traced per outcome
    /// (other words are the same for both).
    fn lane_select(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let picks: Vec<_> = words
            .iter()
            .map(|w| match *w {
                Term::Value(v) => self.lane_picked(v),
                Term::Const(_) => None,
            })
            .collect();
        let Some(mask) = picks.iter().flatten().map(|p| p.0).next() else {
            return Ok(None);
        };
        if picks.iter().flatten().any(|p| p.0 != mask) {
            return Err("a descriptor chosen per lane by several masks");
        }
        for (w, p) in words.iter().zip(&picks) {
            if let (Term::Value(v), Some(_)) = (*w, p)
                && let Some(site) = self.defs[v.0 as usize]
            {
                self.runs(site, known)?;
            }
        }
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(usize::MAX);
        // Each side holds where the lane's bit of the mask is set, or clear.
        let bit = self.lanes.and_then(|l| l.masks.get(&mask).copied());
        let side = |set: bool| -> Vec<Term> {
            words
                .iter()
                .zip(&picks)
                .map(|(w, p)| match p {
                    Some((_, t, f)) => {
                        if set {
                            *t
                        } else {
                            *f
                        }
                    }
                    None => *w,
                })
                .collect()
        };
        let args = [true, false]
            .into_iter()
            .map(|set| {
                let n = known.len();
                known.extend(bit.map(|b| (b, set)));
                let arg = self.per_lane(&side(set), joins, known);
                known.truncate(n);
                arg
            })
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Lanes { mask, args: args? }))
    }

    /// For `v` written by a plain `v_cndmask_b32`: the low dword of the
    /// lane mask it reads, and the dword it is where the lane's bit is set
    /// (src1), then clear (src0).
    fn lane_picked(&self, v: Value) -> Option<(Value, Term, Term)> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        let Some(Operand::Vector { reg, .. }) = f.dst else {
            return None;
        };
        if inst.name() != "v_cndmask_b32"
            || inst.write(Reg::V(reg)) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.clamp
            || f.omod != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        let [clear, set, Operand::Scalar { code, .. }] = f.src[..] else {
            return None;
        };
        let mask = inst.read(Reg::S(code))?;
        Some((mask, self.raw(inst, &set)?, self.raw(inst, &clear)?))
    }

    /// Per-lane words that phis of one block join, as an if/else on a
    /// per-lane condition leaves them: traced on each edge into the block
    /// that lanes in `known` can take (other words hold on every edge).
    /// Lanes take their own paths ([`Structured`]), so each gets the value
    /// of the edge it came by; an edge they cannot take is `None` (a flow
    /// block's phi joins an arm's dwords with what lanes that skipped the
    /// arm held before, which only they read).
    fn lane_phi(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let phi_of = |w: &Term| match *w {
            Term::Value(v) => self.phis[v.0 as usize],
            Term::Const(_) => None,
        };
        let Some((block, _)) = words.iter().find_map(phi_of) else {
            return Ok(None);
        };
        if words.iter().filter_map(phi_of).any(|(b, _)| b != block) {
            return Err("a descriptor joined in several blocks");
        }
        if joins.contains(&block) {
            return Err("a descriptor that changes in a loop");
        }
        let preds = &self.p.blocks[block].preds;
        if preds.iter().any(|&q| self.rpo[q] >= self.rpo[block]) {
            // shortcut: what a lane holds after some passes is not traced.
            return Err("a descriptor read from the first lane of a phi at a loop header");
        }
        let Some(lanes) = self.lanes else {
            return Err("a descriptor read from the first lane of a phi, without the lanes' paths");
        };
        // The other words must be there before the join: defined in a
        // block that dominates it, which (both dominating the access) is
        // one the join does not reach.
        for w in words {
            if let Term::Value(v) = *w
                && phi_of(w).is_none()
                && let Some((d, _)) = self.defs[v.0 as usize]
                && (d == block || self.reaches(block, d))
            {
                return Err("a descriptor partly joined by control flow");
            }
        }
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(block);
        let phis = &self.p.blocks[block].phis;
        let mut args = Vec::new();
        for (j, &q) in preds.iter().enumerate() {
            let n = known.len();
            known.push((lanes.reach[q], true));
            if let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = lanes.exits[q]
                && taken != not_taken
            {
                known.push((when, taken == block));
            }
            let arg = if self.meet(lanes, known) {
                let edge: Vec<Term> = words
                    .iter()
                    .map(|w| match phi_of(w) {
                        Some((_, i)) => Term::Value(phis[i].args[j]),
                        None => *w,
                    })
                    .collect();
                self.per_lane(&edge, joins, known).map(Some)
            } else {
                Ok(None)
            };
            known.truncate(n);
            match arg {
                Ok(a) => args.push(a),
                Err(e) => {
                    joins.pop();
                    return Err(e);
                }
            }
        }
        joins.pop();
        Ok(Some(Source::LanePhi { block, args }))
    }

    /// Whether a path leads from block `from` to block `to` (to `from`
    /// itself only through a back edge).
    fn reaches(&self, from: usize, to: usize) -> bool {
        let mut seen = vec![false; self.p.blocks.len()];
        let mut stack = self.p.blocks[from].exit.successors();
        while let Some(x) = stack.pop() {
            if x == to {
                return true;
            }
            if std::mem::replace(&mut seen[x], true) {
                continue;
            }
            stack.extend(self.p.blocks[x].exit.successors());
        }
        false
    }

    /// Prove a single-block waterfall at the descriptor access. In each
    /// iteration the access serves exactly the matching lanes, and the
    /// back edge keeps exactly the other lanes active. Otherwise, why not.
    ///
    // shortcut: only the accesses are proven. Any other per-lane work in the
    // loop is taken to run once for each lane, as the access does; a body
    // whose effect adds up over the passes a lane waits through (a counter
    // before the mask is narrowed) would differ. Compilers put nothing but
    // the reads, the compares and the access in these loops.
    fn waterfall(
        &self,
        b: usize,
        accesses: &[usize],
        reads: &[Value],
    ) -> Result<Waterfall, &'static str> {
        const LOOP: &str = "a descriptor read from the first lane outside a waterfall loop";
        const SPLIT: &str = "a waterfall loop nested in another or split over blocks";
        const SELECT: &str = "a waterfall loop whose compares do not select the access's lanes";
        const RETRY: &str = "a waterfall loop whose retry does not keep the lanes still waiting";
        let access = accesses[0];
        let block = &self.p.blocks[b];
        let lanes = self.lanes.ok_or(LOOP)?;
        let exit = block.insts.last().ok_or(LOOP)?;
        if !matches!(block.exit, Exit::Branch { taken, not_taken } if taken == b && not_taken != b)
            || exit.name() != "s_cbranch_execnz"
        {
            return Err(LOOP);
        }
        let mask = |low: Option<Value>, high: Option<Value>| {
            let high = if self.p.wave64 { Some(high?) } else { None };
            lanes.mask_bits(low?, high)
        };
        let exec = |inst: &Inst| mask(inst.read(Reg::S(EXEC)), inst.read(Reg::S(EXEC + 1)));
        // The reads belong to this loop's own block: a loop around this one
        // (one per operand, as LLVM nests them for a per-lane image and
        // sampler) is not followed.
        if reads
            .iter()
            .any(|v| self.defs[v.0 as usize].is_none_or(|at| at.0 != b))
        {
            return Err(SPLIT);
        }
        let selected = exec(&block.insts[access]).ok_or(SELECT)?;
        if accesses
            .iter()
            .any(|&i| exec(&block.insts[i]) != Some(selected))
        {
            return Err(SELECT);
        }
        let retry = exec(exit).ok_or(RETRY)?;
        let first = self.defs[reads[0].0 as usize].ok_or(LOOP)?;
        let remaining = exec(self.inst(first)).ok_or(SELECT)?;
        if reads.iter().any(|v| {
            self.defs[v.0 as usize]
                .is_none_or(|at| at.1 >= access || exec(self.inst(at)) != Some(remaining))
        }) {
            return Err(SELECT);
        }
        let reach = (lanes.reach[b], true);
        let meet = |conditions: &[(Bool, bool)]| self.meet(lanes, conditions);
        let mut out = Waterfall::default();
        let mut required = vec![(remaining, true)];
        for (i, inst) in block.insts[..access].iter().enumerate() {
            let Fields::Valu(f) = &inst.fields else {
                continue;
            };
            let Some(ty) = inst
                .name()
                .strip_prefix("v_cmp_eq_")
                .or(inst.name().strip_prefix("v_cmpx_eq_"))
            else {
                continue;
            };
            let dwords = match ty {
                "u32" | "i32" => 1,
                "u64" | "i64" => 2,
                _ => continue,
            };
            if f.abs != 0 || f.neg != 0 || f.sdwa.is_some() || f.dpp.is_some() {
                continue;
            }
            // The first lane's dwords on one side, the lane's own on the
            // other.
            let pair = |first: &Operand, own: &Operand| -> Option<Vec<Value>> {
                let (&Operand::Scalar { code, .. }, &Operand::Vector { reg, .. }) = (first, own)
                else {
                    return None;
                };
                (0..dwords)
                    .map(|k| {
                        let Term::Value(x) = self.origin(inst.read(Reg::S(code + k))?) else {
                            return None;
                        };
                        let lane = inst.read(Reg::V(reg + k))?;
                        let (_, read) = self.defs[x.0 as usize]?;
                        let same = reads.contains(&x)
                            && read < i
                            && self.lane_term(Term::Value(self.first_lane_source(x)?))
                                == self.lane_term(Term::Value(lane));
                        same.then_some(x)
                    })
                    .collect()
            };
            let (Some(a), Some(c)) = (f.src.first(), f.src.get(1)) else {
                continue;
            };
            if let Some(xs) = pair(a, c).or_else(|| pair(c, a)) {
                let code = if inst.name().starts_with("v_cmpx_") {
                    EXEC
                } else if let Some(Operand::Scalar { code, .. }) = f.dst {
                    code
                } else {
                    continue;
                };
                let Some(matched) = mask(inst.write(Reg::S(code)), inst.write(Reg::S(code + 1))) else {
                    continue;
                };
                if meet(&[reach, (selected, true), (matched, false)]) {
                    continue;
                }
                required.push((matched, true));
                out.compares.push((b, i));
                for x in xs {
                    if !out.reads.contains(&x) {
                        out.reads.push(x);
                    }
                }
            }
        }
        if reads.iter().any(|v| !out.reads.contains(v))
            || meet(&[reach, (selected, true), (remaining, false)])
        {
            return Err(SELECT);
        }
        // The access must take every match.
        required.extend([reach, (selected, false)]);
        if meet(&required) {
            return Err(SELECT);
        }
        // The retry must keep exactly the lanes that were eligible but did
        // not perform the access.
        if meet(&[reach, (retry, true), (remaining, false)])
            || meet(&[reach, (retry, true), (selected, true)])
            || meet(&[reach, (remaining, true), (selected, false), (retry, false)])
        {
            return Err(RETRY);
        }
        Ok(out)
    }

    /// Consecutive dwords of one scalar load.
    fn loaded(&self, words: &[Term]) -> Option<Source> {
        let Term::Value(v0) = words[0] else {
            return None;
        };
        let load = self.defs[v0.0 as usize]?;
        let inst = self.inst(load);
        if !inst.name().starts_with("s_load_") {
            return None;
        }
        let first = inst.writes.iter().position(|w| w.1 == v0)?;
        let same = words.iter().enumerate().all(|(k, w)| {
            inst.writes
                .get(first + k)
                .is_some_and(|x| *w == Term::Value(x.1))
        });
        same.then_some(Source::Loaded {
            load,
            first: first as u8,
        })
    }

    /// A descriptor the code keeps per lane: every dword of `words` is the
    /// first lane (`v_readfirstlane_b32`) of a VGPR that `v_mov_b32` filled
    /// from SGPRs. The console compiler does this to a descriptor loaded on
    /// both sides of a per-lane branch: the join is per lane, so it lives in
    /// VGPRs, and the instruction that needs it in SGPRs runs in a loop over
    /// the distinct values (the first lane's, the lanes holding the same,
    /// then the rest). When every scalar those moves copy is the same dword
    /// of loads from one constant address, the lanes agree and the
    /// descriptor is any of those loads.
    // shortcut: what else the VGPRs held (numbers, on paths or in lanes that
    // never read them as a descriptor) is ignored, and a move is taken to
    // leave either its source or what a move before it wrote; a descriptor
    // read per lane from memory (`global_load_*`) refuses.
    fn lane_copies(&self, words: &[Term]) -> Option<Source> {
        let mut loads: Vec<Site> = Vec::new();
        let mut first = None;
        for (k, w) in words.iter().enumerate() {
            let Term::Value(v) = *w else { return None };
            let mut copies = Vec::new();
            if !self.scalar_copies(self.first_lane(v)?, &mut copies) {
                return None;
            }
            let mut found = false;
            for c in copies {
                let Term::Value(c) = c else { continue };
                let Some(site) = self.defs[c.0 as usize] else {
                    continue;
                };
                let load = self.inst(site);
                if !load.name().starts_with("s_load_") {
                    continue;
                }
                // Dword k of the operand is dword `first + k` of each load.
                let dword = load.writes.iter().position(|x| x.1 == c)?;
                let f = dword.checked_sub(k)?;
                if *first.get_or_insert(f) != f {
                    return None;
                }
                if !loads.contains(&site) {
                    loads.push(site);
                }
                found = true;
            }
            if !found {
                return None;
            }
        }
        let address = |site: Site| {
            let inst = self.inst(site);
            match &inst.fields {
                Fields::Smem {
                    base,
                    offset,
                    soffset,
                    ..
                } => self.smem_address(inst, base, *offset, soffset).ok(),
                _ => None,
            }
        };
        let a = address(*loads.first()?)?;
        if a.dynamic.is_some() || loads.iter().any(|&l| address(l) != Some(a)) {
            return None;
        }
        // The handle is made where the descriptor is used, from the load's
        // operands: take a load whose base is the root parameter's own SGPR.
        loads.sort_unstable();
        let load = *loads.iter().find(|&&l| self.base_from_entry(l))?;
        self.lane_copied.borrow_mut().extend(&loads);
        Some(Source::Loaded {
            load,
            first: first? as u8,
        })
    }

    /// The scalars (as origins) that `v_mov_b32` copies put in VGPR value
    /// `v`, through joins, moves between VGPRs and per-lane selects; a move
    /// keeps what it wrote over in the lanes it does not run. False if the
    /// VGPR may hold something read per lane from memory.
    fn scalar_copies(&self, v: Value, out: &mut Vec<Term>) -> bool {
        let (mut seen, mut work) = (Vec::new(), vec![v]);
        while let Some(v) = work.pop() {
            if seen.contains(&v) {
                continue;
            }
            if seen.len() == MAX_COPIES {
                return false;
            }
            seen.push(v);
            if let Some((b, i)) = self.phis[v.0 as usize] {
                work.extend(&self.p.blocks[b].phis[i].args);
                continue;
            }
            // An entry value, or a number some instruction computed.
            let Some(inst) = self.def(v) else { continue };
            if inst.name().starts_with("global_load_") {
                return false;
            }
            let Fields::Valu(f) = &inst.fields else {
                continue;
            };
            let sources = match inst.name() {
                "v_mov_b32" => 1,
                "v_cndmask_b32" => 2,
                _ => continue,
            };
            if f.abs != 0 || f.neg != 0 || f.sdwa.is_some() || f.dpp.is_some() {
                continue;
            }
            for op in f.src.iter().take(sources) {
                match *op {
                    Operand::Vector { reg, .. } => work.extend(inst.read(Reg::V(reg))),
                    Operand::Scalar { code, .. } if code != NULL => {
                        out.extend(inst.read(Reg::S(code)).map(|s| self.origin(s)));
                    }
                    _ => {}
                }
            }
            // Lanes outside EXEC keep the old value.
            if let Some(Operand::Vector { reg, .. }) = f.dst {
                work.extend(inst.read(Reg::V(reg)));
            }
        }
        true
    }

    /// The load at `site` has a constant offset and takes its base from a
    /// root parameter's SGPR through moves alone: a value that exists
    /// wherever the descriptor is used.
    fn base_from_entry(&self, site: Site) -> bool {
        let inst = self.inst(site);
        let Fields::Smem {
            base: Operand::Scalar { code, .. },
            soffset: Operand::Scalar { code: NULL, .. },
            ..
        } = &inst.fields
        else {
            return false;
        };
        let Some(mut v) = inst.read(Reg::S(*code)) else {
            return false;
        };
        loop {
            if self.inputs[v.0 as usize].is_some() {
                return true;
            }
            let Some(def) = self.def(v) else { return false };
            let Some(k) = def.writes.iter().position(|w| w.1 == v) else {
                return false;
            };
            match copy_source(def, k) {
                Some(Moved::Value(s)) => v = s,
                _ => return false,
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Moved {
    Const(u32),
    Value(Value),
}

fn plain_lane(v: &crate::gfx10::Valu) -> bool {
    v.abs == 0
        && v.neg == 0
        && v.opsel == 0
        && !v.clamp
        && v.omod == 0
        && v.sdwa.is_none()
        && v.dpp.is_none()
}

impl Tracer<'_> {
    /// For `v_readlane_b32` of a constant lane: the SGPR spilled to that
    /// lane, found by following the VGPR's `v_writelane_b32` writes.
    fn restored(&self, inst: &Inst, block: Option<usize>) -> Option<Moved> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_readlane_b32" || !plain_lane(v) {
            return None;
        }
        let [Operand::Vector { reg, .. }, lane] = v.src[..] else {
            return None;
        };
        let lane = lane_of(lane)?;
        if lane >= if self.p.wave64 { 64 } else { 32 } {
            return None;
        }
        self.spilled(inst.read(Reg::V(reg))?, lane, block, &mut Vec::new())
            .ok()
            .flatten()
    }

    /// What lane `lane` of VGPR value `v` holds: `Ok(None)` for a phi in
    /// `seen` (already being resolved), `Err` if not a spilled SGPR.
    fn spilled(
        &self,
        v: Value,
        lane: u32,
        block: Option<usize>,
        seen: &mut Vec<Value>,
    ) -> Result<Option<Moved>, ()> {
        if let Some((b, i)) = self.phis[v.0 as usize] {
            if block.is_some() {
                return Err(());
            }
            if seen.contains(&v) {
                return Ok(None);
            }
            seen.push(v);
            let mut found = None;
            for &a in &self.p.blocks[b].phis[i].args {
                if let Some(m) = self.spilled(a, lane, block, seen)? {
                    if found.is_some_and(|f| f != m) {
                        return Err(());
                    }
                    found = Some(m);
                }
            }
            return Ok(found);
        }
        let site = self.defs[v.0 as usize].ok_or(())?;
        if block.is_some_and(|b| site.0 != b) {
            return Err(());
        }
        let inst = self.inst(site);
        let Fields::Valu(w) = &inst.fields else {
            return Err(());
        };
        let (Some(Operand::Vector { reg, .. }), [value, at]) = (w.dst, &w.src[..]) else {
            return Err(());
        };
        if inst.name() != "v_writelane_b32" || !plain_lane(w) {
            return Err(());
        }
        let written = lane_of(*at).ok_or(())?;
        if written >= if self.p.wave64 { 64 } else { 32 } {
            return Err(());
        }
        if written != lane {
            return self.spilled(inst.read(Reg::V(reg)).ok_or(())?, lane, block, seen);
        }
        match *value {
            Operand::Scalar { code, .. } if code != NULL => {
                inst.read(Reg::S(code)).map(Moved::Value)
            }
            Operand::Literal(x) => Some(Moved::Const(x)),
            Operand::Const(c) => inline(c).map(Moved::Const),
            _ => None,
        }
        .map(Some)
        .ok_or(())
    }
}

/// A lane select given as a constant.
fn lane_of(op: Operand) -> Option<u32> {
    match op {
        Operand::Const(c) => inline(c),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

/// What dword `k` of a move's result copies.
fn copy_source(inst: &Inst, k: usize) -> Option<Moved> {
    let Fields::Sop { src, imm, .. } = &inst.fields else {
        return None;
    };
    if inst.name() == "s_movk_i32" {
        return Some(Moved::Const(i32::from((*imm)? as i16) as u32));
    }
    // s_bfm_b32 d, width, offset: `width` ones from bit `offset` (the
    // console compiler builds sampler words this way, the upscaler's too),
    // from inline constants or literals.
    if inst.name() == "s_bfm_b32" && k == 0 {
        let constant = |o: &Operand| match *o {
            Operand::Const(c) => inline(c),
            Operand::Literal(x) => Some(x),
            _ => None,
        };
        let width = constant(src.first()?)? & 31;
        let offset = constant(src.get(1)?)? & 31;
        return Some(Moved::Const(((1u32 << width) - 1) << offset));
    }
    let src = src.first()?;
    match (inst.name(), *src) {
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { code, .. }) if code != NULL => {
            inst.read(Reg::S(code + k as u8)).map(Moved::Value)
        }
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { .. }) => Some(Moved::Const(0)),
        ("s_mov_b32", Operand::Literal(x)) => Some(Moved::Const(x)),
        ("s_mov_b32", Operand::Const(c)) => inline(c).map(Moved::Const),
        // shortcut: only the one-operand ways the compiler makes a constant
        // that has no inline form; fold more if descriptors need it.
        ("s_brev_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(x.reverse_bits())),
        ("s_not_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(!x)),
        // shortcut: a literal below 2^31, whose high dword is 0 whether the
        // hardware extends it with zeros or with its sign.
        ("s_mov_b64", Operand::Literal(x)) if x >> 31 == 0 => {
            Some(Moved::Const(if k == 0 { x } else { 0 }))
        }
        // 64-bit inline integers are sign-extended.
        ("s_mov_b64", Operand::Const(c @ 128..=208)) => {
            let low = inline(c)?;
            Some(Moved::Const(if k == 0 {
                low
            } else {
                ((low as i32) >> 31) as u32
            }))
        }
        _ => None,
    }
}

/// The 32-bit value of an inline constant operand.
pub(crate) fn inline(code: u8) -> Option<u32> {
    let float = |f: f32| Some(f.to_bits());
    match code {
        128..=192 => Some(u32::from(code) - 128),
        193..=208 => Some((192 - i32::from(code)) as u32),
        240 => float(0.5),
        241 => float(-0.5),
        242 => float(1.0),
        243 => float(-1.0),
        244 => float(2.0),
        245 => float(-2.0),
        246 => float(4.0),
        247 => float(-4.0),
        248 => Some(0x3E22_F983),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::native::RootType::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn track_words(words: &[u32], layout: &[RootParam], stage: HwStage) -> Resources {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        track(&p, layout, stage, lanes(&p).as_ref()).unwrap()
    }

    /// `p` as one lane runs it, if it structures.
    fn lanes(p: &Program) -> Option<Structured> {
        crate::structure::structure(p).ok()
    }

    fn param(root: RootType, user_data: u8, sgprs: u32) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot: 0,
        }
    }

    fn at(param: usize, offset: u32) -> Address {
        Address {
            param,
            offset,
            dynamic: None,
        }
    }

    #[test]
    fn table_descriptors_are_loaded_from_their_range() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE89_0380, // s_mov_b32 s9, 0
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF408_0204,
                0xFA00_0000, // s_load_dwordx4 s[8:11], s[8:9], null
                0xF40C_0001,
                0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
                0xF080_0F08,
                0x0040_0002, // image_sample v[0:3], v[2:3], s[0:7], s[8:11]
                S_ENDPGM,
            ],
            &[param(Table, 8, 1), param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0)), ((0, 4), at(1, 0))]);
        let roles: Vec<_> = r.descriptors.iter().map(|d| (d.role, &d.source)).collect();
        assert_eq!(
            roles,
            [
                (
                    Role::Resource,
                    &Source::Loaded {
                        load: (0, 4),
                        first: 0
                    }
                ),
                (
                    Role::Sampler,
                    &Source::Loaded {
                        load: (0, 3),
                        first: 0
                    }
                ),
            ]
        );
    }

    #[test]
    fn root_descriptors_are_built_in_code() {
        let r = track_words(
            &[
                0x8801_FF0B,
                0x0010_0000, // s_or_b32 s1, s11, 0x100000
                0xBE80_030A, // s_mov_b32 s0, s10
                0xBE82_03C1, // s_mov_b32 s2, -1
                0xBE83_03FF,
                0x1101_4FAC, // s_mov_b32 s3, 0x11014fac
                0xF430_0000,
                0xFA00_0000, // s_buffer_load_dwordx16 s[0:15], s[0:3], null
                S_ENDPGM,
            ],
            &[param(Descriptor, 10, 2)],
            HwStage::Vs,
        );
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: at(0, 0),
                bits: 0x10_0000,
                words: [u32::MAX, 0x1101_4FAC]
            }
        );
        assert_eq!(
            r.params.iter().map(|p| (p.1, p.2)).collect::<Vec<_>>(),
            [(0, 0), (0, 1)]
        );
    }

    #[test]
    fn heap_indices_from_root_constants_in_a_merged_stage() {
        // User data starts at s8: the constant (index 6) is s14, the heap
        // (index 10) s18.
        let code = [
            0x8F08_850E, // s_lshl_b32 s8, s14, 5
            0xBE93_0380, // s_mov_b32 s19, 0
            0xF408_0209,
            0x1000_0000, // s_load_dwordx4 s[8:11], s[18:19], s8
            0xE000_2000,
            0x8002_0400, // buffer_load_format_x v4, v0, s[8:11], 0 idxen
            S_ENDPGM,
        ];
        let r = track_words(
            &code,
            &[param(Constants, 6, 1), param(ResourceHeap, 10, 1)],
            HwStage::Ms,
        );
        let index = r.params.iter().find(|p| p.1 == 0).unwrap().0;
        let shift = r.scalar[0].1.dynamic.unwrap();
        assert_eq!(r.scalar[0].1.param, 1);
        // The computed term is the shifted constant, not the constant itself.
        assert_ne!(shift, index);
        assert_eq!(
            r.descriptors[0].source,
            Source::Loaded {
                load: (0, 2),
                first: 0
            }
        );
        assert!(r.root_indexed_views.is_empty());
        let table = RootParam {
            slot: 7,
            ..param(Table, 10, 1)
        };
        let indexed = track_words(&code, &[param(Constants, 6, 1), table], HwStage::Ms);
        assert_eq!(
            indexed.root_indexed_views,
            [RootIndexedView {
                table_slot: 7,
                constant_slot: 0,
                constant_dword: 0,
                byte_offset: 0,
                shift: 5,
            }]
        );
        let nonconstant = track_words(&code, &[param(Table, 6, 1), table], HwStage::Ms);
        assert!(nonconstant.root_indexed_views.is_empty());
    }

    #[test]
    fn spilled_sgprs_are_followed_through_vgpr_lanes() {
        let r = track_words(
            &[
                0xD761_0000,
                0x0001_0609, // v_writelane_b32 v0, s9, 3
                0xD760_0002,
                0x0001_0700, // v_readlane_b32 s2, v0, 3
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF40C_0001,
                0xFA00_0020, // s_load_dwordx8 s[0:7], s[2:3], 0x20
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0x20))]);
    }

    /// A VGPR a wave reduction used is free again after it: from the first
    /// save on it holds parked scalars alone (an upscaler program parks a
    /// constant buffer's descriptor that way).
    #[test]
    fn a_vgpr_used_before_holds_spilled_sgprs_from_its_first_save() {
        let before = [
            0x7E00_0281, // v_mov_b32 v0, 1
            0x7E02_0300, // v_mov_b32 v1, v0
        ];
        let park = [
            0xD761_0000,
            0x0001_0609, // v_writelane_b32 v0, s9, 3
            0xD760_0002,
            0x0001_0700, // v_readlane_b32 s2, v0, 3
            0xBE83_0380, // s_mov_b32 s3, 0
        ];
        let load = [
            0xF40C_0001,
            0xFA00_0020, // s_load_dwordx8 s[0:7], s[2:3], 0x20
            S_ENDPGM,
        ];
        let words = [&before[..], &park, &load].concat();
        let r = track_words(&words, &[param(Table, 9, 1)], HwStage::Ps);
        assert_eq!(r.scalar, [((0, 5), at(0, 0x20))]);
        // Still lane reads: a lane restored before it is saved, and a
        // register with an ordinary use after the first save.
        let restored_first = [&before[..], &park[2..4], &park[..2], &load].concat();
        let used_after = [&before[..], &park, &before[1..], &load].concat();
        for words in [restored_first, used_after] {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let p = crate::ir::build(&code, true, 8).unwrap();
            let insts = p.blocks.iter().flat_map(|b| &b.insts);
            assert_eq!(insts.filter(|i| i.name() == "v_readlane_b32").count(), 1);
        }
    }

    /// Two lanes hold a pair (a lane mask, say) and later one of them a
    /// scalar, as in a water volume program that parks more than the lanes
    /// of one VGPR hold: the pair still moves as one.
    #[test]
    fn a_lane_spilled_to_again_later_does_not_split_an_earlier_pair() {
        let pair = [
            0xD761_0000u32,
            0x0001_0004, // v_writelane_b32 v0, s4, 0
            0xD761_0000,
            0x0001_0205, // v_writelane_b32 v0, s5, 1
            0xBE84_0380, // s_mov_b32 s4, 0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xD760_0004,
            0x0001_0100, // v_readlane_b32 s4, v0, 0
            0xD760_0005,
            0x0001_0300, // v_readlane_b32 s5, v0, 1
        ];
        let scalar = [
            0xD761_0000u32,
            0x0001_0006, // v_writelane_b32 v0, s6, 0
            0xBE86_0380, // s_mov_b32 s6, 0
            0xD760_0006,
            0x0001_0100, // v_readlane_b32 s6, v0, 0
            S_ENDPGM,
        ];
        let moves = |words: &[u32]| {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let p = crate::ir::build(&code, true, 8).unwrap();
            let names: Vec<_> = p.blocks[0].insts.iter().map(|i| i.name()).collect();
            assert!(!names.iter().any(|n| n.contains("lane")));
            names.iter().filter(|n| **n == "s_mov_b64").count()
        };
        assert_eq!(moves(&[&pair[..], &scalar].concat()), 2);
        // A restore both saves reach keeps the lane one scalar.
        let joined = [
            &pair[..4],
            &[0xBF84_0002], // s_cbranch_scc0 past the second save
            &scalar[..2],
            &pair[4..],
            &scalar[2..],
        ]
        .concat();
        assert_eq!(moves_anywhere(&joined), 0);
    }

    fn moves_anywhere(words: &[u32]) -> usize {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 8).unwrap();
        let insts = p.blocks.iter().flat_map(|b| &b.insts);
        insts.filter(|i| i.name() == "s_mov_b64").count()
    }

    #[test]
    fn descriptors_joined_by_control_flow_are_traced_per_edge() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE83_0380, // s_mov_b32 s3, 0
                0xBF06_8001, // s_cmp_eq_u32 s1, 0
                0xBF85_0003, // s_cbranch_scc1 -> 0x1c
                0xF408_0001,
                0xFA00_0000, // s_load_dwordx4 s[0:3], s[2:3], null
                0xBF82_0002, // s_branch -> 0x24
                0xF408_0001,
                0xFA00_0010, // 0x1c: s_load_dwordx4 s[0:3], s[2:3], 0x10
                0xF420_0100,
                0xFA00_0000, // 0x24: s_buffer_load_dword s4, s[0:3], null
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        let Source::Phi { args, .. } = &r.descriptors[0].source else {
            panic!("{:?}", r.descriptors)
        };
        let offsets: Vec<_> = args
            .iter()
            .map(|a| match a {
                Source::Loaded { load, .. } => {
                    r.scalar.iter().find(|s| s.0 == *load).unwrap().1.offset
                }
                a => panic!("{a:?}"),
            })
            .collect();
        assert_eq!(offsets.len(), 2);
        assert!(
            offsets.contains(&0) && offsets.contains(&0x10),
            "{offsets:?}"
        );
    }

    /// Two paths add an index to one table in the code and meet before one load
    /// (hair `CS_InterpolateStrands`): the join is the computed term, counted
    /// from the `s_getpc_b64`.
    #[test]
    fn two_paths_into_one_code_table_load() {
        let r = track_words(
            &[
                0xBE82_1F00, // 0x00: s_getpc_b64 s[2:3]
                0x8002_FF02,
                0x0000_0040, // s_add_u32 s2, s2, 0x40
                0x8203_8003, // s_addc_u32 s3, s3, 0
                0xBF06_8001, // 0x10: s_cmp_eq_u32 s1, 0
                0xBF85_0005, // 0x14: s_cbranch_scc1 -> 0x2c
                0xD70F_0457,
                0x0000_054A, // 0x18: v_add_co_u32 v87, s4, v74, s2
                0xD528_0458,
                0x0010_0680, // v_add_co_ci_u32 v88, s4, 0, s3, s4
                0xBF82_0004, // 0x28: s_branch -> 0x3c
                0xD70F_0457,
                0x0000_054B, // 0x2c: v_add_co_u32 v87, s4, v75, s2
                0xD528_0458,
                0x0010_0680, // v_add_co_ci_u32 v88, s4, 0, s3, s4
                0xDC34_8000,
                0x577D_0057, // 0x3c: global_load_dwordx2 v[87:88], v[87:88], null
                S_ENDPGM,
            ],
            &[],
            HwStage::Cs,
        );
        assert_eq!(r.scalar.len(), 1);
        let a = r.scalar[0].1;
        assert_eq!((a.param, a.offset), (CODE, 0));
        assert!(a.dynamic.is_some(), "the join is the computed term, the byte in the code");
    }

    /// An address into a table in the code computed per lane and taken from
    /// the first lane (`avgLuminanceCompute` `ColorDownsample`): the per-lane
    /// sum carries the `s_getpc_b64` base, so the first-lane read is looked
    /// through and the lane's index is the computed term. (A first-lane read
    /// of an index alone stays the computed term itself; see
    /// `the_console_waterfall_over_a_first_lane_offset_is_one`.)
    #[test]
    fn a_code_table_address_taken_from_the_first_lane() {
        let r = track_words(
            &[
                0xBE82_1F00, // 0x00: s_getpc_b64 s[2:3]
                0xD70F_0457,
                0x0000_054A, // v_add_co_u32 v87, s4, v74, s2
                0xD528_0458,
                0x0010_0680, // v_add_co_ci_u32 v88, s4, 0, s3, s4
                0x7E0C_0557, // v_readfirstlane_b32 s6, v87
                0x7E0E_0558, // v_readfirstlane_b32 s7, v88
                0xF400_0203,
                0xFA00_0000, // s_load_dword s8, s[6:7], 0x0
                S_ENDPGM,
            ],
            &[],
            HwStage::Cs,
        );
        assert_eq!(r.scalar.len(), 1);
        let a = r.scalar[0].1;
        assert_eq!((a.param, a.offset), (CODE, 4));
        assert!(a.dynamic.is_some(), "the lane's index is the computed term");
    }

    #[test]
    fn a_descriptor_kept_per_lane_is_its_load_and_needs_no_waterfall() {
        // The console compiler's way with a descriptor it joins per lane:
        // the dwords go to VGPRs and come back from the first lane. Every
        // lane holds the one load's dwords, so the descriptor is that load
        // (`lane_copied`), and no waterfall is asked for.
        let mut words = vec![
            0xBE82_0309, // s_mov_b32 s2, s9
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF40C_0001,
            0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
        ];
        // v_mov_b32 v10.., s0..; v_readfirstlane_b32 s16.., v10..
        words.extend((0..8).map(|k| 0x7E00_0200 | (10 + k) << 17 | k));
        words.extend((0..8).map(|k| 0x7E00_0400 | (16 + k) << 17 | (0x100 + 10 + k)));
        words.extend([
            0xF000_1700,
            0x0004_0D06, // image_load v[13:15], v6, s[16:23]
            S_ENDPGM,
        ]);
        let r = track_words(&words, &[param(Table, 9, 1)], HwStage::Ps);
        let load = (0, 2);
        assert_eq!(r.scalar, [(load, at(0, 0))]);
        let sources: Vec<_> = r.descriptors.iter().map(|d| &d.source).collect();
        assert_eq!(sources, [&Source::Loaded { load, first: 0 }]);
        assert_eq!(r.lane_copied, [load]);
        assert_eq!(r.waterfall, Waterfall::default());
    }

    #[test]
    fn loads_at_one_first_lane_offset_have_one_address() {
        // Two loads of the table entry at the offset v4 holds, each with
        // its own first-lane read, as the console compiler loads a
        // descriptor again wherever it needs it: one address, by which
        // lowering knows them for the same descriptor.
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE83_0380, // s_mov_b32 s3, 0
                0x7E08_0504, // v_readfirstlane_b32 s4, v4
                0xF40C_0201,
                0x0800_0000, // s_load_dwordx8 s[8:15], s[2:3], s4
                0x7E0A_0504, // v_readfirstlane_b32 s5, v4
                0xF40C_0401,
                0x0A00_0000, // s_load_dwordx8 s[16:23], s[2:3], s5
                0xF000_1700,
                0x0004_0D06, // image_load v[13:15], v6, s[16:23]
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        let [(_, a), (_, b)] = r.scalar[..] else {
            panic!("{:?}", r.scalar)
        };
        assert_eq!(a, b);
        assert_eq!((a.param, a.offset), (0, 0));
        assert!(
            a.dynamic.is_some(),
            "the lane's offset is the computed term"
        );
    }

    #[test]
    fn descriptors_selected_by_scc_are_traced_per_outcome() {
        // Two buffer descriptors loaded from the table at s9 and one
        // picked with s_cselect_b64 per half, as the compiler makes a
        // conditional choice of resources without a branch.
        let r = track_words(
            &[
                0xBE8C_0309, // s_mov_b32 s12, s9
                0xBE8D_0380, // s_mov_b32 s13, 0
                0xF408_0006,
                0xFA00_0000, // s_load_dwordx4 s[0:3], s[12:13], null
                0xF408_0106,
                0xFA00_0010, // s_load_dwordx4 s[4:7], s[12:13], 0x10
                0xBF06_800E, // s_cmp_eq_u32 s14, 0
                0x8588_0400, // s_cselect_b64 s[8:9], s[0:1], s[4:5]
                0x858A_0602, // s_cselect_b64 s[10:11], s[2:3], s[6:7]
                0xF420_0404,
                0xFA00_0000, // s_buffer_load_dword s16, s[8:11], null
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        let Source::Select { args, .. } = &r.descriptors[0].source else {
            panic!("{:?}", r.descriptors)
        };
        // SCC set picks the first operand.
        assert_eq!(
            args[..],
            [
                Source::Loaded {
                    load: (0, 2),
                    first: 0
                },
                Source::Loaded {
                    load: (0, 3),
                    first: 0
                },
            ]
        );
        assert_eq!(r.scalar, [((0, 2), at(0, 0)), ((0, 3), at(0, 0x10))]);
    }

    /// The FidelityFX upscaler's byte accesses make a loaded buffer
    /// descriptor raw first (stride cleared, records times stride, bounds
    /// in bytes): the access reads the loaded descriptor. Any other change
    /// still fails.
    #[test]
    fn a_descriptor_made_raw_is_the_loaded_one() {
        let words = |mask: u32| -> Vec<u8> {
            [
                0xBE88_0304, // s_mov_b32 s8, s4
                0xBE89_0380, // s_mov_b32 s9, 0
                0xF408_0204,
                0xFA00_0160, // s_load_dwordx4 s[8:11], s[8:9], 0x160
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                0x938C_FF09,
                0x000E_0010, // s_bfe_u32 s12, s9, 0x000e0010
                0x848C_810C, // s_max_u32 s12, s12, 1
                0x8709_FF09,
                mask, // s_and_b32 s9, s9, mask
                0x870B_FF0B,
                0xCFFF_FFFF, // s_and_b32 s11, s11, 0xcfffffff
                0x930A_0C0A, // s_mul_i32 s10, s10, s12
                0x880B_FF0B,
                0x3000_0000, // s_or_b32 s11, s11, 0x30000000
                0xE078_1000,
                0x8002_0007, // buffer_store_dwordx4 v[0:3], v7, s[8:11], 0 offen
                S_ENDPGM,
            ]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes())
            .collect()
        };
        let run = |mask| {
            let p = crate::ir::build(&words(mask), false, 256).unwrap();
            track(&p, &[param(Table, 4, 1)], HwStage::Cs, lanes(&p).as_ref())
        };
        assert!(run(0xC000_FFFF).is_ok());
        assert_eq!(
            run(0xC000_FFFE).unwrap_err(),
            ShaderError::Unsupported {
                offset: 0x3c,
                what: "a descriptor changed after loading"
            }
        );
    }

    /// Immediate sampler words the upscaler's compiler makes with a bit
    /// mask and a 64-bit move of a literal.
    #[test]
    fn sampler_words_from_bit_masks_and_wide_literals_are_constants() {
        let code: Vec<u8> = [
            0xB010_3092, // s_movk_i32 s16, 0x3092
            0x9211_8C8C, // s_bfm_b32 s17, 12, 12
            0xBE92_04FF,
            0x1850_0000, // s_mov_b64 s[18:19], 0x18500000
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w: &u32| w.to_le_bytes())
        .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let layout = [param(Table, 4, 1)];
        let t = Tracer::new(&p, &layout, HwStage::Cs, None).unwrap();
        let block = &p.blocks[0];
        let value = |n: usize, k: usize| t.origin(block.insts[n].writes[k].1);
        assert_eq!(value(1, 0), Term::Const(0x00FF_F000));
        assert_eq!(value(2, 0), Term::Const(0x1850_0000));
        assert_eq!(value(2, 1), Term::Const(0));
    }


    #[test]
    fn a_selection_that_mixes_two_descriptors_fails() {
        // The high half picked from the first load whatever SCC is: when
        // clear, the descriptor is half of one load and half of the other.
        let code: Vec<u8> = [
            0xBE8C_0309, // s_mov_b32 s12, s9
            0xBE8D_0380, // s_mov_b32 s13, 0
            0xF408_0006,
            0xFA00_0000, // s_load_dwordx4 s[0:3], s[12:13], null
            0xF408_0106,
            0xFA00_0010, // s_load_dwordx4 s[4:7], s[12:13], 0x10
            0xBF06_800E, // s_cmp_eq_u32 s14, 0
            0x8588_0400, // s_cselect_b64 s[8:9], s[0:1], s[4:5]
            0x858A_0202, // s_cselect_b64 s[10:11], s[2:3], s[2:3]
            0xF420_0404,
            0xFA00_0000, // s_buffer_load_dword s16, s[8:11], null
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w: &u32| w.to_le_bytes())
        .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let err = track(&p, &[param(Table, 9, 1)], HwStage::Ps, lanes(&p).as_ref()).unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 36,
                what: "a descriptor changed after loading"
            }
        );
    }

    #[test]
    fn untraceable_addresses_fail_with_a_reason() {
        let code: Vec<u8> = [0xF408_0001, 0xFA00_0000, S_ENDPGM] // s_load_dwordx4 s[0:3], s[2:3]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes())
            .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let err = track(&p, &[param(Table, 9, 1)], HwStage::Ps, lanes(&p).as_ref()).unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 0,
                what: "an address in an SGPR that is no root parameter"
            }
        );
    }

    /// A waterfall loop over an image descriptor each lane loads from the
    /// table at s0 (wave32, assembled with llvm-mc for gfx1030): lane i
    /// reads entry i with two `global_load_dwordx4`, the loop reads the
    /// first active lane's descriptor into s[12:19], samples with the lanes
    /// whose descriptor it is, and repeats for the others. The sampler comes
    /// from the table at s1; the result goes to LDS so the sample is needed.
    pub(crate) const WATERFALL_FETCHED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0x3402_0085, // v_lshlrev_b32 v1, 5, v0
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7E18_0504, // 0x3c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0x3c
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// [`WATERFALL_FETCHED_WAVE32`] in wave64: 64-bit lane masks.
    pub(crate) const WATERFALL_FETCHED_WAVE64: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0x3402_0085, // v_lshlrev_b32 v1, 5, v0
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x7E18_0504, // 0x3c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0016,
        0x0002_080C, // v_cmp_eq_u64_e64 s[22:23], s[12:13], v[4:5]
        0xD4E2_0018,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s[24:25], s[14:15], v[6:7]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xD4E2_0018,
        0x0002_1010, // v_cmp_eq_u64_e64 s[24:25], s[16:17], v[8:9]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xD4E2_0018,
        0x0002_1412, // v_cmp_eq_u64_e64 s[24:25], s[18:19], v[10:11]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xBE96_2416, // s_and_saveexec_b64 s[22:23], s[22:23]
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x89FE_167E, // s_xor_b64 exec, exec, s[22:23]
        0xBF89_FFE6, // s_cbranch_execnz -> 0x3c
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// The same waterfall over one of two image descriptors loaded from the
    /// table at s0, picked per lane by `v_cndmask_b32` (lanes below 16 take
    /// the first).
    pub(crate) const WATERFALL_PICKED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E08_0218, // v_mov_b32 v4, s24
        0x0208_0820, // v_cndmask_b32 v4, s32, v4, vcc_lo
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x020A_0A21, // v_cndmask_b32 v5, s33, v5, vcc_lo
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x020C_0C22, // v_cndmask_b32 v6, s34, v6, vcc_lo
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x020E_0E23, // v_cndmask_b32 v7, s35, v7, vcc_lo
        0x7E10_021C, // v_mov_b32 v8, s28
        0x0210_1024, // v_cndmask_b32 v8, s36, v8, vcc_lo
        0x7E12_021D, // v_mov_b32 v9, s29
        0x0212_1225, // v_cndmask_b32 v9, s37, v9, vcc_lo
        0x7E14_021E, // v_mov_b32 v10, s30
        0x0214_1426, // v_cndmask_b32 v10, s38, v10, vcc_lo
        0x7E16_021F, // v_mov_b32 v11, s31
        0x0216_1627, // v_cndmask_b32 v11, s39, v11, vcc_lo
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7E18_0504, // 0x7c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0x7c
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// The waterfall loop of [`WATERFALL_PICKED_WAVE32`] over one of two
    /// image descriptors loaded from the table at s0, chosen by control flow
    /// as the compiler lowers an if/else on a per-lane condition: lanes below
    /// 16 move the first into v[4:11], the others the second, each arm
    /// skipped when no lane takes it. The sample goes to the buffer at dword
    /// 16 of the table, at lane * 4.
    pub(crate) const WATERFALL_JOINED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0008, // s_cbranch_execz -> 0x70
        0x7E08_0218, // v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x7E10_021C, // v_mov_b32 v8, s28
        0x7E12_021D, // v_mov_b32 v9, s29
        0x7E14_021E, // v_mov_b32 v10, s30
        0x7E16_021F, // v_mov_b32 v11, s31
        0xBE94_3D14, // 0x70: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0008, // s_cbranch_execz -> 0x9c
        0x7E08_0220, // v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x7E10_0224, // v_mov_b32 v8, s36
        0x7E12_0225, // v_mov_b32 v9, s37
        0x7E14_0226, // v_mov_b32 v10, s38
        0x7E16_0227, // v_mov_b32 v11, s39
        0x887E_147E, // 0x9c: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0x7E18_0504, // 0xa4: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0xa4
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// The console compiler's waterfall (the title's ray-tracing
    /// resolve and others): each lane holds its entry's byte offset in the
    /// table at s0 (v7 = (lane & 1) * 32); the loop reads the first active
    /// lane's offset, narrows EXEC to the lanes with the same one
    /// (`v_cmpx_eq_u32`), loads that entry with `s_load_dwordx8` and samples,
    /// then drops those lanes (`s_andn2_wrexec_b32`). The sample goes to the
    /// buffer at dword 16 of the table, at lane * 4.
    pub(crate) const WATERFALL_INDEXED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x360E_0081, // v_and_b32 v7, 1, v0
        0x340E_0E85, // v_lshlrev_b32 v7, 5, v7
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0xBE95_037E, // s_mov_b32 s21, exec_lo
        0x7E28_0507, // 0x40: v_readfirstlane_b32 s20, v7
        0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
        0xF40C_0302,
        0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
        0xBF89_FFF6, // s_cbranch_execnz -> 0x40
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    #[test]
    fn the_console_waterfall_over_a_first_lane_offset_is_one() {
        let r = waterfall(WATERFALL_INDEXED_WAVE32, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| d.role == Role::Resource)
            .unwrap();
        let Source::Loaded { load, first: 0 } = image.source else {
            panic!("{:?}", image.source)
        };
        let (_, address) = r.scalar.iter().find(|s| s.0 == load).unwrap();
        assert!(address.dynamic.is_some(), "{address:?}");
        assert_eq!(r.waterfall.compares.len(), 1);
        assert_eq!(r.waterfall.reads.len(), 1);
        assert_eq!(r.waterfall.loads, [load]);
    }

    /// [`WATERFALL_JOINED_WAVE32`] with other sources on each path: lanes
    /// below 16 load table entry (lane & 1) with two `global_load_dwordx4`,
    /// the others pick per lane with `v_cndmask_b32` (lanes below 24 the
    /// first entry, the rest the second).
    pub(crate) const WATERFALL_JOINED_MIXED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x3602_0081, // v_and_b32 v1, 1, v0
        0x3402_0285, // v_lshlrev_b32 v1, 5, v1
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0004, // s_cbranch_execz -> 0x68
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xBE94_3D14, // 0x68: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0011, // s_cbranch_execz -> 0xb8
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0x7E08_0218, // v_mov_b32 v4, s24
        0x0208_0820, // v_cndmask_b32 v4, s32, v4, vcc_lo
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x020A_0A21, // v_cndmask_b32 v5, s33, v5, vcc_lo
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x020C_0C22, // v_cndmask_b32 v6, s34, v6, vcc_lo
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x020E_0E23, // v_cndmask_b32 v7, s35, v7, vcc_lo
        0x7E10_021C, // v_mov_b32 v8, s28
        0x0210_1024, // v_cndmask_b32 v8, s36, v8, vcc_lo
        0x7E12_021D, // v_mov_b32 v9, s29
        0x0212_1225, // v_cndmask_b32 v9, s37, v9, vcc_lo
        0x7E14_021E, // v_mov_b32 v10, s30
        0x0214_1426, // v_cndmask_b32 v10, s38, v10, vcc_lo
        0x7E16_021F, // v_mov_b32 v11, s31
        0x0216_1627, // v_cndmask_b32 v11, s39, v11, vcc_lo
        0x887E_147E, // 0xb8: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E18_0504, // 0xc4: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0xc4
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// A waterfall loop whose descriptor changes on each pass: v[4:11] holds
    /// the first table entry on entry and the second after every pass, so
    /// the first lane's dwords are phis of the loop header.
    pub(crate) const WATERFALL_LOOPED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7E08_0218, // v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x7E10_021C, // v_mov_b32 v8, s28
        0x7E12_021D, // v_mov_b32 v9, s29
        0x7E14_021E, // v_mov_b32 v10, s30
        0x7E16_021F, // v_mov_b32 v11, s31
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0x7E18_0504, // 0x64: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0x7E08_0220, // v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x7E10_0224, // v_mov_b32 v8, s36
        0x7E12_0225, // v_mov_b32 v9, s37
        0x7E14_0226, // v_mov_b32 v10, s38
        0x7E16_0227, // v_mov_b32 v11, s39
        0xBF89_FFDE, // s_cbranch_execnz -> 0x64
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// A waterfall loop over one of two buffer descriptors (four dwords)
    /// loaded from the table at s0, chosen by nested if/else: lanes below 16
    /// move the first into v[4:7], lanes from 16 to 23 the second, and the
    /// others keep what v[4:7] held. Only lanes below 24 (the same mask)
    /// run the loop, which loads dword `lane` of the chosen buffer, and
    /// store it to the buffer at dword 16 of the table.
    pub(crate) const WATERFALL_JOINED_BUFFER_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0602,
        0xFA00_0000, // s_load_dwordx4 s[24:27], s[4:5], 0x0
        0xF408_0802,
        0xFA00_0020, // s_load_dwordx4 s[32:35], s[4:5], 0x20
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xD4C4_001E,
        0x0002_0098, // v_cmp_gt_u32_e64 s30, 24, v0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0004, // s_cbranch_execz -> 0x50
        0x7E08_0218, // 0x40: v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0xBE94_3D14, // 0x50: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0007, // s_cbranch_execz -> 0x78
        0xBE9F_3C1E, // s_and_saveexec_b32 s31, s30
        0xBF88_0004, // s_cbranch_execz -> 0x74
        0x7E08_0220, // 0x64: v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x887E_1F7E, // 0x74: s_or_b32 exec_lo, exec_lo, s31
        0x887E_147E, // 0x78: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_3C1E, // s_and_saveexec_b32 s23, s30
        0xBF88_0013, // s_cbranch_execz -> 0xd0
        0xBE9D_037E, // s_mov_b32 s29, exec_lo
        0x7E18_0504, // 0x88: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xE030_1000,
        0x8003_0C10, // buffer_load_dword v12, v16, s[12:15], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFF1, // s_cbranch_execnz -> 0x88
        0xBEFE_031D, // s_mov_b32 exec_lo, s29
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0x887E_177E, // 0xd0: s_or_b32 exec_lo, exec_lo, s23
        0xBF81_0000, // s_endpgm
    ];

    fn waterfall(words: &[u32], wave64: bool) -> Result<Resources, ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, wave64, 256).unwrap();
        let layout = [param(Table, 0, 1), param(Table, 1, 1)];
        track(&p, &layout, HwStage::Cs, lanes(&p).as_ref())
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_per_lane_loads_are_traced_to_them() {
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
        ] {
            let r = waterfall(words, wave64).unwrap();
            let image = r
                .descriptors
                .iter()
                .find(|d| d.role == Role::Resource)
                .unwrap();
            let Source::Fetched { load, first: 0 } = image.source else {
                panic!("{:?}", image.source)
            };
            // The first of the two loads: entry v0 * 32 of the table at s0.
            let a = r.scalar.iter().find(|s| s.0 == load).unwrap().1;
            assert_eq!((a.param, a.offset), (0, 0));
            assert!(a.dynamic.is_some());
            // The sampler is the same for every lane.
            let sampler = r.descriptors.iter().find(|d| d.role == Role::Sampler);
            assert!(matches!(
                sampler.unwrap().source,
                Source::Loaded { first: 0, .. }
            ));
            // Four 64-bit compares of the first lane's dwords with the lane's
            // own; the eight reads become the lane's own dwords.
            assert_eq!(r.waterfall.compares.len(), 4);
            assert_eq!(r.waterfall.reads.len(), 8);
        }
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_a_per_lane_choice_are_traced_per_lane() {
        let r = waterfall(WATERFALL_PICKED_WAVE32, false).unwrap();
        let image = &r.descriptors[0];
        let Source::Lanes { args, .. } = &image.source else {
            panic!("{:?}", image.source)
        };
        // Lanes with their bit set take the first load (offset 0), the
        // others the second (0x20).
        let offsets: Vec<u32> = args
            .iter()
            .map(|a| match *a {
                Source::Loaded { load, first: 0 } => {
                    r.scalar.iter().find(|s| s.0 == load).unwrap().1.offset
                }
                ref a => panic!("{a:?}"),
            })
            .collect();
        assert_eq!(offsets, [0, 0x20]);
        assert_eq!(r.waterfall.compares.len(), 4);
    }

    #[test]
    fn a_first_lane_read_outside_a_loop_is_no_waterfall() {
        // The waterfall without its back edge: the s_cbranch_execnz is an
        // s_nop, so lanes the first one does not match never get a pass.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        let back = words.iter().position(|&w| w == 0xBF89_FFE6).unwrap();
        words[back] = 0xBF80_0000;
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane outside a waterfall loop",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// Why resource tracking refuses `words`.
    fn refusal(words: &[u32], wave64: bool) -> &'static str {
        match waterfall(words, wave64) {
            Err(ShaderError::Unsupported { what, .. }) => what,
            other => panic!("not refused: {other:?}"),
        }
    }

    const NO_LOOP: &str = "a descriptor read from the first lane outside a waterfall loop";
    const NOT_SELECTED: &str = "a waterfall loop whose compares do not select the access's lanes";
    const BAD_RETRY: &str = "a waterfall loop whose retry does not keep the lanes still waiting";
    const SPLIT: &str = "a waterfall loop nested in another or split over blocks";

    /// The index of `word` in `words`.
    fn word_at(words: &[u32], word: u32) -> usize {
        words.iter().position(|&w| w == word).unwrap()
    }

    #[test]
    fn a_first_lane_read_without_its_compare_is_no_waterfall() {
        // The loop without the compare of dwords 0 and 1: nothing makes
        // the first lane's dwords the lane's own there.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        let cmp = word_at(&words, 0x0002_080C);
        words[cmp - 1..=cmp].copy_from_slice(&[0xBF80_0000, 0xBF80_0000]);
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn a_compare_that_does_not_mask_the_access_is_no_waterfall() {
        // The compares are all there, but their result never narrows EXEC:
        // every lane samples with the first lane's descriptor (reported by
        // ifBars on upstream PR #566).
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let save = word_at(&words, 0xBE95_3C15);
        words[save] = 0xBE95_037E; // s_mov_b32 s21, exec_lo
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn an_access_under_a_further_mask_is_no_waterfall() {
        // The access runs for the matching lanes that VCC also holds, yet
        // the retry removes every matching lane.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let nop = word_at(&words, 0xBE95_3C15) + 1;
        assert_eq!(words[nop], 0xBF80_0000);
        words[nop] = 0x877E_6A7E; // s_and_b32 exec_lo, exec_lo, vcc_lo
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn a_loop_that_drops_unmatched_lanes_is_no_waterfall() {
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let retry = word_at(&words, 0x897E_157E);
        words[retry] = 0xBEFE_0380; // s_mov_b32 exec_lo, 0
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_retry_that_keeps_other_lanes_than_those_waiting_is_no_waterfall() {
        // The retry keeps the lanes of VCC that did not just sample, not
        // the waiting ones: lanes 16 and up never sample on the console.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let retry = word_at(&words, 0x897E_157E);
        words[retry] = 0x8A7E_7E6A; // s_andn2_b32 exec_lo, vcc_lo, exec_lo
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_wave64_retry_of_one_half_is_no_waterfall() {
        // Only the low half of EXEC loses the lanes just served.
        let mut words = WATERFALL_FETCHED_WAVE64.to_vec();
        let retry = word_at(&words, 0x89FE_167E);
        words[retry] = 0x897E_167E; // s_xor_b32 exec_lo, exec_lo, s22
        assert_eq!(refusal(&words, true), BAD_RETRY);
    }

    #[test]
    fn a_descriptor_used_after_exec_is_restored_is_no_waterfall() {
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let sample = word_at(&words, 0xF09C_0F08);
        let restore = word_at(&words, 0xBEFE_0317);
        let image = [words[sample], words[sample + 1]];
        words[sample..sample + 2].fill(0xBF80_0000); // Keep the branch offsets.
        words.splice(restore + 1..restore + 1, image);
        assert_eq!(refusal(&words, false), NO_LOOP);
    }

    #[test]
    fn a_wave64_access_with_exec_hi_overwritten_is_no_waterfall() {
        let mut words = WATERFALL_FETCHED_WAVE64.to_vec();
        let save = word_at(&words, 0xBE96_2416);
        words[save + 1] = 0xBEFF_03C1; // s_mov_b32 exec_hi, -1
        assert_eq!(refusal(&words, true), NOT_SELECTED);
    }

    /// The waterfall with `v_cmpx_eq` narrowing EXEC itself, and the retry
    /// as `todo & !EXEC` (`folded`: in one `s_andn2_wrexec`), as LLVM's
    /// AMDGPU backend writes it for GFX10 and later.
    pub(crate) fn cmpx_waterfall(wave64: bool, folded: bool) -> Vec<u32> {
        let (base, todo, restore): (&[u32], u32, u32) = if wave64 {
            (WATERFALL_FETCHED_WAVE64, 0xBE94_047E, 0xBEFE_0414)
        } else {
            (WATERFALL_JOINED_WAVE32, 0xBE97_037E, 0xBEFE_0317)
        };
        let reads = word_at(base, 0x7E18_0504);
        let sample = word_at(base, 0xF09C_0F08);
        let mut w = base[..=word_at(base, todo)].to_vec();
        // s_mov_b32 s21, exec_lo / s_mov_b64 s[22:23], exec
        w.push(if wave64 { 0xBE96_047E } else { 0xBE95_037E });
        w.extend(&base[reads..reads + 8]); // v_readfirstlane_b32 s12..s19
        w.extend([
            0x7DE4_080C, // v_cmpx_eq_u64 s[12:13], v[4:5]
            0x7DE4_0C0E, // v_cmpx_eq_u64 s[14:15], v[6:7]
            0x7DE4_1010, // v_cmpx_eq_u64 s[16:17], v[8:9]
            0x7DE4_1412, // v_cmpx_eq_u64 s[18:19], v[10:11]
        ]);
        w.extend(&base[sample..sample + 2]); // image_sample_lz
        match (wave64, folded) {
            // s_andn2_wrexec_b32 s21, s21; s_cbranch_execnz
            (false, true) => w.extend([0xBE95_4715, 0xBF89_FFF0]),
            // s_andn2_b32 s21, s21, exec_lo; s_mov_b32 exec_lo, s21
            (false, false) => w.extend([0x8A15_7E15, 0xBEFE_0315, 0xBF89_FFEF]),
            // s_andn2_wrexec_b64 s[22:23], s[22:23]
            (true, true) => w.extend([0xBE96_3A16, 0xBF89_FFF0]),
            // s_andn2_b64 s[22:23], s[22:23], exec; s_mov_b64 exec, s[22:23]
            (true, false) => w.extend([0x8A96_7E16, 0xBEFE_0416, 0xBF89_FFEF]),
        }
        w.extend(&base[word_at(base, restore)..]);
        w
    }

    #[test]
    fn a_waterfall_narrowed_by_v_cmpx_is_one() {
        for wave64 in [false, true] {
            for folded in [false, true] {
                let r = waterfall(&cmpx_waterfall(wave64, folded), wave64)
                    .unwrap_or_else(|e| panic!("wave64 {wave64} folded {folded}: {e:?}"));
                assert_eq!(r.waterfall.compares.len(), 4, "{wave64} {folded}");
                assert_eq!(r.waterfall.reads.len(), 8, "{wave64} {folded}");
            }
        }
    }

    #[test]
    fn a_v_cmpx_waterfall_whose_retry_drops_lanes_is_none() {
        let mut words = cmpx_waterfall(false, false);
        let retry = word_at(&words, 0x8A15_7E15);
        words[retry] = 0x8A15_7E6A; // s_andn2_b32 s21, vcc_lo, exec_lo
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_waterfall_nested_in_another_is_refused_by_name() {
        // An image whose resource and sampler are both per lane, with one
        // loop per operand, the one for the sampler inside the one for the
        // resource (what the SelectionDAG legalisation in LLVM would give).
        // The console gives each lane its own pair. Refused for now: the
        // access of the inner loop reads the first-lane dwords of the outer
        // loop, which another block holds.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0x7E04_0D00, // v_cvt_f32_u32 v2, v0
            0x7E06_0302, // v_mov_b32 v3, v2
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0x3602_0081, // v_and_b32 v1, 1, v0
            0x3402_0285, // v_lshlrev_b32 v1, 5, v1
            0x3622_0080, // v_and_b32 v17, 0, v0
            0x3422_2284, // v_lshlrev_b32 v17, 4, v17
            0xDC38_8000,
            0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
            0xDC38_8010,
            0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
            0xDC38_8000,
            0x1406_0011, // global_load_dwordx4 v[20:23], v17, s[6:7]
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0xBE94_037E, // s_mov_b32 s20, exec_lo
            // L1, the loop over the resource:
            0x7E18_0504, // v_readfirstlane_b32 s12, v4
            0x7E1A_0505,
            0x7E1C_0506,
            0x7E1E_0507,
            0x7E20_0508,
            0x7E22_0509,
            0x7E24_050A,
            0x7E26_050B, // v_readfirstlane_b32 s19, v11
            0xD4E2_0015,
            0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
            0xD4E2_0016,
            0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xD4E2_0016,
            0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xD4E2_0016,
            0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xBE95_3C15, // s_and_saveexec_b32 s21, s21
            0xBE9A_037E, // s_mov_b32 s26, exec_lo
            // L2, the loop over the sampler:
            0x7E10_0514, // v_readfirstlane_b32 s8, v20
            0x7E12_0515, // v_readfirstlane_b32 s9, v21
            0x7E14_0516, // v_readfirstlane_b32 s10, v22
            0x7E16_0517, // v_readfirstlane_b32 s11, v23
            0xD4E2_0018,
            0x0002_2808, // v_cmp_eq_u64_e64 s24, s[8:9], v[20:21]
            0xD4E2_0019,
            0x0002_2C0A, // v_cmp_eq_u64_e64 s25, s[10:11], v[22:23]
            0x8718_1918, // s_and_b32 s24, s24, s25
            0xBE98_3C18, // s_and_saveexec_b32 s24, s24
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
            0x897E_187E, // s_xor_b32 exec_lo, exec_lo, s24
            0xBF89_FFF2, // s_cbranch_execnz L2
            0xBEFE_031A, // s_mov_b32 exec_lo, s26
            0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
            0xBF89_FFDA, // s_cbranch_execnz L1
            0xBEFE_0314, // s_mov_b32 exec_lo, s20
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        assert_eq!(refusal(&words, false), SPLIT);
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_lds_fail_naming_it() {
        // The waterfall over descriptors each lane read from LDS:
        // ds_read_b128 v[4:7], v1 and ds_read_b128 v[8:11], v1 offset:16
        // in place of the global loads.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        words[5..9].copy_from_slice(&[0xDBFC_0000, 0x0400_0001, 0xDBFC_0010, 0x0800_0001]);
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of an LDS read",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// The block of `p` at byte `offset`.
    fn block_at(words: &[u32], offset: usize) -> usize {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        p.blocks.iter().position(|b| b.offset == offset).unwrap()
    }

    /// The table offset of what the scalar load of `s` reads.
    fn table_offset(r: &Resources, s: &Source) -> u32 {
        match *s {
            Source::Loaded { load, first: 0 } => {
                r.scalar.iter().find(|x| x.0 == load).unwrap().1.offset
            }
            ref s => panic!("{s:?}"),
        }
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_an_if_else_join_are_traced_per_path() {
        let words = WATERFALL_JOINED_WAVE32;
        let r = waterfall(words, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        // The join after the else arm: lanes that ran it (lanes from 16 on)
        // take the second entry (0x20); the others come through the flow
        // block, whose join gives lanes that ran the then arm the first one
        // (0). No lane that skipped the then arm comes that way, so the
        // dwords they held before (the inputs) are not traced.
        let Source::LanePhi { block, args } = &image.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0x9c));
        let [Some(flow), Some(other)] = &args[..] else {
            panic!("{args:?}")
        };
        assert_eq!(table_offset(&r, other), 0x20);
        let Source::LanePhi { block, args } = flow else {
            panic!("{flow:?}")
        };
        assert_eq!(*block, block_at(words, 0x70));
        let [None, Some(then)] = &args[..] else {
            panic!("{args:?}")
        };
        assert_eq!(table_offset(&r, then), 0);
        // The loop's compares are of the joined dwords; still recognised.
        assert_eq!(r.waterfall.compares.len(), 4);
        assert_eq!(r.waterfall.reads.len(), 8);
        // Without the lanes' paths the join is refused.
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let layout = [param(Table, 0, 1), param(Table, 1, 1)];
        let err = track(&p, &layout, HwStage::Cs, None).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of a phi, without the lanes' paths",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn descriptors_joined_from_a_per_lane_load_and_a_per_lane_choice_are_traced_per_path() {
        let words = WATERFALL_JOINED_MIXED_WAVE32;
        let r = waterfall(words, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        let Source::LanePhi { block, args } = &image.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0xb8));
        let [
            Some(Source::LanePhi { args: flow, .. }),
            Some(Source::Lanes { args: picked, .. }),
        ] = &args[..]
        else {
            panic!("{args:?}")
        };
        // The then arm loads each lane's entry from the table at s0.
        let [None, Some(Source::Fetched { load, first: 0 })] = flow[..] else {
            panic!("{flow:?}")
        };
        let a = r.scalar.iter().find(|s| s.0 == load).unwrap().1;
        assert_eq!((a.param, a.offset), (0, 0));
        assert!(a.dynamic.is_some());
        // The else arm picks the first entry where vcc is set.
        let offsets: Vec<u32> = picked.iter().map(|s| table_offset(&r, s)).collect();
        assert_eq!(offsets, [0, 0x20]);
        assert_eq!(r.waterfall.compares.len(), 4);
    }

    #[test]
    fn a_descriptor_read_from_the_first_lane_of_a_loop_phi_fails_naming_it() {
        let err = waterfall(WATERFALL_LOOPED_WAVE32, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of a phi at a loop header",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_per_lane_descriptor_some_lanes_do_not_write_fails() {
        // The else arm of WATERFALL_JOINED_WAVE32 not skipped (its
        // s_cbranch_execz an s_nop): it runs with EXEC clear in the lanes
        // of the then arm, which keep the first entry; the second is not
        // every lane's.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let skip = words.iter().rposition(|&w| w == 0xBF88_0008).unwrap();
        words[skip] = 0xBF80_0000;
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a per-lane descriptor written where some lanes may keep the old one",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_buffer_descriptor_joined_per_lane_is_traced_on_the_paths_of_lanes_that_load() {
        let words = WATERFALL_JOINED_BUFFER_WAVE32;
        let r = waterfall(words, false).unwrap();
        let buffer = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        // The outer join: lanes from the else arm's inner join, or (below
        // 16) through the flow block.
        let Source::LanePhi { block, args } = &buffer.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0x78));
        let [
            Some(Source::LanePhi { args: flow, .. }),
            Some(Source::LanePhi {
                block: inner,
                args: nested,
            }),
        ] = &args[..]
        else {
            panic!("{args:?}")
        };
        let [None, Some(then)] = &flow[..] else {
            panic!("{flow:?}")
        };
        assert_eq!(table_offset(&r, then), 0);
        // Lanes from 24 on skip the inner arm and keep the inputs, but do
        // not load: that edge is not traced.
        assert_eq!(*inner, block_at(words, 0x74));
        let [None, Some(other)] = &nested[..] else {
            panic!("{nested:?}")
        };
        assert_eq!(table_offset(&r, other), 0x20);
        assert_eq!(r.waterfall.compares.len(), 2);
        assert_eq!(r.waterfall.reads.len(), 4);
    }

    #[test]
    fn a_guarded_access_excludes_dead_lane_edges_at_a_join() {
        // Lanes failing `v_cmp_gt_u32` clear EXEC and take `s_cbranch_execz`
        // directly to the join block without initializing the immediate
        // sampler `s[8:11]`. The join's `image_sample_lz` is guarded by EXEC,
        // so only the arm where `s[8:11]` was built can execute it.
        let code: Vec<u8> = [
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xF40C_0302,
            0xFA00_0000,    // s_load_dwordx8 s[12:19], s[4:5], 0x0
            0x7E04_0D00,    // v_cvt_f32_u32 v2, v0
            0x7E06_0302,    // v_mov_b32 v3, v2
            0x7D88_0090,    // v_cmp_gt_u32 vcc_lo, 16, v0
            0xBF8C_C07F,    // s_waitcnt lgkmcnt(0)
            0x877E_6A7E,    // s_and_b32 exec_lo, exec_lo, vcc_lo
            0xBF88_0004,    // s_cbranch_execz -> join (skips 4 dwords)
            0xBE88_0380,    // s_mov_b32 s8, 0
            0xBE89_0380,    // s_mov_b32 s9, 0
            0xBE8A_0380,    // s_mov_b32 s10, 0
            0xBE8B_0380,    // s_mov_b32 s11, 0
            0xF09C_0F08,
            0x0043_0C02,    // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let structured = lanes(&p).expect("structured form");
        assert!(structured.guards[2][0].is_some(), "join access must carry an EXEC guard");
        let r = track(&p, &[param(Table, 0, 1)], HwStage::Cs, Some(&structured)).unwrap();
        assert!(
            r.descriptors
                .iter()
                .any(|d| d.role == Role::Sampler && d.source == Source::Constant(vec![0; 4])),
            "guarded join should resolve the immediate sampler from the live arm: {:?}",
            r.descriptors
        );
    }

    #[test]
    fn root_constant_zero_offsets_tables_and_root_descriptors() {
        // RootType::Constants dword 0 (s0) as an SMEM soffset and via s_add_u32
        // on a descriptor table base (s2).
        let r = track_words(
            &[
                0xBE84_0302, // s_mov_b32 s4, s2
                0xBE85_0380, // s_mov_b32 s5, 0
                0xF408_0202,
                0x0000_0020, // s_load_dwordx4 s[8:11], s[4:5], s0 offset:0x20
                0x8006_0002, // s_add_u32 s6, s2, s0
                0xBE87_0380, // s_mov_b32 s7, 0
                0xF408_0303,
                0xFA00_0040, // s_load_dwordx4 s[12:15], s[6:7], 0x40
                0xE030_1000,
                0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 offen
                0xE070_1000,
                0x8003_0100, // buffer_store_dword v1, v0, s[12:15], 0 offen
                S_ENDPGM,
            ],
            &[param(Constants, 0, 2), param(Table, 2, 1)],
            HwStage::Cs,
        );
        let c0 = r.params.iter().find(|p| (p.1, p.2) == (0, 0)).unwrap().0;
        assert_eq!(
            r.scalar,
            [
                (
                    (0, 2),
                    Address {
                        param: 1,
                        offset: 0x20,
                        dynamic: Some(c0),
                    },
                ),
                (
                    (0, 5),
                    Address {
                        param: 1,
                        offset: 0x40,
                        dynamic: Some(c0),
                    },
                ),
            ]
        );
        assert_eq!(r.descriptors.len(), 2);

        // RootType::Constants dword 0 spilled through a VGPR lane and used as a
        // table soffset.
        let spilled = track_words(
            &[
                0xD761_0000,
                0x0001_0600, // v_writelane_b32 v0, s0, 3
                0xD760_0004,
                0x0001_0700, // v_readlane_b32 s4, v0, 3
                0xBE87_0380, // s_mov_b32 s7, 0
                0xF408_0203,
                0x0800_0000, // s_load_dwordx4 s[8:11], s[6:7], s4
                S_ENDPGM,
            ],
            &[param(Constants, 0, 2), param(Table, 6, 1)],
            HwStage::Cs,
        );
        let sc0 = spilled
            .params
            .iter()
            .find(|p| (p.1, p.2) == (0, 0))
            .unwrap()
            .0;
        assert_eq!(
            spilled.scalar,
            [(
                (0, 3),
                Address {
                    param: 1,
                    offset: 0,
                    dynamic: Some(sc0),
                },
            )]
        );

        // RootType::Constants dword 0 as byte offset on RootType::Descriptor
        // (s_load_dword and global_load_dword), while (dword 0, dword 1) as a
        // 64-bit pointer still classifies as MEMORY.
        let desc = track_words(
            &[
                0xF400_0101,
                0x0000_0008, // s_load_dword s4, s[2:3], s0 offset:0x8
                0x7E02_0200, // v_mov_b32 v1, s0
                0xDC30_800C,
                0x0202_0001, // global_load_dword v2, v1, s[2:3] offset:12
                0x4A06_0290, // v_add_nc_u32 v3, 16, v1
                0xDC30_8004,
                0x0402_0003, // global_load_dword v4, v3, s[2:3] offset:4
                0xF400_0180,
                0xFA00_0000, // s_load_dword s6, s[0:1], null
                0xDC30_8000,
                0x0500_0000, // global_load_dword v5, v0, s[0:1]
                S_ENDPGM,
            ],
            &[param(Constants, 0, 2), param(Descriptor, 2, 2)],
            HwStage::Cs,
        );
        let dc0 = desc.params.iter().find(|p| (p.1, p.2) == (0, 0)).unwrap().0;
        assert_eq!(
            desc.scalar,
            [
                (
                    (0, 0),
                    Address {
                        param: 1,
                        offset: 8,
                        dynamic: Some(dc0),
                    },
                ),
                (
                    (0, 2),
                    Address {
                        param: 1,
                        offset: 12,
                        dynamic: Some(dc0),
                    },
                ),
                (
                    (0, 4),
                    Address {
                        param: 1,
                        offset: 20,
                        dynamic: Some(dc0),
                    },
                ),
                ((0, 5), at(MEMORY, 0)),
                ((0, 6), at(MEMORY, 0)),
            ]
        );
    }

    #[test]
    fn root_constant_zero_offsets_emit_to_dxil() {
        use crate::native::{ComputeIo, NativeProgram, ResourceType, StageIo, UsedResource};

        let words: &[u32] = &[
            0xBE86_0302, // s_mov_b32 s6, s2
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF408_0203,
            0x0000_0000, // s_load_dwordx4 s[8:11], s[6:7], s0
            0x800C_0002, // s_add_u32 s12, s2, s0
            0xBE8D_0380, // s_mov_b32 s13, 0
            0xF408_0306,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[12:13], 0x20
            0x7E02_0200, // v_mov_b32 v1, s0
            0xDC30_8008,
            0x0204_0001, // global_load_dword v2, v1, s[4:5] offset:8
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0xE030_1000,
            0x8002_0300, // buffer_load_dword v3, v0, s[8:11], 0 offen
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x4A04_0702, // v_add_nc_u32 v2, v2, v3
            0xE070_1000,
            0x8003_0200, // buffer_store_dword v2, v0, s[12:15], 0 offen
            S_ENDPGM,
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![
                RootParam {
                    slot: 0,
                    ..param(Constants, 0, 2)
                },
                RootParam {
                    slot: 1,
                    ..param(Table, 2, 1)
                },
                RootParam {
                    slot: 2,
                    ..param(Descriptor, 4, 2)
                },
            ],
            used: vec![
                UsedResource {
                    kind: ResourceType::Srv,
                    root: Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 2,
                    slot: 1,
                },
                UsedResource {
                    kind: ResourceType::Uav,
                    root: Table,
                    table_offset_dwords: 8,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 2,
                    slot: 1,
                },
                UsedResource {
                    kind: ResourceType::Srv,
                    root: Descriptor,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 2,
                    user_data: 4,
                    slot: 2,
                },
            ],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 6 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        assert!(!res.scalar.iter().any(|(_, a)| a.param == MEMORY));
        let io = crate::io::raise(&p, &native).unwrap();
        let bytes = crate::dxbc::build(&[]);
        let empty = crate::Container::parse(&bytes).unwrap();
        let shader = crate::emit::emit(&empty, &native, &p, &st, &io, &res).unwrap();
        assert!(shader.user_data.contains(&crate::emit::UserData {
            index: 0,
            slot: 0,
            root: Constants,
            dword: 0,
        }));
        #[cfg(windows)]
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut shader.dxil.clone())
                .expect("DXIL validation failed");
        }
    }
}
