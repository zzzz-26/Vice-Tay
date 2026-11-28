//! One-to-one SSA form of a native program (decision 0003, step 1).
//!
//! Every machine instruction becomes one [`Inst`], except scheduling-only
//! ones (`s_nop`, `s_waitcnt*`, `s_inst_prefetch`...) and redundant conditional
//! branches.
//! Each register dword is a variable: a read names the [`Value`] it sees, a
//! write defines a new one, and [`Phi`]s merge values where control flow
//! joins. Nothing is raised: an instruction keeps its opcode and decoded
//! fields, so its exact semantics stay with it.
//!
//! Implicit operands are explicit: SCC, M0, and EXEC, which every vector
//! instruction reads. A write to a VGPR also reads the register's old value,
//! because lanes disabled in EXEC keep it.
//!
//! M0-relative VGPR access (`v_movrels/d/sd_b32`, `v_swaprel_b32`) names its
//! operand's whole possible range: from the encoded register up to the
//! program's VGPR count, all read (and all written, for a destination).

use std::collections::{HashMap, HashSet};

use crate::ShaderError;
use crate::gfx10::{self, Fields, Flow, Instruction, Operand};

/// One register dword. Scalar codes as in [`Operand::Scalar`]: 0-105 SGPRs,
/// 106/107 VCC, 108-123 trap temporaries, 124 M0, 126/127 EXEC. Codes from
/// [`PARKED`] up are scalars the compiler parked in VGPR lanes (`unspill`):
/// no machine instruction names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reg {
    S(u8),
    Scc,
    V(u8),
}

pub const VCC: u8 = 106;
pub const M0: u8 = 124;
pub const EXEC: u8 = 126;
/// The first scalar code past the machine's (0-127): see [`Reg`].
pub const PARKED: u8 = 128;

/// An SSA value: the content of one register dword (SCC: one bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Value(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub wave64: bool,
    /// VGPRs v0 up to this exist; bounds M0-relative access.
    pub vgprs: u16,
    /// In code order, then copies made to give every loop one entry; block 0
    /// is the entry.
    pub blocks: Vec<Block>,
    /// Registers read before any write, with their value on entry: what the
    /// hardware or a called fetch routine set up (user SGPRs, system values,
    /// vertex inputs).
    pub inputs: Vec<(Reg, Value)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// Byte offset of the block's first machine instruction (shared by a
    /// block and its copies).
    pub offset: usize,
    pub preds: Vec<usize>,
    pub phis: Vec<Phi>,
    pub insts: Vec<Inst>,
    pub exit: Exit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    Jump(usize),
    /// `s_cbranch_*`, the block's last instruction.
    Branch {
        taken: usize,
        not_taken: usize,
    },
    /// `s_endpgm` and its variants.
    End,
    /// `s_setpc_b64` to an address held in registers, e.g. the handover to
    /// the next half of a merged hardware stage.
    Indirect,
}

impl Exit {
    pub fn successors(&self) -> Vec<usize> {
        match *self {
            Exit::Jump(b) => vec![b],
            Exit::Branch { taken, not_taken } => vec![taken, not_taken],
            Exit::End | Exit::Indirect => vec![],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi {
    pub reg: Reg,
    pub value: Value,
    /// One per predecessor, in [`Block::preds`] order.
    pub args: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inst {
    pub machine: Instruction,
    pub fields: Fields,
    /// Every register dword read, with the value it holds.
    pub reads: Vec<(Reg, Value)>,
    /// Every register dword written, with the value it receives.
    pub writes: Vec<(Reg, Value)>,
}

impl Inst {
    pub fn name(&self) -> &'static str {
        self.machine.name().unwrap_or("")
    }

    /// The value `reg` holds when this instruction reads it.
    pub fn read(&self, reg: Reg) -> Option<Value> {
        self.reads.iter().find(|r| r.0 == reg).map(|r| r.1)
    }

    /// The value this instruction writes to `reg`.
    pub fn write(&self, reg: Reg) -> Option<Value> {
        self.writes.iter().find(|w| w.0 == reg).map(|w| w.1)
    }
}

/// Decodes `code` and builds its SSA form; `vgprs` is the program's VGPR
/// count. Fails on instructions the IR cannot express yet
/// ([`ShaderError::Unsupported`]).
pub fn build(code: &[u8], wave64: bool, vgprs: u16) -> Result<Program, ShaderError> {
    let mut list = gfx10::decode(code)?;
    let lanes = 1 + u8::from(wave64);
    let isa = |offset, reason| ShaderError::Isa { offset, reason };

    let mut flows = Vec::with_capacity(list.len());
    let mut leaders = vec![0];
    for (i, inst) in list.iter().enumerate() {
        let flow = gfx10::control(inst, code.len(), |at| {
            list[..list.partition_point(|i| i.offset < at)].last()
        })?;
        if let Flow::Jump(t) | Flow::Branch(t) = flow {
            leaders.push(t);
        }
        if flow != Flow::Next
            && let Some(next) = list.get(i + 1)
        {
            leaders.push(next.offset);
        }
        flows.push(flow);
    }
    leaders.sort_unstable();
    leaders.dedup();
    let parked = unspill(&mut list, &flows, &leaders, wave64, lanes, vgprs);
    let block_at = |offset: usize| {
        leaders
            .binary_search(&offset)
            .map_err(|_| isa(offset, "jump into a block"))
    };

    let mut blocks: Vec<Block> = leaders
        .iter()
        .map(|&offset| Block {
            offset,
            preds: Vec::new(),
            phis: Vec::new(),
            insts: Vec::new(),
            exit: Exit::End,
        })
        .collect();
    let next_offsets: Vec<Option<usize>> = (0..list.len())
        .map(|i| list.get(i + 1).map(|n| n.offset))
        .collect();
    for (n, ((machine, flow), next)) in list.into_iter().zip(flows).zip(next_offsets).enumerate() {
        let b = leaders.partition_point(|&l| l <= machine.offset) - 1;
        let name = machine.name().ok_or(ShaderError::Unsupported {
            offset: machine.offset,
            what: "unknown opcode",
        })?;
        let mut fields = machine.fields(wave64);
        // A move to or from a scalar parked in a VGPR lane (`unspill`): its register
        // codes may be ones the machine word cannot hold.
        if let Ok(k) = parked.binary_search_by_key(&n, |m| m.0)
            && let Fields::Sop { dst: Some(Operand::Scalar { code: to, .. }), src, .. } = &mut fields
            && let Some(Operand::Scalar { code: from, .. }) = src.first_mut()
        {
            (*to, *from) = (parked[k].1, parked[k].2);
        }
        let fallthrough = || block_at(next.ok_or(isa(machine.offset, "falls off the code"))?);
        // A conditional SOPP branch to fallthrough has no register effects
        // and contributes one edge to the join. Calls also use Flow::Branch
        // but write a return address.
        if machine.encoding == gfx10::Encoding::Sopp
            && matches!(flow, Flow::Branch(target) if Some(target) == next)
        {
            blocks[b].exit = Exit::Jump(fallthrough()?);
            continue;
        }
        let ends = flow != Flow::Next || next.is_none_or(|n| leaders.binary_search(&n).is_ok());
        if ends {
            blocks[b].exit = match flow {
                Flow::Next => Exit::Jump(fallthrough()?),
                Flow::Jump(t) => Exit::Jump(block_at(t)?),
                Flow::Branch(t) => Exit::Branch {
                    taken: block_at(t)?,
                    not_taken: fallthrough()?,
                },
                Flow::Stop if name == "s_setpc_b64" => Exit::Indirect,
                Flow::Stop => Exit::End,
            };
        }
        if let Some((mut reads, writes)) = effects(&machine, &fields, name, lanes, vgprs)? {
            // Lanes disabled in EXEC keep the old value.
            for w in writes.iter().filter(|w| matches!(w, Reg::V(_))) {
                if !reads.contains(w) {
                    reads.push(*w);
                }
            }
            let unset = |regs: Vec<Reg>| regs.into_iter().map(|r| (r, NONE)).collect();
            blocks[b].insts.push(Inst {
                machine,
                fields,
                reads: unset(reads),
                writes: unset(writes),
            });
        }
    }
    link(&mut blocks);
    if !blocks[0].preds.is_empty() {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "a branch to the program entry",
        });
    }
    split_irreducible(&mut blocks)?;

    Ok(rename(blocks, wave64, vgprs))
}

/// Scalar registers the compiler parked in lanes of a VGPR, as scalar moves.
///
/// When it runs out of SGPRs the console compiler saves one in a lane of a
/// spare VGPR (`v_writelane_b32 vN, sX, K`) and takes it back later
/// (`v_readlane_b32 sX, vN, K`). A VGPR whose every use in the program is one
/// of those two forms with a constant lane is such a store: lane K of it is a
/// scalar register by another name. Each (VGPR, lane) gets a register code
/// the program does not use (the trap temporaries first, then SGPRs it never
/// names, then codes from [`PARKED`] up, which no machine word can hold), and
/// the save and the restore become `s_mov_b32` to and from it. Returns those
/// moves (index in `list`, destination code, source code) in order, for
/// `build` to put the codes into the decoded fields. They are then copies for
/// every later stage: a parked descriptor is traced to its load, and nothing
/// is left of a cross-lane operation, which a console wave run as two host
/// waves could not do where not every lane runs (the lane may be in the other
/// wave, or not running).
/// A VGPR of other uses is such a store too from its first save on, when every
/// use from there is one of the two forms, in one block, and saves a lane
/// before it restores it (a register free again after a wave reduction).
/// A register pair parked in two neighbouring lanes (a lane mask, a 64-bit
/// address) moves as a pair (`s_mov_b64`) when every save and every restore
/// of the two lanes come together, so a parked lane mask comes back as the
/// mask it is; otherwise its halves come back as two scalars.
///
/// The compiler reuses a lane once what it held is dead, for a pair first and
/// a scalar later, say. Where that keeps a pair from being one, each life of a
/// lane (the saves that reach a restore in common, and those restores) is a
/// store of its own.
fn unspill(
    list: &mut [gfx10::Instruction],
    flows: &[Flow],
    leaders: &[usize],
    wave64: bool,
    lanes: u8,
    vgprs: u16,
) -> Vec<(usize, u8, u8)> {
    /// `v_writelane_b32 vN, sX, K` / `v_readlane_b32 sX, vN, K`: (save, N, X, K).
    fn parked(inst: &gfx10::Instruction, fields: &Fields) -> Option<(bool, u8, u8, u32)> {
        let Fields::Valu(v) = fields else { return None };
        if v.abs != 0 || v.neg != 0 || v.opsel != 0 || v.clamp || v.omod != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return None;
        }
        let lane = |op: &Operand| match *op {
            Operand::Const(c @ 128..=191) => Some(u32::from(c) - 128),
            _ => None,
        };
        match (inst.name()?, v.dst?, v.src.as_slice()) {
            ("v_writelane_b32", Operand::Vector { reg, .. }, [Operand::Scalar { code, .. }, k]) if *code < 105 => {
                Some((true, reg, *code, lane(k)?))
            }
            ("v_readlane_b32", Operand::Scalar { code, .. }, [Operand::Vector { reg, .. }, k]) if code < 105 => {
                Some((false, *reg, code, lane(k)?))
            }
            _ => None,
        }
    }
    // VGPRs used by parking alone, the scalar codes the program uses, and
    // per instruction the scalar codes it writes.
    let mut stores = [false; 256];
    let mut other: [Option<usize>; 256] = [None; 256]; // the last such use
    let mut used = [false; 256];
    let mut parks: Vec<Option<(bool, u8, u8, u32)>> = Vec::with_capacity(list.len());
    let mut written: Vec<Vec<u8>> = Vec::with_capacity(list.len());
    let mut read: Vec<Vec<u8>> = Vec::with_capacity(list.len());
    for inst in list.iter() {
        let Some(name) = inst.name() else { return Vec::new() };
        let fields = inst.fields(wave64);
        let park = parked(inst, &fields);
        let Ok(effects) = effects(inst, &fields, name, lanes, vgprs) else {
            return Vec::new();
        };
        let mut codes = Vec::new();
        let mut from = Vec::new();
        if let Some((reads, writes)) = effects {
            for &reg in &writes {
                if let Reg::S(c) = reg {
                    codes.push(c);
                }
            }
            for &reg in &reads {
                if let Reg::S(c) = reg {
                    from.push(c);
                }
            }
            // An M0-relative read (`v_movrels_b32`) names every VGPR from its
            // base up; the compiler indexes a small array with it, never the
            // lanes it parks scalars in, so it does not make those VGPRs
            // ordinary ones. (Relative writes still do.)
            let relative = name == "v_movrels_b32";
            let n_reads = reads.len();
            for (k, reg) in reads.into_iter().chain(writes).enumerate() {
                if relative && k < n_reads && matches!(reg, Reg::V(_)) {
                    continue;
                }
                match reg {
                    Reg::S(c) => used[usize::from(c)] = true,
                    Reg::V(r) if park.is_some_and(|p| p.1 == r) => stores[usize::from(r)] = true,
                    Reg::V(r) => other[usize::from(r)] = Some(parks.len()),
                    Reg::Scc => {}
                }
            }
        }
        parks.push(park);
        written.push(codes);
        read.push(from);
    }
    let leader = |i: usize| leaders.binary_search(&list[i].offset).is_ok();
    for reg in 0..256 {
        let Some(last) = other[reg] else { continue };
        let of = |p: &Option<(bool, u8, u8, u32)>| p.filter(|p| usize::from(p.1) == reg);
        let first = parks.iter().position(|p| of(p).is_some_and(|p| p.0));
        let late = first.is_some_and(|first| {
            let end = parks.iter().rposition(|p| of(p).is_some()).unwrap_or(first);
            let mut saved = Vec::new();
            last < first
                && !(first + 1..=end).any(leader)
                && parks[first..=end].iter().filter_map(of).all(|(save, _, _, lane)| {
                    if save {
                        saved.push(lane);
                    }
                    saved.contains(&lane)
                })
        });
        let from = if late { first.unwrap_or(0) } else { parks.len() };
        for park in &mut parks[..from] {
            if of(park).is_some() {
                *park = None;
            }
        }
    }
    if parks.iter().all(Option::is_none) {
        return Vec::new();
    }
    // Lane pairs (register, lower lane) that always move together: each
    // save of the lower lane is followed, in the same block and before its
    // register changes, by the save of the next register into the next
    // lane; each restore of the lower lane is followed at once by the
    // restore of the next lane into the next register.
    // The second save may come any number of instructions after the first (the
    // compiler puts loads and constants between the two: five instructions later
    // in the ped mesh programs). Only wave64 programs have pairs: a wave32 mask
    // is one dword, and a pair move of it with its neighbour would hide it.
    // instruction -> the instruction that completes its pair
    let partner = |i: usize| -> Option<usize> {
        let (save, reg, code, lane) = parks[i]?;
        if save {
            let mut j = i + 1;
            while j < list.len() && !leader(j) {
                if parks[j] == Some((true, reg, code + 1, lane + 1)) {
                    return Some(j);
                }
                if parks[j].is_some_and(|p| p.1 == reg && (p.3 == lane || p.3 == lane + 1))
                    || written[j].contains(&code)
                    || written[j].contains(&(code + 1))
                {
                    return None;
                }
                j += 1;
            }
            None
        } else {
            // The second restore may come a few instructions later (the
            // compiler schedules other work between the two): the pair is
            // restored at the first, so nothing between may read or write
            // the second register, write the first, or save into the lanes.
            let mut j = i + 1;
            while j < list.len() && !leader(j) {
                if parks[j] == Some((false, reg, code + 1, lane + 1)) {
                    return Some(j);
                }
                if parks[j].is_some_and(|p| p.0 && p.1 == reg && (p.3 == lane || p.3 == lane + 1))
                    || written[j].contains(&code)
                    || written[j].contains(&(code + 1))
                    || read[j].contains(&(code + 1))
                {
                    return None;
                }
                j += 1;
            }
            None
        }
    };
    // The store each parking instruction uses, and what the stores are lanes
    // of: one per (VGPR, lane), or (`split`) one per life of it.
    let stores_of = |split: bool| -> (Vec<usize>, Vec<(u8, u32)>) {
        let slot = |i: usize| parks[i].map(|p| (p.1, p.3));
        // Instructions of one store, by the first of them.
        let mut first: Vec<usize> = (0..list.len()).collect();
        if split {
            fn root(first: &mut [usize], mut i: usize) -> usize {
                while first[i] != i {
                    first[i] = first[first[i]];
                    i = first[i];
                }
                i
            }
            let at = |offset: usize| list.partition_point(|i| i.offset < offset);
            for save in (0..list.len()).filter(|&i| parks[i].is_some_and(|p| p.0)) {
                let mut seen = HashSet::new();
                let mut work = vec![save + 1];
                while let Some(from) = work.pop() {
                    for i in from..list.len() {
                        if slot(i).is_some() && slot(i) == slot(save) {
                            if parks[i].is_some_and(|p| p.0) {
                                break;
                            }
                            let (a, b) = (root(&mut first, save), root(&mut first, i));
                            first[a.max(b)] = a.min(b);
                        }
                        let mut go = |to: usize| {
                            if seen.insert(to) {
                                work.push(to);
                            }
                        };
                        match flows[i] {
                            Flow::Next if i + 1 < list.len() && leader(i + 1) => go(i + 1),
                            Flow::Next => continue,
                            Flow::Stop => {}
                            Flow::Jump(t) => go(at(t)),
                            Flow::Branch(t) => {
                                go(at(t));
                                go(i + 1);
                            }
                        }
                        break;
                    }
                }
            }
            for i in 0..list.len() {
                first[i] = root(&mut first, i);
            }
        } else {
            let mut of: HashMap<(u8, u32), usize> = HashMap::new();
            for i in 0..list.len() {
                if let Some(s) = slot(i) {
                    first[i] = *of.entry(s).or_insert(i);
                }
            }
        }
        let mut stores: Vec<((u8, u32), usize)> = (0..list.len())
            .filter(|&i| first[i] == i)
            .filter_map(|i| Some((slot(i)?, i)))
            .collect();
        stores.sort_unstable();
        let number = |i: usize| stores.iter().position(|s| s.1 == first[i]);
        let of = (0..list.len())
            .map(|i| parks[i].and_then(|_| number(i)).unwrap_or(usize::MAX))
            .collect();
        (of, stores.into_iter().map(|s| s.0).collect())
    };
    // The stores that move as pairs (lower, upper) and every store's code;
    // none if the spare codes do not suffice.
    let plan = |split: bool| -> Option<(Vec<usize>, Vec<(usize, usize)>, Vec<u8>)> {
        let (mut of, stores) = stores_of(split);
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for (n, &(reg, lane)) in stores.iter().enumerate() {
            if !wave64 || pairs.iter().any(|&(_, upper)| upper == n) {
                continue;
            }
            let lower: Vec<usize> = (0..list.len()).filter(|&i| of[i] == n).collect();
            let partners: Vec<Option<usize>> = lower.iter().map(|&i| partner(i)).collect();
            let mut distinct: Vec<usize> = partners.iter().flatten().copied().collect();
            distinct.sort_unstable();
            distinct.dedup();
            let Some(upper) = distinct.first().map(|&j| of[j]) else {
                continue;
            };
            if stores[upper] == (reg, lane + 1)
                && partners.iter().all(Option::is_some)
                && distinct.len() == lower.len()
                && distinct.iter().all(|&j| of[j] == upper)
                && of.iter().filter(|&&o| o == upper).count() == lower.len()
            {
                pairs.push((n, upper));
            }
        }
        // Lives of one lane that move alike share a code, as they share the lane.
        let role = |n: usize| {
            let (lower, upper) = (pairs.iter().any(|p| p.0 == n), pairs.iter().any(|p| p.1 == n));
            (stores[n], u8::from(lower) + 2 * u8::from(upper))
        };
        let mut shared: Vec<_> = (0..stores.len()).map(role).collect();
        let place: Vec<usize> = {
            let all = shared.clone();
            shared.sort_unstable();
            shared.dedup();
            all.iter().filter_map(|r| shared.binary_search(r).ok()).collect()
        };
        for o in of.iter_mut().filter(|o| **o != usize::MAX) {
            *o = place[*o];
        }
        for p in &mut pairs {
            *p = (place[p.0], place[p.1]);
        }
        pairs.sort_unstable();
        pairs.dedup();
        // Codes for the parked scalars: the trap temporaries (108..=123), then
        // user SGPRs the program never names, from the top, then the codes past
        // the machine's (`PARKED`..=254; 255 stays unused so that the code after a
        // parked one always exists). A material resolve program parks up to 63
        // scalars and leaves about 20 machine codes unused.
        let past = usize::from(PARKED);
        let mut free = [false; 256];
        for c in (0..105).chain(108..=123) {
            free[c] = !used[c];
        }
        free[past..255].fill(true);
        let mut codes: Vec<Option<u8>> = vec![None; shared.len()];
        for &(lower, upper) in &pairs {
            let at = (108..123).chain((0..104).rev()).chain(past..254).find(|&c| free[c] && free[c + 1]);
            // Not enough spare codes: leave the program as it is.
            let c = at?;
            (free[c], free[c + 1]) = (false, false);
            codes[lower] = Some(c as u8);
            codes[upper] = Some(c as u8 + 1);
        }
        for code in codes.iter_mut().filter(|c| c.is_none()) {
            let c = (108..=123).chain((0..105).rev()).chain(past..255).find(|&c| free[c])?;
            free[c] = false;
            *code = Some(c as u8);
        }
        Some((of, pairs, codes.into_iter().flatten().collect()))
    };
    let Some(mut chosen) = plan(false) else { return Vec::new() };
    let lonely = |(of, pairs, _): &(Vec<usize>, Vec<(usize, usize)>, Vec<u8>)| {
        (0..list.len()).any(|i| {
            parks[i].is_some_and(|p| p.0)
                && partner(i).is_some()
                && !pairs.iter().any(|&(lower, upper)| of[i] == lower || of[i] == upper)
        })
    };
    if wave64
        && lonely(&chosen)
        && let Some(split) = plan(true)
        && split.1.len() > chosen.1.len()
    {
        chosen = split;
    }
    let (store, pairs, codes) = chosen;
    let put = |list: &mut [gfx10::Instruction], i: usize, encoding, opcode: u16, word: u32| {
        list[i] = gfx10::Instruction::new(list[i].offset, encoding, opcode, vec![word]);
    };
    // `s_mov_b32` / `s_mov_b64 dst, src` (SOP1, opcodes 3 and 4); `s_nop`. A code
    // from `PARKED` up does not fit the word: s0 stands in for it there.
    let field = |code: u8| if code < PARKED { u32::from(code) } else { 0 };
    let mov = |wide: bool, dst: u8, src: u8| 0xBE80_0000 | field(dst) << 16 | (3 + u32::from(wide)) << 8 | field(src);
    const NOP: u32 = 0xBF80_0000;
    let mut edits: Vec<(usize, Option<(bool, u8, u8)>)> = Vec::new();
    for i in 0..list.len() {
        let Some((save, _, code, _)) = parks[i] else {
            continue;
        };
        let slot = codes[store[i]];
        if pairs.iter().any(|p| p.0 == store[i]) {
            let Some(j) = partner(i) else { return Vec::new() };
            // A save moves the pair where its second half is saved; a
            // restore where its first half is restored.
            let (at, gone) = if save { (j, i) } else { (i, j) };
            let (dst, src) = if save { (slot, code) } else { (code, slot) };
            edits.push((at, Some((true, dst, src))));
            edits.push((gone, None));
        } else if pairs.iter().any(|p| p.1 == store[i]) {
            // the upper half of a pair: moved with the lower
        } else {
            let (dst, src) = if save { (slot, code) } else { (code, slot) };
            edits.push((i, Some((false, dst, src))));
        }
    }
    let mut moves = Vec::new();
    for (i, edit) in edits {
        match edit {
            Some((wide, dst, src)) => {
                put(list, i, gfx10::Encoding::Sop1, 3 + u16::from(wide), mov(wide, dst, src));
                moves.push((i, dst, src));
            }
            None => put(list, i, gfx10::Encoding::Sopp, 0, NOP),
        }
    }
    moves.sort_unstable();
    moves
}

/// Rename register reads and writes after changing edges in a reducible graph.
fn rename(mut blocks: Vec<Block>, wave64: bool, vgprs: u16) -> Program {
    let order = reverse_postorder(&blocks);
    let (live_in, live_out) = liveness(&blocks, &order);
    let mut values = 0u32;
    let mut new_value = || {
        values += 1;
        Value(values - 1)
    };
    let mut inputs = Vec::new();
    let mut outs: Vec<Vec<(Reg, Value)>> = vec![Vec::new(); blocks.len()];
    let mut current = [NONE; REGS];
    for &b in &order {
        current.fill(NONE);
        let block = &mut blocks[b];
        for reg in members(&live_in[b]) {
            current[index(reg)] = match block.preds[..] {
                [] => {
                    let v = new_value();
                    inputs.push((reg, v));
                    v
                }
                // A single predecessor comes earlier in reverse postorder.
                [p] => lookup(&outs[p], reg),
                _ => {
                    let value = new_value();
                    block.phis.push(Phi {
                        reg,
                        value,
                        args: Vec::new(),
                    });
                    value
                }
            };
        }
        for inst in &mut block.insts {
            for (reg, v) in &mut inst.reads {
                *v = current[index(*reg)];
                debug_assert_ne!(*v, NONE, "{reg:?} read at {:#x}", inst.machine.offset);
            }
            for (reg, v) in &mut inst.writes {
                *v = new_value();
                current[index(*reg)] = *v;
            }
        }
        outs[b] = members(&live_out[b])
            .map(|reg| (reg, current[index(reg)]))
            .collect();
    }
    for block in &mut blocks {
        for phi in &mut block.phis {
            phi.args = block
                .preds
                .iter()
                .map(|&p| lookup(&outs[p], phi.reg))
                .collect();
            debug_assert!(!phi.args.contains(&NONE), "{phi:?}");
        }
    }
    remove_trivial_phis(&mut blocks, values);
    Program {
        wave64,
        vgprs,
        blocks,
        inputs,
    }
}

/// Duplicate a join for each proven (predecessor, join, destination) edge.
/// The instructions still execute on that path and define fresh SSA values.
pub(crate) fn thread_edges(
    p: &Program,
    edges: &[(usize, usize, usize)],
) -> Result<Program, ShaderError> {
    let mut blocks = p.blocks.clone();
    for block in &mut blocks {
        block.phis.clear();
    }
    for &(pred, join, dest) in edges {
        let mut copy = blocks[join].clone();
        copy.exit = Exit::Jump(dest);
        let target = blocks.len();
        match &mut blocks[pred].exit {
            Exit::Jump(t) => *t = target,
            Exit::Branch { taken, not_taken } => {
                if *taken == join {
                    *taken = target;
                }
                if *not_taken == join {
                    *not_taken = target;
                }
            }
            _ => unreachable!("threading an incoming edge"),
        }
        blocks.push(copy);
    }
    link(&mut blocks);
    split_irreducible(&mut blocks)?;
    Ok(rename(blocks, p.wave64, p.vgprs))
}

/// The program without its early-outs: an `s_cbranch_execz` to a block that
/// only ends the program (`if (!inBounds) return;`, the outermost branch of
/// most compute programs) is left out, so the lanes that would have left
/// walk the body with EXEC clear. Vector instructions and every effect are
/// masked by EXEC already, as on the console, where the branch is taken
/// only when no lane of the wave is left; scalar code has no effects. The
/// body is then code every lane runs, which a console wave run as two host
/// waves needs for the barriers of its exchanges and LDS accesses.
/// A program too long for the branch to reach its end has the other form of the same
/// early-out: `s_cbranch_execnz` over a far jump to that end, which is left out as well.
/// `None` if the program has no such branch.
pub(crate) fn without_early_outs(p: &Program) -> Option<Program> {
    let mut blocks = p.blocks.clone();
    let mut any = false;
    for (b, block) in blocks.iter_mut().enumerate() {
        if let Some(next) = early_out_continuation(p, b) {
            block.insts.pop();
            block.exit = Exit::Jump(next);
            any = true;
        }
    }
    if !any {
        return None;
    }
    for block in &mut blocks {
        block.phis.clear();
    }
    link(&mut blocks);
    split_irreducible(&mut blocks).ok()?;
    Some(rename(blocks, p.wave64, p.vgprs))
}

/// Whether block `b` only ends the program.
fn ends_program(p: &Program, b: usize) -> bool {
    p.blocks[b].exit == Exit::End
        && p.blocks[b].insts.iter().all(|i| matches!(i.fields, crate::gfx10::Fields::Sopp { .. }))
}

/// Whether block `b` is the far jump of a console branch whose target is too far for the
/// branch's 16-bit offset (`s_getpc_b64`, an address added to it, `s_setpc_b64`) to a block
/// that only ends the program. Code that does any work on the way stays what it is.
fn far_jump_to_end(p: &Program, b: usize) -> bool {
    let block = &p.blocks[b];
    matches!(block.exit, Exit::Jump(target) if ends_program(p, target))
        && block
            .insts
            .iter()
            .all(|i| matches!(i.name(), "s_getpc_b64" | "s_add_u32" | "s_addc_u32" | "s_setpc_b64"))
}

/// Where the lanes that stay go when block `b` ends in an early-out: `s_cbranch_execz` to a
/// block that only ends the program, or `s_cbranch_execnz` whose other way is a far jump to
/// such a block (a program too long for the branch to reach its end). The branch is then
/// left out and the code goes on there.
fn early_out_continuation(p: &Program, b: usize) -> Option<usize> {
    let Exit::Branch { taken, not_taken } = p.blocks[b].exit else {
        return None;
    };
    match p.blocks[b].insts.last()?.name() {
        "s_cbranch_execz" if ends_program(p, taken) => Some(not_taken),
        "s_cbranch_execnz" if far_jump_to_end(p, not_taken) => Some(taken),
        _ => None,
    }
}

/// As [`without_early_outs`], and also without the branches that only skip a
/// region when no lane is left for it: an `s_cbranch_execz` whose target is
/// where the skipped code ends up anyway (`if (c) { ... }` with nothing
/// else leaving the region). Every lane then walks the region with EXEC
/// clear for those that would have skipped it, and LDS accesses and
/// exchanges inside it are in code every lane runs.
/// The console takes such a branch only when no lane of the wave is left,
/// and then its scalar code does not run: a branch is kept if the code
/// after it could see that (a scalar register or SCC that the region
/// changes and the join merges with the value from before; EXEC aside,
/// which is empty either way). `None` if nothing changes.
///
/// `deep` looks closer at what the code after the branch can see. A scalar the region moves
/// away and back (the compiler parks scalars in VGPR lanes around a branch and takes them
/// back before the join, `unspill`) is the value it copies, and so is a scalar phi whose
/// arguments all come to one value (a join of branches that left it alone); a join nothing
/// reads shows no one the difference (a restore of EXEC that the next one overwrites reads
/// nothing); and the join may have other ways in (an inner branch that ends together with an
/// outer one).
/// That leaves out most of the skips of the title's large lighting programs, whose bodies
/// are then code every lane runs.
pub(crate) fn without_skips(p: &Program, deep: bool) -> Option<Program> {
    let n = p.blocks.len();
    // Scalars moved from register to register (a scalar the compiler parked in a VGPR lane
    // comes back by such a move, `unspill`) are the value they copy.
    let mut copy: std::collections::HashMap<Value, Value> = Default::default();
    // Values something reads, directly or through joins. A scalar operation none of whose
    // results is read reads nothing either (the compiler restores EXEC from a saved mask
    // and overwrites it at once where two branches end together).
    let mut live: std::collections::HashSet<Value> = Default::default();
    let idle = |inst: &Inst, live: &std::collections::HashSet<Value>| {
        let name = inst.name();
        matches!(inst.fields, crate::gfx10::Fields::Sop { .. })
            && !(name.starts_with("s_setpc")
                || name.starts_with("s_swappc")
                || name.starts_with("s_call")
                || name.starts_with("s_rfe")
                || name.starts_with("s_movrel")
                || name.starts_with("s_subvector"))
            && inst.writes.iter().all(|w| !live.contains(&w.1))
    };
    for block in &p.blocks {
        for inst in &block.insts {
            if matches!(inst.name(), "s_mov_b32" | "s_mov_b64")
                && inst.reads.len() == inst.writes.len()
                && inst.reads.iter().chain(&inst.writes).all(|r| matches!(r.0, Reg::S(_)))
            {
                copy.extend(inst.writes.iter().zip(&inst.reads).map(|(w, r)| (w.1, r.1)));
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for block in &p.blocks {
            for inst in &block.insts {
                if !idle(inst, &live) {
                    for r in &inst.reads {
                        changed |= live.insert(r.1);
                    }
                }
            }
            for phi in &block.phis {
                if live.contains(&phi.value) {
                    for &a in &phi.args {
                        changed |= live.insert(a);
                    }
                }
            }
        }
    }
    if deep {
        copy_through_phis(p, &mut copy);
    }
    let root = |v: Value| if deep { follow(&copy, v) } else { v };
    // The region a skip of `b` to `target` leaves out, if it is a pure skip.
    let skipped = |b: usize, target: usize, first: usize| -> Option<Vec<bool>> {
        let mut inside = vec![false; n];
        let mut stack = vec![first];
        while let Some(x) = stack.pop() {
            if x == target || inside[x] {
                continue;
            }
            if x == b {
                return None; // a loop back to the branch
            }
            inside[x] = true;
            match p.blocks[x].exit {
                Exit::End | Exit::Indirect => return None,
                exit => stack.extend(exit.successors()),
            }
        }
        // entered only through the branch
        for x in (0..n).filter(|&x| inside[x]) {
            if p.blocks[x].preds.iter().any(|&q| q != b && !inside[q]) {
                return None;
            }
        }
        // the join sees no scalar the region changed
        let join = &p.blocks[target];
        for phi in &join.phis {
            let scalar = match phi.reg {
                Reg::S(c) => c != 126 && c != 127,
                Reg::Scc => true,
                Reg::V(_) => false,
            };
            // (a join nothing reads shows no one the difference)
            if !scalar || deep && !live.contains(&phi.value) {
                continue;
            }
            let before: Vec<Value> =
                join.preds.iter().zip(&phi.args).filter(|(q, _)| **q == b).map(|(_, a)| root(*a)).collect();
            let within = join.preds.iter().zip(&phi.args).filter(|(q, _)| inside[**q]).map(|(_, a)| root(*a));
            if within.into_iter().any(|a| !before.contains(&a)) {
                return None;
            }
        }
        // Other ways into the join than the branch and the region (an inner branch that ends
        // together with an outer one) change nothing in that: the lanes that would have
        // skipped arrive from the region instead, with the same scalars.
        if !deep && join.preds.iter().any(|&q| q != b && !inside[q]) {
            return None;
        }
        Some(inside)
    };
    let mut blocks = p.blocks.clone();
    let mut any = false;
    for b in 0..n {
        let Exit::Branch { taken, not_taken } = p.blocks[b].exit else {
            continue;
        };
        let next = early_out_continuation(p, b).or_else(|| {
            let skip = p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz");
            (skip && skipped(b, taken, not_taken).is_some()).then_some(not_taken)
        });
        if let Some(next) = next {
            blocks[b].insts.pop();
            blocks[b].exit = Exit::Jump(next);
            any = true;
        }
    }
    if !any {
        return None;
    }
    for block in &mut blocks {
        block.phis.clear();
    }
    link(&mut blocks);
    split_irreducible(&mut blocks).ok()?;
    Some(rename(blocks, p.wave64, p.vgprs))
}

/// The value `v` is a copy of, followed to the end of `copy`'s chains.
fn follow(copy: &std::collections::HashMap<Value, Value>, mut v: Value) -> Value {
    while let Some(&c) = copy.get(&v) {
        v = c;
    }
    v
}

/// The one value every argument of the scalar `phi` comes to (the phi's own value left
/// aside, as a loop's back edge brings it), if there is one.
fn single_source(copy: &std::collections::HashMap<Value, Value>, phi: &Phi) -> Option<Value> {
    let mut one = None;
    for x in phi.args.iter().map(|&a| follow(copy, a)).filter(|&x| x != phi.value) {
        if *one.get_or_insert(x) != x {
            return None;
        }
    }
    one
}

/// Adds to `copy` the scalar phis that pass one value through (a join of branches none of
/// which changed it, one after the other, as the console's compiler leaves them): such a phi
/// is a copy of that value. A phi that merges a real change is not. Repeats until no phi is
/// added, for a phi may pass through the ones before it.
fn copy_through_phis(p: &Program, copy: &mut std::collections::HashMap<Value, Value>) {
    let mut added = true;
    while added {
        added = false;
        for phi in p.blocks.iter().flat_map(|b| &b.phis) {
            if copy.contains_key(&phi.value) || !matches!(phi.reg, Reg::S(_) | Reg::Scc) {
                continue;
            }
            if let Some(one) = single_source(copy, phi) {
                copy.insert(phi.value, one);
                added = true;
            }
        }
    }
}

/// Placeholder before renaming.
const NONE: Value = Value(u32::MAX);
/// Scalar codes 0-255 (from [`PARKED`] up: parked scalars), SCC, VGPRs 0-255.
const REGS: usize = 256 + 1 + 256;
type RegSet = [u64; REGS.div_ceil(64)];

fn index(reg: Reg) -> usize {
    match reg {
        Reg::S(c) => usize::from(c),
        Reg::Scc => 256,
        Reg::V(r) => 257 + usize::from(r),
    }
}

fn reg_at(i: usize) -> Reg {
    match i {
        0..256 => Reg::S(i as u8),
        256 => Reg::Scc,
        _ => Reg::V((i - 257) as u8),
    }
}

fn insert(set: &mut RegSet, reg: Reg) {
    let i = index(reg);
    set[i / 64] |= 1 << (i % 64);
}

fn contains(set: &RegSet, reg: Reg) -> bool {
    let i = index(reg);
    set[i / 64] >> (i % 64) & 1 != 0
}

/// In [`Reg`] order, which is also index order.
fn members(set: &RegSet) -> impl Iterator<Item = Reg> + '_ {
    let (mut word, mut bits) = (0, set[0]);
    std::iter::from_fn(move || loop {
        if bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            return Some(reg_at(word * 64 + b));
        }
        word += 1;
        if word >= set.len() {
            return None;
        }
        bits = set[word];
    })
}

/// `outs` is sorted by register, and holds every register live out.
fn lookup(outs: &[(Reg, Value)], reg: Reg) -> Value {
    outs.binary_search_by_key(&reg, |&(r, _)| r)
        .map_or(NONE, |i| outs[i].1)
}

pub(crate) fn reverse_postorder(blocks: &[Block]) -> Vec<usize> {
    let mut seen = vec![false; blocks.len()];
    let mut post = Vec::with_capacity(blocks.len());
    let mut stack = vec![(0, 0)];
    seen[0] = true;
    while let Some(top) = stack.last_mut() {
        let (b, i) = *top;
        top.1 += 1;
        match blocks[b].exit.successors().get(i) {
            Some(&s) if !seen[s] => {
                seen[s] = true;
                stack.push((s, 0));
            }
            Some(_) => {}
            None => {
                post.push(b);
                stack.pop();
            }
        }
    }
    post.reverse();
    post
}

/// Recomputes every block's predecessors from the exits.
fn link(blocks: &mut [Block]) {
    for block in blocks.iter_mut() {
        block.preds.clear();
    }
    for b in 0..blocks.len() {
        for s in blocks[b].exit.successors() {
            blocks[s].preds.push(b);
        }
    }
}

/// An edge to an earlier block in reverse postorder whose target does not
/// dominate its source, if any: there the control flow is irreducible.
/// `index` is each block's position in `order`.
pub(crate) fn irreducible_edge(
    blocks: &[Block],
    order: &[usize],
    index: &[usize],
) -> Option<(usize, usize)> {
    let &entry = order.first()?;
    let idom = idoms(blocks, order, index);
    for &b in order {
        for s in blocks[b].exit.successors() {
            if index[s] > index[b] {
                continue;
            }
            let mut d = b;
            while d != s && d != entry {
                d = idom[d];
            }
            if d != s {
                return Some((b, s));
            }
        }
    }
    None
}

/// Each block's immediate dominator (the entry's is itself, unreachable
/// blocks' `usize::MAX`); `order` is reverse postorder, `index` each
/// block's position in it.
pub(crate) fn idoms(blocks: &[Block], order: &[usize], index: &[usize]) -> Vec<usize> {
    let mut idom = vec![usize::MAX; blocks.len()];
    let Some(&entry) = order.first() else {
        return idom;
    };
    idom[entry] = entry;
    let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
        while a != b {
            while index[a] > index[b] {
                a = idom[a];
            }
            while index[b] > index[a] {
                b = idom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &order[1..] {
            let mut new = usize::MAX;
            for &q in blocks[b].preds.iter().filter(|&&q| idom[q] != usize::MAX) {
                new = if new == usize::MAX {
                    q
                } else {
                    intersect(&idom, q, new)
                };
            }
            if idom[b] != new {
                idom[b] = new;
                changed = true;
            }
        }
    }
    idom
}

/// Node splitting, before SSA. While a loop is entered at several blocks
/// (a compiler-rotated loop whose peeled first test jumps into its body),
/// the entry first in reverse postorder stays the header, and another
/// entry's outside predecessors move to a copy of everything that entry
/// reaches in the loop before the header. The copies run the same
/// instructions, so behaviour is unchanged.
fn split_irreducible(blocks: &mut Vec<Block>) -> Result<(), ShaderError> {
    let limit = 4 * blocks.len() + 64;
    loop {
        let order = reverse_postorder(blocks);
        let mut index = vec![usize::MAX; blocks.len()];
        for (i, &b) in order.iter().enumerate() {
            index[b] = i;
        }
        let Some((from, _)) = irreducible_edge(blocks, &order, &index) else {
            return Ok(());
        };
        let found = multi_entry_loop(blocks, &vec![true; blocks.len()]);
        let Some((region, mut entries)) = found.filter(|_| blocks.len() < limit) else {
            return Err(ShaderError::Unsupported {
                offset: blocks[from].offset,
                what: "irreducible control flow",
            });
        };
        entries.sort_by_key(|&e| index[e]);
        let (header, entry) = (entries[0], entries[1]);

        // What `entry` reaches in the loop before the header, and its copies.
        let mut copy = vec![usize::MAX; blocks.len()];
        let mut copied = Vec::new();
        let mut stack = vec![entry];
        while let Some(b) = stack.pop() {
            if copy[b] != usize::MAX {
                continue;
            }
            copy[b] = blocks.len() + copied.len();
            copied.push(b);
            for s in blocks[b].exit.successors() {
                if region[s] && s != header {
                    stack.push(s);
                }
            }
        }
        let retarget = |exit: &mut Exit, to: &dyn Fn(usize) -> usize| match exit {
            Exit::Jump(s) => *s = to(*s),
            Exit::Branch { taken, not_taken } => {
                *taken = to(*taken);
                *not_taken = to(*not_taken);
            }
            Exit::End | Exit::Indirect => {}
        };
        for &b in &copied {
            let mut block = blocks[b].clone();
            retarget(&mut block.exit, &|s| {
                if copy[s] == usize::MAX { s } else { copy[s] }
            });
            blocks.push(block);
        }
        let outside: Vec<usize> = blocks[entry]
            .preds
            .iter()
            .copied()
            .filter(|&p| !region[p])
            .collect();
        for p in outside {
            retarget(&mut blocks[p].exit, &|s| {
                if s == entry { copy[entry] } else { s }
            });
        }
        link(blocks);
    }
}

/// A loop among the `inside` blocks entered at more than one block, with
/// its entries. Loops nested in a single-entry loop are searched with its
/// header removed, so one is found whenever the control flow is irreducible.
fn multi_entry_loop(blocks: &[Block], inside: &[bool]) -> Option<(Vec<bool>, Vec<usize>)> {
    let reach = |from: usize, forward: bool| {
        let mut seen = vec![false; blocks.len()];
        let mut stack = vec![from];
        while let Some(b) = stack.pop() {
            let next = if forward {
                blocks[b].exit.successors()
            } else {
                blocks[b].preds.clone()
            };
            for n in next {
                if inside[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        seen
    };
    let mut done = vec![false; blocks.len()];
    for v in (0..blocks.len()).filter(|&v| inside[v]) {
        if done[v] {
            continue;
        }
        // Blocks on a cycle through `v`; `v` itself only if there is one.
        let (ahead, behind) = (reach(v, true), reach(v, false));
        let region: Vec<bool> = ahead.iter().zip(&behind).map(|(a, b)| a & b).collect();
        if !region[v] {
            continue;
        }
        let members = (0..blocks.len()).filter(|&b| region[b]);
        for b in members.clone() {
            done[b] = true;
        }
        let entries: Vec<usize> = members
            .filter(|&b| blocks[b].preds.iter().any(|&p| !region[p]))
            .collect();
        match entries[..] {
            [] => {}
            [header] => {
                let mut nested = region;
                nested[header] = false;
                if let Some(found) = multi_entry_loop(blocks, &nested) {
                    return Some(found);
                }
            }
            _ => return Some((region, entries)),
        }
    }
    None
}

/// Registers live into and out of each block.
fn liveness(blocks: &[Block], order: &[usize]) -> (Vec<RegSet>, Vec<RegSet>) {
    let empty = [0; REGS.div_ceil(64)];
    let mut uses = vec![empty; blocks.len()];
    let mut defs = vec![empty; blocks.len()];
    for (b, block) in blocks.iter().enumerate() {
        for inst in &block.insts {
            for &(reg, _) in &inst.reads {
                if !contains(&defs[b], reg) {
                    insert(&mut uses[b], reg);
                }
            }
            for &(reg, _) in &inst.writes {
                insert(&mut defs[b], reg);
            }
        }
    }
    let mut live_in = vec![empty; blocks.len()];
    let mut live_out = vec![empty; blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().rev() {
            let mut out = empty;
            for s in blocks[b].exit.successors() {
                for (o, i) in out.iter_mut().zip(&live_in[s]) {
                    *o |= i;
                }
            }
            let mut inn = empty;
            for w in 0..inn.len() {
                inn[w] = uses[b][w] | out[w] & !defs[b][w];
            }
            if inn != live_in[b] || out != live_out[b] {
                (live_in[b], live_out[b]) = (inn, out);
                changed = true;
            }
        }
    }
    (live_in, live_out)
}

/// Phis are placed for every live register at every join; those whose
/// arguments are all one value (or the phi itself) are replaced by it.
fn remove_trivial_phis(blocks: &mut [Block], values: u32) {
    remove_trivial_phis_steps(blocks, values);
}

fn remove_trivial_phis_steps(blocks: &mut [Block], values: u32) -> usize {
    fn find(replaced: &mut [Value], v: Value, steps: &mut usize) -> Value {
        let mut root = v;
        while replaced[root.0 as usize] != root {
            *steps += 1;
            root = replaced[root.0 as usize];
        }
        let mut cur = v;
        while replaced[cur.0 as usize] != root {
            *steps += 1;
            let next = replaced[cur.0 as usize];
            replaced[cur.0 as usize] = root;
            cur = next;
        }
        root
    }

    let mut replaced: Vec<Value> = (0..values).map(Value).collect();
    let mut steps = 0usize;
    let mut changed = true;
    while changed {
        changed = false;
        for block in blocks.iter_mut() {
            block.phis.retain(|phi| {
                let mut only = None;
                for &a in &phi.args {
                    let a = find(&mut replaced, a, &mut steps);
                    if a == phi.value || Some(a) == only {
                        continue;
                    }
                    if only.is_some() {
                        return true;
                    }
                    only = Some(a);
                }
                let Some(v) = only else { return true };
                replaced[phi.value.0 as usize] = v;
                changed = true;
                false
            });
        }
    }
    for block in blocks {
        for phi in &mut block.phis {
            for a in &mut phi.args {
                *a = find(&mut replaced, *a, &mut steps);
            }
        }
        for inst in &mut block.insts {
            for (_, v) in &mut inst.reads {
                *v = find(&mut replaced, *v, &mut steps);
            }
        }
    }
    steps
}

/// Scheduling hints, performance counters and tracing: no effect on results.
const DROPPED: &[&str] = &[
    "s_nop",
    "s_waitcnt",
    "s_waitcnt_vscnt",
    "s_waitcnt_vmcnt",
    "s_waitcnt_expcnt",
    "s_waitcnt_lgkmcnt",
    "s_waitcnt_depctr",
    "s_wait_idle",
    "s_inst_prefetch",
    "s_clause",
    "s_setprio",
    "s_sleep",
    "s_ttracedata",
    "s_ttracedata_imm",
    "s_incperflevel",
    "s_decperflevel",
    "v_nop",
    "v_pipeflush",
    "v_clrexcp",
];

/// Registers read, registers written.
type Effects = (Vec<Reg>, Vec<Reg>);

/// The registers `inst` of `p` reads as operands: its reads without the old
/// value of a VGPR it only writes.
pub(crate) fn operand_reads(p: &Program, inst: &Inst) -> Vec<Reg> {
    let lanes = 1 + u8::from(p.wave64);
    effects(&inst.machine, &inst.fields, inst.name(), lanes, p.vgprs)
        .ok()
        .flatten()
        .map_or_else(Vec::new, |e| e.0)
}

/// Registers an instruction reads and writes, one per dword, or `None` if
/// the IR drops it.
fn effects(
    inst: &Instruction,
    fields: &Fields,
    name: &'static str,
    lanes: u8,
    vgprs: u16,
) -> Result<Option<Effects>, ShaderError> {
    if DROPPED.contains(&name) {
        return Ok(None);
    }
    let unsupported = || ShaderError::Unsupported {
        offset: inst.offset,
        what: name,
    };
    let mask = |code: u8| (0..lanes).map(move |i| Reg::S(code + i));
    let (mut r, mut w) = (Vec::new(), Vec::new());
    let regs = |op: &Operand, out: &mut Vec<Reg>| {
        add_regs(op, lanes, out).map_err(|_| ShaderError::Isa {
            offset: inst.offset,
            reason: "register range out of bounds",
        })
    };
    let vector = !matches!(
        fields,
        Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. }
    );
    match fields {
        Fields::Sop { dst, src, .. } => {
            if name.starts_with("s_movrel")
                || name.starts_with("s_subvector_loop")
                || matches!(name, "s_call_b64" | "s_rfe_b64")
            {
                return Err(unsupported());
            }
            for s in src {
                regs(s, &mut r)?;
            }
            if let Some(d) = dst {
                regs(d, &mut w)?;
                // These change (or may keep) the destination's old value.
                if name.starts_with("s_bitset")
                    || name.starts_with("s_cmov")
                    || matches!(name, "s_addk_i32" | "s_mulk_i32")
                {
                    regs(d, &mut r)?;
                }
            }
            if matches!(name, "s_addc_u32" | "s_subb_u32" | "s_cmovk_i32")
                || name.starts_with("s_cselect_")
                || name.starts_with("s_cmov_")
            {
                r.push(Reg::Scc);
            }
            if name.contains("saveexec") || name.contains("wrexec") {
                r.extend(mask(EXEC));
                w.extend(mask(EXEC));
            }
            if writes_scc(inst.encoding, name) {
                w.push(Reg::Scc);
            }
        }
        Fields::Sopp { .. } => match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => r.push(Reg::Scc),
            "s_cbranch_vccz" | "s_cbranch_vccnz" => r.extend(mask(VCC)),
            "s_cbranch_execz" | "s_cbranch_execnz" => r.extend(mask(EXEC)),
            "s_sendmsg" | "s_sendmsghalt" => r.push(Reg::S(M0)),
            "s_endpgm"
            | "s_endpgm_saved"
            | "s_endpgm_ordered_ps_done"
            | "s_code_end"
            | "s_branch"
            | "s_barrier"
            | "s_icache_inv"
            | "s_wakeup"
            | "s_round_mode"
            | "s_denorm_mode" => {}
            _ => return Err(unsupported()),
        },
        Fields::Smem {
            data,
            base,
            soffset,
            ..
        } => match name {
            "s_memtime" | "s_memrealtime" | "s_get_waveid_in_workgroup" => {
                regs(data.as_ref().ok_or_else(unsupported)?, &mut w)?;
            }
            "s_gl1_inv" | "s_dcache_inv" | "s_dcache_wb" => {}
            _ if name.contains("_load_") || name.contains("_store_") => {
                regs(base, &mut r)?;
                regs(soffset, &mut r)?;
                let data = data.as_ref().ok_or_else(unsupported)?;
                regs(
                    data,
                    if name.contains("_load_") {
                        &mut w
                    } else {
                        &mut r
                    },
                )?;
            }
            _ => return Err(unsupported()),
        },
        Fields::Valu(v) => {
            let (src_indexed, dst_indexed) = match name {
                "v_movrels_b32" => (true, false),
                "v_movreld_b32" => (false, true),
                "v_movrelsd_b32" | "v_movrelsd_2_b32" | "v_swaprel_b32" => (true, true),
                _ => (false, false),
            };
            // v[reg + M0] for any M0 that stays inside the program's VGPRs.
            let indexed = |op: &Operand, out: &mut Vec<Reg>| match *op {
                Operand::Vector { reg, .. } if u16::from(reg) < vgprs => {
                    out.extend((u16::from(reg)..vgprs).map(|r| Reg::V(r as u8)));
                    Ok(())
                }
                _ => Err(ShaderError::Isa {
                    offset: inst.offset,
                    reason: "relative VGPR base outside the program's VGPRs",
                }),
            };
            for (i, s) in v.src.iter().enumerate() {
                if i == 0 && src_indexed {
                    indexed(s, &mut r)?;
                } else {
                    regs(s, &mut r)?;
                }
            }
            // GFX10 compares into EXEC write nothing else.
            if name.starts_with("v_cmpx") {
                w.extend(mask(EXEC));
            } else if let Some(d) = &v.dst {
                if dst_indexed {
                    indexed(d, &mut w)?;
                } else {
                    regs(d, &mut w)?;
                }
            }
            if let Some(c) = &v.carry {
                regs(c, &mut w)?;
            }
            match name {
                "v_swap_b32" => {
                    regs(&v.src[0], &mut w)?;
                    if let Some(d) = &v.dst {
                        regs(d, &mut r)?;
                    }
                }
                "v_swaprel_b32" => {
                    indexed(&v.src[0], &mut w)?;
                    if let Some(d) = &v.dst {
                        indexed(d, &mut r)?;
                    }
                }
                "v_div_fmas_f32" | "v_div_fmas_f64" => r.extend(mask(VCC)),
                _ => {}
            }
            if name.starts_with("v_interp_") || src_indexed || dst_indexed {
                r.push(Reg::S(M0));
            }
        }
        Fields::Vintrp { dst, src, .. } => {
            regs(src, &mut r)?;
            r.push(Reg::S(M0));
            regs(dst, &mut w)?;
        }
        Fields::Ds {
            dst,
            addr,
            data,
            gds,
            ..
        } => {
            for o in addr.iter().chain(data) {
                regs(o, &mut r)?;
            }
            if let Some(d) = dst {
                regs(d, &mut w)?;
            }
            if *gds
                || name.starts_with("ds_gws")
                || name.contains("addtid")
                || matches!(name, "ds_ordered_count" | "ds_append" | "ds_consume")
            {
                r.push(Reg::S(M0));
            }
        }
        Fields::Buffer {
            data,
            addr,
            rsrc,
            soffset,
            offen,
            idxen,
            lds,
            cache,
            ..
        } => {
            if *offen || *idxen {
                regs(addr, &mut r)?;
            }
            regs(rsrc, &mut r)?;
            regs(soffset, &mut r)?;
            if *lds {
                r.push(Reg::S(M0));
            }
            if let Some(d) = data {
                if name.contains("_load_") {
                    if !lds {
                        regs(d, &mut w)?;
                    }
                } else if name.contains("_store_") {
                    regs(d, &mut r)?;
                } else if name.contains("_atomic_") {
                    regs(d, &mut r)?;
                    if cache.glc {
                        regs(&atomic_result(d, name), &mut w)?;
                    }
                } else {
                    return Err(unsupported());
                }
            }
        }
        Fields::Image {
            data,
            addr,
            rsrc,
            sampler,
            cache,
            ..
        } => {
            for a in addr.iter().chain([rsrc]).chain(sampler) {
                regs(a, &mut r)?;
            }
            if name.contains("_store") {
                regs(data, &mut r)?;
            } else if name.contains("_atomic_") {
                regs(data, &mut r)?;
                if cache.glc {
                    regs(&atomic_result(data, name), &mut w)?;
                }
            } else {
                regs(data, &mut w)?;
            }
        }
        Fields::Flat {
            dst,
            addr,
            data,
            saddr,
            ..
        } => {
            for o in [addr, data, saddr].into_iter().flatten() {
                regs(o, &mut r)?;
            }
            if let Some(d) = dst {
                regs(d, &mut w)?;
            }
        }
        Fields::Exp { src, .. } => {
            for s in src.iter().flatten() {
                regs(s, &mut r)?;
            }
        }
    }
    if vector {
        r.extend(mask(EXEC));
    }
    let mut seen = Vec::with_capacity(r.len());
    r.retain(|reg| {
        let new = !seen.contains(reg);
        seen.push(*reg);
        new
    });
    Ok(Some((r, w)))
}

/// Scalar ALU opcodes that set SCC (carry, overflow, comparison, or "result
/// is not zero").
fn writes_scc(encoding: gfx10::Encoding, name: &str) -> bool {
    use gfx10::Encoding::*;
    match encoding {
        Sopc => true,
        Sopk => name.starts_with("s_cmpk_") || name == "s_addk_i32",
        _ => {
            let stem = name.split('_').nth(1).unwrap_or("");
            name.contains("saveexec")
                || name.contains("wrexec")
                || matches!(
                    stem,
                    "add"
                        | "sub"
                        | "addc"
                        | "subb"
                        | "min"
                        | "max"
                        | "and"
                        | "or"
                        | "xor"
                        | "andn2"
                        | "orn2"
                        | "nand"
                        | "nor"
                        | "xnor"
                        | "lshl"
                        | "lshr"
                        | "ashr"
                        | "bfe"
                        | "absdiff"
                        | "lshl1"
                        | "lshl2"
                        | "lshl3"
                        | "lshl4"
                        | "not"
                        | "wqm"
                        | "bcnt0"
                        | "bcnt1"
                        | "quadmask"
                        | "abs"
                )
        }
    }
}

/// What an atomic with `glc` returns: the old value (for compare-and-swap,
/// half the data operand).
fn atomic_result(data: &Operand, name: &str) -> Operand {
    match *data {
        Operand::Vector { reg, dwords } if name.contains("cmpswap") => Operand::Vector {
            reg,
            dwords: dwords / 2,
        },
        op => op,
    }
}

/// Appends the register dwords an operand names; reading `vccz`, `execz`,
/// `scc` or `lds_direct` reads the register behind it.
fn add_regs(op: &Operand, lanes: u8, out: &mut Vec<Reg>) -> Result<(), ()> {
    match *op {
        Operand::Scalar { code, .. } if code == gfx10::NULL => {}
        Operand::Scalar { code, dwords } => {
            let end = u16::from(code) + u16::from(dwords);
            // A machine operand ends within the machine's codes, a parked
            // scalar (`unspill`) within those past them.
            let limit = if code < PARKED { 128 } else { 256 };
            if end > limit {
                return Err(());
            }
            out.extend((u16::from(code)..end).map(|c| Reg::S(c as u8)));
        }
        Operand::Vector { reg, dwords } => {
            if u16::from(reg) + u16::from(dwords) > 256 {
                return Err(());
            }
            out.extend((0..dwords).map(|i| Reg::V(reg + i)));
        }
        Operand::Const(251) => out.extend((0..lanes).map(|i| Reg::S(VCC + i))),
        Operand::Const(252) => out.extend((0..lanes).map(|i| Reg::S(EXEC + i))),
        Operand::Const(253) => out.push(Reg::Scc),
        Operand::Const(254) => out.push(Reg::S(M0)),
        Operand::Const(_) | Operand::Literal(_) | Operand::Attr { .. } => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn build_words(words: &[u32], wave64: bool) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        build(&code, wave64, 256).unwrap()
    }

    /// A scalar destination field of 128 or more (a constant's code, not a
    /// register) is refused or ignored, never an index past the scalar
    /// codes the parking search keeps (found by fuzzing the translator).
    #[test]
    fn scalar_codes_past_the_registers_do_not_panic() {
        // v_writelane_b32 v1, s0, 0: a program that parks a scalar, so the
        // parking search runs.
        const PARK: [u32; 2] = [0xD761_0001, 0x0001_0000];
        for (words, wave64) in [
            ([0xD4C5_A8D7, 0x9342_BFA0], true),
            ([0xD401_53FF, 0x0509_8D04], false),
            ([0xD4EC_FE8B, 0x1234_5678], false),
        ] {
            let words = [&PARK[..], &words[..], &[S_ENDPGM]].concat();
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let _ = build(&code, wave64, 256);
        }
    }

    fn input(p: &Program, reg: Reg) -> Value {
        p.inputs.iter().find(|i| i.0 == reg).expect("input").1
    }

    /// `if (v0 < s0) { ... }` whose body changes s6: `body` are its words, `after` what
    /// follows the join.
    fn skip(body: &[u32], after: &[u32]) -> Program {
        let mut w = vec![
            0xBE94_0306,                     // s_mov_b32 s20, s6
            0x7D88_0000,                     // v_cmp_gt_u32 vcc, s0, v0
            0xBE8C_246A,                     // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0000 | body.len() as u32, // s_cbranch_execz past the body
        ];
        w.extend_from_slice(body);
        w.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
        w.extend_from_slice(after);
        w.push(S_ENDPGM);
        build_words(&w, true)
    }

    fn skips(p: &Program) -> usize {
        p.blocks
            .iter()
            .filter(|b| b.insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
            .count()
    }

    #[test]
    fn a_skip_is_left_out_when_the_scalars_it_changes_come_back_or_are_not_read() {
        const SET: u32 = 0xBE86_0385; // s_mov_b32 s6, 5
        const USE: u32 = 0x7E02_0206; // v_mov_b32 v1, s6
        const BACK: u32 = 0xBE86_0314; // s_mov_b32 s6, s20
        const READ: u32 = 0x7E04_0206; // v_mov_b32 v2, s6
        // s6 is changed in the body and read after the join: the branch stays
        let kept = skip(&[SET, USE], &[READ]);
        assert_eq!(skips(&kept), 1);
        assert!(without_skips(&kept, false).is_none());
        assert!(without_skips(&kept, true).is_none());
        // ... put back by a copy of what it held (as a scalar parked in a VGPR lane comes
        // back): every lane may walk the body
        let back = skip(&[SET, USE, BACK], &[READ]);
        assert!(without_skips(&back, false).is_none());
        assert_eq!(skips(&without_skips(&back, true).expect("left out")), 0);
        // ... or not read after the join at all
        let unread = skip(&[SET, USE], &[]);
        assert_eq!(skips(&without_skips(&unread, false).expect("left out")), 0);
        assert_eq!(skips(&without_skips(&unread, true).expect("left out")), 0);
        // ... or read only by a restore of EXEC that nothing uses before the next one
        // (s_mov_b64 exec, s[6:7]; s_mov_b64 exec, s[12:13]; v_mov_b32 v2, s0)
        let idle = skip(&[SET, USE], &[0xBEFE_0406, 0xBEFE_040C, 0x7E04_0200]);
        assert!(without_skips(&idle, false).is_none());
        assert_eq!(skips(&without_skips(&idle, true).expect("left out")), 0);
    }

    #[test]
    fn joins_get_phis_only_where_values_differ() {
        let code = [
            0xBF06_0100, // s_cmp_eq_u32 s0, s1
            0xBF85_0001, // s_cbranch_scc1 -> 12
            0xBE82_0380, // s_mov_b32 s2, 0
            0x7E00_0202, // 12: v_mov_b32 v0, s2
            0x7E02_0200, // v_mov_b32 v1, s0
            S_ENDPGM,
        ];
        for wave64 in [false, true] {
            let p = build_words(&code, wave64);
            let [b0, b1, b2] = &p.blocks[..] else {
                panic!("{:?}", p.blocks)
            };
            assert_eq!(
                b0.exit,
                Exit::Branch {
                    taken: 2,
                    not_taken: 1
                }
            );
            assert_eq!(b1.exit, Exit::Jump(2));
            assert_eq!(b2.preds, [0, 1]);
            assert_eq!(b0.insts[1].read(Reg::Scc), b0.insts[0].write(Reg::Scc));

            // s2 differs per path; s0, EXEC, v0 and v1 do not.
            let [phi] = &b2.phis[..] else {
                panic!("{:?}", b2.phis)
            };
            assert_eq!(phi.reg, Reg::S(2));
            assert_eq!(
                phi.args,
                [input(&p, Reg::S(2)), b1.insts[0].write(Reg::S(2)).unwrap()]
            );
            let [mov0, mov1, end] = &b2.insts[..] else {
                panic!()
            };
            assert_eq!(mov0.read(Reg::S(2)), Some(phi.value));
            assert_eq!(mov1.read(Reg::S(0)), Some(input(&p, Reg::S(0))));
            // The old v1 is read: lanes outside EXEC keep it.
            assert_eq!(mov1.read(Reg::V(1)), Some(input(&p, Reg::V(1))));
            assert_eq!(mov1.read(Reg::S(EXEC)), Some(input(&p, Reg::S(EXEC))));
            assert_eq!(mov1.read(Reg::S(EXEC + 1)).is_some(), wave64);
            assert_eq!((end.name(), b2.exit), ("s_endpgm", Exit::End));
        }
    }

    #[test]
    fn in_place_scalar_ops_read_their_destination() {
        let p = build_words(
            &[
                0xBE8D_1D94, // s_bitset1_b32 s13, 20
                0xB78E_0010, // s_addk_i32 s14, 0x10
                0xBE8F_0380, // s_mov_b32 s15, 0
                S_ENDPGM,
            ],
            false,
        );
        let [bitset, addk, mov, _] = &p.blocks[0].insts[..] else {
            panic!("{:?}", p.blocks[0].insts)
        };
        assert_eq!(bitset.read(Reg::S(13)), Some(input(&p, Reg::S(13))));
        assert_eq!(addk.read(Reg::S(14)), Some(input(&p, Reg::S(14))));
        assert_eq!(mov.read(Reg::S(15)), None);
    }

    #[test]
    fn loops_carry_values_through_phis() {
        let p = build_words(
            &[
                0xBE80_0380, // s_mov_b32 s0, 0
                0x8000_8100, // 4: s_add_u32 s0, s0, 1
                0xBF07_8400, // s_cmp_lg_u32 s0, 4
                0xBF85_FFFD, // s_cbranch_scc1 -> 4
                S_ENDPGM,
            ],
            true,
        );
        assert!(p.inputs.is_empty(), "{:?}", p.inputs);
        let [b0, b1, b2] = &p.blocks[..] else {
            panic!()
        };
        assert_eq!(b1.preds, [0, 1]);
        assert_eq!(
            b1.exit,
            Exit::Branch {
                taken: 1,
                not_taken: 2
            }
        );
        let [phi] = &b1.phis[..] else {
            panic!("{:?}", b1.phis)
        };
        let add = &b1.insts[0];
        assert_eq!(phi.reg, Reg::S(0));
        assert_eq!(
            phi.args,
            [
                b0.insts[0].write(Reg::S(0)).unwrap(),
                add.write(Reg::S(0)).unwrap()
            ]
        );
        assert_eq!(add.read(Reg::S(0)), Some(phi.value));
        assert_eq!(b1.insts[1].read(Reg::S(0)), add.write(Reg::S(0)));
        assert_eq!(b2.insts[0].name(), "s_endpgm");
    }

    #[test]
    fn drops_scheduling_and_follows_far_jumps() {
        let p = build_words(
            &[
                0xBF80_0000, // s_nop 0
                0xBE88_1F00, // s_getpc_b64 s[8:9]
                0x8008_FF08,
                20,          // s_add_u32 s8, s8, 20  -> 8 + 20 = 28
                0x8209_8009, // s_addc_u32 s9, s9, 0
                0xBE80_2008, // s_setpc_b64 s[8:9]
                0xFFFF_FFFF, // not code
                S_ENDPGM,
            ],
            false,
        );
        let names: Vec<_> = p.blocks[0].insts.iter().map(Inst::name).collect();
        assert_eq!(
            names,
            ["s_getpc_b64", "s_add_u32", "s_addc_u32", "s_setpc_b64"]
        );
        assert_eq!(p.blocks[0].exit, Exit::Jump(1));
        assert_eq!(p.blocks[1].offset, 28);
        // s_addc_u32 reads the carry s_add_u32 left in SCC.
        let i = &p.blocks[0].insts;
        assert_eq!(i[2].read(Reg::Scc), i[1].write(Reg::Scc));
    }

    #[test]
    fn reports_what_it_cannot_express() {
        let code: Vec<u8> = [0xBE80_2E01_u32, S_ENDPGM] // s_movrels_b32 s0, s1
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        assert_eq!(
            build(&code, false, 256),
            Err(ShaderError::Unsupported {
                offset: 0,
                what: "s_movrels_b32"
            })
        );
    }

    #[test]
    fn relative_vgpr_access_spans_up_to_the_vgpr_count() {
        let code: Vec<u8> = [
            0x7E04_8501_u32, // v_movreld_b32 v2, v1: v[2 + m0] = v1
            0x7E00_8704,     // v_movrels_b32 v0, v4: v0 = v[4 + m0]
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = build(&code, false, 6).unwrap();
        let [movreld, movrels, _] = &p.blocks[0].insts[..] else {
            panic!()
        };
        let vgprs = |regs: &[(Reg, Value)]| -> Vec<u8> {
            regs.iter()
                .filter_map(|r| match r.0 {
                    Reg::V(v) => Some(v),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(vgprs(&movreld.writes), [2, 3, 4, 5]);
        // v1 is the source; v2-v5 are read for the lanes and indices not written.
        assert_eq!(vgprs(&movreld.reads), [1, 2, 3, 4, 5]);
        assert!(movreld.read(Reg::S(M0)).is_some());
        assert_eq!(vgprs(&movrels.writes), [0]);
        assert_eq!(vgprs(&movrels.reads), [4, 5, 0]);
        assert_eq!(movrels.read(Reg::V(4)), movreld.write(Reg::V(4)));
        assert!(build(&code, false, 4).is_err(), "v4 is past 4 VGPRs");
    }

    const fn s_branch(simm16: i16) -> u32 {
        0xBF82_0000 | simm16 as u16 as u32
    }

    const fn s_cbranch_scc1(simm16: i16) -> u32 {
        0xBF85_0000 | simm16 as u16 as u32
    }

    #[test]
    fn reverse_order_diamond_chain_eliminates_trivial_phis() {
        const LAYERS: usize = 4096;
        let mut reverse = vec![
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBE82_0382, // s_mov_b32 s2, 2
            s_branch(((LAYERS - 1) * 4) as i16),
        ];
        for k in 0..LAYERS {
            reverse.extend([
                0xBF06_0201, // s_cmp_eq_u32 s1, s2
                s_cbranch_scc1(1),
                0xBF80_0000, // s_nop 0
                if k == 0 {
                    s_branch((LAYERS * 4 - 4) as i16)
                } else {
                    s_branch(-8)
                },
            ]);
        }
        reverse.extend([0x8000_0201, S_ENDPGM]); // s_add_u32 s0, s1, s2

        let mut forward = vec![
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBE82_0382, // s_mov_b32 s2, 2
            s_branch(0),
        ];
        for _ in 0..LAYERS {
            forward.extend([
                0xBF06_0201, // s_cmp_eq_u32 s1, s2
                s_cbranch_scc1(1),
                0xBF80_0000, // s_nop 0
                s_branch(0),
            ]);
        }
        forward.extend([0x8000_0201, S_ENDPGM]); // s_add_u32 s0, s1, s2

        let rev = build_words(&reverse, false);
        let fwd = build_words(&forward, false);
        assert!(rev.inputs.is_empty(), "{:?}", rev.inputs);
        assert!(fwd.inputs.is_empty(), "{:?}", fwd.inputs);
        assert!(rev.blocks.iter().all(|b| b.phis.is_empty()));
        assert!(fwd.blocks.iter().all(|b| b.phis.is_empty()));
        assert_eq!(rev.blocks.len(), fwd.blocks.len());

        let rev_s1 = rev.blocks[0].insts[0].write(Reg::S(1)).unwrap();
        let rev_s2 = rev.blocks[0].insts[1].write(Reg::S(2)).unwrap();
        let rev_exit = rev.blocks.last().unwrap();
        assert_eq!(rev_exit.insts[0].name(), "s_add_u32");
        assert_eq!(rev_exit.insts[0].read(Reg::S(1)), Some(rev_s1));
        assert_eq!(rev_exit.insts[0].read(Reg::S(2)), Some(rev_s2));

        let fwd_s1 = fwd.blocks[0].insts[0].write(Reg::S(1)).unwrap();
        let fwd_s2 = fwd.blocks[0].insts[1].write(Reg::S(2)).unwrap();
        let fwd_exit = fwd.blocks.last().unwrap();
        assert_eq!(fwd_exit.insts[0].name(), "s_add_u32");
        assert_eq!(fwd_exit.insts[0].read(Reg::S(1)), Some(fwd_s1));
        assert_eq!(fwd_exit.insts[0].read(Reg::S(2)), Some(fwd_s2));
    }

    #[test]
    fn remove_trivial_phis_compresses_reverse_chain_in_bounded_steps() {
        const CHAIN: usize = 4096;
        let mut template = build_words(&[0xBE80_0380, S_ENDPGM], false).blocks[0].insts[0].clone();
        template.writes.clear();
        let mut blocks = Vec::with_capacity(CHAIN);
        for k in 0..CHAIN {
            let v1 = Value(2 + 2 * k as u32);
            let v2 = Value(3 + 2 * k as u32);
            let (n1, n2) = if k + 1 < CHAIN {
                (Value(2 + 2 * (k + 1) as u32), Value(3 + 2 * (k + 1) as u32))
            } else {
                (Value(0), Value(1))
            };
            let mut inst = template.clone();
            inst.reads = vec![(Reg::S(1), v1), (Reg::S(2), v2)];
            blocks.push(Block {
                offset: k * 4,
                preds: vec![],
                phis: vec![
                    Phi {
                        reg: Reg::S(1),
                        value: v1,
                        args: vec![n1, n1],
                    },
                    Phi {
                        reg: Reg::S(2),
                        value: v2,
                        args: vec![n2, n2],
                    },
                ],
                insts: vec![inst],
                exit: Exit::End,
            });
        }
        let steps = remove_trivial_phis_steps(&mut blocks, 2 + 2 * CHAIN as u32);
        assert!(blocks.iter().all(|b| b.phis.is_empty()));
        assert!(
            blocks
                .iter()
                .all(|b| b.insts[0].reads == [(Reg::S(1), Value(0)), (Reg::S(2), Value(1))])
        );
        assert!(steps < 100_000, "steps = {steps}");
    }

    #[test]
    fn non_trivial_phis_preserve_compressed_representatives() {
        const LAYERS: usize = 4;
        let mut code = vec![
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBE82_0382, // s_mov_b32 s2, 2
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            s_cbranch_scc1((2 + (LAYERS - 1) * 4) as i16),
            0xBE81_0383, // s_mov_b32 s1, 3
            s_branch((LAYERS * 4) as i16),
        ];
        for k in 0..LAYERS {
            code.extend([
                0xBF06_0201, // s_cmp_eq_u32 s1, s2
                s_cbranch_scc1(1),
                0xBF80_0000, // s_nop 0
                if k == 0 {
                    s_branch((LAYERS * 4 - 4) as i16)
                } else {
                    s_branch(-8)
                },
            ]);
        }
        code.extend([0x8000_0201, S_ENDPGM]); // s_add_u32 s0, s1, s2

        let p = build_words(&code, false);
        let entry_s1 = p.blocks[0].insts[0].write(Reg::S(1)).unwrap();
        let entry_s2 = p.blocks[0].insts[1].write(Reg::S(2)).unwrap();
        let arm_s1 = p.blocks[1].insts[0].write(Reg::S(1)).unwrap();
        let merge = p.blocks.last().unwrap();
        for b in &p.blocks[..p.blocks.len() - 1] {
            assert!(b.phis.is_empty(), "{:?}", b.phis);
        }
        let [phi] = &merge.phis[..] else {
            panic!("{:?}", merge.phis)
        };
        assert_eq!(phi.reg, Reg::S(1));
        assert_eq!(phi.args, [arm_s1, entry_s1]);
        assert_eq!(merge.insts[0].read(Reg::S(1)), Some(phi.value));
        assert_eq!(merge.insts[0].read(Reg::S(2)), Some(entry_s2));

        let mut loop_code = vec![
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBE82_0382, // s_mov_b32 s2, 2
            s_branch(((LAYERS - 1) * 4) as i16),
        ];
        for k in 0..LAYERS {
            loop_code.extend([
                0xBF06_0201, // s_cmp_eq_u32 s1, s2
                s_cbranch_scc1(1),
                0xBF80_0000, // s_nop 0
                if k == 0 {
                    s_branch((LAYERS * 4 - 4) as i16)
                } else {
                    s_branch(-8)
                },
            ]);
        }
        loop_code.extend([
            0x8001_0201, // s_add_u32 s1, s1, s2
            0xBF07_8401, // s_cmp_lg_u32 s1, 4
            s_cbranch_scc1(-3),
            0x8000_0201, // s_add_u32 s0, s1, s2
            S_ENDPGM,
        ]);

        let lp = build_words(&loop_code, false);
        let lp_s1 = lp.blocks[0].insts[0].write(Reg::S(1)).unwrap();
        let lp_s2 = lp.blocks[0].insts[1].write(Reg::S(2)).unwrap();
        let header = &lp.blocks[lp.blocks.len() - 2];
        let exit = lp.blocks.last().unwrap();
        for b in &lp.blocks[..lp.blocks.len() - 2] {
            assert!(b.phis.is_empty(), "{:?}", b.phis);
        }
        assert!(exit.phis.is_empty(), "{:?}", exit.phis);
        let [loop_phi] = &header.phis[..] else {
            panic!("{:?}", header.phis)
        };
        let next_s1 = header.insts[0].write(Reg::S(1)).unwrap();
        assert_eq!(loop_phi.reg, Reg::S(1));
        assert_eq!(loop_phi.args, [lp_s1, next_s1]);
        assert_eq!(header.insts[0].read(Reg::S(1)), Some(loop_phi.value));
        assert_eq!(header.insts[0].read(Reg::S(2)), Some(lp_s2));
        assert_eq!(exit.insts[0].read(Reg::S(1)), Some(next_s1));
        assert_eq!(exit.insts[0].read(Reg::S(2)), Some(lp_s2));
    }
}
