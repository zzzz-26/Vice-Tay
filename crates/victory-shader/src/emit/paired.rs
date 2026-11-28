use super::*;

pub(in super::super) fn accesses_lds(inst: &Inst) -> bool {
    let name = inst.name();
    matches!(inst.fields, Fields::Ds { .. })
        && !name.starts_with("ds_swizzle")
        && !name.contains("permute")
}

pub(in super::super) fn supports_paired_lds(name: &str) -> bool {
    matches!(name, "ds_read_b32" | "ds_write_b32" | "ds_read2_b32" | "ds_write2_b32") || paired_rmw(name).is_some()
}

/// The LDS atomics without a returned value the paired form takes: each lane's is a host
/// atomic of its own, and these integer operations give the same dword in any order
/// (`ds_add_u32`: the depth scan's counters, CS_ScanMip0..3).
fn paired_rmw(name: &str) -> Option<Rmw> {
    Some(match name {
        "ds_min_u32" => Rmw::UMin,
        "ds_max_u32" => Rmw::UMax,
        "ds_min_i32" => Rmw::Min,
        "ds_max_i32" => Rmw::Max,
        "ds_add_u32" => Rmw::Add,
        "ds_sub_u32" => Rmw::Sub,
        "ds_and_b32" => Rmw::And,
        "ds_or_b32" => Rmw::Or,
        "ds_xor_b32" => Rmw::Xor,
        _ => return None,
    })
}

impl<'a> Lower<'a> {
    pub(super) fn paired_lds(&mut self, inst: &Inst, address: V, guard: Option<V>, dwords: u32) -> Result<(), Why> {
        let Fields::Ds { dst, data, offset0, offset1, .. } = &inst.fields else {
            return Err("a paired LDS operation");
        };
        let name = inst.name();
        if !supports_paired_lds(name) {
            return Err("an LDS operation not supported by paired Wave64");
        }
        let pair = matches!(name, "ds_read2_b32" | "ds_write2_b32");
        let offsets = if pair {
            [u32::from(*offset0) * 4, u32::from(*offset1) * 4]
        } else {
            [u32::from(*offset0) | u32::from(*offset1) << 8, 0]
        };
        let read = name.starts_with("ds_read");
        let rmw = paired_rmw(name);
        let dst = if read {
            let Some(Operand::Vector { reg, .. }) = dst else {
                return Err("an LDS result");
            };
            Some(*reg)
        } else {
            None
        };
        for (n, offset) in offsets.into_iter().take(if pair { 2 } else { 1 }).enumerate() {
            let value = if read { None } else {
                Some(self.vgprs(inst, data.get(n).ok_or("an LDS write")?, 1)?[0])
            };
            let c = self.cur;
            let off = self.s.m.i32(offset);
            let at = self.s.m.bin(c, Bin::Add, address, off);
            let two = self.s.m.i32(2);
            let index = self.s.m.bin(c, Bin::LShr, at, two);
            let bound = self.s.m.i32(dwords);
            let inside = self.s.m.cmp(c, pred::IULT, index, bound);
            let on = guard.map_or(inside, |g| self.s.m.bin(c, Bin::And, g, inside));
            let join = self.begin_guarded(Some(on));
            let lds = self.s.lds();
            let ptr = self.s.m.element(self.cur, lds, index);
            if let Some(value) = value {
                if let Some(op) = rmw {
                    self.s.m.atomic(self.cur, op, ptr, value);
                } else {
                    self.s.m.store(self.cur, ptr, value);
                }
                self.end_guarded(join, &[]);
            } else {
                let result = self.s.m.load(self.cur, ptr);
                let zero = self.s.m.i32(0);
                let result = self.end_guarded(join, &[(result, zero)])[0];
                self.write(inst, Reg::V(dst.ok_or("an LDS result")? + n as u8), result, guard);
            }
        }
        Ok(())
    }

    /// The paired form: makes the other console lane of this thread the one being lowered.
    pub(super) fn swap_lanes(&mut self) {
        let Some(t) = self.twin.as_deref_mut() else {
            return;
        };
        std::mem::swap(&mut self.vals, &mut t.vals);
        std::mem::swap(&mut self.floats, &mut t.floats);
        std::mem::swap(&mut self.bools, &mut t.bools);
        std::mem::swap(&mut self.lane_bits, &mut t.lane_bits);
        std::mem::swap(&mut self.lane_fail, &mut t.lane_fail);
        std::mem::swap(&mut self.mask_phis, &mut t.mask_phis);
        std::mem::swap(&mut self.pending_masks, &mut t.pending_masks);
        std::mem::swap(&mut self.data_phis, &mut t.data_phis);
        std::mem::swap(&mut self.handles, &mut t.handles);
        std::mem::swap(&mut self.strides, &mut t.strides);
        std::mem::swap(&mut self.clane, &mut t.clane);
        self.upper = !self.upper;
    }

    /// The paired form: `f` for each console lane of this thread, the lower one first; the lane
    /// being lowered stays the same.
    pub(super) fn both<T>(
        &mut self,
        mut f: impl FnMut(&mut Self) -> Result<T, Why>,
    ) -> Result<[T; 2], Why> {
        let was = self.upper;
        if was {
            self.swap_lanes();
        }
        let lo = f(self);
        self.swap_lanes();
        let hi = f(self);
        if !was {
            self.swap_lanes();
        }
        Ok([lo?, hi?])
    }

    /// The paired form: `lo` as the lower lane's value of `v`, `hi` as the upper lane's.
    fn put_lanes(&mut self, v: Value, lo: Val, hi: Val) {
        let (mine, other) = if self.upper { (hi, lo) } else { (lo, hi) };
        self.vals.insert(v, mine);
        if let Some(t) = self.twin.as_deref_mut() {
            t.vals.insert(v, other);
        }
    }

    /// Instruction `i` of IR block `b` in the paired form ([`with_paired_lanes`]), for both
    /// console lanes of the thread: what a lane does is lowered for the lower lane, then for
    /// the upper one, which keeps the console's order of the wave's instructions; scalar
    /// control (the title's barrier) and reads of other lanes once
    /// with what both lanes hold ([`Self::paired_cross`]). An LDS access that follows another
    /// gets a fence first unless both read: what one lane wrote (or still read) must not move
    /// past another lane's access the console made after it.
    pub(super) fn paired_inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        if matches!(inst.fields, Fields::Ds { gds: true, .. }) {
            return Err(unsupported(inst.machine.offset, "GDS"));
        }
        let name = inst.name();
        if matches!(inst.fields, Fields::Smem { .. })
            && (name.contains("store") || name.contains("atomic"))
        {
            return Err(unsupported(
                inst.machine.offset,
                "a scalar memory write in paired Wave64",
            ));
        }
        if matches!(inst.fields, Fields::Sopp { .. }) {
            self.inst(b, i)?;
            self.mirror(inst);
            return Ok(());
        }
        let cross = match name {
            "v_readlane_b32" | "ds_bpermute_b32" => true,
            // (not a waterfall loop's read, the lane's own dword: `Self::valu`)
            "v_readfirstlane_b32" => !inst
                .writes
                .first()
                .is_some_and(|w| self.x.res.waterfall.reads.contains(&w.1)),
            _ => false,
        };
        if cross {
            if let Err(why) = self.paired_cross(inst, b, i) {
                for &(_, w) in &inst.writes {
                    self.put_lanes(w, Val::Missing(why), Val::Missing(why));
                }
            }
            return Ok(());
        }
        if accesses_lds(inst) {
            let write = !name.starts_with("ds_read");
            if write || self.lds_last != Some(false) {
                self.group_sync(self.cur);
            }
            self.lds_last = Some(write);
        }
        self.inst(b, i)?;
        self.swap_lanes();
        let r = self.inst(b, i);
        self.swap_lanes();
        r
    }

    /// The paired form: what `inst` wrote in the lane being lowered, for the other lane too (a
    /// scalar result, the same in both).
    fn mirror(&mut self, inst: &Inst) {
        let Some(t) = self.twin.as_deref_mut() else {
            return;
        };
        for &(_, w) in &inst.writes {
            if let Some(&v) = self.vals.get(&w) {
                t.vals.insert(w, v);
            }
        }
    }

    /// The paired form: an instruction that reads other lanes of the console wave, any of its
    /// 64: `v_readlane_b32` and `v_readfirstlane_b32` (scalar results, the same in both lanes of
    /// the thread). `ds_bpermute_b32` stays within its own 32-lane half on GFX10.
    /// Lane n and lane n + 32 are the lower and the upper lane of host lane n.
    fn paired_cross(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        self.cross_lane(false)?;
        let c = self.cur;
        let (k31, k32, z) = (self.s.m.i32(31), self.s.m.i32(32), self.s.m.i32(0));
        // a lane's value from the lower or upper value of the host lane `at` (of 0..63)
        let pick = |l: &mut Self, lo: V, hi: V, at: V| {
            let host = l.s.m.bin(c, Bin::And, at, k31);
            let upper = l.s.m.bin(c, Bin::And, at, k32);
            let upper = l.s.m.cmp(c, pred::NE, upper, z);
            let (lo, hi) = (l.read_lane(lo, host), l.read_lane(hi, host));
            l.s.m.select(c, upper, hi, lo)
        };
        match inst.name() {
            "v_readlane_b32" => {
                // the lane: a scalar or a constant
                if inst.reads.iter().any(|r| self.lane_scalars.contains(&r.1)) {
                    return Err("a paired lane read with a lane-varying scalar index");
                }
                let lane = self.src(inst, 1)?;
                let [lo, hi] = self.both(|l| l.src(inst, 0))?;
                let r = pick(self, lo, hi, lane);
                self.write_i(inst, r, None)?;
            }
            "v_readfirstlane_b32" => {
                let [(lo, in_lo), (hi, in_hi)] = self.both(|l| {
                    let x = l.src(inst, 0)?;
                    let guard = l.guard(b, i)?;
                    Ok(l.first_lane(x, guard))
                })?;
                // the first lane in EXEC is a lower one if there is one; with none, lane 0
                let hi = self.s.m.select(c, in_hi, hi, lo);
                let r = self.s.m.select(c, in_lo, lo, hi);
                self.write_i(inst, r, None)?;
            }
            _ => {
                // GFX10 bpermute addresses only the destination lane's own half.
                // An inactive source lane supplies zero.
                let Fields::Ds {
                    dst: Some(Operand::Vector { reg, .. }),
                    addr: Some(addr),
                    data,
                    offset0,
                    offset1,
                    ..
                } = &inst.fields
                else {
                    return Err("a permute's operands");
                };
                let data = data.first().ok_or("a permute source")?;
                let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
                let [(from_lo, x_lo, g_lo), (from_hi, x_hi, g_hi)] = self.both(|l| {
                    let at = l.vgprs(inst, addr, 1)?[0];
                    let x = l.vgprs(inst, data, 1)?[0];
                    let guard = l.guard(b, i)?;
                    let (o, two) = (l.s.m.i32(offset), l.s.m.i32(2));
                    let at = l.s.m.bin(c, Bin::Add, at, o);
                    let from = l.s.m.bin(c, Bin::LShr, at, two);
                    let x = match guard {
                        Some(g) => l.s.m.select(c, g, x, z),
                        None => x,
                    };
                    Ok((from, x, guard))
                })?;
                let from_lo = self.s.m.bin(c, Bin::And, from_lo, k31);
                let from_hi = self.s.m.bin(c, Bin::And, from_hi, k31);
                let read = [self.read_lane(x_lo, from_lo), self.read_lane(x_hi, from_hi)];
                let guards = [g_lo, g_hi];
                let reg = *reg;
                self.both(|l| {
                    let n = usize::from(l.upper);
                    l.write(inst, Reg::V(reg), read[n], guards[n]);
                    Ok(())
                })?;
                return Ok(());
            }
        }
        self.mirror(inst);
        Ok(())
    }

    /// [`Self::section_inst`] in the paired form: a value over the wave's lanes takes both
    /// lanes of every thread ([`Self::paired_shape`]), a lane's own value each lane's.
    pub(super) fn paired_section_inst(&mut self, b: usize, i: usize, start: usize) {
        use super::super::wave::Shape;
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        for &(_, w) in &inst.writes {
            let shape = match self.section_out.contains(&w) {
                true => self.sections[&b][&start].shapes.get(&w).copied(),
                false => None,
            };
            // A value that leaves the section with no shape but that nothing after it sees
            // (a reduction's scratch partial sums, which the next tree overwrites in every
            // lane before reading: [`super::paired_fp32::unseen_after`]): any value will do.
            // This line's IR names a register's old value as read by every write to it, so
            // without this the next section's replay reads a missing value (5dc3b25986622949:
            // the FP64 count's partial sums in v2-v5, then the FP32 count s15).
            if shape.is_none()
                && self.section_out.contains(&w)
                && let Some(&(reg, _)) = inst.writes.iter().find(|x| x.1 == w)
                && super::paired_fp32::unseen_after(
                    self.x.p,
                    &self.sections,
                    (b, start, self.sections[&b][&start].end),
                    reg,
                    w,
                )
            {
                let z = self.s.m.i32(0);
                self.put_lanes(w, Val::I32(z), Val::I32(z));
                continue;
            }
            if shape.is_none() && self.section_out.contains(&w) && std::env::var_os("VICTORY_WAVE_WHY").is_some() {
                eprintln!(
                    "WAVE64-PAIRS {} at {:#x} leaves {w:?} unknown and seen later",
                    inst.name(),
                    inst.machine.offset
                );
            }
            let made = match shape {
                None => Err(UNKNOWN),
                Some(Shape::ActiveF64(lo, hi, mask, high)) => {
                    self.paired_f64_constant_sum(b, start, lo, hi, mask, high)
                }
                Some(sh @ (Shape::Active(..) | Shape::Prefix(..))) => self.paired_shape(sh),
                Some(sh) => self.both(|l| l.shape(sh, false)),
            };
            match made {
                Ok([lo, hi]) => self.put_lanes(w, Val::I32(lo), Val::I32(hi)),
                Err(why) => self.put_lanes(w, Val::Missing(why), Val::Missing(why)),
            }
        }
    }

    /// A section's 64-bit float sum over the console wave whose every term is one constant
    /// with exact sums ([`super::paired_fp32::exact_f64_term`]): the number of lanes in the
    /// mask, both lanes of every thread, times the constant; +0.0 for none. Any other paired
    /// FP64 sum is refused: its additions round, so their order counts.
    fn paired_f64_constant_sum(
        &mut self,
        b: usize,
        start: usize,
        lo: Value,
        hi: Value,
        mask: Value,
        high: bool,
    ) -> Result<[V; 2], Why> {
        use super::super::wave::Op;
        let term = super::paired_fp32::exact_f64_term(self.x.p, b, start, lo, hi)
            .ok_or("a paired FP64 reduction")?;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let c = self.cur;
        let [lo_on, hi_on] = self.both(|l| {
            let on = l.boolean(m, c)?;
            let (one, zero) = (l.s.m.i32(1), l.s.m.i32(0));
            Ok(l.s.m.select(c, on, one, zero))
        })?;
        self.s.flags |= flags::WAVE_OPS;
        let lo_n = self.host_wave_op(Op::AddU, lo_on, false, None)?;
        let hi_n = self.host_wave_op(Op::AddU, hi_on, false, None)?;
        let n = self.s.m.bin(c, Bin::Add, lo_n, hi_n);
        let zero = self.s.m.i32(0);
        if term == 0 {
            return Ok([zero, zero]);
        }
        self.s.flags |= flags::DOUBLES;
        let f64t = self.s.m.double();
        let i32t = self.s.t.i32;
        let split_ty = self
            .s
            .m
            .named_struct("dx.types.splitdouble", vec![i32t, i32t]);
        let (klo, khi) = (self.s.m.i32(term as u32), self.s.m.i32((term >> 32) as u32));
        let k = self
            .s
            .op(c, "makeDouble.f64", Attr::ReadNone, f64t, 101, &[klo, khi]);
        let f32t = self.s.t.f32;
        let count = self.s.m.cast(c, Cast::SiToFp, n, f32t);
        let count = self.s.m.cast(c, Cast::FpExt, count, f64t);
        // (exact: n is at most 64 and the term has at most 47 significant bits)
        let sum = self.s.m.bin(c, Bin::Mul, count, k);
        let split = self
            .s
            .op(c, "splitDouble.f64", Attr::ReadNone, split_ty, 102, &[sum]);
        let word = self.s.m.extract(c, split, u32::from(high));
        let none = self.s.m.cmp(c, pred::EQ, n, zero);
        let r = self.s.m.select(c, none, zero, word);
        Ok([r, r])
    }

    /// The first `steps` levels of [`Self::half_sum`]'s tree over this host wave: its lanes'
    /// sum when they are all of its waves' lanes (6 steps for 64, 5 for 32), in the low half;
    /// the high half lane 0's.
    pub(super) fn half_tree(&mut self, x: V, steps: u32) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let lane = self
            .s
            .op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let mut sum = x;
        for step in 0..steps {
            let bit = self.s.m.i32(1 << step);
            let partner = self.s.m.bin(c, Bin::Xor, lane, bit);
            let theirs = self.read_lane(sum, partner);
            sum = self.add_halves(sum, theirs);
        }
        let (high, low, zero) = (
            self.s.m.i32(0xFFFF_0000),
            self.s.m.i32(0xFFFF),
            self.s.m.i32(0),
        );
        let first = self.read_lane(x, zero);
        let first = self.s.m.bin(c, Bin::And, first, high);
        let sum = self.s.m.bin(c, Bin::And, sum, low);
        self.s.m.bin(c, Bin::Or, first, sum)
    }

    /// The last addition of [`Self::half_sum`]'s tree: the sums of a console wave's two halves
    /// (from [`Self::half_tree`]), with the high half of the lower one's (lane 0's).
    pub(super) fn join_half_sums(&mut self, lo: V, hi: V) -> V {
        let c = self.cur;
        let total = self.add_halves(lo, hi);
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let total = self.s.m.bin(c, Bin::And, total, low);
        let first = self.s.m.bin(c, Bin::And, lo, high);
        self.s.m.bin(c, Bin::Or, first, total)
    }

    fn host_wave_op(
        &mut self,
        op: super::super::wave::Op,
        arg: V,
        float: bool,
        prefix: Option<bool>,
    ) -> Result<V, Why> {
        use super::super::wave::Op;
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        Ok(match (op, prefix) {
            (Op::AddU | Op::AddF, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    &format!("wavePrefixOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    121,
                    &[arg, k0, k1],
                );
                if inclusive {
                    self.s.m.bin(c, Bin::Add, r, arg)
                } else {
                    r
                }
            }
            (_, Some(_)) => return Err("a wave prefix of an operation other than a sum"),
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s
                    .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, k, u],
                )
            }
        })
    }

    /// The results of a reduction by `op` (`float`: floats, else bits) over the two halves of
    /// a console wave, joined into the wave's.
    fn join_halves(&mut self, op: super::super::wave::Op, lo: V, hi: V, float: bool) -> V {
        use super::super::wave::Op;
        let c = self.cur;
        let ty = if float { self.s.t.f32 } else { self.s.t.i32 };
        let sfx = if float { "f32" } else { "i32" };
        match op {
            Op::AddU | Op::AddF => self.s.m.bin(c, Bin::Add, lo, hi),
            Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
            Op::And => self.s.m.bin(c, Bin::And, lo, hi),
            Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
            _ => {
                // FMax 35, FMin 36, IMax 37, IMin 38, UMax 39, UMin 40
                let code = match op {
                    Op::MaxF => 35,
                    Op::MinF => 36,
                    Op::MinU => 40,
                    Op::MaxU => 39,
                    Op::MinI => 38,
                    _ => 37,
                };
                self.s.op(
                    c,
                    &format!("binary.{sfx}"),
                    Attr::ReadNone,
                    ty,
                    code,
                    &[lo, hi],
                )
            }
        }
    }

    /// [`Self::shape`] of a reduction or prefix in the paired form, over the console wave's 64
    /// lanes: for the lower and the upper lane of this thread. Each half is a host wave
    /// operation over the lower or the upper lanes, and the halves join as two host waves' do
    /// in `shape`: a reduction of the two results, a prefix of the upper lanes after the lower
    /// lanes' total.
    fn paired_shape(&mut self, shape: super::super::wave::Shape) -> Result<[V; 2], Why> {
        use super::super::wave::{Op, Shape};
        let (op, src, mask, ident, prefix) = match shape {
            Shape::ActiveF64(..) => return Err("a paired FP64 reduction"),
            Shape::Prefix(Op::AddF | Op::AddH | Op::MinF | Op::MaxF, ..) => {
                return Err("a paired floating-point prefix");
            }
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
            _ => return Err("a whole-wave section value of another shape"),
        };
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let c = self.cur;
        let [lo, hi] = self.both(|l| {
            let on = l.boolean(m, c)?;
            let x = l.i32_of(src)?;
            let id = l.s.m.i32(ident);
            Ok(l.s.m.select(c, on, x, id))
        })?;
        self.s.flags |= flags::WAVE_OPS;
        if op == Op::AddH {
            if prefix.is_some() {
                return Err("a wave prefix of an operation other than a sum");
            }
            let (lo, hi) = (self.half_tree(lo, 5), self.half_tree(hi, 5));
            let r = self.join_half_sums(lo, hi);
            return Ok([r, r]);
        }
        if matches!(op, Op::AddF | Op::MinF | Op::MaxF) {
            return Err("a floating-point reduction without an exact paired tree");
        }
        let r = match prefix {
            Some(inclusive) => {
                let mine = self.host_wave_op(op, lo, false, Some(inclusive))?;
                let theirs = self.host_wave_op(op, hi, false, Some(inclusive))?;
                let below = self.host_wave_op(op, lo, false, None)?;
                [mine, self.s.m.bin(c, Bin::Add, theirs, below)]
            }
            None => {
                let lo = self.host_wave_op(op, lo, false, None)?;
                let hi = self.host_wave_op(op, hi, false, None)?;
                let r = self.join_halves(op, lo, hi, false);
                [r, r]
            }
        };
        Ok(r)
    }
}
